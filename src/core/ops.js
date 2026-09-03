// The numeric kernels the forward pass is built from.
//
// Storage is f32 throughout, matching how the weights arrive and how the GPU
// backend will hold them, but every reduction accumulates in a JavaScript
// number, which is f64. That is deliberate: this is the reference the GPU gets
// compared against, so it should be the more accurate of the two. A dot product
// over 896 terms accumulated in f32 carries about sqrt(896) * 2^-24 of relative
// error, and there is no reason to inherit that here when f64 is free.

/**
 * Root-mean-square normalisation, then a per-channel gain.
 *
 * Qwen2 uses this in place of LayerNorm: no mean subtraction and no bias, so
 * the only statistic is the mean square. The epsilon is inside the square root,
 * which matters -- putting it outside changes the result for small inputs.
 */
export function rmsNorm(x, weight, eps, out = new Float32Array(x.length)) {
  const n = x.length;
  let sumSquares = 0;
  for (let i = 0; i < n; i++) sumSquares += x[i] * x[i];
  const scale = 1 / Math.sqrt(sumSquares / n + eps);
  for (let i = 0; i < n; i++) out[i] = x[i] * scale * weight[i];
  return out;
}

/**
 * out = W x + bias, where W is [rows, cols] stored row-major.
 *
 * This is the shape the weights already have: a linear layer's weight is
 * [out_features, in_features], so row r holds every input weight for output r
 * contiguously and each output is one contiguous dot product.
 */
export function matVec(W, x, rows, cols, bias = null, out = new Float32Array(rows)) {
  for (let r = 0; r < rows; r++) {
    const base = r * cols;
    let acc = bias ? bias[r] : 0;
    for (let c = 0; c < cols; c++) acc += W[base + c] * x[c];
    out[r] = acc;
  }
  return out;
}

/**
 * As matVec, but W arrives as bfloat16 and is widened per element.
 *
 * Used by the streaming output projection, where materialising the widened
 * matrix would cost 544 MB for a saving of one shift per multiply.
 */
export function matVecBf16(Wbits, x, rows, cols, out = new Float32Array(rows)) {
  const conv = new ArrayBuffer(4);
  const asU32 = new Uint32Array(conv);
  const asF32 = new Float32Array(conv);
  for (let r = 0; r < rows; r++) {
    const base = r * cols;
    let acc = 0;
    for (let c = 0; c < cols; c++) {
      asU32[0] = Wbits[base + c] << 16;
      acc += asF32[0] * x[c];
    }
    out[r] = acc;
  }
  return out;
}

/** Softmax over x[0..n), in place, shifted by the max for stability. */
export function softmaxInPlace(x, n = x.length, offset = 0) {
  let max = -Infinity;
  for (let i = 0; i < n; i++) if (x[offset + i] > max) max = x[offset + i];
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const e = Math.exp(x[offset + i] - max);
    x[offset + i] = e;
    sum += e;
  }
  const inv = 1 / sum;
  for (let i = 0; i < n; i++) x[offset + i] *= inv;
  return x;
}

/** SiLU, also called swish: x * sigmoid(x). */
export function silu(x) {
  return x / (1 + Math.exp(-x));
}

/**
 * SwiGLU: silu(gate) * up, elementwise, in place over `gate`.
 */
export function swigluInPlace(gate, up) {
  for (let i = 0; i < gate.length; i++) gate[i] = silu(gate[i]) * up[i];
  return gate;
}

/**
 * Rotary position embedding frequencies for one head width.
 *
 * inv_freq[j] = theta^(-2j/headDim). Qwen2.5 uses theta = 1e6 rather than the
 * original 1e4, which is what lets it address 32768 positions.
 */
export function ropeFrequencies(headDim, theta) {
  const half = headDim >> 1;
  const inv = new Float64Array(half);
  for (let j = 0; j < half; j++) inv[j] = Math.pow(theta, (-2 * j) / headDim);
  return inv;
}

/**
 * Apply RoPE to one head vector, in place, at a given position.
 *
 * This is the half-split convention: element j pairs with element j + headDim/2,
 * not with its neighbour j+1. The interleaved convention is the other common
 * one and produces a different, wrong-looking-but-plausible model, so it is
 * worth being explicit. It follows from the reference implementation forming
 * cos/sin as concat(f, f) and rotating with concat(-x[half:], x[:half]).
 */
export function ropeInPlace(vec, offset, headDim, position, invFreq) {
  const half = headDim >> 1;
  for (let j = 0; j < half; j++) {
    const angle = position * invFreq[j];
    const cos = Math.cos(angle);
    const sin = Math.sin(angle);
    const lo = vec[offset + j];
    const hi = vec[offset + j + half];
    vec[offset + j] = lo * cos - hi * sin;
    vec[offset + j + half] = hi * cos + lo * sin;
  }
  return vec;
}
