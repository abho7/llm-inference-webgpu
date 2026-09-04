// Weight quantization.
//
// Two schemes, both weight-only: activations stay in f32. That is the usual
// arrangement for inference at batch one, where the weights are the thing being
// moved and the activations are a rounding error in the memory budget.
//
//   INT8, symmetric, per output channel
//     One scale per row of the weight matrix. Symmetric means no zero point:
//     the quantization grid is centred on zero, which costs half a level and
//     buys an inner loop with no offset in it.
//
//   INT4, asymmetric, per group along the input dimension
//     One scale and one zero point per run of `groupSize` weights within a
//     row. Four bits is too few for a symmetric grid to be usable, so this one
//     carries a zero point and spends its 16 levels on the actual range of the
//     group rather than on a range symmetric about zero.
//
// The granularity is the whole design. A single scale for a 151936 x 896 matrix
// would be set by its largest entry and would quantize everything else to
// nothing; a scale per row, or per group within a row, keeps the grid close to
// the values it has to represent. Finer granularity costs storage, and
// bitsPerWeight() reports what each scheme actually spends including it.

/** Round half away from zero, which is what quantization conventionally uses. */
function roundHalfAwayFromZero(x) {
  return x < 0 ? -Math.round(-x) : Math.round(x);
}

/**
 * Symmetric INT8, one scale per row.
 *
 * The grid is [-127, 127] rather than [-128, 127]: giving up the extra negative
 * level keeps it exactly symmetric, so that negating a weight negates its code.
 * A row that is entirely zero gets scale 0 and dequantizes back to zeros.
 */
export function quantizeInt8PerChannel(W, rows, cols) {
  const q = new Int8Array(rows * cols);
  const scales = new Float32Array(rows);

  for (let r = 0; r < rows; r++) {
    const base = r * cols;
    let peak = 0;
    for (let c = 0; c < cols; c++) {
      const magnitude = Math.abs(W[base + c]);
      if (magnitude > peak) peak = magnitude;
    }
    const scale = peak / 127;
    scales[r] = scale;
    if (scale === 0) continue;

    const inverse = 1 / scale;
    for (let c = 0; c < cols; c++) {
      const level = roundHalfAwayFromZero(W[base + c] * inverse);
      q[base + c] = Math.max(-127, Math.min(127, level));
    }
  }
  return { q, scales, rows, cols, scheme: 'int8-per-channel' };
}

export function dequantizeInt8PerChannel({ q, scales, rows, cols }, out) {
  const dst = out ?? new Float32Array(rows * cols);
  for (let r = 0; r < rows; r++) {
    const base = r * cols;
    const scale = scales[r];
    for (let c = 0; c < cols; c++) dst[base + c] = q[base + c] * scale;
  }
  return dst;
}

/**
 * Asymmetric INT4, one scale and zero point per group along a row.
 *
 * Codes are packed two per byte, low nibble first, because storing four-bit
 * values in bytes would spend twice the memory the scheme exists to save.
 *
 * `groupSize` must divide `cols`. Allowing a ragged final group would mean a
 * group whose scale is set by many fewer samples, which is a quiet accuracy
 * cliff rather than a convenience.
 */
export function quantizeInt4Group(W, rows, cols, groupSize = 128) {
  if (!Number.isInteger(groupSize) || groupSize < 2) {
    throw new RangeError(`groupSize must be an integer of at least 2, got ${groupSize}`);
  }
  if (cols % groupSize !== 0) {
    throw new RangeError(`groupSize ${groupSize} does not divide ${cols} columns`);
  }
  const groupsPerRow = cols / groupSize;
  const q = new Uint8Array((rows * cols) / 2);
  const scales = new Float32Array(rows * groupsPerRow);
  const zeros = new Float32Array(rows * groupsPerRow);

  for (let r = 0; r < rows; r++) {
    for (let g = 0; g < groupsPerRow; g++) {
      const base = r * cols + g * groupSize;
      let lo = Infinity;
      let hi = -Infinity;
      for (let i = 0; i < groupSize; i++) {
        const value = W[base + i];
        if (value < lo) lo = value;
        if (value > hi) hi = value;
      }
      // Include zero in the represented range. A group whose values are all
      // positive still has to be able to encode zero, or the dequantized
      // weights carry a bias.
      lo = Math.min(lo, 0);
      hi = Math.max(hi, 0);

      const scale = (hi - lo) / 15;
      const index = r * groupsPerRow + g;
      scales[index] = scale;
      if (scale === 0) {
        zeros[index] = 0;
        continue;
      }
      // Zero point kept as an integer level, so that the value zero is
      // represented exactly rather than to within half a step.
      const zero = Math.max(0, Math.min(15, roundHalfAwayFromZero(-lo / scale)));
      zeros[index] = zero;

      const inverse = 1 / scale;
      for (let i = 0; i < groupSize; i++) {
        const level = Math.max(0, Math.min(15,
          roundHalfAwayFromZero(W[base + i] * inverse) + zero));
        const at = base + i;
        const byte = at >> 1;
        if ((at & 1) === 0) q[byte] = (q[byte] & 0xf0) | level;
        else q[byte] = (q[byte] & 0x0f) | (level << 4);
      }
    }
  }
  return { q, scales, zeros, rows, cols, groupSize, scheme: `int4-g${groupSize}` };
}

export function dequantizeInt4Group({ q, scales, zeros, rows, cols, groupSize }, out) {
  const dst = out ?? new Float32Array(rows * cols);
  const groupsPerRow = cols / groupSize;
  for (let r = 0; r < rows; r++) {
    for (let g = 0; g < groupsPerRow; g++) {
      const base = r * cols + g * groupSize;
      const index = r * groupsPerRow + g;
      const scale = scales[index];
      const zero = zeros[index];
      for (let i = 0; i < groupSize; i++) {
        const at = base + i;
        const byte = q[at >> 1];
        const level = (at & 1) === 0 ? (byte & 0x0f) : (byte >> 4);
        dst[at] = (level - zero) * scale;
      }
    }
  }
  return dst;
}

/** Round-trip a matrix through a scheme, returning the dequantized weights. */
export function roundTrip(W, rows, cols, scheme, groupSize = 128) {
  if (scheme === 'int8') {
    return dequantizeInt8PerChannel(quantizeInt8PerChannel(W, rows, cols));
  }
  if (scheme === 'int4') {
    return dequantizeInt4Group(quantizeInt4Group(W, rows, cols, groupSize));
  }
  throw new Error(`unknown scheme ${scheme}`);
}

/**
 * Storage cost per weight, counting the scales and zero points.
 *
 * Quoting "4-bit" while carrying an f32 scale for every 128 weights is quoting
 * 4.25 bits, and the difference is the whole argument for a larger group.
 */
export function bitsPerWeight(scheme, cols, groupSize = 128) {
  if (scheme === 'int8') return 8 + 32 / cols;            // one f32 scale per row
  if (scheme === 'int4') return 4 + (32 + 32) / groupSize; // scale and zero per group
  if (scheme === 'f16' || scheme === 'bf16') return 16;
  if (scheme === 'f32') return 32;
  throw new Error(`unknown scheme ${scheme}`);
}

/**
 * How far a quantized matrix is from the original.
 *
 * Relative Frobenius error is the headline: it is scale-free and it is what the
 * literature reports. Max absolute error is kept alongside because Frobenius
 * error hides a single badly-mangled weight, and a single badly-mangled weight
 * in an outlier channel is exactly the failure mode worth catching.
 */
export function quantizationError(original, approx) {
  let sumSquaredError = 0;
  let sumSquared = 0;
  let maxAbs = 0;
  let at = -1;
  for (let i = 0; i < original.length; i++) {
    const diff = approx[i] - original[i];
    sumSquaredError += diff * diff;
    sumSquared += original[i] * original[i];
    const magnitude = Math.abs(diff);
    if (magnitude > maxAbs) { maxAbs = magnitude; at = i; }
  }
  return {
    frobenius: sumSquared > 0 ? Math.sqrt(sumSquaredError / sumSquared) : 0,
    rmse: Math.sqrt(sumSquaredError / original.length),
    maxAbs,
    at,
  };
}
