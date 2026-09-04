// The compute kernels, in WGSL.
//
// One kernel per operation in the forward pass, each a direct translation of
// the corresponding function in src/core/ops.js. They are written to be
// comparable to that code rather than to be fast: no fusion, no tiling, no
// clever memory layout. Phase 3's claim is that the GPU agrees with the CPU
// reference, and every optimisation added before that is established is a
// place for a discrepancy to hide.
//
// Two conventions throughout:
//
//   Weights are f16 in storage and widened to f32 on load; accumulation is
//   always f32. `enable f16` is what makes the storage declaration legal, and
//   it is why the engine requires the shader-f16 feature.
//
//   Row indices come from a two-dimensional workgroup grid, recovered as
//   x + y * gridWidth. The output projection has 151936 rows and
//   maxComputeWorkgroupsPerDimension is 65535, so one dimension is not enough.

const F16 = 'enable f16;\n';

/** Threads per workgroup for the reduction kernels. */
export const GROUP = 256;

/**
 * y = W x + bias, with W stored [rows, cols] row-major in f16.
 *
 * One workgroup per output row. Each thread strides over the row accumulating
 * in f32, then the workgroup reduces its partial sums in a tree. The tree is
 * why GPU and CPU results differ in the last bits: the same products are added
 * in a different order, and floating-point addition is not associative.
 */
export const MATVEC = `${F16}
struct Params { rows: u32, cols: u32, hasBias: u32, gridWidth: u32 };

@group(0) @binding(0) var<storage, read>       W      : array<f16>;
@group(0) @binding(1) var<storage, read>       x      : array<f32>;
@group(0) @binding(2) var<storage, read>       bias   : array<f32>;
@group(0) @binding(3) var<storage, read_write> y      : array<f32>;
@group(0) @binding(4) var<uniform>             params : Params;

var<workgroup> partial : array<f32, ${GROUP}>;

@compute @workgroup_size(${GROUP})
fn main(@builtin(workgroup_id) wg : vec3<u32>,
        @builtin(local_invocation_id) lid : vec3<u32>) {
  let row = wg.x + wg.y * params.gridWidth;
  // Uniform across the workgroup, so the barriers below are still uniform.
  if (row >= params.rows) { return; }

  let base = row * params.cols;
  var acc = 0.0;
  var c = lid.x;
  loop {
    if (c >= params.cols) { break; }
    acc = acc + f32(W[base + c]) * x[c];
    c = c + ${GROUP}u;
  }
  partial[lid.x] = acc;
  workgroupBarrier();

  var stride = ${GROUP / 2}u;
  loop {
    if (stride == 0u) { break; }
    if (lid.x < stride) { partial[lid.x] = partial[lid.x] + partial[lid.x + stride]; }
    workgroupBarrier();
    stride = stride >> 1u;
  }

  if (lid.x == 0u) {
    var v = partial[0];
    if (params.hasBias == 1u) { v = v + bias[row]; }
    y[row] = v;
  }
}`;

/**
 * RMS normalisation with a per-channel gain, one workgroup per row.
 *
 * The epsilon goes inside the square root, matching ops.js. Putting it outside
 * agrees for ordinary inputs and diverges for small ones, which is exactly the
 * kind of difference a tolerance-based comparison would not catch.
 */
export const RMSNORM = `${F16}
struct Params { rows: u32, cols: u32, eps: f32, pad: u32 };

@group(0) @binding(0) var<storage, read>       x      : array<f32>;
@group(0) @binding(1) var<storage, read>       gain   : array<f16>;
@group(0) @binding(2) var<storage, read_write> y      : array<f32>;
@group(0) @binding(3) var<uniform>             params : Params;

var<workgroup> partial : array<f32, ${GROUP}>;

@compute @workgroup_size(${GROUP})
fn main(@builtin(workgroup_id) wg : vec3<u32>,
        @builtin(local_invocation_id) lid : vec3<u32>) {
  let row = wg.x;
  if (row >= params.rows) { return; }
  let base = row * params.cols;

  var acc = 0.0;
  var c = lid.x;
  loop {
    if (c >= params.cols) { break; }
    let v = x[base + c];
    acc = acc + v * v;
    c = c + ${GROUP}u;
  }
  partial[lid.x] = acc;
  workgroupBarrier();

  var stride = ${GROUP / 2}u;
  loop {
    if (stride == 0u) { break; }
    if (lid.x < stride) { partial[lid.x] = partial[lid.x] + partial[lid.x + stride]; }
    workgroupBarrier();
    stride = stride >> 1u;
  }

  let scale = inverseSqrt(partial[0] / f32(params.cols) + params.eps);
  c = lid.x;
  loop {
    if (c >= params.cols) { break; }
    y[base + c] = x[base + c] * scale * f32(gain[c]);
    c = c + ${GROUP}u;
  }
}`;

/**
 * Rotary position embedding, in place over a packed [positions, heads * dim].
 *
 * The half-split convention: element j pairs with j + headDim/2. One thread per
 * pair.
 *
 * The cosines and sines arrive precomputed rather than being evaluated here.
 * That is not an optimisation, it is a correctness requirement: WGSL permits
 * relaxed precision on sin and cos, and on this adapter they carry about 3.0e-5
 * of absolute error even within one turn, which is 300 times the error of every
 * other kernel. web/precision.html measures it. The tables are built in f64 by
 * ropeTables() in ops.js and indexed by absolute position, so decoding from a
 * cache rotates by the position in the conversation rather than the offset
 * within the call.
 */
export const ROPE = `
struct Params { rows: u32, heads: u32, headDim: u32, basePos: u32 };

@group(0) @binding(0) var<storage, read_write> v      : array<f32>;
@group(0) @binding(1) var<storage, read>       cosTab : array<f32>;
@group(0) @binding(2) var<storage, read>       sinTab : array<f32>;
@group(0) @binding(3) var<uniform>             params : Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let half = params.headDim / 2u;
  let perRow = params.heads * half;
  let idx = gid.x;
  if (idx >= params.rows * perRow) { return; }

  let row = idx / perRow;
  let within = idx % perRow;
  let head = within / half;
  let j = within % half;

  let tableAt = (params.basePos + row) * half + j;
  let c = cosTab[tableAt];
  let s = sinTab[tableAt];

  let base = row * params.heads * params.headDim + head * params.headDim + j;
  let lo = v[base];
  let hi = v[base + half];
  v[base] = lo * c - hi * s;
  v[base + half] = hi * c + lo * s;
}`;

/**
 * Causal grouped-query attention: one workgroup per (query position, head).
 *
 * Scores are computed into a scratch buffer, softmaxed with the usual max
 * shift, then used to weight the values. Query head h reads key/value head
 * floor(h / groupSize), the blocked mapping asserted in
 * test/attention.test.js.
 *
 * Softmax runs over a slice of a shared scratch buffer rather than workgroup
 * memory because the context can be longer than 32 KB of shared storage
 * allows; correctness first, and the tiled online-softmax version belongs
 * after the GPU is known to agree with the CPU.
 */
export const ATTENTION = `
struct Params { seq: u32, past: u32, numHeads: u32, numKVHeads: u32,
                headDim: u32, groupSize: u32, total: u32, pad: u32 };

@group(0) @binding(0) var<storage, read>       q      : array<f32>;
@group(0) @binding(1) var<storage, read>       keys   : array<f32>;
@group(0) @binding(2) var<storage, read>       values : array<f32>;
@group(0) @binding(3) var<storage, read_write> out    : array<f32>;
@group(0) @binding(4) var<storage, read_write> scores : array<f32>;
@group(0) @binding(5) var<uniform>             params : Params;

var<workgroup> reduce : array<f32, ${GROUP}>;

@compute @workgroup_size(${GROUP})
fn main(@builtin(workgroup_id) wg : vec3<u32>,
        @builtin(local_invocation_id) lid : vec3<u32>) {
  let t = wg.x;                 // query position within this call
  let h = wg.y;                 // query head
  if (t >= params.seq || h >= params.numHeads) { return; }

  let qDim = params.numHeads * params.headDim;
  let kvDim = params.numKVHeads * params.headDim;
  let kvBase = (h / params.groupSize) * params.headDim;
  let qBase = t * qDim + h * params.headDim;
  let upTo = params.past + t;   // causal horizon, inclusive
  let scoreBase = (h * params.seq + t) * params.total;
  let scale = 1.0 / sqrt(f32(params.headDim));

  // scores[s] = dot(q, k_s) * scale
  var s = lid.x;
  loop {
    if (s > upTo) { break; }
    var dot = 0.0;
    let kBase = s * kvDim + kvBase;
    for (var i = 0u; i < params.headDim; i = i + 1u) {
      dot = dot + q[qBase + i] * keys[kBase + i];
    }
    scores[scoreBase + s] = dot * scale;
    s = s + ${GROUP}u;
  }
  workgroupBarrier();

  // max, for the shift that keeps exp() finite
  var localMax = -3.4e38;
  s = lid.x;
  loop {
    if (s > upTo) { break; }
    localMax = max(localMax, scores[scoreBase + s]);
    s = s + ${GROUP}u;
  }
  reduce[lid.x] = localMax;
  workgroupBarrier();
  var stride = ${GROUP / 2}u;
  loop {
    if (stride == 0u) { break; }
    if (lid.x < stride) { reduce[lid.x] = max(reduce[lid.x], reduce[lid.x + stride]); }
    workgroupBarrier();
    stride = stride >> 1u;
  }
  let rowMax = reduce[0];
  workgroupBarrier();

  // exp and sum
  var localSum = 0.0;
  s = lid.x;
  loop {
    if (s > upTo) { break; }
    let e = exp(scores[scoreBase + s] - rowMax);
    scores[scoreBase + s] = e;
    localSum = localSum + e;
    s = s + ${GROUP}u;
  }
  reduce[lid.x] = localSum;
  workgroupBarrier();
  stride = ${GROUP / 2}u;
  loop {
    if (stride == 0u) { break; }
    if (lid.x < stride) { reduce[lid.x] = reduce[lid.x] + reduce[lid.x + stride]; }
    workgroupBarrier();
    stride = stride >> 1u;
  }
  let inv = 1.0 / reduce[0];
  workgroupBarrier();

  // out = sum_s p_s * v_s, one thread per element of the head
  var i = lid.x;
  loop {
    if (i >= params.headDim) { break; }
    var acc = 0.0;
    for (var u = 0u; u <= upTo; u = u + 1u) {
      acc = acc + scores[scoreBase + u] * inv * values[u * kvDim + kvBase + i];
    }
    out[t * qDim + h * params.headDim + i] = acc;
    i = i + ${GROUP}u;
  }
}`;

/** gate = silu(gate) * up, elementwise. */
export const SWIGLU = `
struct Params { n: u32, pad0: u32, pad1: u32, pad2: u32 };

@group(0) @binding(0) var<storage, read_write> gate   : array<f32>;
@group(0) @binding(1) var<storage, read>       up     : array<f32>;
@group(0) @binding(2) var<uniform>             params : Params;

@compute @workgroup_size(${GROUP})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= params.n) { return; }
  let g = gate[i];
  gate[i] = (g / (1.0 + exp(-g))) * up[i];
}`;

/** x += delta, the residual connection. */
export const ADD = `
struct Params { n: u32, pad0: u32, pad1: u32, pad2: u32 };

@group(0) @binding(0) var<storage, read_write> x      : array<f32>;
@group(0) @binding(1) var<storage, read>       delta  : array<f32>;
@group(0) @binding(2) var<uniform>             params : Params;

@compute @workgroup_size(${GROUP})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= params.n) { return; }
  x[i] = x[i] + delta[i];
}`;
