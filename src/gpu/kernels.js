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
  // The third dispatch dimension is the position within the sequence, so one
  // kernel serves both prefill and single-token decode. Uniform across the
  // workgroup, like the row, so the barriers below stay uniform.
  let b = wg.z;
  if (row >= params.rows) { return; }

  let base = row * params.cols;
  let xBase = b * params.cols;
  var acc = 0.0;
  var c = lid.x;
  loop {
    if (c >= params.cols) { break; }
    acc = acc + f32(W[base + c]) * x[xBase + c];
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
    y[b * params.rows + row] = v;
  }
}`;

/** Rows of the weight matrix one workgroup owns. */
export const TILE_ROWS = 32;
/** Columns staged into workgroup memory per step. */
export const TILE_K = 128;

/**
 * y = W X + bias, with a tile of W staged in workgroup memory and reused
 * across `batchTile` sequence positions at once.
 *
 * MATVEC computes one output row for one position per workgroup, so it
 * re-reads the entire weight matrix once per position. For a prefill of S
 * tokens that is S passes over 716 MB of weights, and measurement put prefill
 * at the memory system's practical limit while doing 52x more traffic than
 * the arithmetic needs. Staging a tile and reusing it across positions divides
 * the weight traffic by `batchTile`.
 *
 * Two properties this is written to preserve, both of them correctness rather
 * than speed:
 *
 * The order in which a given output accumulates its products does not depend
 * on `batchTile`. Every output is the sum, in column order, of per-tile sums
 * each accumulated in column order. batchTile decides only which outputs a
 * workgroup computes, never how any one of them is summed. That is what keeps
 * prefill and single-token decode agreeing bit for bit rather than merely to
 * a tolerance -- they run the same kernel over the same numbers in the same
 * order.
 *
 * The summation is two-level on purpose. A flat sequential sum over 4864
 * columns in f32 grows error like n*eps, around 3e-4 relative, which is close
 * enough to the project's 5e-4 per-layer bound to be uncomfortable.
 * Accumulating each 128-column tile separately and adding the tile totals
 * makes it (TILE_K + cols/TILE_K)*eps instead: 166 terms rather than 4864.
 *
 * Every thread runs every barrier. Out-of-range rows and positions are
 * handled by zero-filling the tiles and guarding the final store, never by
 * returning early, since a workgroup that loses threads at a barrier
 * deadlocks or reads uninitialised memory.
 */
export function tiledMatmul(batchTile) {
  if (TILE_ROWS * batchTile > 256) {
    throw new Error(
      `batchTile ${batchTile} needs ${TILE_ROWS * batchTile} outputs, more than the 256 threads`,
    );
  }
  return `${F16}
struct Params { rows: u32, cols: u32, hasBias: u32, gridWidth: u32,
                batch: u32, pad0: u32, pad1: u32, pad2: u32 };

@group(0) @binding(0) var<storage, read>       W      : array<f16>;
@group(0) @binding(1) var<storage, read>       x      : array<f32>;
@group(0) @binding(2) var<storage, read>       bias   : array<f32>;
@group(0) @binding(3) var<storage, read_write> y      : array<f32>;
@group(0) @binding(4) var<uniform>             params : Params;

const TR : u32 = ${TILE_ROWS}u;
const TB : u32 = ${batchTile}u;
const TK : u32 = ${TILE_K}u;

var<workgroup> wTile : array<f16, ${TILE_ROWS * TILE_K}>;
var<workgroup> xTile : array<f32, ${batchTile * TILE_K}>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg : vec3<u32>,
        @builtin(local_invocation_id) lid : vec3<u32>) {
  let rowTile = (wg.x + wg.y * params.gridWidth) * TR;
  let batTile = wg.z * TB;

  let t = lid.x;
  // 256 threads over a TR x TB output tile, one output each. The tile holds
  // TR*TB outputs, which is 256 only when TB is 8; for smaller tiles the
  // surplus threads have no output of their own. They still load and still
  // reach every barrier -- they simply do not accumulate or store. Letting
  // them compute an index anyway ran myRow up to 255 against a 32-row tile
  // and read whatever followed it in workgroup memory.
  // 'active' is a WGSL reserved keyword; naming it that compiles to nothing
  // and the kernel silently writes zeros.
  let hasOutput = t < TR * TB;
  let myRow = t / TB;
  let myBat = t % TB;
  let row = rowTile + myRow;
  let bat = batTile + myBat;

  var acc = 0.0;
  var k0 = 0u;
  loop {
    if (k0 >= params.cols) { break; }

    // Stage the weight tile. Out of range reads become zero, which is exact
    // under addition and so cannot perturb the sum.
    var i = t;
    loop {
      if (i >= TR * TK) { break; }
      let r = rowTile + i / TK;
      let c = k0 + i % TK;
      var wv = 0.0h;
      if (r < params.rows && c < params.cols) { wv = W[r * params.cols + c]; }
      wTile[i] = wv;
      i = i + 256u;
    }
    var j = t;
    loop {
      if (j >= TB * TK) { break; }
      let b = batTile + j / TK;
      let c = k0 + j % TK;
      var xv = 0.0;
      if (b < params.batch && c < params.cols) { xv = x[b * params.cols + c]; }
      xTile[j] = xv;
      j = j + 256u;
    }
    workgroupBarrier();

    // No barrier inside, so this may diverge safely.
    if (hasOutput) {
      var tileAcc = 0.0;
      var c2 = 0u;
      loop {
        if (c2 >= TK) { break; }
        tileAcc = tileAcc + f32(wTile[myRow * TK + c2]) * xTile[myBat * TK + c2];
        c2 = c2 + 1u;
      }
      acc = acc + tileAcc;
    }

    // Before overwriting the tiles on the next pass.
    workgroupBarrier();
    k0 = k0 + TK;
  }

  if (hasOutput && row < params.rows && bat < params.batch) {
    var v = acc;
    if (params.hasBias == 1u) { v = v + bias[row]; }
    y[bat * params.rows + row] = v;
  }
}`;
}

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
