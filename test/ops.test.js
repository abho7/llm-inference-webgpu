import test from 'node:test';
import assert from 'node:assert/strict';

import {
  rmsNorm, matVec, matVecBf16, softmaxInPlace, silu, swigluInPlace,
  ropeFrequencies, ropeInPlace,
} from '../src/core/ops.js';
import { f32ToBf16 } from '../src/core/dtype.js';

const close = (a, b, tol, what) =>
  assert.ok(Math.abs(a - b) <= tol, `${what}: ${a} vs ${b} (tolerance ${tol})`);

// ---------------------------------------------------------------- rmsNorm --

test('rmsNorm gives the output unit root-mean-square when the gain is 1', () => {
  const x = Float32Array.from([1, -2, 3, -4, 5]);
  const ones = new Float32Array(5).fill(1);
  const out = rmsNorm(x, ones, 0);
  let sumSquares = 0;
  for (const v of out) sumSquares += v * v;
  close(Math.sqrt(sumSquares / out.length), 1, 1e-6, 'rms of the output');
});

test('rmsNorm applies the gain per channel', () => {
  const x = Float32Array.from([3, 4]);
  const gain = Float32Array.from([2, 10]);
  const out = rmsNorm(x, gain, 0);
  // rms = sqrt((9 + 16) / 2) = 3.5355..., so x / rms = [0.8485, 1.1314]
  const rms = Math.sqrt(25 / 2);
  close(out[0], (3 / rms) * 2, 1e-6, 'channel 0');
  close(out[1], (4 / rms) * 10, 1e-6, 'channel 1');
});

test('rmsNorm puts epsilon inside the square root, not outside', () => {
  // The two placements agree for ordinary inputs and diverge sharply for tiny
  // ones, which is exactly where a silent mistake would hide.
  const x = Float32Array.from([1e-4, 1e-4]);
  const ones = Float32Array.from([1, 1]);
  const eps = 1e-6;
  const out = rmsNorm(x, ones, eps);
  const meanSquare = 1e-8;
  const expected = 1e-4 / Math.sqrt(meanSquare + eps);
  close(out[0], expected, 1e-9, 'epsilon inside the root');
  assert.ok(Math.abs(out[0] - 1e-4 / (Math.sqrt(meanSquare) + eps)) > 0.5,
    'the outside-the-root placement should give a visibly different answer');
});

test('rmsNorm is scale-equivariant only through the gain', () => {
  // Doubling the input leaves the normalised output unchanged, which is the
  // property that makes RMSNorm a normalisation at all.
  const x = Float32Array.from([1, 2, 3, 4]);
  const doubled = Float32Array.from([2, 4, 6, 8]);
  const ones = new Float32Array(4).fill(1);
  const a = rmsNorm(x, ones, 0);
  const b = rmsNorm(doubled, ones, 0);
  for (let i = 0; i < 4; i++) close(a[i], b[i], 1e-6, `element ${i}`);
});

// ----------------------------------------------------------------- matVec --

test('matVec computes rows of W times x', () => {
  // W = [[1, 2, 3], [4, 5, 6]], x = [1, 0, -1] -> [1 - 3, 4 - 6] = [-2, -2]
  const W = Float32Array.from([1, 2, 3, 4, 5, 6]);
  const x = Float32Array.from([1, 0, -1]);
  assert.deepEqual([...matVec(W, x, 2, 3)], [-2, -2]);
});

test('matVec adds the bias', () => {
  const W = Float32Array.from([1, 2, 3, 4, 5, 6]);
  const x = Float32Array.from([1, 0, -1]);
  const bias = Float32Array.from([10, 100]);
  assert.deepEqual([...matVec(W, x, 2, 3, bias)], [8, 98]);
});

test('matVec treats W as [out, in] row-major, matching the stored weights', () => {
  // A transposed reading would produce [1*1 + 4*0, ...] instead. Using a
  // non-square W means the transposed reading does not even have valid shapes,
  // and using distinct values means it cannot coincidentally agree.
  const W = Float32Array.from([1, 2, 3, 4, 5, 6, 7, 8]); // 4 rows, 2 cols
  const x = Float32Array.from([1, 10]);
  assert.deepEqual([...matVec(W, x, 4, 2)], [21, 43, 65, 87]);
});

test('matVecBf16 agrees with matVec on the widened matrix', () => {
  const rows = 7;
  const cols = 5;
  const bits = new Uint16Array(rows * cols);
  const wide = new Float32Array(rows * cols);
  for (let i = 0; i < rows * cols; i++) {
    // Values that are exactly representable in bf16 so the two paths are
    // comparable at all: bf16 has 8 mantissa bits.
    const v = ((i % 9) - 4) * 0.25;
    bits[i] = f32ToBf16(v);
    wide[i] = v;
  }
  const x = Float32Array.from([1, -2, 0.5, 3, -0.25]);
  const a = matVec(wide, x, rows, cols);
  const b = matVecBf16(bits, x, rows, cols);
  for (let r = 0; r < rows; r++) close(a[r], b[r], 1e-6, `row ${r}`);
});

// ---------------------------------------------------------------- softmax --

test('softmax produces a distribution', () => {
  const x = Float32Array.from([1, 2, 3, 4]);
  softmaxInPlace(x);
  let sum = 0;
  for (const v of x) {
    assert.ok(v > 0 && v < 1, 'every probability is in (0, 1)');
    sum += v;
  }
  close(sum, 1, 1e-6, 'total probability');
  assert.ok(x[3] > x[2] && x[2] > x[1], 'order is preserved');
});

test('softmax is invariant to adding a constant', () => {
  const a = Float32Array.from([1, 2, 3]);
  const b = Float32Array.from([101, 102, 103]);
  softmaxInPlace(a);
  softmaxInPlace(b);
  for (let i = 0; i < 3; i++) close(a[i], b[i], 1e-6, `element ${i}`);
});

test('softmax survives inputs that would overflow without the max shift', () => {
  // exp(800) is Infinity in f64, so an implementation without the shift returns
  // NaN here. Attention scores do reach this range.
  const x = Float32Array.from([800, 801, 802]);
  softmaxInPlace(x);
  let sum = 0;
  for (const v of x) {
    assert.ok(Number.isFinite(v), 'no infinities');
    sum += v;
  }
  close(sum, 1, 1e-6, 'total probability');
});

test('softmax over a prefix leaves the rest of the buffer alone', () => {
  // The attention loop reuses one scores buffer and softmaxes only the causal
  // prefix, so stale values beyond it must not be touched or read.
  const x = Float32Array.from([1, 1, 999, 999]);
  softmaxInPlace(x, 2);
  close(x[0], 0.5, 1e-6, 'first');
  close(x[1], 0.5, 1e-6, 'second');
  assert.equal(x[2], 999, 'beyond the prefix is untouched');
});

test('a one-element softmax is exactly 1', () => {
  const x = Float32Array.from([-12345]);
  softmaxInPlace(x, 1);
  assert.equal(x[0], 1);
});

// ------------------------------------------------------------------- silu --

test('silu matches its definition at known points', () => {
  close(silu(0), 0, 1e-12, 'silu(0)');
  close(silu(1), 1 / (1 + Math.exp(-1)), 1e-12, 'silu(1)');
  close(silu(-1), -1 / (1 + Math.exp(1)), 1e-12, 'silu(-1)');
  // Unlike ReLU, silu dips below zero before recovering; its minimum is near
  // x = -1.2784 with value about -0.2785.
  assert.ok(silu(-1.2784) < -0.278 && silu(-1.2784) > -0.279, 'the negative lobe exists');
});

test('swiglu multiplies the gated branch by the linear one', () => {
  const gate = Float32Array.from([1, -1, 0]);
  const up = Float32Array.from([2, 3, 5]);
  swigluInPlace(gate, up);
  close(gate[0], silu(1) * 2, 1e-6, 'first');
  close(gate[1], silu(-1) * 3, 1e-6, 'second');
  close(gate[2], 0, 1e-12, 'third');
});

// ------------------------------------------------------------------- rope --

test('rope frequencies decay geometrically from 1', () => {
  const inv = ropeFrequencies(8, 10000);
  assert.equal(inv.length, 4);
  close(inv[0], 1, 1e-12, 'first frequency is 1');
  for (let j = 1; j < inv.length; j++) {
    assert.ok(inv[j] < inv[j - 1], 'frequencies decrease');
  }
  close(inv[1], Math.pow(10000, -2 / 8), 1e-12, 'second frequency');
});

test('rope is a rotation, so it preserves the length of every pair', () => {
  const headDim = 8;
  const inv = ropeFrequencies(headDim, 10000);
  const v = Float32Array.from([1, 2, 3, 4, 5, 6, 7, 8]);
  const before = Array.from(v);
  ropeInPlace(v, 0, headDim, 37, inv);
  const half = headDim >> 1;
  for (let j = 0; j < half; j++) {
    const lenBefore = Math.hypot(before[j], before[j + half]);
    const lenAfter = Math.hypot(v[j], v[j + half]);
    close(lenAfter, lenBefore, 1e-5, `pair ${j} length`);
  }
});

test('rope pairs element j with j + headDim/2, not with j + 1', () => {
  // The half-split and interleaved conventions both "work" and produce
  // different models. Rotating a vector that is 1 in slot 0 and 0 elsewhere
  // moves mass into slot headDim/2 under the half-split convention and into
  // slot 1 under the interleaved one.
  const headDim = 8;
  const inv = ropeFrequencies(headDim, 10000);
  const v = new Float32Array(headDim);
  v[0] = 1;
  ropeInPlace(v, 0, headDim, 1, inv);
  close(v[0], Math.cos(1), 1e-6, 'slot 0 keeps cos');
  close(v[4], Math.sin(1), 1e-6, 'slot headDim/2 receives sin');
  close(v[1], 0, 1e-12, 'slot 1 is untouched, so this is not the interleaved form');
});

test('rope at position 0 is the identity', () => {
  const headDim = 8;
  const inv = ropeFrequencies(headDim, 10000);
  const v = Float32Array.from([1, 2, 3, 4, 5, 6, 7, 8]);
  const before = Array.from(v);
  ropeInPlace(v, 0, headDim, 0, inv);
  for (let i = 0; i < headDim; i++) close(v[i], before[i], 1e-12, `element ${i}`);
});

test('rope makes the query-key dot product depend only on relative position', () => {
  // This is the defining property of rotary embeddings and the reason to use
  // them, so it is worth asserting directly rather than inferring it from the
  // formula. <R(q, m), R(k, n)> must depend on m - n alone.
  const headDim = 16;
  const theta = 1e6;
  const inv = ropeFrequencies(headDim, theta);
  const q0 = Float32Array.from(Array.from({ length: headDim }, (_, i) => Math.sin(i * 1.7)));
  const k0 = Float32Array.from(Array.from({ length: headDim }, (_, i) => Math.cos(i * 0.9)));

  const dotAt = (m, n) => {
    const q = Float32Array.from(q0);
    const k = Float32Array.from(k0);
    ropeInPlace(q, 0, headDim, m, inv);
    ropeInPlace(k, 0, headDim, n, inv);
    let d = 0;
    for (let i = 0; i < headDim; i++) d += q[i] * k[i];
    return d;
  };

  for (const [m, n] of [[0, 0], [5, 2], [100, 97], [1000, 997]]) {
    close(dotAt(m, n), dotAt(m - n, 0), 1e-4, `offset ${m}-${n} should match ${m - n}-0`);
  }
  // And a different relative offset really does give a different answer, so
  // the test above is not vacuously true.
  assert.ok(Math.abs(dotAt(5, 2) - dotAt(5, 1)) > 1e-3, 'different offsets differ');
});

test('rope applies to a slice at an offset, leaving neighbours alone', () => {
  // The forward pass rotates head h in place inside a packed [heads * headDim]
  // buffer, so an off-by-one in the offset would silently rotate the wrong head.
  const headDim = 4;
  const inv = ropeFrequencies(headDim, 10000);
  const buf = Float32Array.from([9, 9, 9, 9, 1, 0, 0, 0, 7, 7, 7, 7]);
  ropeInPlace(buf, 4, headDim, 1, inv);
  assert.deepEqual([...buf.slice(0, 4)], [9, 9, 9, 9], 'the head before is untouched');
  assert.deepEqual([...buf.slice(8)], [7, 7, 7, 7], 'the head after is untouched');
  close(buf[4], Math.cos(1), 1e-6, 'the target head rotated');
});
