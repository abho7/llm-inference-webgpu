import test from 'node:test';
import assert from 'node:assert/strict';

import {
  quantizeInt8PerChannel, dequantizeInt8PerChannel,
  quantizeInt4Group, dequantizeInt4Group,
  roundTrip, bitsPerWeight, quantizationError,
} from '../src/core/quantize.js';

function rand(n, seed) {
  let s = seed >>> 0;
  const a = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    s = (s * 1664525 + 1013904223) >>> 0;
    a[i] = (s / 0xffffffff) * 2 - 1;
  }
  return a;
}

// -------------------------------------------------------------------- int8 --

test('int8 codes stay inside a symmetric grid', () => {
  const W = rand(4 * 64, 1);
  const { q } = quantizeInt8PerChannel(W, 4, 64);
  for (let i = 0; i < q.length; i++) {
    assert.ok(q[i] >= -127 && q[i] <= 127, `code ${q[i]} outside [-127, 127]`);
  }
});

test('int8 puts the largest weight of each row at the top of the grid', () => {
  // The scale is defined by the row peak, so the peak must land on 127 exactly.
  // If it landed lower the grid would be wasted; higher and it would clip.
  const W = Float32Array.from([
    1, -2, 0.5, 0,      // row peak 2
    0.1, 0.05, -0.1, 0, // row peak 0.1
  ]);
  const packed = quantizeInt8PerChannel(W, 2, 4);
  assert.equal(packed.q[1], -127, 'row 0 peak is negative and maps to -127');
  // Scales are stored f32, so compare against the f32 rounding of 2/127.
  assert.equal(packed.scales[0], Math.fround(2 / 127));
  assert.equal(Math.max(...[...packed.q.slice(4)].map(Math.abs)), 127, 'row 1 peak maps to 127');
});

test('int8 scales are per row, so a loud row cannot flatten a quiet one', () => {
  // The point of per-channel granularity. One scale for the whole matrix would
  // be set by the loud row and would quantize the quiet row almost to zero.
  const cols = 32;
  const W = new Float32Array(2 * cols);
  for (let c = 0; c < cols; c++) {
    W[c] = 100 * Math.sin(c);          // loud row
    W[cols + c] = 0.001 * Math.cos(c);  // quiet row
  }
  const back = roundTrip(W, 2, cols, 'int8');
  const quietOriginal = W.subarray(cols);
  const quietBack = back.subarray(cols);
  const error = quantizationError(quietOriginal, quietBack);
  assert.ok(error.frobenius < 0.01,
    `the quiet row should survive, got ${error.frobenius.toExponential(2)} relative error`);
});

test('int8 represents zero exactly', () => {
  const W = Float32Array.from([0, 1, -1, 0, 0.5, 0]);
  const back = roundTrip(W, 2, 3, 'int8');
  assert.equal(back[0], 0);
  assert.equal(back[3], 0);
  assert.equal(back[5], 0);
});

test('int8 handles an all-zero row without dividing by zero', () => {
  const W = Float32Array.from([0, 0, 0, 0, 1, 2, 3, 4]);
  const packed = quantizeInt8PerChannel(W, 2, 4);
  assert.equal(packed.scales[0], 0);
  const back = dequantizeInt8PerChannel(packed);
  for (let i = 0; i < 4; i++) assert.equal(back[i], 0);
  assert.ok(Number.isFinite(back[7]));
});

test('int8 error is bounded by half a step, per row', () => {
  // The guarantee the scheme actually makes. Anything larger means the codes
  // are being computed wrongly, not that quantization is lossy.
  const rows = 6;
  const cols = 128;
  const W = rand(rows * cols, 7);
  const packed = quantizeInt8PerChannel(W, rows, cols);
  const back = dequantizeInt8PerChannel(packed);
  for (let r = 0; r < rows; r++) {
    const halfStep = packed.scales[r] / 2;
    for (let c = 0; c < cols; c++) {
      const i = r * cols + c;
      assert.ok(Math.abs(back[i] - W[i]) <= halfStep * 1.0001,
        `row ${r} col ${c}: error ${Math.abs(back[i] - W[i])} exceeds half-step ${halfStep}`);
    }
  }
});

test('int8 negates codes when weights are negated', () => {
  // What symmetry buys, and a check that no zero point crept in.
  const W = rand(3 * 16, 11);
  const negated = Float32Array.from(W, (v) => -v);
  const a = quantizeInt8PerChannel(W, 3, 16);
  const b = quantizeInt8PerChannel(negated, 3, 16);
  for (let i = 0; i < a.q.length; i++) assert.equal(b.q[i], -a.q[i], `code ${i}`);
});

// -------------------------------------------------------------------- int4 --

test('int4 packs two codes per byte and unpacks them in the right order', () => {
  // Sixteen weights spanning the group range, so every level is exercised.
  const cols = 16;
  const W = Float32Array.from({ length: cols }, (_, i) => i - 8);
  const packed = quantizeInt4Group(W, 1, cols, cols);
  assert.equal(packed.q.length, cols / 2, 'two codes per byte');

  const levels = [];
  for (let i = 0; i < cols; i++) {
    const byte = packed.q[i >> 1];
    levels.push((i & 1) === 0 ? (byte & 0x0f) : (byte >> 4));
  }
  for (const level of levels) assert.ok(level >= 0 && level <= 15, `level ${level} out of range`);
  // Monotone input must give monotone codes.
  for (let i = 1; i < levels.length; i++) {
    assert.ok(levels[i] >= levels[i - 1], `codes should not decrease at ${i}`);
  }
});

test('int4 represents zero exactly, which is what the zero point is for', () => {
  // An all-positive group still has to encode zero without bias.
  const cols = 32;
  const W = Float32Array.from({ length: cols }, (_, i) => (i === 0 ? 0 : 1 + i * 0.1));
  const back = roundTrip(W, 1, cols, 'int4', cols);
  assert.equal(back[0], 0, 'zero must survive a group of positive weights');
});

test('int4 groups are independent, so a loud group cannot flatten a quiet one', () => {
  const groupSize = 16;
  const cols = 32;
  const W = new Float32Array(cols);
  for (let i = 0; i < groupSize; i++) {
    W[i] = 50 * Math.sin(i);
    W[groupSize + i] = 0.01 * Math.cos(i);
  }
  const back = roundTrip(W, 1, cols, 'int4', groupSize);
  const error = quantizationError(W.subarray(groupSize), back.subarray(groupSize));
  assert.ok(error.frobenius < 0.1,
    `the quiet group should survive, got ${error.frobenius.toExponential(2)}`);
});

test('int4 rejects a group size that does not divide the row', () => {
  // A ragged last group would set its scale from fewer samples, which is an
  // accuracy cliff rather than a convenience.
  assert.throws(() => quantizeInt4Group(new Float32Array(100), 1, 100, 32), /does not divide/);
  assert.throws(() => quantizeInt4Group(new Float32Array(8), 1, 8, 1), /at least 2/);
});

test('int4 handles a constant group without dividing by zero', () => {
  const W = new Float32Array(16).fill(0);
  const packed = quantizeInt4Group(W, 1, 16, 16);
  assert.equal(packed.scales[0], 0);
  const back = dequantizeInt4Group(packed);
  for (const v of back) assert.equal(v, 0);
});

test('int4 error is bounded by half a step, per group', () => {
  const cols = 128;
  const groupSize = 32;
  const W = rand(cols, 21);
  const packed = quantizeInt4Group(W, 1, cols, groupSize);
  const back = dequantizeInt4Group(packed);
  for (let g = 0; g < cols / groupSize; g++) {
    const halfStep = packed.scales[g] / 2;
    for (let i = 0; i < groupSize; i++) {
      const at = g * groupSize + i;
      assert.ok(Math.abs(back[at] - W[at]) <= halfStep * 1.0001,
        `group ${g} element ${i}: ${Math.abs(back[at] - W[at])} exceeds ${halfStep}`);
    }
  }
});

// ------------------------------------------------------------- comparisons --

test('smaller int4 groups are more accurate than larger ones', () => {
  // The trade the scheme exists to expose, asserted rather than assumed.
  const cols = 512;
  const W = rand(cols, 31);
  let previous = Infinity;
  for (const groupSize of [256, 128, 64, 32]) {
    const error = quantizationError(W, roundTrip(W, 1, cols, 'int4', groupSize)).frobenius;
    assert.ok(error < previous, `group ${groupSize} should beat the larger group`);
    previous = error;
  }
});

test('int8 is more accurate than int4 on the same data', () => {
  const rows = 4;
  const cols = 256;
  const W = rand(rows * cols, 41);
  const int8 = quantizationError(W, roundTrip(W, rows, cols, 'int8')).frobenius;
  const int4 = quantizationError(W, roundTrip(W, rows, cols, 'int4', 128)).frobenius;
  assert.ok(int8 < int4, `int8 ${int8} should beat int4 ${int4}`);
  // And by roughly the factor the extra bits imply: 16x the levels.
  assert.ok(int4 / int8 > 4, `expected a wide gap, got ${(int4 / int8).toFixed(1)}x`);
});

test('bits per weight counts the scales, not just the codes', () => {
  // "4-bit" with an f32 scale and zero point per 128 weights is 4.5 bits, and
  // the difference is the entire argument for a larger group.
  assert.equal(bitsPerWeight('int4', 896, 128), 4 + 64 / 128);
  assert.equal(bitsPerWeight('int4', 896, 64), 4 + 64 / 64);
  assert.ok(bitsPerWeight('int8', 896) > 8);
  assert.ok(bitsPerWeight('int8', 896) < 8.1, 'one scale per 896 weights is nearly free');
  assert.equal(bitsPerWeight('f16'), 16);
});

test('quantization error reports both a relative and a worst-case number', () => {
  // Frobenius error hides a single badly-mangled weight; max absolute error is
  // what catches it, and a mangled outlier is the failure mode that matters.
  const original = Float32Array.from([1, 1, 1, 1]);
  const approx = Float32Array.from([1, 1, 1, 5]);
  const error = quantizationError(original, approx);
  assert.equal(error.maxAbs, 4);
  assert.equal(error.at, 3);
  assert.ok(error.frobenius > 1.9 && error.frobenius < 2.1);
});
