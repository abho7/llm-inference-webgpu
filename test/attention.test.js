import test from 'node:test';
import assert from 'node:assert/strict';

import { groupedAttention, kvHeadFor } from '../src/core/attention.js';
import { softmaxInPlace } from '../src/core/ops.js';

// The reference the grouped implementation is checked against.
//
// Repeats each key/value head groupSize times to make numHeads of them, then
// runs plain multi-head attention where head h reads head h. That is how the
// architecture is defined and how the reference implementation computes it; the
// grouped version skips the copy and indexes instead. The two must agree
// exactly -- the same products summed in the same order -- so this is compared
// with strict equality rather than a tolerance.
function expandedAttention(q, keys, values, { seq, past, numHeads, numKVHeads, headDim }) {
  const groupSize = numHeads / numKVHeads;
  const qDim = numHeads * headDim;
  const kvDim = numKVHeads * headDim;
  const total = past + seq;

  // Materialise numHeads key/value heads by repeating each one groupSize times.
  const wideKeys = new Float32Array(total * qDim);
  const wideValues = new Float32Array(total * qDim);
  for (let s = 0; s < total; s++) {
    for (let h = 0; h < numHeads; h++) {
      const from = s * kvDim + Math.floor(h / groupSize) * headDim;
      const to = s * qDim + h * headDim;
      for (let i = 0; i < headDim; i++) {
        wideKeys[to + i] = keys[from + i];
        wideValues[to + i] = values[from + i];
      }
    }
  }

  const out = new Float32Array(seq * qDim);
  const scores = new Float32Array(total);
  const scale = 1 / Math.sqrt(headDim);

  for (let h = 0; h < numHeads; h++) {
    for (let t = 0; t < seq; t++) {
      const upTo = past + t;
      const qBase = t * qDim + h * headDim;
      for (let s = 0; s <= upTo; s++) {
        let dot = 0;
        const kBase = s * qDim + h * headDim;
        for (let i = 0; i < headDim; i++) dot += q[qBase + i] * wideKeys[kBase + i];
        scores[s] = dot * scale;
      }
      softmaxInPlace(scores, upTo + 1);
      const outBase = t * qDim + h * headDim;
      for (let s = 0; s <= upTo; s++) {
        const vBase = s * qDim + h * headDim;
        for (let i = 0; i < headDim; i++) out[outBase + i] += scores[s] * wideValues[vBase + i];
      }
    }
  }
  return out;
}

function randomArray(n, seed) {
  // A small deterministic generator, so a failure is reproducible.
  let state = seed >>> 0;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    state = (state * 1664525 + 1013904223) >>> 0;
    out[i] = (state / 0xffffffff) * 2 - 1;
  }
  return out;
}

function run(shape, seed = 1) {
  const { seq, past, numHeads, numKVHeads, headDim } = shape;
  const total = past + seq;
  const q = randomArray(seq * numHeads * headDim, seed);
  const keys = randomArray(total * numKVHeads * headDim, seed + 1);
  const values = randomArray(total * numKVHeads * headDim, seed + 2);

  const mine = new Float32Array(seq * numHeads * headDim);
  groupedAttention(q, keys, values, mine, new Float32Array(total), shape);
  return { mine, reference: expandedAttention(q, keys, values, shape) };
}

const QWEN = { numHeads: 14, numKVHeads: 2, headDim: 64 };

test('the query-to-kv head mapping is blocked, not interleaved', () => {
  // Query heads 0..6 read key/value head 0 and 7..13 read head 1. The
  // interleaved alternative (h % numKVHeads) is the plausible-looking wrong
  // answer, and it agrees with the correct one only at head 0.
  const mapping = Array.from({ length: 14 }, (_, h) => kvHeadFor(h, 14, 2));
  assert.deepEqual(mapping, [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]);

  const interleaved = Array.from({ length: 14 }, (_, h) => h % 2);
  assert.notDeepEqual(mapping, interleaved, 'the two conventions must differ');
});

test('the mapping covers every kv head evenly', () => {
  const counts = new Array(2).fill(0);
  for (let h = 0; h < 14; h++) counts[kvHeadFor(h, 14, 2)]++;
  assert.deepEqual(counts, [7, 7]);
});

test('grouped attention equals attention over explicitly repeated kv heads', () => {
  // Exactly, not approximately: skipping the copy must not change the sums.
  const { mine, reference } = run({ seq: 5, past: 0, ...QWEN });
  assert.deepEqual([...mine], [...reference]);
});

test('grouped attention matches the expanded reference when decoding from a cache', () => {
  const { mine, reference } = run({ seq: 1, past: 11, ...QWEN }, 99);
  assert.deepEqual([...mine], [...reference]);
});

test('grouped attention matches on a mixed prefill against existing context', () => {
  const { mine, reference } = run({ seq: 3, past: 7, ...QWEN }, 4242);
  assert.deepEqual([...mine], [...reference]);
});

test('grouped attention matches when there is no grouping at all', () => {
  // numKVHeads === numHeads is ordinary multi-head attention; the grouped path
  // must not special-case its way into being wrong there.
  const { mine, reference } = run({ seq: 4, past: 2, numHeads: 4, numKVHeads: 4, headDim: 8 });
  assert.deepEqual([...mine], [...reference]);
});

test('grouped attention matches when every head shares one kv head', () => {
  // The other extreme: multi-query attention.
  const { mine, reference } = run({ seq: 4, past: 3, numHeads: 6, numKVHeads: 1, headDim: 8 });
  assert.deepEqual([...mine], [...reference]);
});

test('attention is causal: a position cannot see the future', () => {
  // Changing the key and value at the last position must leave every earlier
  // output untouched. This is the property a masking bug breaks, and it is
  // invisible in the aggregate error.
  const shape = { seq: 4, past: 0, numHeads: 4, numKVHeads: 2, headDim: 8 };
  const total = shape.seq;
  const qDim = shape.numHeads * shape.headDim;
  const kvDim = shape.numKVHeads * shape.headDim;

  const q = randomArray(shape.seq * qDim, 7);
  const keys = randomArray(total * kvDim, 8);
  const values = randomArray(total * kvDim, 9);

  const before = new Float32Array(shape.seq * qDim);
  groupedAttention(q, keys, values, before, new Float32Array(total), shape);

  // Perturb only the final position.
  const keys2 = Float32Array.from(keys);
  const values2 = Float32Array.from(values);
  for (let i = 0; i < kvDim; i++) {
    keys2[(total - 1) * kvDim + i] += 3.5;
    values2[(total - 1) * kvDim + i] -= 2.25;
  }
  const after = new Float32Array(shape.seq * qDim);
  groupedAttention(q, keys2, values2, after, new Float32Array(total), shape);

  for (let t = 0; t < shape.seq - 1; t++) {
    for (let i = 0; i < qDim; i++) {
      assert.equal(after[t * qDim + i], before[t * qDim + i],
        `position ${t} changed when only position ${total - 1} was perturbed`);
    }
  }
  let moved = false;
  for (let i = 0; i < qDim; i++) {
    if (after[(shape.seq - 1) * qDim + i] !== before[(shape.seq - 1) * qDim + i]) moved = true;
  }
  assert.ok(moved, 'the last position should have changed, or the test proves nothing');
});

test('the first position attends only to itself, so it returns its own value', () => {
  // With one visible key the softmax is exactly 1, so the output is value[0]
  // verbatim. A useful anchor: it depends on the mask, the softmax and the
  // value indexing all at once, and it has a closed-form answer.
  const shape = { seq: 1, past: 0, numHeads: 4, numKVHeads: 2, headDim: 8 };
  const qDim = shape.numHeads * shape.headDim;
  const kvDim = shape.numKVHeads * shape.headDim;
  const q = randomArray(qDim, 21);
  const keys = randomArray(kvDim, 22);
  const values = randomArray(kvDim, 23);

  const out = new Float32Array(qDim);
  groupedAttention(q, keys, values, out, new Float32Array(1), shape);

  for (let h = 0; h < shape.numHeads; h++) {
    const kvBase = kvHeadFor(h, shape.numHeads, shape.numKVHeads) * shape.headDim;
    for (let i = 0; i < shape.headDim; i++) {
      assert.equal(out[h * shape.headDim + i], values[kvBase + i],
        `head ${h} element ${i} should be its kv head's value verbatim`);
    }
  }
});

test('attention output is a convex combination of the values it can see', () => {
  // Softmax weights are non-negative and sum to one, so every output element
  // must lie inside the range of the values available to that head. A sign
  // error or an unnormalised softmax escapes this immediately.
  const shape = { seq: 6, past: 0, numHeads: 4, numKVHeads: 2, headDim: 8 };
  const qDim = shape.numHeads * shape.headDim;
  const kvDim = shape.numKVHeads * shape.headDim;
  const q = randomArray(shape.seq * qDim, 31);
  const keys = randomArray(shape.seq * kvDim, 32);
  const values = randomArray(shape.seq * kvDim, 33);

  const out = new Float32Array(shape.seq * qDim);
  groupedAttention(q, keys, values, out, new Float32Array(shape.seq), shape);

  for (let h = 0; h < shape.numHeads; h++) {
    const kvBase = kvHeadFor(h, shape.numHeads, shape.numKVHeads) * shape.headDim;
    for (let t = 0; t < shape.seq; t++) {
      for (let i = 0; i < shape.headDim; i++) {
        let lo = Infinity;
        let hi = -Infinity;
        for (let s = 0; s <= t; s++) {
          const value = values[s * kvDim + kvBase + i];
          if (value < lo) lo = value;
          if (value > hi) hi = value;
        }
        const got = out[t * qDim + h * shape.headDim + i];
        assert.ok(got >= lo - 1e-6 && got <= hi + 1e-6,
          `head ${h} position ${t} element ${i}: ${got} outside [${lo}, ${hi}]`);
      }
    }
  }
});
