// Causal grouped-query attention.
//
// Pulled out of the forward pass so it can be tested on its own, because the
// head grouping is the part most able to be subtly wrong: mapping query head h
// to the wrong key/value head produces a model that still runs, still emits
// fluent-looking text, and is quietly incorrect.
//
// Qwen2.5-0.5B has 14 query heads and 2 key/value heads, so seven query heads
// share each key/value head. The reference implementation expresses this by
// materialising 14 key/value heads -- repeating each one seven times -- and
// running ordinary multi-head attention. Repeating is pure waste at inference
// time, so the mapping is done by indexing instead, and
// test/attention.test.js checks the two against each other.
//
// The grouping is *blocked*, not interleaved: query heads 0..6 share key/value
// head 0, and 7..13 share head 1. Interleaving instead (head h uses h % 2)
// is the natural-looking alternative and is wrong.

import { softmaxInPlace } from './ops.js';

/**
 * Attention for `seq` query positions against `past + seq` cached positions.
 *
 * @param q        [seq, numHeads * headDim] queries, already rotated
 * @param keys     [past + seq, numKVHeads * headDim] from the cache
 * @param values   [past + seq, numKVHeads * headDim] from the cache
 * @param out      [seq, numHeads * headDim], overwritten
 * @param scores   scratch of at least past + seq elements
 *
 * Query position t has absolute position past + t and attends to 0..past+t.
 */
export function groupedAttention(q, keys, values, out, scores, {
  seq, past, numHeads, numKVHeads, headDim,
}) {
  const groupSize = numHeads / numKVHeads;
  const qDim = numHeads * headDim;
  const kvDim = numKVHeads * headDim;
  const scale = 1 / Math.sqrt(headDim);

  out.fill(0, 0, seq * qDim);

  for (let h = 0; h < numHeads; h++) {
    const kvBase = Math.floor(h / groupSize) * headDim;

    for (let t = 0; t < seq; t++) {
      const qBase = t * qDim + h * headDim;
      const upTo = past + t;   // causal horizon, inclusive

      for (let s = 0; s <= upTo; s++) {
        const kBase = s * kvDim + kvBase;
        let dot = 0;
        for (let i = 0; i < headDim; i++) dot += q[qBase + i] * keys[kBase + i];
        scores[s] = dot * scale;
      }
      softmaxInPlace(scores, upTo + 1);

      const outBase = t * qDim + h * headDim;
      for (let s = 0; s <= upTo; s++) {
        const weight = scores[s];
        const vBase = s * kvDim + kvBase;
        for (let i = 0; i < headDim; i++) out[outBase + i] += weight * values[vBase + i];
      }
    }
  }
  return out;
}

/**
 * Which key/value head a query head reads from.
 *
 * A single expression, exported so the test can state the rule independently of
 * the loop that uses it.
 */
export function kvHeadFor(queryHead, numHeads, numKVHeads) {
  return Math.floor(queryHead / (numHeads / numKVHeads));
}
