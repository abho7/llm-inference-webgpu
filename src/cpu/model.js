// The Qwen2 forward pass, as a reference implementation.
//
// Correctness first and speed nowhere: this is the thing the GPU backend gets
// compared against, so it is written to be read and to be obviously the
// architecture on the page. Every reduction accumulates in f64 (see ops.js),
// and no attempt is made to fuse or tile anything.
//
// One layer, in the order it runs:
//
//   h  = x + O( attn( RoPE(Q(rms(x))), RoPE(K(rms(x))), V(rms(x)) ) )
//   y  = h + Down( silu(Gate(rms(h))) * Up(rms(h)) )
//
// The norms are RMSNorm with no bias, the MLP is SwiGLU, and the attention is
// grouped-query: 14 query heads share 2 key/value heads, seven to one.
//
// Every pass goes through a KV cache, including the ones that do not reuse it.
// Passing no cache means "use a fresh one", not "take a different path". That
// is what makes the phase 2 gate exact rather than approximate: there is only
// one attention implementation, so incremental decode and full recomputation
// cannot drift apart -- they run the same code over the same numbers in the
// same order, and must agree bit for bit.

import {
  rmsNorm, matVec, matVecBf16, swigluInPlace, ropeFrequencies, ropeInPlace,
} from '../core/ops.js';
import { groupedAttention } from '../core/attention.js';
import { KVCache } from './cache.js';

/** Rows of the embedding matrix to widen at a time when computing logits. */
const LM_HEAD_BLOCK = 4096;

export class ReferenceModel {
  constructor(config, weights) {
    this.config = config;
    this.weights = weights;
    this.invFreq = ropeFrequencies(config.headDim, config.ropeTheta);
  }

  /** A cache sized for `capacity` positions, to pass to forward(). */
  newCache(capacity) {
    return new KVCache(this.config, capacity);
  }

  /**
   * Run the model over `tokenIds`, continuing from whatever `cache` holds.
   *
   * With no cache this is a prefill over the whole sequence. With one, the
   * tokens are appended: their positions start at the cache's current length,
   * and attention sees everything already stored.
   *
   * `collectPresent` returns the per-layer keys and values in the layout the
   * ONNX export publishes. Those 48 tensors are the comparison surface for
   * phase 1, so matching them checks every layer rather than only the output.
   */
  async forward(tokenIds, {
    cache = null,
    collectPresent = false,
    logitsForAllPositions = false,
  } = {}) {
    const cfg = this.config;
    const seq = tokenIds.length;
    if (seq === 0) throw new Error('forward: need at least one token');
    const d = cfg.hiddenSize;
    const headDim = cfg.headDim;
    const kvDim = cfg.kvDim;

    const kv = cache ?? new KVCache(cfg, seq);
    const past = kv.allocate(seq);
    const total = past + seq;

    const x = await this.#embed(tokenIds);
    const present = collectPresent ? [] : null;

    const normed = new Float32Array(d);
    const q = new Float32Array(seq * d);
    const k = new Float32Array(seq * kvDim);
    const v = new Float32Array(seq * kvDim);
    const attnOut = new Float32Array(seq * d);
    const scores = new Float32Array(total);
    const gate = new Float32Array(cfg.intermediateSize);
    const up = new Float32Array(cfg.intermediateSize);
    const projected = new Float32Array(d);

    for (let layer = 0; layer < cfg.numLayers; layer++) {
      const w = await this.weights.layer(layer);

      // ---- projections for the new tokens ----
      for (let t = 0; t < seq; t++) {
        rmsNorm(x.subarray(t * d, (t + 1) * d), w.inputNorm, cfg.rmsNormEps, normed);
        matVec(w.wq, normed, d, d, w.bq, q.subarray(t * d, (t + 1) * d));
        matVec(w.wk, normed, kvDim, d, w.bk, k.subarray(t * kvDim, (t + 1) * kvDim));
        matVec(w.wv, normed, kvDim, d, w.bv, v.subarray(t * kvDim, (t + 1) * kvDim));

        // The rotary position is absolute, so it counts from the start of the
        // conversation and not from the start of this call. Using t here
        // instead of past + t is the classic KV-cache bug: it works perfectly
        // on the first pass and silently corrupts every one after it.
        const position = past + t;
        for (let h = 0; h < cfg.numHeads; h++) {
          ropeInPlace(q, t * d + h * headDim, headDim, position, this.invFreq);
        }
        for (let h = 0; h < cfg.numKVHeads; h++) {
          ropeInPlace(k, t * kvDim + h * headDim, headDim, position, this.invFreq);
        }
      }

      kv.writeAt(layer, past, k, v, seq);
      const allKeys = kv.keysUpTo(layer, total);
      const allValues = kv.valuesUpTo(layer, total);

      // ---- attention over everything the cache holds ----
      groupedAttention(q, allKeys, allValues, attnOut, scores, {
        seq, past, numHeads: cfg.numHeads, numKVHeads: cfg.numKVHeads, headDim,
      });

      for (let t = 0; t < seq; t++) {
        matVec(w.wo, attnOut.subarray(t * d, (t + 1) * d), d, d, null, projected);
        for (let i = 0; i < d; i++) x[t * d + i] += projected[i];
      }

      // ---- feed-forward ----
      for (let t = 0; t < seq; t++) {
        const row = x.subarray(t * d, (t + 1) * d);
        rmsNorm(row, w.postAttnNorm, cfg.rmsNormEps, normed);
        matVec(w.wGate, normed, cfg.intermediateSize, d, null, gate);
        matVec(w.wUp, normed, cfg.intermediateSize, d, null, up);
        swigluInPlace(gate, up);
        matVec(w.wDown, gate, d, cfg.intermediateSize, null, projected);
        for (let i = 0; i < d; i++) row[i] += projected[i];
      }
    }

    // Only now are the new positions visible, once every layer has written them.
    kv.commit(seq);
    if (collectPresent) {
      for (let layer = 0; layer < cfg.numLayers; layer++) present.push(kv.present(layer));
    }

    const finalNorm = await this.weights.finalNorm();
    const positions = logitsForAllPositions ? [...Array(seq).keys()] : [seq - 1];
    const logits = [];
    for (const t of positions) {
      rmsNorm(x.subarray(t * d, (t + 1) * d), finalNorm, cfg.rmsNormEps, normed);
      logits.push(await this.#logits(normed));
    }

    return {
      logits: logitsForAllPositions ? logits : logits[0],
      present,
      hidden: x,
      cache: kv,
    };
  }

  /**
   * Greedy decoding with a cache.
   *
   * The prompt goes through in one pass, then each new token is a pass of
   * length one. Yields as it goes so a caller can stream.
   */
  async *generate(promptIds, { maxTokens = 32, stopTokens = [], capacity = null } = {}) {
    const cache = this.newCache(capacity ?? promptIds.length + maxTokens);
    const stop = new Set(stopTokens);
    let next = promptIds;

    for (let step = 0; step < maxTokens; step++) {
      const { logits } = await this.forward(next, { cache });
      let best = 0;
      for (let i = 1; i < logits.length; i++) if (logits[i] > logits[best]) best = i;
      yield { token: best, logit: logits[best], step, context: cache.length };
      if (stop.has(best)) return;
      next = [best];
    }
  }

  /** Look up one embedding row per token and widen it. */
  async #embed(tokenIds) {
    const d = this.config.hiddenSize;
    const out = new Float32Array(tokenIds.length * d);
    const conv = new ArrayBuffer(4);
    const asU32 = new Uint32Array(conv);
    const asF32 = new Float32Array(conv);

    for (let t = 0; t < tokenIds.length; t++) {
      const id = tokenIds[t];
      if (!Number.isInteger(id) || id < 0 || id >= this.config.vocabSize) {
        throw new RangeError(`token ${id} at position ${t} is outside the vocabulary`);
      }
      const row = await this.weights.embeddingRows(id, id + 1);
      for (let i = 0; i < d; i++) {
        asU32[0] = row[i] << 16;
        out[t * d + i] = asF32[0];
      }
    }
    return out;
  }

  /**
   * The output projection, streamed.
   *
   * Embeddings are tied, so this multiplies by the same 151936 x 896 matrix the
   * input lookup used. Widened that is 544 MB, which this machine does not
   * have, so it is walked in blocks and multiplied straight from bfloat16.
   */
  async #logits(normed) {
    const vocab = this.config.vocabSize;
    const d = this.config.hiddenSize;
    const out = new Float32Array(vocab);
    for (let start = 0; start < vocab; start += LM_HEAD_BLOCK) {
      const end = Math.min(start + LM_HEAD_BLOCK, vocab);
      const block = await this.weights.embeddingRows(start, end);
      matVecBf16(block, normed, end - start, d, out.subarray(start, end));
    }
    return out;
  }
}
