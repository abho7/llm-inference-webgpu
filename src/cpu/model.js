// The Qwen2 forward pass, as a reference implementation.
//
// Correctness first and speed nowhere: this is the thing the GPU backend gets
// compared against, so it is written to be read and to be obviously the
// architecture on the page. Every reduction accumulates in f64 (see ops.js),
// and no attempt is made to fuse, tile or cache anything.
//
// Phase 1 recomputes the whole sequence on every call. The KV cache arrives in
// phase 2, and the gate for it is that incremental decode must reproduce what
// this does exactly.
//
// One layer, in the order it runs:
//
//   h  = x + O( attn( RoPE(Q(rms(x))), RoPE(K(rms(x))), V(rms(x)) ) )
//   y  = h + Down( silu(Gate(rms(h))) * Up(rms(h)) )
//
// The norms are RMSNorm with no bias, the MLP is SwiGLU, and the attention is
// grouped-query: 14 query heads share 2 key/value heads, seven to one.

import {
  rmsNorm, matVec, matVecBf16, softmaxInPlace, swigluInPlace,
  ropeFrequencies, ropeInPlace,
} from '../core/ops.js';

/** Rows of the embedding matrix to widen at a time when computing logits. */
const LM_HEAD_BLOCK = 4096;

export class ReferenceModel {
  constructor(config, weights) {
    this.config = config;
    this.weights = weights;
    this.invFreq = ropeFrequencies(config.headDim, config.ropeTheta);
  }

  /**
   * Run the model over a whole sequence.
   *
   * Returns the logits for the final position, and optionally the per-layer
   * key and value tensors. Those are not a debugging convenience: they are the
   * comparison surface. The ONNX export publishes the same 48 tensors as
   * `present.N.key` / `present.N.value`, so matching them checks every layer
   * rather than only the output, and it does so without touching the graph.
   */
  async forward(tokenIds, { collectPresent = false, logitsForAllPositions = false } = {}) {
    const cfg = this.config;
    const seq = tokenIds.length;
    if (seq === 0) throw new Error('forward: need at least one token');
    const d = cfg.hiddenSize;
    const headDim = cfg.headDim;
    const kvDim = cfg.kvDim;

    let x = await this.#embed(tokenIds);
    const present = collectPresent ? [] : null;

    const normed = new Float32Array(d);
    const q = new Float32Array(seq * d);
    const k = new Float32Array(seq * kvDim);
    const v = new Float32Array(seq * kvDim);
    const attnOut = new Float32Array(seq * d);
    const scores = new Float32Array(seq);
    const gate = new Float32Array(cfg.intermediateSize);
    const up = new Float32Array(cfg.intermediateSize);
    const projected = new Float32Array(d);

    for (let layer = 0; layer < cfg.numLayers; layer++) {
      const w = await this.weights.layer(layer);

      // ---- attention ----
      for (let t = 0; t < seq; t++) {
        rmsNorm(x.subarray(t * d, (t + 1) * d), w.inputNorm, cfg.rmsNormEps, normed);
        matVec(w.wq, normed, d, d, w.bq, q.subarray(t * d, (t + 1) * d));
        matVec(w.wk, normed, kvDim, d, w.bk, k.subarray(t * kvDim, (t + 1) * kvDim));
        matVec(w.wv, normed, kvDim, d, w.bv, v.subarray(t * kvDim, (t + 1) * kvDim));

        // Position is the index in the sequence. Rotation is applied per head,
        // to queries and keys but never to values.
        for (let h = 0; h < cfg.numHeads; h++) {
          ropeInPlace(q, t * d + h * headDim, headDim, t, this.invFreq);
        }
        for (let h = 0; h < cfg.numKVHeads; h++) {
          ropeInPlace(k, t * kvDim + h * headDim, headDim, t, this.invFreq);
        }
      }

      if (collectPresent) present.push(this.#packPresent(k, v, seq));

      const scale = 1 / Math.sqrt(headDim);
      attnOut.fill(0);
      for (let h = 0; h < cfg.numHeads; h++) {
        // Grouped-query attention: query heads are laid out so that a run of
        // kvGroupSize consecutive heads shares one key/value head.
        const kvHead = Math.floor(h / cfg.kvGroupSize);
        const kvBase = kvHead * headDim;

        for (let t = 0; t < seq; t++) {
          const qBase = t * d + h * headDim;
          // Causal: position t attends to 0..t inclusive and nothing later.
          for (let s = 0; s <= t; s++) {
            const kBase = s * kvDim + kvBase;
            let dot = 0;
            for (let i = 0; i < headDim; i++) dot += q[qBase + i] * k[kBase + i];
            scores[s] = dot * scale;
          }
          softmaxInPlace(scores, t + 1);

          const outBase = t * d + h * headDim;
          for (let s = 0; s <= t; s++) {
            const weight = scores[s];
            if (weight === 0) continue;
            const vBase = s * kvDim + kvBase;
            for (let i = 0; i < headDim; i++) attnOut[outBase + i] += weight * v[vBase + i];
          }
        }
      }

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
    };
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

  /**
   * Reshape keys and values into the layout the ONNX export publishes:
   * [kvHeads, seq, headDim], where ours are stored [seq, kvHeads * headDim].
   */
  #packPresent(k, v, seq) {
    const { numKVHeads, headDim, kvDim } = this.config;
    const key = new Float32Array(numKVHeads * seq * headDim);
    const value = new Float32Array(numKVHeads * seq * headDim);
    for (let h = 0; h < numKVHeads; h++) {
      for (let t = 0; t < seq; t++) {
        const from = t * kvDim + h * headDim;
        const to = (h * seq + t) * headDim;
        for (let i = 0; i < headDim; i++) {
          key[to + i] = k[from + i];
          value[to + i] = v[from + i];
        }
      }
    }
    return { key, value };
  }
}
