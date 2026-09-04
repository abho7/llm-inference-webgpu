// The forward pass on the GPU.
//
// Same architecture as src/cpu/model.js, same order of operations, same KV
// cache semantics -- deliberately so, because phase 3's claim is that the two
// agree per layer and the comparison is only meaningful if they are really
// doing the same thing. Where this differs it is because the GPU forces it:
//
//   Weights are uploaded once as f16 and stay resident: 988 MB, since f16 is
//   two bytes per weight exactly as the stored bfloat16 is. The point of a GPU
//   backend is not that the weights get smaller, it is that they stop moving.
//
//   The conversion is close to free in accuracy as well. bfloat16 carries 7
//   mantissa bits and f16 carries 10, so no mantissa precision can be lost;
//   only f16's narrower exponent range costs anything, and it changes 0.0249%
//   of the weights, all of them below 6.1e-5 in magnitude.
//   validate/f16-fidelity.js measures it.
//
//   The embedding lookup stays on the CPU. It is a gather of `seq` rows of 896
//   floats; writing a kernel for it would add a failure mode to save nothing.
//
//   The rotary cosines and sines are precomputed on the CPU in f64 and
//   uploaded as a table, because this adapter's sin and cos are only accurate
//   to about 3e-5. See the comment in ops.js.
//
// Prefill runs the same kernels as decode, with the sequence in the third
// dispatch dimension. That keeps one code path rather than two, at the cost of
// a matrix-vector product per position instead of a real matrix multiply --
// which is a phase 5 problem, not a correctness one.

import { MATVEC, RMSNORM, ROPE, ATTENTION, SWIGLU, ADD, GROUP } from './kernels.js';
import { grid2d } from './device.js';
import { ropeTables } from '../core/ops.js';
import { bf16ToF32Array, f32ToF16Array } from '../core/dtype.js';
import { roundTrip } from '../core/quantize.js';

const LAYER_TENSORS = {
  inputNorm: 'input_layernorm.weight',
  wq: 'self_attn.q_proj.weight', bq: 'self_attn.q_proj.bias',
  wk: 'self_attn.k_proj.weight', bk: 'self_attn.k_proj.bias',
  wv: 'self_attn.v_proj.weight', bv: 'self_attn.v_proj.bias',
  wo: 'self_attn.o_proj.weight',
  postAttnNorm: 'post_attention_layernorm.weight',
  wGate: 'mlp.gate_proj.weight', wUp: 'mlp.up_proj.weight',
  wDown: 'mlp.down_proj.weight',
};

/** Biases stay f32: there are 896 of them per layer and they are added last. */
const F32_TENSORS = new Set(['bq', 'bk', 'bv']);

function pack(ctx, fields, label) {
  const size = Math.max(16, Math.ceil(fields.length / 4) * 16);
  const buf = new ArrayBuffer(size);
  const view = new DataView(buf);
  fields.forEach(([kind, value], i) => {
    if (kind === 'u32') view.setUint32(i * 4, value, true);
    else view.setFloat32(i * 4, value, true);
  });
  return ctx.uniform(new Uint32Array(buf), label);
}

export class GpuKVCache {
  constructor(ctx, config, capacity) {
    this.config = config;
    this.capacity = capacity;
    this.length = 0;
    const bytes = capacity * config.kvDim * 4;
    this.keys = [];
    this.values = [];
    for (let i = 0; i < config.numLayers; i++) {
      this.keys.push(ctx.empty(bytes, `kvKeys${i}`));
      this.values.push(ctx.empty(bytes, `kvValues${i}`));
    }
    this.byteLength = bytes * 2 * config.numLayers;
  }
  reset() { this.length = 0; }
}

export class GpuModel {
  constructor(ctx, config, weights, ropeTable, maxPositions) {
    this.ctx = ctx;
    this.config = config;
    this.w = weights;
    this.ropeTable = ropeTable;
    this.maxPositions = maxPositions;
    this.scratch = new Map();
  }

  /**
   * Upload every weight, converting bfloat16 to f16 one tensor at a time.
   *
   * One tensor at a time matters: the embedding alone is 272 MB of bfloat16,
   * and holding the whole model in both formats at once would need about
   * 1.5 GB of browser heap on a machine that does not have it.
   */
  static async load(ctx, safetensors, config, {
    maxPositions = 2048, onProgress, quantize = null,
  } = {}) {
    // `quantize` runs the weights through a quantization scheme and back before
    // they are uploaded. The kernels still read f16, so this measures what
    // quantization does to the model's answers without needing integer
    // kernels -- and since f16 storage is nearly exact for these bfloat16
    // weights (validate/f16-fidelity.js), the difference it measures really is
    // the quantization and not the container.
    const applyQuantization = (wide, name) => {
      if (!quantize) return wide;
      const shape = safetensors.info(name).shape;
      if (shape.length !== 2) return wide;   // norms and biases stay as they are
      const [rows, cols] = shape;
      if (quantize.groupSize && cols % quantize.groupSize !== 0) return wide;
      return roundTrip(wide, rows, cols, quantize.scheme, quantize.groupSize);
    };

    const upload = async (name, label) => {
      const info = safetensors.info(name);
      const raw = await safetensors.readRaw(name);
      const aligned = raw.byteOffset % 2 === 0 ? raw : new Uint8Array(raw);
      const bits = new Uint16Array(aligned.buffer, aligned.byteOffset, info.numel);
      const wide = applyQuantization(bf16ToF32Array(bits), name);
      const half = f32ToF16Array(wide);
      return ctx.upload(half, label);
    };
    const uploadF32 = async (name, label) => ctx.upload(await safetensors.readF32(name), label);

    const total = config.numLayers * Object.keys(LAYER_TENSORS).length + 2;
    let done = 0;
    const tick = (what) => {
      done++;
      if (onProgress) onProgress(done, total, what);
    };

    const weights = { layers: [] };
    weights.embed = await upload('model.embed_tokens.weight', 'embed');
    tick('embeddings');
    weights.finalNorm = await upload('model.norm.weight', 'finalNorm');
    tick('final norm');

    for (let i = 0; i < config.numLayers; i++) {
      const layer = {};
      for (const [key, suffix] of Object.entries(LAYER_TENSORS)) {
        const name = `model.layers.${i}.${suffix}`;
        layer[key] = F32_TENSORS.has(key)
          ? await uploadF32(name, `L${i}.${key}`)
          : await upload(name, `L${i}.${key}`);
        tick(`layer ${i} ${key}`);
      }
      weights.layers.push(layer);
    }

    const tables = ropeTables(config.headDim, config.ropeTheta, maxPositions);
    const ropeTable = {
      cos: ctx.upload(tables.cos, 'ropeCos'),
      sin: ctx.upload(tables.sin, 'ropeSin'),
    };
    return new GpuModel(ctx, config, weights, ropeTable, maxPositions);
  }

  newCache(capacity) { return new GpuKVCache(this.ctx, this.config, capacity); }

  /** A reusable buffer of `floats`, keyed by name, grown as needed. */
  #buf(name, floats) {
    const held = this.scratch.get(name);
    if (held && held.floats >= floats) return held.buffer;
    if (held) held.buffer.destroy();
    const buffer = this.ctx.empty(floats * 4, name);
    this.scratch.set(name, { buffer, floats });
    return buffer;
  }

  /**
   * Run the model over `tokenIds`, continuing from `cache`.
   *
   * `hidden` is the already-embedded input, [seq, hiddenSize], supplied by the
   * caller so that the embedding gather stays on the CPU.
   */
  async forward(hidden, seq, { cache, wantLogits = true, wantPresent = false } = {}) {
    const ctx = this.ctx;
    const cfg = this.config;
    const d = cfg.hiddenSize;
    const kvDim = cfg.kvDim;
    const ffn = cfg.intermediateSize;

    const past = cache.length;
    const total = past + seq;
    if (total > cache.capacity) {
      throw new RangeError(`cache holds ${past} of ${cache.capacity}; cannot add ${seq}`);
    }
    if (total > this.maxPositions) {
      throw new RangeError(`position ${total} exceeds the ${this.maxPositions}-entry rope table`);
    }

    const x = this.#buf('x', seq * d);
    ctx.device.queue.writeBuffer(x, 0, hidden.buffer, hidden.byteOffset, hidden.byteLength);

    const normed = this.#buf('normed', seq * d);
    const q = this.#buf('q', seq * d);
    const k = this.#buf('k', seq * kvDim);
    const v = this.#buf('v', seq * kvDim);
    const attnOut = this.#buf('attnOut', seq * d);
    const scores = this.#buf('scores', cfg.numHeads * seq * Math.max(total, 1));
    const gate = this.#buf('gate', seq * ffn);
    const up = this.#buf('up', seq * ffn);
    const projected = this.#buf('projected', seq * d);
    const noBias = this.#buf('noBias', 1);

    const normParams = pack(ctx, [['u32', seq], ['u32', d], ['f32', cfg.rmsNormEps], ['u32', 0]]);
    const addParams = pack(ctx, [['u32', seq * d], ['u32', 0], ['u32', 0], ['u32', 0]]);
    const swigluParams = pack(ctx, [['u32', seq * ffn], ['u32', 0], ['u32', 0], ['u32', 0]]);

    const mv = (W, input, output, rows, cols, bias) => {
      const { grid, width } = grid2d(rows);
      const params = pack(ctx, [
        ['u32', rows], ['u32', cols], ['u32', bias ? 1 : 0], ['u32', width],
      ]);
      ctx.dispatch(MATVEC, [W, input, bias ?? noBias, output, params],
        [grid[0], grid[1], seq], 'matvec');
    };

    for (let layer = 0; layer < cfg.numLayers; layer++) {
      const w = this.w.layers[layer];

      ctx.dispatch(RMSNORM, [x, w.inputNorm, normed, normParams], [seq], 'rmsnorm');
      mv(w.wq, normed, q, d, d, w.bq);
      mv(w.wk, normed, k, kvDim, d, w.bk);
      mv(w.wv, normed, v, kvDim, d, w.bv);

      const ropeQ = pack(ctx, [['u32', seq], ['u32', cfg.numHeads], ['u32', cfg.headDim], ['u32', past]]);
      const ropeK = pack(ctx, [['u32', seq], ['u32', cfg.numKVHeads], ['u32', cfg.headDim], ['u32', past]]);
      const qPairs = seq * cfg.numHeads * (cfg.headDim / 2);
      const kPairs = seq * cfg.numKVHeads * (cfg.headDim / 2);
      ctx.dispatch(ROPE, [q, this.ropeTable.cos, this.ropeTable.sin, ropeQ],
        [Math.ceil(qPairs / 64)], 'rope-q');
      ctx.dispatch(ROPE, [k, this.ropeTable.cos, this.ropeTable.sin, ropeK],
        [Math.ceil(kPairs / 64)], 'rope-k');

      // Append this layer's keys and values to the cache.
      const encoder = ctx.device.createCommandEncoder({ label: 'kv-append' });
      encoder.copyBufferToBuffer(k, 0, cache.keys[layer], past * kvDim * 4, seq * kvDim * 4);
      encoder.copyBufferToBuffer(v, 0, cache.values[layer], past * kvDim * 4, seq * kvDim * 4);
      ctx.device.queue.submit([encoder.finish()]);

      const attnParams = pack(ctx, [
        ['u32', seq], ['u32', past], ['u32', cfg.numHeads], ['u32', cfg.numKVHeads],
        ['u32', cfg.headDim], ['u32', cfg.kvGroupSize], ['u32', total], ['u32', 0],
      ]);
      ctx.dispatch(ATTENTION,
        [q, cache.keys[layer], cache.values[layer], attnOut, scores, attnParams],
        [seq, cfg.numHeads], 'attention');

      mv(w.wo, attnOut, projected, d, d, null);
      ctx.dispatch(ADD, [x, projected, addParams], [Math.ceil(seq * d / GROUP)], 'residual');

      ctx.dispatch(RMSNORM, [x, w.postAttnNorm, normed, normParams], [seq], 'rmsnorm2');
      mv(w.wGate, normed, gate, ffn, d, null);
      mv(w.wUp, normed, up, ffn, d, null);
      ctx.dispatch(SWIGLU, [gate, up, swigluParams], [Math.ceil(seq * ffn / GROUP)], 'swiglu');
      mv(w.wDown, gate, projected, d, ffn, null);
      ctx.dispatch(ADD, [x, projected, addParams], [Math.ceil(seq * d / GROUP)], 'residual2');
    }

    cache.length = total;

    let logits = null;
    if (wantLogits) {
      // Only the final position's logits are wanted, so normalise that row and
      // run the output projection over it alone rather than over the sequence.
      const lastRow = this.#buf('lastRow', d);
      const encoder = ctx.device.createCommandEncoder({ label: 'last-row' });
      encoder.copyBufferToBuffer(x, (seq - 1) * d * 4, lastRow, 0, d * 4);
      ctx.device.queue.submit([encoder.finish()]);

      const oneRowNorm = pack(ctx, [['u32', 1], ['u32', d], ['f32', cfg.rmsNormEps], ['u32', 0]]);
      const normedLast = this.#buf('normedLast', d);
      ctx.dispatch(RMSNORM, [lastRow, this.w.finalNorm, normedLast, oneRowNorm], [1], 'finalNorm');

      const logitBuf = this.#buf('logits', cfg.vocabSize);
      const { grid, width } = grid2d(cfg.vocabSize);
      const params = pack(ctx, [
        ['u32', cfg.vocabSize], ['u32', d], ['u32', 0], ['u32', width],
      ]);
      ctx.dispatch(MATVEC, [this.w.embed, normedLast, noBias, logitBuf, params],
        [grid[0], grid[1], 1], 'lm_head');
      await ctx.done();
      logits = await ctx.readF32(logitBuf, cfg.vocabSize);
    }

    let present = null;
    if (wantPresent) {
      await ctx.done();
      present = [];
      for (let layer = 0; layer < cfg.numLayers; layer++) {
        present.push({
          key: await ctx.readF32(cache.keys[layer], total * kvDim),
          value: await ctx.readF32(cache.values[layer], total * kvDim),
        });
      }
    }

    await ctx.done();
    return { logits, present, hiddenBuffer: x };
  }
}
