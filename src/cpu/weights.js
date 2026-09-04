// Weight access for the reference model.
//
// Holding the whole model as f32 would need about 2 GB, and this machine has
// well under that free, so weights are pulled from the safetensors file one
// layer at a time and released. The file descriptor stays open and the
// operating system's page cache does the caching, which is the right layer for
// it to happen at -- a second cache here would just compete with that one for
// the same scarce memory.
//
// The output projection is a separate case. It reuses the 151936 x 896
// embedding matrix, which is 544 MB as f32, so it is never widened at all: the
// logit computation streams it in row blocks straight from bfloat16.

import { bf16ToF32Array } from '../core/dtype.js';
import { roundTrip } from '../core/quantize.js';

const LAYER_TENSORS = [
  'input_layernorm.weight',
  'self_attn.q_proj.weight', 'self_attn.q_proj.bias',
  'self_attn.k_proj.weight', 'self_attn.k_proj.bias',
  'self_attn.v_proj.weight', 'self_attn.v_proj.bias',
  'self_attn.o_proj.weight',
  'post_attention_layernorm.weight',
  'mlp.gate_proj.weight', 'mlp.up_proj.weight', 'mlp.down_proj.weight',
];

/** Short names, so the forward pass reads like the architecture. */
const SHORT = {
  'input_layernorm.weight': 'inputNorm',
  'self_attn.q_proj.weight': 'wq', 'self_attn.q_proj.bias': 'bq',
  'self_attn.k_proj.weight': 'wk', 'self_attn.k_proj.bias': 'bk',
  'self_attn.v_proj.weight': 'wv', 'self_attn.v_proj.bias': 'bv',
  'self_attn.o_proj.weight': 'wo',
  'post_attention_layernorm.weight': 'postAttnNorm',
  'mlp.gate_proj.weight': 'wGate', 'mlp.up_proj.weight': 'wUp',
  'mlp.down_proj.weight': 'wDown',
};

/** The inverse of SHORT, so a scratch buffer can be traced back to its tensor. */
const SHORT_TO_NAME = Object.fromEntries(
  Object.entries(SHORT).map(([name, short]) => [short, name]),
);

export class Weights {
  #st;
  #config;
  #layerCache = new Map();   // layer index -> widened tensors
  #cacheLimit;
  #resident = null;          // layer index -> raw bfloat16 per tensor
  #scratch = null;           // reused f32 buffers, one set for all layers

  /**
   * `resident: true` holds every layer's weights in memory as bfloat16 and
   * widens them into reusable buffers on demand.
   *
   * The trade is 716 MB of residency against not touching the disk. Streaming
   * moves about 1.5 GB per forward pass, and with little free memory the page
   * cache cannot absorb that, so decoding becomes disk-bound. Widening from RAM
   * is a shift over 15M elements per layer, which is far cheaper than a read.
   *
   * It stays opt-in because 716 MB is not obviously affordable here, and the
   * point of the streaming default is that the engine runs at all on a machine
   * that cannot hold the model.
   */
  /**
   * `quantize` runs each 2D weight matrix through a quantization scheme and
   * back before it is used, so the reference implementation can be evaluated
   * under int8 or int4 without integer kernels. Norms and biases are left
   * alone, which is what weight-only quantization means in practice.
   *
   * This re-quantizes on every forward pass rather than once, which makes a
   * quantized run several times slower than an unquantized one -- int8 took
   * 808 seconds against f16's 150 on the same passage. Caching the result would
   * need either 1.5 GB of f32, which this machine does not have, or a second
   * lossy step to store it more compactly, which would contaminate the very
   * measurement the option exists to make. Slow and exact wins.
   */
  constructor(safetensors, config, {
    cacheLayers = 1, resident = false, quantize = null,
  } = {}) {
    this.#st = safetensors;
    this.#config = config;
    this.#cacheLimit = Math.max(1, cacheLayers);
    this.wantResident = resident;
    this.quantize = quantize;
    this.bytesRead = 0;
  }

  /** Apply the configured scheme to one named tensor, in place where possible. */
  #applyQuantization(values, name) {
    if (!this.quantize) return values;
    const shape = this.#st.info(name).shape;
    if (shape.length !== 2) return values;
    const [rows, cols] = shape;
    if (this.quantize.groupSize && cols % this.quantize.groupSize !== 0) return values;
    const back = roundTrip(values, rows, cols, this.quantize.scheme, this.quantize.groupSize);
    values.set(back);
    return values;
  }

  /** Pull every layer's weights into memory as bfloat16. Returns bytes held. */
  async makeResident() {
    if (this.#resident) return this.residentBytes;
    this.#resident = new Map();
    let held = 0;
    for (let layer = 0; layer < this.#config.numLayers; layer++) {
      const prefix = `model.layers.${layer}.`;
      const raw = {};
      for (const name of LAYER_TENSORS) {
        const bytes = await this.#st.readRaw(prefix + name);
        const aligned = bytes.byteOffset % 2 === 0 ? bytes : new Uint8Array(bytes);
        raw[SHORT[name]] = new Uint16Array(
          aligned.buffer, aligned.byteOffset, bytes.byteLength / 2,
        );
        held += bytes.byteLength;
      }
      this.#resident.set(layer, raw);
    }
    this.residentBytes = held;
    return held;
  }

  /** The final RMSNorm gain, small enough to keep resident. */
  async finalNorm() {
    if (!this._finalNorm) this._finalNorm = await this.#st.readF32('model.norm.weight');
    return this._finalNorm;
  }

  /** One layer's weights, widened to f32. */
  async layer(index) {
    if (this.wantResident && !this.#resident) await this.makeResident();

    if (this.#resident) {
      // Widen into buffers shared by every layer. Safe because a layer's
      // weights are only live until the next layer is asked for, which is
      // exactly how the forward pass walks them.
      const raw = this.#resident.get(index);
      if (!this.#scratch) {
        this.#scratch = {};
        for (const key of Object.keys(raw)) {
          this.#scratch[key] = new Float32Array(raw[key].length);
        }
      }
      for (const key of Object.keys(raw)) {
        bf16ToF32Array(raw[key], this.#scratch[key]);
      }
      if (this.quantize) {
        const prefix = `model.layers.${index}.`;
        for (const [key, suffix] of Object.entries(SHORT_TO_NAME)) {
          if (this.#scratch[key]) this.#applyQuantization(this.#scratch[key], prefix + suffix);
        }
      }
      return this.#scratch;
    }

    const cached = this.#layerCache.get(index);
    if (cached) return cached;

    const prefix = `model.layers.${index}.`;
    const out = {};
    for (const name of LAYER_TENSORS) {
      out[SHORT[name]] = await this.#st.readF32(prefix + name);
      this.bytesRead += this.#st.info(prefix + name).byteLength;
    }

    // Evict in insertion order. The forward pass walks layers 0..23 in
    // sequence, so anything held is the least likely to be wanted next.
    if (this.#layerCache.size >= this.#cacheLimit) {
      const oldest = this.#layerCache.keys().next().value;
      this.#layerCache.delete(oldest);
    }
    this.#layerCache.set(index, out);
    return out;
  }

  /**
   * Rows [start, end) of the embedding matrix, still as bfloat16.
   *
   * Used both for looking up input embeddings and for the tied output
   * projection. Returning raw bits rather than f32 halves the bytes moved and
   * lets the caller widen only what it multiplies.
   */
  async embeddingRows(start, end) {
    const info = this.#st.info('model.embed_tokens.weight');
    const cols = info.shape[1];
    const bytesPerRow = cols * 2;
    const bytes = await this.#st.readRawRange(
      'model.embed_tokens.weight', start * bytesPerRow, (end - start) * bytesPerRow,
    );
    this.bytesRead += bytes.byteLength;
    const aligned = bytes.byteOffset % 2 === 0 ? bytes : new Uint8Array(bytes);
    return new Uint16Array(aligned.buffer, aligned.byteOffset, (end - start) * cols);
  }

  get vocabSize() { return this.#st.info('model.embed_tokens.weight').shape[0]; }
  get hiddenSize() { return this.#st.info('model.embed_tokens.weight').shape[1]; }
}
