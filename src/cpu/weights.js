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

export class Weights {
  #st;
  #config;
  #layerCache = new Map();   // layer index -> widened tensors
  #cacheLimit;

  constructor(safetensors, config, { cacheLayers = 1 } = {}) {
    this.#st = safetensors;
    this.#config = config;
    this.#cacheLimit = Math.max(1, cacheLayers);
    this.bytesRead = 0;
  }

  /** The final RMSNorm gain, small enough to keep resident. */
  async finalNorm() {
    if (!this._finalNorm) this._finalNorm = await this.#st.readF32('model.norm.weight');
    return this._finalNorm;
  }

  /** One layer's weights, widened to f32. */
  async layer(index) {
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
