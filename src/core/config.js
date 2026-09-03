// The model's architecture, parsed from config.json and checked against the
// weights that claim to implement it.
//
// The useful part is not the parsing, it is expectedTensors(): the config
// implies exactly which tensors must exist and what shape each one has, so the
// file can be checked against the architecture instead of trusted. A config
// that disagrees with its weights otherwise shows up much later as logits that
// are subtly wrong, which is a far worse place to find out.

const REQUIRED_NUMBERS = [
  'hidden_size', 'intermediate_size', 'num_hidden_layers',
  'num_attention_heads', 'num_key_value_heads', 'vocab_size',
  'max_position_embeddings', 'rope_theta', 'rms_norm_eps',
];

export class ModelConfig {
  constructor(raw) {
    for (const key of REQUIRED_NUMBERS) {
      if (typeof raw[key] !== 'number' || !Number.isFinite(raw[key])) {
        throw new Error(`config: ${key} must be a finite number, got ${JSON.stringify(raw[key])}`);
      }
    }

    this.modelType = raw.model_type;
    this.architectures = raw.architectures ?? [];
    this.hiddenSize = raw.hidden_size;
    this.intermediateSize = raw.intermediate_size;
    this.numLayers = raw.num_hidden_layers;
    this.numHeads = raw.num_attention_heads;
    this.numKVHeads = raw.num_key_value_heads;
    this.vocabSize = raw.vocab_size;
    this.maxPositionEmbeddings = raw.max_position_embeddings;
    this.ropeTheta = raw.rope_theta;
    this.rmsNormEps = raw.rms_norm_eps;
    this.hiddenAct = raw.hidden_act ?? 'silu';
    this.tieWordEmbeddings = raw.tie_word_embeddings === true;
    this.useSlidingWindow = raw.use_sliding_window === true;
    this.slidingWindow = raw.sliding_window ?? null;
    this.bosTokenId = raw.bos_token_id ?? null;
    this.eosTokenId = raw.eos_token_id ?? null;

    // Refuse anything this engine does not actually implement. Silently running
    // the wrong architecture is the failure mode worth spending an error on.
    if (this.modelType !== 'qwen2') {
      throw new Error(`config: only qwen2 is implemented, got ${JSON.stringify(this.modelType)}`);
    }
    if (this.hiddenAct !== 'silu') {
      throw new Error(`config: only the silu/SwiGLU MLP is implemented, got ${this.hiddenAct}`);
    }
    if (this.useSlidingWindow) {
      throw new Error('config: sliding-window attention is not implemented');
    }
    if (this.hiddenSize % this.numHeads !== 0) {
      throw new Error(
        `config: hidden_size ${this.hiddenSize} is not divisible by ` +
        `num_attention_heads ${this.numHeads}`,
      );
    }
    if (this.numHeads % this.numKVHeads !== 0) {
      throw new Error(
        `config: num_attention_heads ${this.numHeads} is not divisible by ` +
        `num_key_value_heads ${this.numKVHeads}, so the query heads cannot be ` +
        'grouped evenly over the key/value heads',
      );
    }

    /** Width of one attention head. */
    this.headDim = this.hiddenSize / this.numHeads;
    /** Query heads sharing each key/value head. 1 means plain multi-head. */
    this.kvGroupSize = this.numHeads / this.numKVHeads;
    /** Width of the packed key (or value) projection output. */
    this.kvDim = this.numKVHeads * this.headDim;

    Object.freeze(this);
  }

  static parse(text) {
    return new ModelConfig(JSON.parse(text));
  }

  /**
   * Every tensor this architecture requires, mapped to its exact shape.
   *
   * Qwen2 carries biases on the query, key and value projections but not on the
   * output projection or the MLP, which is a real asymmetry in the architecture
   * rather than an oversight here.
   */
  expectedTensors() {
    const shapes = new Map();
    const d = this.hiddenSize;

    shapes.set('model.embed_tokens.weight', [this.vocabSize, d]);
    shapes.set('model.norm.weight', [d]);

    for (let i = 0; i < this.numLayers; i++) {
      const p = `model.layers.${i}.`;
      shapes.set(`${p}input_layernorm.weight`, [d]);
      shapes.set(`${p}post_attention_layernorm.weight`, [d]);

      shapes.set(`${p}self_attn.q_proj.weight`, [d, d]);
      shapes.set(`${p}self_attn.q_proj.bias`, [d]);
      shapes.set(`${p}self_attn.k_proj.weight`, [this.kvDim, d]);
      shapes.set(`${p}self_attn.k_proj.bias`, [this.kvDim]);
      shapes.set(`${p}self_attn.v_proj.weight`, [this.kvDim, d]);
      shapes.set(`${p}self_attn.v_proj.bias`, [this.kvDim]);
      shapes.set(`${p}self_attn.o_proj.weight`, [d, d]);

      shapes.set(`${p}mlp.gate_proj.weight`, [this.intermediateSize, d]);
      shapes.set(`${p}mlp.up_proj.weight`, [this.intermediateSize, d]);
      shapes.set(`${p}mlp.down_proj.weight`, [d, this.intermediateSize]);
    }

    // Untied models carry a separate output projection; this one reuses the
    // embedding matrix, which is why lm_head.weight is absent from the file.
    if (!this.tieWordEmbeddings) {
      shapes.set('lm_head.weight', [this.vocabSize, d]);
    }
    return shapes;
  }

  /**
   * Assert that `safetensors` contains exactly the tensors this config implies,
   * each with exactly the implied shape. Extra tensors are an error too: an
   * unread tensor means the engine is ignoring part of the model.
   */
  checkAgainst(safetensors) {
    const expected = this.expectedTensors();
    const problems = [];

    for (const [name, shape] of expected) {
      if (!safetensors.has(name)) {
        problems.push(`missing ${name}`);
        continue;
      }
      const got = safetensors.info(name).shape;
      if (got.length !== shape.length || got.some((d, i) => d !== shape[i])) {
        problems.push(`${name}: expected shape [${shape}], file has [${got}]`);
      }
    }
    for (const name of safetensors.names()) {
      if (!expected.has(name)) problems.push(`unexpected tensor ${name}`);
    }

    if (problems.length) {
      throw new Error(
        `weights do not match config (${problems.length} problem` +
        `${problems.length === 1 ? '' : 's'}):\n  ${problems.slice(0, 10).join('\n  ')}` +
        (problems.length > 10 ? `\n  ... and ${problems.length - 10} more` : ''),
      );
    }
    return expected.size;
  }

  /** Total parameters implied by the config, for reporting. */
  parameterCount() {
    let total = 0;
    for (const shape of this.expectedTensors().values()) {
      total += shape.reduce((a, b) => a * b, 1);
    }
    return total;
  }

  describe() {
    return [
      `${this.modelType} / ${this.architectures.join(',')}`,
      `${this.numLayers} layers, d=${this.hiddenSize}, ffn=${this.intermediateSize}`,
      `${this.numHeads} query heads / ${this.numKVHeads} kv heads ` +
      `(${this.kvGroupSize}:1 GQA), head_dim=${this.headDim}`,
      `vocab=${this.vocabSize}, rope_theta=${this.ropeTheta}, ` +
      `tied_embeddings=${this.tieWordEmbeddings}`,
      `${(this.parameterCount() / 1e6).toFixed(1)}M parameters`,
    ].join('\n');
  }
}
