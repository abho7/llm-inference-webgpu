// The key/value cache.
//
// Attention at position t needs the keys and values of every position up to t.
// Without a cache, generating token n recomputes all of them, which makes
// decoding quadratic in the output length: phase 1 measured 0.06 tokens per
// second, and most of that was recomputing work it had already done.
//
// The cache holds them instead. What makes this checkable rather than merely
// plausible is that caching changes nothing about the arithmetic. Position s's
// key depends only on positions up to s, so the value computed while s was the
// newest token is the same value a full recomputation would produce later --
// not close to it, the same bits. That gives phase 2 an exact gate rather than
// a tolerance: incremental decode must equal full recomputation exactly.
//
// Storage is [position, kvHead * headDim] per layer, matching the layout the
// projections already write, so appending is a copy and nothing is transposed.

export class KVCache {
  #keys;      // Float32Array per layer, capacity * kvDim
  #values;
  #length = 0;

  constructor(config, capacity) {
    if (!Number.isInteger(capacity) || capacity < 1) {
      throw new RangeError(`cache capacity must be a positive integer, got ${capacity}`);
    }
    this.config = config;
    this.capacity = capacity;
    this.#keys = [];
    this.#values = [];
    for (let layer = 0; layer < config.numLayers; layer++) {
      this.#keys.push(new Float32Array(capacity * config.kvDim));
      this.#values.push(new Float32Array(capacity * config.kvDim));
    }
  }

  /** Positions currently held. */
  get length() { return this.#length; }

  /** Bytes held, for reporting: this is the memory a longer context costs. */
  get byteLength() {
    return this.config.numLayers * this.capacity * this.config.kvDim * 4 * 2;
  }

  /**
   * Reserve room for `count` more positions and return where they start.
   *
   * Deliberately separate from commit(): every layer writes the same positions,
   * so the visible length must not advance until all of them have. Reserving up
   * front means a sequence that does not fit fails before anything is written.
   */
  allocate(count) {
    if (this.#length + count > this.capacity) {
      throw new RangeError(
        `cache holds ${this.#length} of ${this.capacity} positions; ` +
        `cannot add ${count} more`,
      );
    }
    return this.#length;
  }

  /** Write one layer's keys and values for `count` positions starting at `at`. */
  writeAt(layer, at, keys, values, count) {
    const { kvDim } = this.config;
    this.#keys[layer].set(keys.subarray(0, count * kvDim), at * kvDim);
    this.#values[layer].set(values.subarray(0, count * kvDim), at * kvDim);
  }

  /**
   * A layer's keys over the first `n` positions.
   *
   * Takes an explicit count because attention has to see the positions written
   * during this pass, which are reserved but not yet committed.
   */
  keysUpTo(layer, n) {
    return this.#keys[layer].subarray(0, n * this.config.kvDim);
  }

  valuesUpTo(layer, n) {
    return this.#values[layer].subarray(0, n * this.config.kvDim);
  }

  /** Make the reserved positions visible, once every layer has written them. */
  commit(count) {
    this.#length += count;
  }

  /** All keys held for a layer: [length, kvDim]. */
  keys(layer) { return this.keysUpTo(layer, this.#length); }

  values(layer) { return this.valuesUpTo(layer, this.#length); }

  /**
   * Reshape a layer's cache into the layout the ONNX export publishes:
   * [kvHeads, length, headDim].
   */
  present(layer) {
    const { numKVHeads, headDim, kvDim } = this.config;
    const n = this.#length;
    const key = new Float32Array(numKVHeads * n * headDim);
    const value = new Float32Array(numKVHeads * n * headDim);
    const k = this.#keys[layer];
    const v = this.#values[layer];
    for (let h = 0; h < numKVHeads; h++) {
      for (let t = 0; t < n; t++) {
        const from = t * kvDim + h * headDim;
        const to = (h * n + t) * headDim;
        for (let i = 0; i < headDim; i++) {
          key[to + i] = k[from + i];
          value[to + i] = v[from + i];
        }
      }
    }
    return { key, value };
  }

  /** Forget everything, keeping the allocation. */
  reset() { this.#length = 0; }
}
