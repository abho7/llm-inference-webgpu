import test from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import { KVCache } from '../src/cpu/cache.js';
import { ModelConfig } from '../src/core/config.js';
import { Safetensors } from '../src/core/safetensors.js';
import { FileSource } from '../src/core/source-node.js';
import { Weights } from '../src/cpu/weights.js';
import { ReferenceModel } from '../src/cpu/model.js';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const MODEL = join(ROOT, 'weights', 'model.safetensors');
const CONFIG = join(ROOT, 'weights', 'config.json');

// A miniature config, so the cache can be tested without a gigabyte of weights.
const tiny = new ModelConfig({
  model_type: 'qwen2',
  hidden_size: 8,
  intermediate_size: 16,
  num_hidden_layers: 2,
  num_attention_heads: 4,
  num_key_value_heads: 2,
  vocab_size: 32,
  max_position_embeddings: 128,
  rope_theta: 10000,
  rms_norm_eps: 1e-6,
  tie_word_embeddings: true,
});

// kvDim = 2 kv heads * 2 head width = 4
const KV = tiny.kvDim;

function filled(value, count = KV) {
  return Float32Array.from({ length: count }, (_, i) => value + i / 100);
}

test('a new cache is empty and sized as configured', () => {
  const cache = new KVCache(tiny, 10);
  assert.equal(cache.length, 0);
  assert.equal(cache.capacity, 10);
  // 2 layers * 10 positions * 4 kvDim * 4 bytes * 2 (keys and values)
  assert.equal(cache.byteLength, 2 * 10 * KV * 4 * 2);
});

test('rejects a nonsensical capacity instead of allocating nothing', () => {
  assert.throws(() => new KVCache(tiny, 0), /positive integer/);
  assert.throws(() => new KVCache(tiny, -1), /positive integer/);
  assert.throws(() => new KVCache(tiny, 1.5), /positive integer/);
});

test('length advances only on commit, not on write', () => {
  // This is the whole reason the two are separate. Every layer writes the same
  // positions, so exposing them after the first layer had written would let
  // layer 1 attend to keys layer 0 has not produced yet.
  const cache = new KVCache(tiny, 4);
  const at = cache.allocate(1);
  assert.equal(at, 0);
  cache.writeAt(0, at, filled(1), filled(2), 1);
  assert.equal(cache.length, 0, 'still invisible');
  cache.commit(1);
  assert.equal(cache.length, 1);
});

test('allocate refuses to overflow the cache, before anything is written', () => {
  const cache = new KVCache(tiny, 3);
  cache.allocate(3);
  cache.commit(3);
  assert.throws(() => cache.allocate(1), /holds 3 of 3 positions/);
  assert.equal(cache.length, 3, 'the failed allocation changed nothing');
});

test('appends land at consecutive positions and read back unchanged', () => {
  const cache = new KVCache(tiny, 4);
  for (let step = 0; step < 3; step++) {
    const at = cache.allocate(1);
    assert.equal(at, step);
    cache.writeAt(0, at, filled(10 * step), filled(100 * step), 1);
    cache.writeAt(1, at, filled(20 * step), filled(200 * step), 1);
    cache.commit(1);
  }
  assert.equal(cache.length, 3);

  const keys = cache.keys(0);
  assert.equal(keys.length, 3 * KV);
  for (let step = 0; step < 3; step++) {
    for (let i = 0; i < KV; i++) {
      assert.equal(keys[step * KV + i], Math.fround(10 * step + i / 100),
        `layer 0 key at position ${step}, element ${i}`);
    }
  }
  // Layers must not share storage.
  assert.notEqual(cache.keys(1)[KV], cache.keys(0)[KV]);
});

test('a multi-token write is equivalent to the same tokens written one by one', () => {
  const bulk = new KVCache(tiny, 4);
  const single = new KVCache(tiny, 4);

  const keys = new Float32Array(3 * KV);
  const values = new Float32Array(3 * KV);
  for (let i = 0; i < 3 * KV; i++) {
    keys[i] = i * 1.5;
    values[i] = -i;
  }

  const at = bulk.allocate(3);
  bulk.writeAt(0, at, keys, values, 3);
  bulk.writeAt(1, at, keys, values, 3);
  bulk.commit(3);

  for (let t = 0; t < 3; t++) {
    const where = single.allocate(1);
    const k = keys.subarray(t * KV, (t + 1) * KV);
    const v = values.subarray(t * KV, (t + 1) * KV);
    single.writeAt(0, where, k, v, 1);
    single.writeAt(1, where, k, v, 1);
    single.commit(1);
  }

  assert.deepEqual([...bulk.keys(0)], [...single.keys(0)]);
  assert.deepEqual([...bulk.values(1)], [...single.values(1)]);
});

test('keysUpTo exposes positions that are reserved but not yet committed', () => {
  // Attention during a pass has to see the tokens written by that same pass.
  const cache = new KVCache(tiny, 4);
  const at = cache.allocate(2);
  cache.writeAt(0, at, filled(7, 2 * KV), filled(8, 2 * KV), 2);
  assert.equal(cache.keys(0).length, 0, 'committed view is still empty');
  assert.equal(cache.keysUpTo(0, 2).length, 2 * KV, 'the pass can see its own writes');
});

test('present reshapes into the [kvHeads, positions, headDim] layout', () => {
  // Stored as [position, kvHead * headDim]; published as [kvHead, position,
  // headDim]. Getting this transpose wrong is invisible until it is compared
  // against the ONNX export, so it gets its own test.
  const cache = new KVCache(tiny, 2);
  const at = cache.allocate(2);
  // position 0 -> [0, 1, 2, 3], position 1 -> [10, 11, 12, 13]
  const keys = Float32Array.from([0, 1, 2, 3, 10, 11, 12, 13]);
  cache.writeAt(0, at, keys, keys, 2);
  cache.commit(2);

  const { key } = cache.present(0);
  // head 0 takes elements 0..1 of each position, head 1 takes 2..3.
  assert.deepEqual([...key], [0, 1, 10, 11, 2, 3, 12, 13]);
});

test('reset forgets the contents while keeping the allocation', () => {
  const cache = new KVCache(tiny, 4);
  cache.writeAt(0, cache.allocate(2), filled(1, 2 * KV), filled(2, 2 * KV), 2);
  cache.commit(2);
  assert.equal(cache.length, 2);
  cache.reset();
  assert.equal(cache.length, 0);
  assert.equal(cache.capacity, 4, 'capacity is unchanged');
  assert.equal(cache.allocate(4), 0, 'the whole cache is available again');
});

// ---------------------------------------------------------------------------
// The Phase 2 gate. Exact equality, not a tolerance: caching must not change
// the arithmetic at all.
// ---------------------------------------------------------------------------

const skip = existsSync(MODEL) && existsSync(CONFIG)
  ? false
  : 'run oracle/fetch_weights.py first';

test('incremental decoding is bit-identical to full recomputation', { skip }, async (t) => {
  const config = ModelConfig.parse(readFileSync(CONFIG, 'utf-8'));
  const st = await Safetensors.open(await FileSource.open(MODEL));
  const weights = new Weights(st, config, { resident: true });
  const model = new ReferenceModel(config, weights);

  const ids = [785, 6722, 315, 9625, 374, 12095]; // "The capital of France is Paris"

  const full = await model.forward(ids, { collectPresent: true });

  // Feed the same tokens as a prefill followed by single steps, which is
  // exactly the shape real decoding takes.
  const cache = model.newCache(ids.length);
  let incremental = null;
  incremental = await model.forward(ids.slice(0, 3), { cache });
  for (const id of ids.slice(3)) {
    incremental = await model.forward([id], { cache, collectPresent: true });
  }

  const bits = (a) => new Uint32Array(a.buffer, a.byteOffset, a.length);

  for (let layer = 0; layer < config.numLayers; layer++) {
    for (const kind of ['key', 'value']) {
      assert.deepEqual(
        [...bits(incremental.present[layer][kind])],
        [...bits(full.present[layer][kind])],
        `layer ${layer} ${kind} differs between cached and recomputed`,
      );
    }
  }
  assert.deepEqual([...bits(incremental.logits)], [...bits(full.logits)],
    'logits differ between cached and recomputed');

  t.diagnostic(`${ids.length} tokens, ${config.numLayers} layers, ` +
    `${incremental.logits.length} logits, all bit-identical`);
  await st.close();
});

test('a cached generation produces the same tokens as an uncached one', { skip }, async () => {
  const config = ModelConfig.parse(readFileSync(CONFIG, 'utf-8'));
  const st = await Safetensors.open(await FileSource.open(MODEL));
  const model = new ReferenceModel(config, new Weights(st, config, { resident: true }));
  const prompt = [785, 6722, 315, 9625, 374];

  const cached = [];
  for await (const step of model.generate(prompt, { maxTokens: 4 })) cached.push(step.token);

  const uncached = [];
  for (let i = 0; i < 4; i++) {
    const { logits } = await model.forward([...prompt, ...uncached]);
    let best = 0;
    for (let j = 1; j < logits.length; j++) if (logits[j] > logits[best]) best = j;
    uncached.push(best);
  }

  assert.deepEqual(cached, uncached);
  await st.close();
});
