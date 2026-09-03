// Phase 2 gate: is the KV cache correct, and does it actually help?
//
// The correctness claim admits no tolerance. Position s's key depends only on
// positions up to s, so the value computed while s was the newest token is the
// same value a full recomputation produces later -- the same bits, not merely
// close. Anything less than exact equality means the cache changed the
// arithmetic, and the usual suspect is the rotary position: using the offset
// within the current call instead of the absolute position works perfectly on
// the first pass and corrupts every one after it.
//
// So the gate compares f32 bit patterns, not magnitudes.
//
//   node validate/phase2.js [--stream]

// Durations use performance.now(), which is monotonic. Date.now() is wall
// clock: it jumps when the system clock is corrected, and a run of this script
// once reported a 3-pass comparison as taking 14.7 hours because of exactly
// that. A benchmark must never measure itself with a clock that can move.
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import { Safetensors } from '../src/core/safetensors.js';
import { FileSource } from '../src/core/source-node.js';
import { ModelConfig } from '../src/core/config.js';
import { Tokenizer } from '../src/core/tokenizer.js';
import { Weights } from '../src/cpu/weights.js';
import { ReferenceModel } from '../src/cpu/model.js';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const w = (p) => join(ROOT, 'weights', p);
const resident = !process.argv.includes('--stream');

const config = ModelConfig.parse(readFileSync(w('config.json'), 'utf-8'));
const tokenizer = Tokenizer.fromJSON(readFileSync(w('tokenizer.json'), 'utf-8'));
const st = await Safetensors.open(await FileSource.open(w('model.safetensors')));
const weights = new Weights(st, config, { resident });
const model = new ReferenceModel(config, weights);

if (resident) {
  const held = await weights.makeResident();
  console.log(`weights resident: ${(held / 1e6).toFixed(0)} MB of bfloat16 ` +
    `(pass --stream to read from disk instead)\n`);
} else {
  console.log('weights streamed from disk\n');
}

/** Exact equality, compared as bits so that -0 and NaN cannot slip through. */
function bitsDiffer(a, b) {
  if (a.length !== b.length) return { count: a.length, first: -1 };
  const ba = new Uint32Array(a.buffer, a.byteOffset, a.length);
  const bb = new Uint32Array(b.buffer, b.byteOffset, b.length);
  let count = 0;
  let first = -1;
  for (let i = 0; i < ba.length; i++) {
    if (ba[i] !== bb[i]) {
      count++;
      if (first < 0) first = i;
    }
  }
  return { count, first };
}

const prompt = 'The capital of France is Paris and';
const ids = tokenizer.encode(prompt);
console.log(`prompt: ${JSON.stringify(prompt)} (${ids.length} tokens)\n`);

// ---------------------------------------------------------------------------
// One pass over the whole sequence: the answer everything else must reproduce.
// ---------------------------------------------------------------------------

let t0 = performance.now();
const full = await model.forward(ids, { collectPresent: true });
const fullSeconds = (performance.now() - t0) / 1000;
console.log(`full recomputation of ${ids.length} tokens: ${fullSeconds.toFixed(1)}s`);

// ---------------------------------------------------------------------------
// The same sequence fed in pieces. Every split must land on the same bits, and
// the splits are chosen to cover the shapes that behave differently: one token
// at a time, a prefill then single steps, and an uneven pair.
// ---------------------------------------------------------------------------

const splits = [
  { name: 'one token at a time', chunks: ids.map((id) => [id]) },
  { name: 'prefill then decode', chunks: [ids.slice(0, 5), ...ids.slice(5).map((id) => [id])] },
  { name: 'uneven chunks', chunks: [ids.slice(0, 3), ids.slice(3, 4), ids.slice(4)] },
];

let allExact = true;

for (const split of splits) {
  const cache = model.newCache(ids.length);
  let last = null;
  t0 = performance.now();
  for (const chunk of split.chunks) {
    last = await model.forward(chunk, { cache, collectPresent: true });
  }
  const seconds = (performance.now() - t0) / 1000;

  const logitDiff = bitsDiffer(full.logits, last.logits);
  let layerDiffs = 0;
  let worstLayer = -1;
  for (let layer = 0; layer < config.numLayers; layer++) {
    for (const kind of ['key', 'value']) {
      const d = bitsDiffer(full.present[layer][kind], last.present[layer][kind]);
      if (d.count > 0) {
        layerDiffs += d.count;
        if (worstLayer < 0) worstLayer = layer;
      }
    }
  }

  const exact = logitDiff.count === 0 && layerDiffs === 0;
  allExact = allExact && exact;
  console.log(`\n${split.name} (${split.chunks.length} passes, ${seconds.toFixed(1)}s)`);
  console.log(`  cache length:   ${cache.length}`);
  console.log(`  keys and values: ${layerDiffs === 0 ? 'identical' :
    `${layerDiffs} floats differ, first at layer ${worstLayer}`}`);
  console.log(`  logits:          ${logitDiff.count === 0 ? 'identical' :
    `${logitDiff.count} of ${full.logits.length} differ, first at ${logitDiff.first}`}`);
}

// ---------------------------------------------------------------------------
// What it buys. Decoding with a cache is one pass of length 1 per token;
// without one it is a pass over the whole sequence so far.
// ---------------------------------------------------------------------------

console.log('\ngeneration, 6 tokens');
const promptIds = tokenizer.encode('The capital of France is');

t0 = performance.now();
const cachedTokens = [];
for await (const step of model.generate(promptIds, { maxTokens: 6 })) {
  cachedTokens.push(step.token);
}
const cachedSeconds = (performance.now() - t0) / 1000;

t0 = performance.now();
const uncachedTokens = [];
for (let i = 0; i < 6; i++) {
  const { logits } = await model.forward([...promptIds, ...uncachedTokens]);
  let best = 0;
  for (let j = 1; j < logits.length; j++) if (logits[j] > logits[best]) best = j;
  uncachedTokens.push(best);
}
const uncachedSeconds = (performance.now() - t0) / 1000;

const sameTokens = cachedTokens.length === uncachedTokens.length
  && cachedTokens.every((t, i) => t === uncachedTokens[i]);

console.log(`  with cache:    ${JSON.stringify(tokenizer.decode(cachedTokens))}  ` +
  `${cachedSeconds.toFixed(1)}s (${(6 / cachedSeconds).toFixed(2)} tok/s)`);
console.log(`  without cache: ${JSON.stringify(tokenizer.decode(uncachedTokens))}  ` +
  `${uncachedSeconds.toFixed(1)}s (${(6 / uncachedSeconds).toFixed(2)} tok/s)`);
console.log(`  same tokens:   ${sameTokens ? 'yes' : 'NO'}`);
console.log(`  speedup:       ${(uncachedSeconds / cachedSeconds).toFixed(2)}x`);

const cache = model.newCache(2048);
console.log(`\na 2048-position cache costs ${(cache.byteLength / 1e6).toFixed(0)} MB`);

await st.close();

console.log('\ngate');
console.log('----');
const pass = allExact && sameTokens;
console.log(pass
  ? 'PASS: incremental decoding is bit-identical to full recomputation, on every split'
  : 'FAIL');
process.exit(pass ? 0 : 1);
