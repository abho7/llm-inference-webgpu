import test from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import { Safetensors } from '../src/core/safetensors.js';
import { FileSource } from '../src/core/source-node.js';
import { ModelConfig } from '../src/core/config.js';
import { Weights } from '../src/cpu/weights.js';
import { ReferenceModel } from '../src/cpu/model.js';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const MODEL = join(ROOT, 'weights', 'model.safetensors');
const CONFIG = join(ROOT, 'weights', 'config.json');
const REF = join(ROOT, 'golden', 'reference');

const haveAll = existsSync(MODEL) && existsSync(CONFIG) && existsSync(join(REF, 'manifest.json'));
const skip = haveAll ? false : 'run oracle/fetch_weights.py and oracle/dump_reference.py first';

// Tolerances, and why these numbers.
//
// Both implementations store activations in f32 but sum in different orders,
// so exact agreement is impossible and the only question is how much drift is
// explainable. A dot product over 896 terms carries roughly sqrt(896) * 2^-24,
// about 2e-6 relative, and 24 layers of that compounds.
//
// The measured worst case across all five prompts is 9.1e-5 relative on the
// per-layer tensors and 6.6e-6 relative on the logits. The bounds below sit a
// few times above those: loose enough that float non-associativity cannot trip
// them, tight enough that a real defect -- a swapped RoPE convention, a
// mis-grouped GQA head, an epsilon outside the square root -- moves the error
// by orders of magnitude and fails immediately.
//
// The per-layer bound is the looser of the two on purpose. Qwen2.5 concentrates
// its hidden state in a few channels (channel 490 reaches 27x the RMS by layer
// 22), so the projection there sums 896 terms in which one dominates, and
// absolute error rises accordingly. validate/activation-scale.js measures it.
const PER_LAYER_REL = 5e-4;
const LOGIT_REL = 1e-4;

async function openModel() {
  const config = ModelConfig.parse(readFileSync(CONFIG, 'utf-8'));
  const st = await Safetensors.open(await FileSource.open(MODEL));
  return { config, st, model: new ReferenceModel(config, new Weights(st, config)) };
}

function readCase(manifest, entry) {
  const bytes = readFileSync(join(REF, entry.bin));
  const all = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
  const per = entry.present_floats_per_tensor;
  const present = [];
  let at = 0;
  for (let layer = 0; layer < manifest.layers; layer++) {
    present.push({
      key: all.subarray(at, at + per),
      value: all.subarray(at + per, at + 2 * per),
    });
    at += 2 * per;
  }
  return { present, logits: all.subarray(at, at + entry.logits_floats) };
}

function worstRelative(mine, theirs) {
  let maxAbs = 0;
  let scale = 0;
  let at = -1;
  for (let i = 0; i < theirs.length; i++) {
    const diff = Math.abs(mine[i] - theirs[i]);
    if (diff > maxAbs) { maxAbs = diff; at = i; }
    const magnitude = Math.abs(theirs[i]);
    if (magnitude > scale) scale = magnitude;
  }
  return { maxAbs, scale, at, relative: scale > 0 ? maxAbs / scale : 0 };
}

// Run every prompt once, and share the results with all four assertions below.
//
// This matters more than it looks. Weights are streamed from disk rather than
// held in memory, so one forward pass moves about 1.5 GB, and there is not
// enough free memory for the page cache to absorb it. Recomputing per
// assertion turned a 77-second run into a 54-minute one.
let cached = null;
function runAll() {
  if (cached) return cached;
  cached = (async () => {
    const manifest = JSON.parse(readFileSync(join(REF, 'manifest.json'), 'utf-8'));
    const { st, model } = await openModel();
    const results = [];
    for (const entry of manifest.cases) {
      results.push({
        entry,
        golden: readCase(manifest, entry),
        out: await model.forward(entry.input_ids, { collectPresent: true }),
      });
    }
    await st.close();
    return { manifest, results };
  })();
  return cached;
}

test('the forward pass matches ONNX Runtime per layer and at the output', { skip }, async (t) => {
  const { manifest, results } = await runAll();

  let worstLayer = 0;
  let worstLogit = 0;

  for (const { entry, golden, out } of results) {
    assert.equal(out.present.length, manifest.layers, 'one present pair per layer');

    for (let layer = 0; layer < manifest.layers; layer++) {
      for (const kind of ['key', 'value']) {
        const stat = worstRelative(out.present[layer][kind], golden.present[layer][kind]);
        worstLayer = Math.max(worstLayer, stat.relative);
        assert.ok(
          stat.relative <= PER_LAYER_REL,
          `case ${entry.index} layer ${layer} ${kind}: ${stat.relative.toExponential(2)} ` +
          `relative (${stat.maxAbs.toExponential(2)} absolute on a +-${stat.scale.toFixed(1)} ` +
          `tensor) exceeds ${PER_LAYER_REL}`,
        );
      }
    }

    const logitStat = worstRelative(out.logits, golden.logits);
    worstLogit = Math.max(worstLogit, logitStat.relative);
    assert.ok(
      logitStat.relative <= LOGIT_REL,
      `case ${entry.index} logits: ${logitStat.relative.toExponential(2)} relative ` +
      `exceeds ${LOGIT_REL}`,
    );
  }

  t.diagnostic(`worst per-layer ${worstLayer.toExponential(2)} relative, ` +
    `worst logit ${worstLogit.toExponential(2)} relative, over ${results.length} cases`);
});

test('greedy decoding picks the same token as the oracle, on every case', { skip }, async () => {
  // The tolerance test above allows small numeric drift. This one does not
  // admit any: the argmax is what actually decides the generated token, so it
  // has to be identical rather than close.
  const { results } = await runAll();
  for (const { entry, out } of results) {
    let argmax = 0;
    for (let i = 1; i < out.logits.length; i++) {
      if (out.logits[i] > out.logits[argmax]) argmax = i;
    }
    assert.equal(argmax, entry.argmax,
      `case ${entry.index} ${JSON.stringify(entry.prompt.slice(0, 40))}: ` +
      `chose ${argmax}, oracle chose ${entry.argmax}`);
  }
});

test('the ten most likely tokens come out in the same order', { skip }, async () => {
  // Stronger than the argmax and much stronger than a tolerance: it constrains
  // the ordering of ten separate values that are often within a few tenths of
  // each other, so a systematic bias would break it even while every individual
  // logit stayed inside tolerance.
  const { results } = await runAll();
  for (const { entry, out } of results) {
    const order = Array.from(out.logits.keys())
      .sort((a, b) => out.logits[b] - out.logits[a])
      .slice(0, 10);
    assert.deepEqual(order, entry.top10.map(([id]) => id),
      `case ${entry.index}: top-10 ordering differs`);
  }
});

test('rejects a token id outside the vocabulary instead of reading past the table',
  { skip }, async () => {
    const { st, model } = await openModel();
    await assert.rejects(() => model.forward([151936]), /outside the vocabulary/);
    await assert.rejects(() => model.forward([-1]), /outside the vocabulary/);
    await assert.rejects(() => model.forward([]), /at least one token/);
    await st.close();
  });
