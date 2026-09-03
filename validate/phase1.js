// Phase 1 gate: does the reference forward pass agree with ONNX Runtime?
//
// Two claims, checked separately, because they fail for different reasons:
//
//   per layer   the 48 present.N.key / present.N.value tensors, which are the
//               post-RoPE keys and the values at every position of every
//               layer. A disagreement at layer L means the fault is in layer L
//               or in what feeds it, which turns "the logits are wrong" into a
//               located bug.
//
//   output      the logits at the final position, plus the argmax, which is
//               what actually decides the generated token.
//
//   node validate/phase1.js [caseIndex...]

// Durations use performance.now(), which is monotonic. Date.now() is wall
// clock: it jumps when the system clock is corrected, and a run of this script
// once reported a 3-pass comparison as taking 14.7 hours because of exactly
// that. A benchmark must never measure itself with a clock that can move.
import { readFileSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import { Safetensors } from '../src/core/safetensors.js';
import { FileSource } from '../src/core/source-node.js';
import { ModelConfig } from '../src/core/config.js';
import { Weights } from '../src/cpu/weights.js';
import { ReferenceModel } from '../src/cpu/model.js';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const REF = join(ROOT, 'golden', 'reference');

if (!existsSync(join(REF, 'manifest.json'))) {
  console.error('no golden/reference/manifest.json; run oracle/dump_reference.py first');
  process.exit(1);
}

const manifest = JSON.parse(readFileSync(join(REF, 'manifest.json'), 'utf-8'));
const config = ModelConfig.parse(readFileSync(join(ROOT, 'weights', 'config.json'), 'utf-8'));
const st = await Safetensors.open(await FileSource.open(join(ROOT, 'weights', 'model.safetensors')));
const model = new ReferenceModel(config, new Weights(st, config));

/** Split one case's .bin into the 48 present tensors and the final logits. */
function readCase(entry) {
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
  const logits = all.subarray(at, at + entry.logits_floats);
  if (at + entry.logits_floats !== all.length) {
    throw new Error(`${entry.bin}: ${all.length} floats, expected ${at + entry.logits_floats}`);
  }
  return { present, logits };
}

/** Largest absolute difference, and the largest relative to the reference scale. */
function compare(mine, theirs) {
  let maxAbs = 0;
  let scale = 0;
  for (let i = 0; i < theirs.length; i++) {
    const diff = Math.abs(mine[i] - theirs[i]);
    if (diff > maxAbs) maxAbs = diff;
    const magnitude = Math.abs(theirs[i]);
    if (magnitude > scale) scale = magnitude;
  }
  return { maxAbs, scale, relative: scale > 0 ? maxAbs / scale : 0 };
}

const wanted = process.argv.slice(2).map(Number);
const cases = wanted.length
  ? manifest.cases.filter((c) => wanted.includes(c.index))
  : manifest.cases;

console.log(`oracle: ${manifest.source}`);
console.log(`        onnxruntime ${manifest.onnxruntime_version}\n`);

let worstLayer = 0;
let worstLayerRel = 0;
let worstLogit = 0;
let worstLogitRel = 0;
let argmaxAgreements = 0;

for (const entry of cases) {
  const golden = readCase(entry);
  const started = performance.now();
  const out = await model.forward(entry.input_ids, { collectPresent: true });
  const seconds = (performance.now() - started) / 1000;

  // Per layer.
  let layerWorst = { maxAbs: 0, layer: -1, kind: '' };
  for (let layer = 0; layer < manifest.layers; layer++) {
    for (const kind of ['key', 'value']) {
      const stat = compare(out.present[layer][kind], golden.present[layer][kind]);
      if (stat.maxAbs > layerWorst.maxAbs) layerWorst = { ...stat, layer, kind };
    }
  }

  const logitStat = compare(out.logits, golden.logits);

  let mineArgmax = 0;
  for (let i = 1; i < out.logits.length; i++) {
    if (out.logits[i] > out.logits[mineArgmax]) mineArgmax = i;
  }
  const agree = mineArgmax === entry.argmax;
  if (agree) argmaxAgreements++;

  worstLayer = Math.max(worstLayer, layerWorst.maxAbs);
  worstLayerRel = Math.max(worstLayerRel, layerWorst.relative ?? 0);
  worstLogit = Math.max(worstLogit, logitStat.maxAbs);
  worstLogitRel = Math.max(worstLogitRel, logitStat.relative);

  console.log(`[${entry.index}] ${JSON.stringify(entry.prompt.slice(0, 46))}`);
  console.log(`     ${entry.seq} tokens, ${seconds.toFixed(1)}s`);
  console.log(`     per-layer worst: ${layerWorst.maxAbs.toExponential(2)} abs ` +
    `(${(layerWorst.relative ?? 0).toExponential(2)} rel) at layer ${layerWorst.layer} ` +
    `${layerWorst.kind}`);
  console.log(`     logits:          ${logitStat.maxAbs.toExponential(2)} abs ` +
    `(${logitStat.relative.toExponential(2)} rel), range +-${logitStat.scale.toFixed(1)}`);
  console.log(`     argmax:          ${mineArgmax} vs ${entry.argmax} ` +
    `${agree ? 'agree' : 'DISAGREE'}\n`);
}

await st.close();

// Tolerances, stated rather than tuned after the fact. Both implementations
// store f32 and sum in different orders, so a dot product over 896 terms
// carries about 2e-6 relative and 24 layers compound it. These bounds sit a few
// times above the measured worst case: float non-associativity cannot reach
// them, and a real defect overshoots them by orders of magnitude.
const PER_LAYER_REL = 5e-4;
const LOGIT_REL = 1e-4;

console.log('summary');
console.log('-------');
console.log(`worst per-layer: ${worstLayerRel.toExponential(2)} relative ` +
  `(${worstLayer.toExponential(2)} absolute), bound ${PER_LAYER_REL}`);
console.log(`worst logit:     ${worstLogitRel.toExponential(2)} relative ` +
  `(${worstLogit.toExponential(2)} absolute), bound ${LOGIT_REL}`);
console.log(`argmax agreement: ${argmaxAgreements}/${cases.length}`);

const pass = worstLayerRel <= PER_LAYER_REL
  && worstLogitRel <= LOGIT_REL
  && argmaxAgreements === cases.length;
console.log('\n' + (pass
  ? 'PASS: every layer and every output within tolerance, every argmax identical'
  : 'FAIL'));
process.exit(pass ? 0 : 1);
