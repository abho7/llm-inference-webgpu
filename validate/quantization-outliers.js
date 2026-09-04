// Which input channels carry the damage when weights are quantized?
//
// Weight-only quantization perturbs the weights and leaves the activations
// alone, so the error that reaches a layer's output is
//
//     dy[r] = sum_c dW[r,c] * x[c]
//
// Every channel's weight error is weighted by that channel's activation. Phase
// 1 found that this model concentrates its hidden state in a few channels --
// channel 490 reaches 27 times the RMS of the whole vector by layer 22 -- so
// the damage should not be spread evenly across the 896 input channels. It
// should pile up in the loud ones.
//
// This measures that directly. For each projection it computes, per input
// channel, the column's quantization error times that channel's typical
// activation, which is the channel's contribution to the output error. Then it
// asks how concentrated the result is.
//
// The answer turned out to contradict the phase 1 prediction, which is why the
// script also prints the normalisation gain at channel 490 -- that is where the
// explanation lives.
//
//   node validate/quantization-outliers.js [layer]

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import { Safetensors } from '../src/core/safetensors.js';
import { FileSource } from '../src/core/source-node.js';
import { ModelConfig } from '../src/core/config.js';
import { Tokenizer } from '../src/core/tokenizer.js';
import { Weights } from '../src/cpu/weights.js';
import { ReferenceModel } from '../src/cpu/model.js';
import { roundTrip } from '../src/core/quantize.js';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const w = (p) => join(ROOT, 'weights', p);

// A handful of prompts, so the activation statistics are not one sentence's
// idiosyncrasy. Calibration sets in the literature are far larger; this is
// enough to establish where the mass sits, and the report says so.
const CALIBRATION = [
  'The capital of France is Paris, and the largest city in Germany is',
  'def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n - 1)',
  'In 1994 the team published a paper describing a new algorithm for',
  'She had never seen anything like it before, and she never would again.',
];

const config = ModelConfig.parse(readFileSync(w('config.json'), 'utf-8'));
const tokenizer = Tokenizer.fromJSON(readFileSync(w('tokenizer.json'), 'utf-8'));
const st = await Safetensors.open(await FileSource.open(w('model.safetensors')));
const weights = new Weights(st, config, { resident: true });
const model = new ReferenceModel(config, weights);

const d = config.hiddenSize;

// ---------------------------------------------------------------------------
// Collect the mean absolute activation per input channel, per layer, for the
// two places a projection reads from.
// ---------------------------------------------------------------------------

console.log(`calibrating on ${CALIBRATION.length} prompts...`);
const attnAct = Array.from({ length: config.numLayers }, () => new Float64Array(d));
const mlpAct = Array.from({ length: config.numLayers }, () => new Float64Array(d));
let positions = 0;

for (const prompt of CALIBRATION) {
  const ids = tokenizer.encode(prompt);
  const out = await model.forward(ids, { captureNorms: true });
  positions += ids.length;
  for (let layer = 0; layer < config.numLayers; layer++) {
    for (let t = 0; t < ids.length; t++) {
      for (let c = 0; c < d; c++) {
        attnAct[layer][c] += Math.abs(out.norms[layer].attn[t * d + c]);
        mlpAct[layer][c] += Math.abs(out.norms[layer].mlp[t * d + c]);
      }
    }
  }
}
for (let layer = 0; layer < config.numLayers; layer++) {
  for (let c = 0; c < d; c++) {
    attnAct[layer][c] /= positions;
    mlpAct[layer][c] /= positions;
  }
}
console.log(`${positions} token positions\n`);

// ---------------------------------------------------------------------------
// Per-channel contribution to output error, under int4.
// ---------------------------------------------------------------------------

/** Column-wise L2 norm of the quantization error of one matrix. */
function columnError(W, rows, cols, scheme, groupSize) {
  const back = roundTrip(W, rows, cols, scheme, groupSize);
  const perColumn = new Float64Array(cols);
  for (let r = 0; r < rows; r++) {
    const base = r * cols;
    for (let c = 0; c < cols; c++) {
      const diff = back[base + c] - W[base + c];
      perColumn[c] += diff * diff;
    }
  }
  for (let c = 0; c < cols; c++) perColumn[c] = Math.sqrt(perColumn[c]);
  return perColumn;
}

/** What share of the total sits in the top `k` entries? */
function concentration(values, k) {
  const sorted = [...values].sort((a, b) => b - a);
  const total = sorted.reduce((a, b) => a + b, 0);
  if (total === 0) return { share: 0, top: [] };
  const head = sorted.slice(0, k).reduce((a, b) => a + b, 0);
  return { share: head / total, total };
}

const PROJECTIONS = [
  { key: 'wq', suffix: 'self_attn.q_proj.weight', reads: 'attn' },
  { key: 'wk', suffix: 'self_attn.k_proj.weight', reads: 'attn' },
  { key: 'wv', suffix: 'self_attn.v_proj.weight', reads: 'attn' },
  { key: 'wGate', suffix: 'mlp.gate_proj.weight', reads: 'mlp' },
  { key: 'wUp', suffix: 'mlp.up_proj.weight', reads: 'mlp' },
];

const only = process.argv[2] !== undefined ? Number(process.argv[2]) : null;
const layers = only !== null ? [only] : [0, 8, 15, 22, 23];

const pad = (s, n) => String(s).padEnd(n);

console.log('For each projection, the top input channels by contribution to output error');
console.log('(column error times that channel\'s mean activation), under int4-g128.\n');

const summary = [];

for (const layer of layers) {
  console.log(`layer ${layer}`);
  console.log(`  ${pad('projection', 12)}${pad('top channel', 13)}${pad('share of damage', 17)}` +
    `${pad('top-5 share', 13)}${pad('activation ratio', 16)}`);
  for (const projection of PROJECTIONS) {
    const name = `model.layers.${layer}.${projection.suffix}`;
    const info = st.info(name);
    const [rows, cols] = info.shape;
    const W = await st.readF32(name);

    const colErr = columnError(W, rows, cols, 'int4', 128);
    const act = projection.reads === 'attn' ? attnAct[layer] : mlpAct[layer];

    // Contribution of channel c to the output error.
    const damage = new Float64Array(cols);
    for (let c = 0; c < cols; c++) damage[c] = colErr[c] * act[c];

    let topChannel = 0;
    for (let c = 1; c < cols; c++) if (damage[c] > damage[topChannel]) topChannel = c;
    const totalDamage = damage.reduce((a, b) => a + b, 0);
    const top1 = damage[topChannel] / totalDamage;
    const top5 = concentration(damage, 5).share;

    // How loud is that channel compared with a typical one?
    let actSum = 0;
    for (let c = 0; c < cols; c++) actSum += act[c];
    const actRatio = act[topChannel] / (actSum / cols);

    console.log(`  ${pad(projection.key, 12)}${pad(topChannel, 13)}` +
      `${pad(`${(top1 * 100).toFixed(1)}%`, 17)}${pad(`${(top5 * 100).toFixed(1)}%`, 13)}` +
      `${pad(`${actRatio.toFixed(1)}x`, 16)}`);

    summary.push({ layer, projection: projection.key, topChannel, top1, top5, actRatio });
  }
  console.log();
}

// ---------------------------------------------------------------------------
// Where did the massive-activation channel go?
//
// Phase 1 found channel 490 growing to 27 times the RMS of the hidden state by
// layer 22, and predicted it would dominate quantization damage. It does not.
// The reason is visible in the normalisation gain: a projection never sees the
// residual stream directly, it sees the residual stream after RMSNorm, and
// RMSNorm applies a learned per-channel gain.
// ---------------------------------------------------------------------------

console.log('the normalisation gain at channel 490, the massive-activation channel');
console.log(`  ${pad('layer', 8)}${pad('gain[490]', 12)}${pad('mean |gain|', 14)}` +
  `${pad('ratio', 9)}${pad('rank of 896', 12)}`);
for (const layer of layers) {
  const gain = await st.readF32(`model.layers.${layer}.input_layernorm.weight`);
  let sum = 0;
  for (let i = 0; i < gain.length; i++) sum += Math.abs(gain[i]);
  const mean = sum / gain.length;
  let rank = 1;
  for (let i = 0; i < gain.length; i++) {
    if (Math.abs(gain[i]) > Math.abs(gain[490])) rank++;
  }
  console.log(`  ${pad(layer, 8)}${pad(gain[490].toFixed(4), 12)}${pad(mean.toFixed(4), 14)}` +
    `${pad((Math.abs(gain[490]) / mean).toFixed(3), 9)}${pad(rank, 12)}`);
}
console.log();

await st.close();

// ---------------------------------------------------------------------------

console.log('reading');
console.log('-------');
const uniform = 1 / d;
const worst = summary.reduce((a, b) => (b.top1 > a.top1 ? b : a));
console.log(`If the damage were spread evenly over ${d} input channels, one channel would`);
console.log(`carry ${(uniform * 100).toFixed(3)}% of it and five would carry ` +
  `${(uniform * 5 * 100).toFixed(2)}%.`);
console.log(`The worst case measured is layer ${worst.layer} ${worst.projection}, where channel ` +
  `${worst.topChannel} alone`);
console.log(`carries ${(worst.top1 * 100).toFixed(1)}% -- ` +
  `${(worst.top1 / uniform).toFixed(0)} times its even share.`);

const channels = new Map();
for (const row of summary) channels.set(row.topChannel, (channels.get(row.topChannel) ?? 0) + 1);
const repeated = [...channels.entries()].sort((a, b) => b[1] - a[1]).slice(0, 4);
console.log(`\nChannels that top the list most often: ` +
  repeated.map(([c, n]) => `${c} (${n} of ${summary.length})`).join(', '));
