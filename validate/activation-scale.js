// Where does the hidden state get big, and how concentrated is it?
//
// Written to explain something the Phase 1 profile turned up: agreement with
// ONNX Runtime degrades smoothly through the stack and then jumps by about an
// order of magnitude at layer 22, in every case rather than one. A jump like
// that is either a bug in that layer or a property of the model, and the two
// look identical from the error alone.
//
// The statistic that separates them is concentration. If a handful of channels
// carry a hidden state thousands of times larger than the rest, then RMSNorm
// divides everything by a number those few channels set, and the projection
// that follows sums 896 terms in which one dwarfs the others. That is
// catastrophic cancellation, and it inflates absolute error without anything
// being wrong.
//
// This also front-runs phase 4: outlier channels are exactly what makes
// per-tensor quantization fail, so it is worth knowing which ones they are.
//
//   node validate/activation-scale.js [caseIndex]

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import { Safetensors } from '../src/core/safetensors.js';
import { FileSource } from '../src/core/source-node.js';
import { ModelConfig } from '../src/core/config.js';
import { Weights } from '../src/cpu/weights.js';
import { rmsNorm, matVec, swigluInPlace, ropeFrequencies, ropeInPlace, softmaxInPlace } from '../src/core/ops.js';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const manifest = JSON.parse(readFileSync(join(ROOT, 'golden', 'reference', 'manifest.json'), 'utf-8'));
const entry = manifest.cases.find((c) => c.index === Number(process.argv[2] ?? 3));

const config = ModelConfig.parse(readFileSync(join(ROOT, 'weights', 'config.json'), 'utf-8'));
const st = await Safetensors.open(await FileSource.open(join(ROOT, 'weights', 'model.safetensors')));
const weights = new Weights(st, config);

const d = config.hiddenSize;
const kvDim = config.kvDim;
const headDim = config.headDim;
const invFreq = ropeFrequencies(headDim, config.ropeTheta);
const seq = entry.input_ids.length;

// Embed.
const x = new Float32Array(seq * d);
{
  const conv = new ArrayBuffer(4);
  const u = new Uint32Array(conv);
  const f = new Float32Array(conv);
  for (let t = 0; t < seq; t++) {
    const row = await weights.embeddingRows(entry.input_ids[t], entry.input_ids[t] + 1);
    for (let i = 0; i < d; i++) { u[0] = row[i] << 16; x[t * d + i] = f[0]; }
  }
}

/** Concentration: how much larger is the biggest channel than a typical one? */
function stats(vec) {
  let max = 0;
  let maxAt = -1;
  let sumSquares = 0;
  for (let i = 0; i < vec.length; i++) {
    const a = Math.abs(vec[i]);
    if (a > max) { max = a; maxAt = i; }
    sumSquares += vec[i] * vec[i];
  }
  const rms = Math.sqrt(sumSquares / vec.length);
  return { max, maxAt, rms, ratio: rms > 0 ? max / rms : 0 };
}

console.log(`case ${entry.index}: ${JSON.stringify(entry.prompt)}  (${seq} tokens)`);
console.log('statistics are for the last position\n');
console.log('layer   hidden rms   hidden max   max/rms   channel   k-proj rms   k-proj max');
console.log('-----   ----------   ----------   -------   -------   ----------   ----------');

const normed = new Float32Array(d);
const q = new Float32Array(seq * d);
const k = new Float32Array(seq * kvDim);
const v = new Float32Array(seq * kvDim);
const attnOut = new Float32Array(seq * d);
const scores = new Float32Array(seq);
const gate = new Float32Array(config.intermediateSize);
const up = new Float32Array(config.intermediateSize);
const projected = new Float32Array(d);

for (let layer = 0; layer < config.numLayers; layer++) {
  const w = await weights.layer(layer);
  const before = stats(x.subarray((seq - 1) * d, seq * d));

  for (let t = 0; t < seq; t++) {
    rmsNorm(x.subarray(t * d, (t + 1) * d), w.inputNorm, config.rmsNormEps, normed);
    matVec(w.wq, normed, d, d, w.bq, q.subarray(t * d, (t + 1) * d));
    matVec(w.wk, normed, kvDim, d, w.bk, k.subarray(t * kvDim, (t + 1) * kvDim));
    matVec(w.wv, normed, kvDim, d, w.bv, v.subarray(t * kvDim, (t + 1) * kvDim));
    for (let h = 0; h < config.numHeads; h++) ropeInPlace(q, t * d + h * headDim, headDim, t, invFreq);
    for (let h = 0; h < config.numKVHeads; h++) ropeInPlace(k, t * kvDim + h * headDim, headDim, t, invFreq);
  }

  const kStats = stats(k.subarray((seq - 1) * kvDim, seq * kvDim));
  console.log(
    `${String(layer).padStart(5)}   ${before.rms.toFixed(3).padStart(10)}   ` +
    `${before.max.toFixed(1).padStart(10)}   ${before.ratio.toFixed(1).padStart(7)}   ` +
    `${String(before.maxAt).padStart(7)}   ${kStats.rms.toFixed(3).padStart(10)}   ` +
    `${kStats.max.toFixed(2).padStart(10)}`,
  );

  const scale = 1 / Math.sqrt(headDim);
  attnOut.fill(0);
  for (let h = 0; h < config.numHeads; h++) {
    const kvBase = Math.floor(h / config.kvGroupSize) * headDim;
    for (let t = 0; t < seq; t++) {
      const qBase = t * d + h * headDim;
      for (let s = 0; s <= t; s++) {
        let dot = 0;
        const kBase = s * kvDim + kvBase;
        for (let i = 0; i < headDim; i++) dot += q[qBase + i] * k[kBase + i];
        scores[s] = dot * scale;
      }
      softmaxInPlace(scores, t + 1);
      const outBase = t * d + h * headDim;
      for (let s = 0; s <= t; s++) {
        const vBase = s * kvDim + kvBase;
        for (let i = 0; i < headDim; i++) attnOut[outBase + i] += scores[s] * v[vBase + i];
      }
    }
  }
  for (let t = 0; t < seq; t++) {
    matVec(w.wo, attnOut.subarray(t * d, (t + 1) * d), d, d, null, projected);
    for (let i = 0; i < d; i++) x[t * d + i] += projected[i];
  }
  for (let t = 0; t < seq; t++) {
    const row = x.subarray(t * d, (t + 1) * d);
    rmsNorm(row, w.postAttnNorm, config.rmsNormEps, normed);
    matVec(w.wGate, normed, config.intermediateSize, d, null, gate);
    matVec(w.wUp, normed, config.intermediateSize, d, null, up);
    swigluInPlace(gate, up);
    matVec(w.wDown, gate, d, config.intermediateSize, null, projected);
    for (let i = 0; i < d; i++) row[i] += projected[i];
  }
}

const final = stats(x.subarray((seq - 1) * d, seq * d));
console.log(`\nafter the last layer: rms ${final.rms.toFixed(3)}, ` +
  `max ${final.max.toFixed(1)} at channel ${final.maxAt}, ratio ${final.ratio.toFixed(1)}`);

await st.close();
