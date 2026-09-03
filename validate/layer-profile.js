// How does the disagreement with ONNX Runtime grow through the stack?
//
// The point is to tell two stories apart. Error that accumulates smoothly
// layer by layer is float non-associativity doing what it does. Error that
// jumps at one layer is a bug in that layer. The Phase 1 gate reports only the
// worst layer, which cannot distinguish them.
//
//   node validate/layer-profile.js [caseIndex]

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import { Safetensors } from '../src/core/safetensors.js';
import { FileSource } from '../src/core/source-node.js';
import { ModelConfig } from '../src/core/config.js';
import { Weights } from '../src/cpu/weights.js';
import { ReferenceModel } from '../src/cpu/model.js';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const REF = join(ROOT, 'golden', 'reference');
const manifest = JSON.parse(readFileSync(join(REF, 'manifest.json'), 'utf-8'));

const caseIndex = Number(process.argv[2] ?? 3);
const entry = manifest.cases.find((c) => c.index === caseIndex);
if (!entry) throw new Error(`no case ${caseIndex}`);

const config = ModelConfig.parse(readFileSync(join(ROOT, 'weights', 'config.json'), 'utf-8'));
const st = await Safetensors.open(await FileSource.open(join(ROOT, 'weights', 'model.safetensors')));
const model = new ReferenceModel(config, new Weights(st, config));

const bytes = readFileSync(join(REF, entry.bin));
const all = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
const per = entry.present_floats_per_tensor;

const out = await model.forward(entry.input_ids, { collectPresent: true });

console.log(`case ${caseIndex}: ${JSON.stringify(entry.prompt)}  (${entry.seq} tokens)\n`);
console.log('layer   key |max|   key maxabs   key rel     value |max|  value maxabs');
console.log('-----   ---------   ----------   --------    ----------  -------------');

for (let layer = 0; layer < manifest.layers; layer++) {
  const row = [];
  for (const kind of ['key', 'value']) {
    const theirs = all.subarray(
      (layer * 2 + (kind === 'key' ? 0 : 1)) * per,
      (layer * 2 + (kind === 'key' ? 0 : 1)) * per + per,
    );
    const mine = out.present[layer][kind];
    let maxAbs = 0;
    let scale = 0;
    for (let i = 0; i < per; i++) {
      const d = Math.abs(mine[i] - theirs[i]);
      if (d > maxAbs) maxAbs = d;
      if (Math.abs(theirs[i]) > scale) scale = Math.abs(theirs[i]);
    }
    row.push({ maxAbs, scale, rel: scale > 0 ? maxAbs / scale : 0 });
  }
  const [k, v] = row;
  console.log(
    `${String(layer).padStart(5)}   ${k.scale.toFixed(2).padStart(9)}   ` +
    `${k.maxAbs.toExponential(2).padStart(10)}   ${k.rel.toExponential(2).padStart(8)}    ` +
    `${v.scale.toFixed(2).padStart(10)}  ${v.maxAbs.toExponential(2).padStart(13)}`,
  );
}

await st.close();
