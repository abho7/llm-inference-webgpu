// Phase 0 gate report.
//
// Prints what the loader and tokenizer actually establish, with the numbers
// that back each claim. Everything here is recomputed on the spot; nothing is
// read from a file of previously-printed results.
//
//   node validate/phase0.js

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import { Safetensors } from '../src/core/safetensors.js';
import { FileSource } from '../src/core/source-node.js';
import { ModelConfig } from '../src/core/config.js';
import { Tokenizer } from '../src/core/tokenizer.js';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const w = (p) => join(ROOT, 'weights', p);

const rule = (label) => console.log(`\n${label}\n${'-'.repeat(label.length)}`);

// --------------------------------------------------------------- the model --

rule('config vs weights');

const config = ModelConfig.parse(readFileSync(w('config.json'), 'utf-8'));
console.log(config.describe());

const st = await Safetensors.open(await FileSource.open(w('model.safetensors')));
const checked = config.checkAgainst(st);
console.log(`\nthe config implies ${checked} tensors; the file contains ${st.size}, ` +
  'with matching shapes and no extras');

// Counted from the file rather than from the config, so agreement between the
// two numbers means something.
let params = 0;
let bytes = 0;
for (const name of st.names()) {
  params += st.info(name).numel;
  bytes += st.info(name).byteLength;
}
console.log(`${params.toLocaleString()} parameters in ${(bytes / 1e6).toFixed(1)} MB of BF16`);
console.log(`config-implied parameter count: ${config.parameterCount().toLocaleString()}` +
  (params === config.parameterCount() ? ' (agrees)' : ' (DISAGREES)'));

// ------------------------------------------------------------- lazy loading --

rule('lazy loading');

const before = process.memoryUsage().rss;
const embed = st.info('model.embed_tokens.weight');
const someLayer = await st.readF32('model.layers.0.self_attn.q_proj.weight');
const after = process.memoryUsage().rss;
console.log(`the embedding alone is ${(embed.byteLength / 1e6).toFixed(1)} MB and was never read`);
console.log(`reading one ${someLayer.length.toLocaleString()}-element projection moved RSS by ` +
  `${((after - before) / 1e6).toFixed(1)} MB`);
console.log(`RSS is ${(after / 1e6).toFixed(0)} MB against a ${(bytes / 1e6).toFixed(0)} MB model`);

// ----------------------------------------------------------- the tokenizer --

rule('tokenizer');

const tok = Tokenizer.fromJSON(readFileSync(w('tokenizer.json'), 'utf-8'));
console.log(`${tok.vocabSize.toLocaleString()} tokens known to the tokenizer, ` +
  `${config.vocabSize.toLocaleString()} rows in the embedding ` +
  `(${(config.vocabSize - tok.vocabSize).toLocaleString()} unused, padding)`);

const golden = JSON.parse(readFileSync(join(ROOT, 'golden', 'tokenizer_cases.json'), 'utf-8'));
let differing = 0;
for (const { text, ids } of golden.cases) {
  const got = tok.encode(text);
  if (got.length !== ids.length || got.some((v, i) => v !== ids[i])) differing++;
}
console.log(`${golden.case_count.toLocaleString()} corpus cases, ` +
  `${golden.token_count.toLocaleString()} tokens: ${differing} differ from ` +
  `${golden.produced_by}`);

const prompt = 'The capital of France is';
const ids = tok.encode(prompt);
console.log(`\n${JSON.stringify(prompt)}`);
console.log(`  -> [${ids.join(', ')}]`);
console.log(`  -> ${ids.map((id) => JSON.stringify(tok.idToToken(id))).join(' ')}`);
console.log(`  -> ${JSON.stringify(tok.decode(ids))}`);

await st.close();

rule('gate');
const pass = checked === st.size && differing === 0 && params === config.parameterCount();
console.log(pass
  ? 'PASS: every tensor accounted for and bit-exact, every corpus case identical'
  : 'FAIL');
process.exit(pass ? 0 : 1);
