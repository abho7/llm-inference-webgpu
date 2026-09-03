// Greedy decoding, the slow way.
//
// Phase 1 has no KV cache, so generating token n means recomputing the whole
// sequence from scratch: the work is quadratic in the output length and each
// step re-streams every weight from disk. That is the point of running it
// anyway. It produces a text continuation whose every token can be checked, and
// it establishes the baseline that phase 2 has to reproduce exactly -- the KV
// cache is only correct if it gives these same tokens.
//
//   node validate/generate.js "The capital of France is" 8

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

const prompt = process.argv[2] ?? 'The capital of France is';
const wanted = Number(process.argv[3] ?? 8);

const config = ModelConfig.parse(readFileSync(w('config.json'), 'utf-8'));
const tokenizer = Tokenizer.fromJSON(readFileSync(w('tokenizer.json'), 'utf-8'));
const st = await Safetensors.open(await FileSource.open(w('model.safetensors')));
const model = new ReferenceModel(config, new Weights(st, config));

const ids = tokenizer.encode(prompt);
console.log(`prompt: ${JSON.stringify(prompt)}`);
console.log(`        ${ids.length} tokens [${ids.join(', ')}]\n`);

const eos = new Set([151645, 151643]); // <|im_end|>, <|endoftext|>
const generated = [];
const started = performance.now();

for (let step = 0; step < wanted; step++) {
  const stepStarted = performance.now();
  const { logits } = await model.forward([...ids, ...generated]);

  let best = 0;
  for (let i = 1; i < logits.length; i++) if (logits[i] > logits[best]) best = i;

  const seconds = (performance.now() - stepStarted) / 1000;
  const context = ids.length + generated.length;
  console.log(`  ${String(step).padStart(2)}  ${String(best).padStart(6)}  ` +
    `${JSON.stringify(tokenizer.idToToken(best)).padEnd(16)}  ` +
    `logit ${logits[best].toFixed(2).padStart(7)}  ` +
    `${context} tokens of context, ${seconds.toFixed(1)}s`);

  generated.push(best);
  if (eos.has(best)) {
    console.log('  (end of sequence)');
    break;
  }
}

const elapsed = (performance.now() - started) / 1000;
console.log(`\ncontinuation: ${JSON.stringify(tokenizer.decode(generated))}`);
console.log(`full text:    ${JSON.stringify(tokenizer.decode([...ids, ...generated]))}`);
console.log(`\n${generated.length} tokens in ${elapsed.toFixed(1)}s ` +
  `(${(generated.length / elapsed).toFixed(2)} tok/s, no KV cache, weights streamed from disk)`);

await st.close();
