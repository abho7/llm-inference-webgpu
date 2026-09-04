// Paired perplexity comparison between weight schemes.
//
// The browser harness (web/perplexity.html) measures the same thing far faster,
// but reported only aggregate perplexity, and an aggregate difference of a
// fraction of a percent on a couple of hundred positions is not evidence of
// anything. This one is slower and answers the question properly: every scheme
// scores exactly the same positions, so the schemes can be compared position by
// position, and the standard error of the mean difference says whether a small
// gap is real.
//
// Protocol, fixed before running and identical for every scheme:
//
//   text      a fixed passage held in this file
//   window    each token after the first is predicted from all tokens before
//             it, no sliding window, no stride, nothing scored twice
//   scored    every position except the first
//   metric    exp(mean negative log-likelihood), natural log
//
//   node validate/perplexity.js [tokenLimit]

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import { Safetensors } from '../src/core/safetensors.js';
import { FileSource } from '../src/core/source-node.js';
import { ModelConfig } from '../src/core/config.js';
import { Tokenizer } from '../src/core/tokenizer.js';
import { Weights } from '../src/cpu/weights.js';
import { ReferenceModel } from '../src/cpu/model.js';
import { bitsPerWeight } from '../src/core/quantize.js';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const w = (p) => join(ROOT, 'weights', p);

const TEXT = `The theory of computation asks what problems can be solved by machines, and how
much time and memory the solutions need. A Turing machine is deliberately impoverished: a
tape, a head that reads and writes one symbol at a time, and a finite table of rules. That
poverty is the point. If a problem can be solved at all by any mechanical procedure, it can
be solved by this one, so a statement about Turing machines is a statement about computation
itself rather than about any particular computer.`;

const SCHEMES = [
  { label: 'f16', quantize: null, bits: 16 },
  { label: 'int8', quantize: { scheme: 'int8' }, bits: bitsPerWeight('int8', 896) },
  { label: 'int4-g128', quantize: { scheme: 'int4', groupSize: 128 }, bits: bitsPerWeight('int4', 896, 128) },
];

const limit = Number(process.argv[2] ?? 48);

const config = ModelConfig.parse(readFileSync(w('config.json'), 'utf-8'));
const tokenizer = Tokenizer.fromJSON(readFileSync(w('tokenizer.json'), 'utf-8'));

const allIds = tokenizer.encode(TEXT.replace(/\s+/g, ' ').trim());
const ids = allIds.slice(0, limit);
console.log(`scoring ${ids.length - 1} positions of a ${allIds.length}-token passage\n`);

const results = [];
let baselineTokens = null;

for (const scheme of SCHEMES) {
  const st = await Safetensors.open(await FileSource.open(w('model.safetensors')));
  const weights = new Weights(st, config, { resident: true, quantize: scheme.quantize });
  const model = new ReferenceModel(config, weights);

  const cache = model.newCache(ids.length + 2);
  const perToken = new Float64Array(ids.length - 1);
  const started = performance.now();

  for (let t = 0; t < ids.length - 1; t++) {
    const { logits } = await model.forward([ids[t]], { cache });
    let max = -Infinity;
    for (let i = 0; i < logits.length; i++) if (logits[i] > max) max = logits[i];
    let sumExp = 0;
    for (let i = 0; i < logits.length; i++) sumExp += Math.exp(logits[i] - max);
    perToken[t] = -(logits[ids[t + 1]] - max - Math.log(sumExp));
    if (t % 8 === 0) process.stderr.write(`  ${scheme.label}: ${t}/${ids.length - 1}\r`);
  }
  process.stderr.write('                          \r');

  let sum = 0;
  for (const v of perToken) sum += v;
  const ppl = Math.exp(sum / perToken.length);
  const seconds = (performance.now() - started) / 1000;

  // Paired against the first scheme: the mean of per-position differences and
  // the standard error of that mean.
  let paired = null;
  if (baselineTokens === null) {
    baselineTokens = perToken;
  } else {
    const n = perToken.length;
    let meanDiff = 0;
    for (let i = 0; i < n; i++) meanDiff += perToken[i] - baselineTokens[i];
    meanDiff /= n;
    let variance = 0;
    for (let i = 0; i < n; i++) {
      const centred = (perToken[i] - baselineTokens[i]) - meanDiff;
      variance += centred * centred;
    }
    variance /= n - 1;
    const stderr = Math.sqrt(variance / n);
    paired = { meanDiff, stderr, ratio: stderr > 0 ? meanDiff / stderr : 0 };
  }

  results.push({ ...scheme, ppl, paired, seconds });
  await st.close();

  const line = `${scheme.label.padEnd(11)}${scheme.bits.toFixed(3).padStart(7)} bits  ` +
    `ppl ${ppl.toFixed(4).padStart(9)}  ${seconds.toFixed(0).padStart(4)}s`;
  console.log(paired
    ? `${line}   mean NLL ${paired.meanDiff >= 0 ? '+' : ''}${paired.meanDiff.toFixed(4)}` +
      ` +- ${paired.stderr.toFixed(4)}  (${paired.ratio.toFixed(1)} standard errors)`
    : `${line}   baseline`);
}

console.log('\nreading');
console.log('-------');
const n = ids.length - 1;
for (const r of results.slice(1)) {
  const significant = Math.abs(r.paired.ratio) >= 2;
  const direction = r.paired.meanDiff > 0 ? 'worse' : 'better';
  console.log(`${r.label}: ${significant
    ? `${direction} than f16 by ${Math.abs(r.paired.meanDiff).toFixed(4)} nats per token, ` +
      `${Math.abs(r.paired.ratio).toFixed(1)} standard errors -- a real difference`
    : `indistinguishable from f16 at ${n} positions (${Math.abs(r.paired.ratio).toFixed(1)} ` +
      'standard errors, and two would be the usual threshold)'}`);
}
console.log(`\n${n} scored positions is a small sample. It is enough to separate a large`);
console.log('effect from nothing, which is what the int4 result needs, and not enough to');
console.log('resolve a fraction of a percent, which is what the int8 result would need.');
