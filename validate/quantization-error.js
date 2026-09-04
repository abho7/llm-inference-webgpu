// Per-tensor quantization error across the whole model.
//
// The headline number for a quantization scheme is usually a single perplexity
// figure, which says nothing about where the damage is. This measures every
// weight matrix separately under every scheme, so the question "which parts of
// this model are hard to quantize" has an answer.
//
// Rows are processed in blocks and the dequantized matrix is never
// materialised. Both schemes are defined within a row -- int8 has one scale per
// row, int4 one per group inside a row -- so blocking by rows changes nothing,
// and it keeps the embedding matrix from needing a gigabyte of headroom.
//
//   node validate/quantization-error.js [--json]

import { readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import { Safetensors } from '../src/core/safetensors.js';
import { FileSource } from '../src/core/source-node.js';
import { ModelConfig } from '../src/core/config.js';
import { roundTrip, bitsPerWeight } from '../src/core/quantize.js';
import { f32ToF16, f16ToF32 } from '../src/core/dtype.js';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const emitJson = process.argv.includes('--json');

const SCHEMES = [
  { name: 'f16', kind: 'f16' },
  { name: 'int8', kind: 'int8' },
  { name: 'int4-g128', kind: 'int4', groupSize: 128 },
  { name: 'int4-g64', kind: 'int4', groupSize: 64 },
  { name: 'int4-g32', kind: 'int4', groupSize: 32 },
];

const ROW_BLOCK = 2048;

const config = ModelConfig.parse(readFileSync(join(ROOT, 'weights', 'config.json'), 'utf-8'));
const st = await Safetensors.open(
  await FileSource.open(join(ROOT, 'weights', 'model.safetensors')),
);

/** Frobenius error of one tensor under one scheme, computed in row blocks. */
function measure(W, rows, cols, scheme) {
  let sumSquaredError = 0;
  let sumSquared = 0;
  let maxAbs = 0;

  for (let start = 0; start < rows; start += ROW_BLOCK) {
    const end = Math.min(start + ROW_BLOCK, rows);
    const block = W.subarray(start * cols, end * cols);
    let back;
    if (scheme.kind === 'f16') {
      back = new Float32Array(block.length);
      for (let i = 0; i < block.length; i++) back[i] = f16ToF32(f32ToF16(block[i]));
    } else {
      back = roundTrip(block, end - start, cols, scheme.kind, scheme.groupSize);
    }
    for (let i = 0; i < block.length; i++) {
      const diff = back[i] - block[i];
      sumSquaredError += diff * diff;
      sumSquared += block[i] * block[i];
      const magnitude = Math.abs(diff);
      if (magnitude > maxAbs) maxAbs = magnitude;
    }
  }
  return {
    frobenius: sumSquared > 0 ? Math.sqrt(sumSquaredError / sumSquared) : 0,
    maxAbs,
  };
}

/** Group tensors by their role, since that is what the pattern turns out to follow. */
function roleOf(name) {
  if (name === 'model.embed_tokens.weight') return 'embed';
  if (name === 'model.norm.weight') return 'norm';
  const match = /model\.layers\.\d+\.(.*)\.weight$/.exec(name);
  if (!match) return 'bias/norm';
  const part = match[1];
  if (part.endsWith('layernorm')) return 'norm';
  return part.replace('self_attn.', '').replace('mlp.', '');
}

const results = [];
const names = st.names().filter((n) => st.info(n).shape.length === 2);
console.log(`measuring ${names.length} weight matrices under ${SCHEMES.length} schemes\n`);

let done = 0;
for (const name of names) {
  const info = st.info(name);
  const [rows, cols] = info.shape;
  const W = await st.readF32(name);
  const entry = { name, role: roleOf(name), rows, cols, schemes: {} };
  for (const scheme of SCHEMES) {
    if (scheme.groupSize && cols % scheme.groupSize !== 0) continue;
    entry.schemes[scheme.name] = measure(W, rows, cols, scheme);
  }
  results.push(entry);
  done++;
  if (done % 24 === 0) process.stderr.write(`  ${done}/${names.length}\r`);
}
process.stderr.write('                    \r');
await st.close();

// ------------------------------------------------------------- by role ----

const roles = new Map();
for (const entry of results) {
  if (!roles.has(entry.role)) roles.set(entry.role, []);
  roles.get(entry.role).push(entry);
}

const pad = (s, n) => String(s).padEnd(n);
const num = (v, n = 9) => v.toExponential(2).padStart(n);

console.log('relative Frobenius error, averaged over the tensors of each role');
console.log();
console.log(`${pad('role', 12)}${pad('count', 6)}${SCHEMES.map((s) => pad(s.name, 12)).join('')}`);
console.log('-'.repeat(12 + 6 + SCHEMES.length * 12));

const order = ['embed', 'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'];
for (const role of order) {
  const group = roles.get(role);
  if (!group) continue;
  const cells = SCHEMES.map((scheme) => {
    const values = group.map((e) => e.schemes[scheme.name]?.frobenius).filter((v) => v !== undefined);
    if (!values.length) return pad('-', 12);
    return pad(num(values.reduce((a, b) => a + b, 0) / values.length, 8), 12);
  });
  console.log(`${pad(role, 12)}${pad(group.length, 6)}${cells.join('')}`);
}

// ------------------------------------------------------- hardest tensors ----

console.log('\nhardest tensors under int4-g128');
const hardest = results
  .filter((e) => e.schemes['int4-g128'])
  .sort((a, b) => b.schemes['int4-g128'].frobenius - a.schemes['int4-g128'].frobenius)
  .slice(0, 8);
for (const entry of hardest) {
  console.log(`  ${pad(entry.name, 48)} ${num(entry.schemes['int4-g128'].frobenius)}`);
}

console.log('\neasiest tensors under int4-g128');
for (const entry of hardest.length ? results
  .filter((e) => e.schemes['int4-g128'])
  .sort((a, b) => a.schemes['int4-g128'].frobenius - b.schemes['int4-g128'].frobenius)
  .slice(0, 4) : []) {
  console.log(`  ${pad(entry.name, 48)} ${num(entry.schemes['int4-g128'].frobenius)}`);
}

// -------------------------------------------------------------- storage ----

console.log('\nstorage, counting scales and zero points');
const totalWeights = results.reduce((a, e) => a + e.rows * e.cols, 0);
console.log(`${pad('scheme', 12)}${pad('bits/weight', 14)}${pad('model size', 12)}`);
console.log('-'.repeat(38));
for (const scheme of SCHEMES) {
  // Quoted against the 896-column projections, the common case.
  const bits = bitsPerWeight(scheme.kind === 'f16' ? 'f16' : scheme.kind, 896, scheme.groupSize);
  const bytes = (totalWeights * bits) / 8;
  console.log(`${pad(scheme.name, 12)}${pad(bits.toFixed(3), 14)}${pad(`${(bytes / 1e6).toFixed(0)} MB`, 12)}`);
}

if (emitJson) {
  const path = join(ROOT, 'golden', 'quantization-error.json');
  writeFileSync(path, `${JSON.stringify({
    model: 'Qwen2.5-0.5B-Instruct',
    schemes: SCHEMES.map((s) => s.name),
    totalWeights,
    tensors: results,
  }, null, 1)}\n`);
  console.log(`\nwrote ${path}`);
}
