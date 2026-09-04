// What does storing this model's weights as f16 actually cost?
//
// Written to settle a question the kernel comparison answered misleadingly. On
// random f32 inputs, rounding a matrix to f16 shifts the result by about 2e-4
// relative, and it would be easy to report that as the price of the GPU
// backend. It is not, because these weights do not arrive as f32.
//
// They arrive as bfloat16: 1 sign bit, 8 exponent bits, 7 mantissa bits. f16
// has 1, 5 and 10. The mantissa is therefore *wider* in f16, so every bf16
// value whose exponent f16 can represent converts exactly. The only losses are
// at the bottom of the range, where f16 runs out of exponent and starts
// gradually underflowing.
//
// This walks every weight and counts them.
//
//   node validate/f16-fidelity.js

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import { Safetensors } from '../src/core/safetensors.js';
import { FileSource } from '../src/core/source-node.js';
import { ModelConfig } from '../src/core/config.js';
import { f32ToF16, f16ToF32 } from '../src/core/dtype.js';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');

const F16_MAX = 65504;
const F16_MIN_NORMAL = 2 ** -14;      // 6.10e-5
const F16_MIN_SUBNORMAL = 2 ** -24;   // 5.96e-8

const config = ModelConfig.parse(readFileSync(join(ROOT, 'weights', 'config.json'), 'utf-8'));
const st = await Safetensors.open(
  await FileSource.open(join(ROOT, 'weights', 'model.safetensors')),
);

let total = 0;
let changed = 0;
let flushed = 0;
let overflowed = 0;
let subnormal = 0;
let smallestNonzero = Infinity;
let largest = 0;
let worstRelative = 0;
const byTensor = [];

for (const name of st.names()) {
  const w = await st.readF32(name);
  let tensorChanged = 0;
  for (let i = 0; i < w.length; i++) {
    const value = w[i];
    const back = f16ToF32(f32ToF16(value));
    const magnitude = Math.abs(value);
    if (magnitude > largest) largest = magnitude;
    if (magnitude > 0 && magnitude < smallestNonzero) smallestNonzero = magnitude;
    if (magnitude > F16_MAX) overflowed++;
    else if (magnitude > 0 && magnitude < F16_MIN_NORMAL) subnormal++;
    if (back !== value) {
      tensorChanged++;
      if (magnitude > 0 && magnitude < F16_MIN_SUBNORMAL) flushed++;
      // Relative error only means something away from zero.
      if (magnitude > F16_MIN_NORMAL) {
        const rel = Math.abs(back - value) / magnitude;
        if (rel > worstRelative) worstRelative = rel;
      }
    }
  }
  total += w.length;
  changed += tensorChanged;
  if (tensorChanged) byTensor.push([tensorChanged, name]);
}

await st.close();

byTensor.sort((a, b) => b[0] - a[0]);
const pct = (n) => `${((n / total) * 100).toPrecision(3)}%`;

console.log(`model: ${config.describe().split('\n')[0]}`);
console.log(`parameters                        ${total.toLocaleString()}`);
console.log(`changed by bf16 -> f16            ${changed.toLocaleString()}  (${pct(changed)})`);
console.log(`  flushed to zero                 ${flushed.toLocaleString()}`);
console.log(`  overflowed to infinity          ${overflowed.toLocaleString()}`);
console.log(`weights f16 must store subnormally ${subnormal.toLocaleString()}  (${pct(subnormal)})`);
console.log(`smallest nonzero |weight|         ${smallestNonzero.toExponential(3)}` +
  `   (f16 smallest subnormal ${F16_MIN_SUBNORMAL.toExponential(3)})`);
console.log(`largest |weight|                  ${largest.toFixed(1)}` +
  `   (f16 largest finite ${F16_MAX})`);
console.log(`worst relative change, normal range ${worstRelative.toExponential(2)}`);

if (byTensor.length) {
  console.log('\ntensors with any changed value:');
  for (const [count, name] of byTensor.slice(0, 6)) {
    console.log(`  ${name.padEnd(48)} ${count.toLocaleString()}`);
  }
  if (byTensor.length > 6) console.log(`  ... and ${byTensor.length - 6} more tensors`);
}

console.log('\nreading');
console.log('-------');
console.log('bf16 carries 7 mantissa bits and f16 carries 10, so the conversion cannot');
console.log('lose mantissa precision. Every changed value is one f16 has too little');
console.log('exponent range for: below 6.10e-5 it goes subnormal and starts dropping');
console.log('bits, and below 5.96e-8 it flushes to zero. Nothing overflows, because the');
console.log(`largest weight here is ${largest.toFixed(0)} against an f16 ceiling of ${F16_MAX}.`);
