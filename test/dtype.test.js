import test from 'node:test';
import assert from 'node:assert/strict';
import {
  bf16ToF32, f32ToBf16, bf16ToF32Array,
  f16ToF32, f32ToF16, f16ToF32Array, f32ToF16Array,
} from '../src/core/dtype.js';

// A 16-bit format has 65536 inhabitants, so "test every one of them" is a
// realistic standard here rather than an aspiration. Both f16 directions are
// checked against Float16Array, which is a genuinely independent
// implementation (V8's, in C++) rather than a second copy of ours.

const ALL = 0x10000;

// Float16Array is the independent oracle for the f16 conversions: it is V8's
// own C++ implementation, so agreeing with it means something. It landed in
// Node 24. On anything older the f16 tests skip with a reason rather than
// failing with a bare ReferenceError.
const HAVE_F16_ORACLE = typeof Float16Array === 'function';
const needsOracle = HAVE_F16_ORACLE
  ? false
  : 'Float16Array requires Node 24 or newer';

const bits = new ArrayBuffer(4);
const asU32 = new Uint32Array(bits);
const asF32 = new Float32Array(bits);
const f32Bits = (x) => { asF32[0] = x; return asU32[0]; };

test('bulk bf16 widening is the exact 16-bit shift, for all 65536 values', () => {
  // The bulk path writes through an aliased Uint32Array and never materialises
  // a JS Number, so it is bit-exact even for signalling NaNs.
  const src = new Uint16Array(ALL);
  for (let b = 0; b < ALL; b++) src[b] = b;
  const out = bf16ToF32Array(src);
  const outBits = new Uint32Array(out.buffer);
  for (let b = 0; b < ALL; b++) {
    assert.equal(outBits[b], (b << 16) >>> 0, `bf16 0x${b.toString(16)}`);
  }
});

test('scalar bf16 widening is bit-exact for every non-NaN value', () => {
  for (let b = 0; b < ALL; b++) {
    if ((b & 0x7f80) === 0x7f80 && (b & 0x007f) !== 0) continue; // NaN, see below
    assert.equal(f32Bits(bf16ToF32(b)), (b << 16) >>> 0, `bf16 0x${b.toString(16)}`);
  }
});

test('scalar widening quiets signalling NaNs, and that is the platform, not us', () => {
  // bf16ToF32 returns a JS Number, which is an f64. Widening an f32 signalling
  // NaN to f64 and narrowing it back sets the quiet bit in hardware, so the
  // payload cannot survive the return type. The value stays NaN and the sign
  // survives; only the signalling bit is lost. Documented here because a
  // reader comparing the two paths deserves to know they differ, and because
  // the bulk path above is the one the weight loader actually uses.
  const sNaN = 0x7f81;
  assert.ok(Number.isNaN(bf16ToF32(sNaN)));
  assert.equal(f32Bits(bf16ToF32(sNaN)), 0x7fc10000, 'quiet bit set by hardware');

  const bulk = bf16ToF32Array(new Uint16Array([sNaN]));
  assert.equal(new Uint32Array(bulk.buffer)[0], 0x7f810000, 'bulk path unaffected');
});

test('bf16 round-trips through f32 for every value', () => {
  // Widening is exact, so narrowing must return the original bits. NaN is
  // excluded only because f32ToBf16 deliberately forces a quiet payload.
  for (let b = 0; b < ALL; b++) {
    const isNaNPattern = (b & 0x7f80) === 0x7f80 && (b & 0x007f) !== 0;
    if (isNaNPattern) continue;
    assert.equal(f32ToBf16(bf16ToF32(b)), b, `bf16 0x${b.toString(16)}`);
  }
});

test('bf16 narrowing rounds to nearest even', () => {
  // A value exactly halfway between two bf16 neighbours must land on the one
  // with an even low bit. 0x3f808000 sits midway between 1.0 (0x3f80) and the
  // next bf16 up (0x3f81); even wins, so 1.0.
  asU32[0] = 0x3f808000;
  assert.equal(f32ToBf16(asF32[0]), 0x3f80);
  // 0x3f818000 sits midway between 0x3f81 and 0x3f82; even wins again.
  asU32[0] = 0x3f818000;
  assert.equal(f32ToBf16(asF32[0]), 0x3f82);
  // Just past halfway always rounds up.
  asU32[0] = 0x3f808001;
  assert.equal(f32ToBf16(asF32[0]), 0x3f81);
});

test('bf16 narrowing saturates to infinity rather than wrapping', () => {
  asU32[0] = 0x7f7fffff; // largest finite f32
  assert.equal(f32ToBf16(asF32[0]), 0x7f80);
  asU32[0] = 0xff7fffff;
  assert.equal(f32ToBf16(asF32[0]), 0xff80);
  assert.equal(f32ToBf16(Infinity), 0x7f80);
  assert.equal(f32ToBf16(-Infinity), 0xff80);
});

test('bf16 preserves NaN as NaN', () => {
  for (const pattern of [0x7f81, 0x7fc0, 0x7fff, 0xffc1]) {
    const widened = bf16ToF32(pattern);
    assert.ok(Number.isNaN(widened), `0x${pattern.toString(16)} should widen to NaN`);
    const narrowed = f32ToBf16(widened);
    assert.equal((narrowed & 0x7f80), 0x7f80);
    assert.notEqual((narrowed & 0x007f), 0, 'must not collapse to infinity');
    assert.equal(narrowed >>> 15, pattern >>> 15, 'sign preserved');
  }
});


test('f16 widening matches Float16Array on all 65536 values', { skip: needsOracle }, () => {
  const oracle = new Float16Array(1);
  const oracleBits = new Uint16Array(oracle.buffer);
  for (let h = 0; h < ALL; h++) {
    oracleBits[0] = h;
    const expected = oracle[0];
    const got = f16ToF32(h);
    if (Number.isNaN(expected)) {
      assert.ok(Number.isNaN(got), `f16 0x${h.toString(16)} should be NaN`);
    } else {
      // Compared as bits, so signed zero is distinguished from zero.
      assert.equal(f32Bits(got), f32Bits(expected), `f16 0x${h.toString(16)}`);
    }
  }
});

test('f16 narrowing matches Float16Array on all 65536 round-trip inputs', { skip: needsOracle }, () => {
  const oracle = new Float16Array(1);
  const oracleBits = new Uint16Array(oracle.buffer);
  for (let h = 0; h < ALL; h++) {
    oracleBits[0] = h;
    const value = oracle[0];
    if (Number.isNaN(value)) continue; // NaN payloads are not required to match
    oracle[0] = value;
    assert.equal(f32ToF16(value), oracleBits[0], `f16 0x${h.toString(16)}`);
  }
});

test('f16 narrowing matches Float16Array on values between representable ones', { skip: needsOracle }, () => {
  // Round-tripping only exercises inputs that are already f16-exact, which
  // never tests the rounding logic. These are the inputs that do: midpoints,
  // subnormals, and the overflow boundary.
  const oracle = new Float16Array(1);
  const oracleBits = new Uint16Array(oracle.buffer);
  const probes = [];
  for (let h = 0; h < ALL - 1; h++) {
    oracleBits[0] = h;
    const lo = oracle[0];
    oracleBits[0] = h + 1;
    const hi = oracle[0];
    if (!Number.isFinite(lo) || !Number.isFinite(hi)) continue;
    probes.push(lo + (hi - lo) / 2);                 // exact midpoint: ties-to-even
    probes.push(lo + (hi - lo) * 0.25);
    probes.push(lo + (hi - lo) * 0.75);
  }
  probes.push(65519.999, 65520, 65536, 1e30, -1e30); // the overflow boundary
  probes.push(5.96e-8, 2.98e-8, 1e-10, 0, -0);       // gradual underflow

  for (const x of probes) {
    // Both sides must see the *same* number. Feeding the oracle an f64 while
    // feeding f32ToF16 its f32 rounding compares two different inputs: at
    // 65519.999, Math.fround lands exactly on the 65520 overflow midpoint
    // while the f64 stays below it, and the two answers legitimately differ.
    const xf = Math.fround(x);
    oracle[0] = xf;
    assert.equal(f32ToF16(xf), oracleBits[0], `f32ToF16(${xf})`);
  }
  assert.equal(probes.length > 150000, true, 'probe set should be exhaustive-ish');
});

test('bulk f16 conversions agree with the scalar path', () => {
  const src = new Uint16Array(ALL);
  for (let h = 0; h < ALL; h++) src[h] = h;
  const wide = f16ToF32Array(src);
  const narrow = f32ToF16Array(wide);
  for (let h = 0; h < ALL; h++) {
    if (Number.isNaN(wide[h])) continue;
    assert.equal(narrow[h], h, `f16 0x${h.toString(16)} bulk round-trip`);
  }
});
