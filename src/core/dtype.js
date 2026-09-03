// Float format conversions.
//
// The weights ship as bfloat16, which no JavaScript typed array understands, so
// the conversion is ours. f16 is here too because the GPU backend stores
// activations in it, and both directions of both formats are cheap to test
// exhaustively -- there are only 65536 inputs.

const _buf = new ArrayBuffer(4);
const _u32 = new Uint32Array(_buf);
const _f32 = new Float32Array(_buf);

// ---------------------------------------------------------------- bfloat16 --
// bfloat16 is the top 16 bits of an IEEE-754 binary32 with the low 16 discarded.
// Widening is therefore exact and total: every bf16 value, including the
// infinities, the NaNs and the subnormals, has an exact f32 counterpart.

export function bf16ToF32(bits) {
  _u32[0] = (bits & 0xffff) << 16;
  return _f32[0];
}

/** Round-to-nearest-even, the mode PyTorch and the hardware both use. */
export function f32ToBf16(x) {
  _f32[0] = x;
  const u = _u32[0];
  // NaN must stay NaN. Truncation alone can land on an infinity, so force a
  // payload bit and keep the sign.
  if ((u & 0x7f800000) === 0x7f800000 && (u & 0x007fffff) !== 0) {
    return ((u >>> 16) | 0x0040) & 0xffff;
  }
  const lsb = (u >>> 16) & 1;
  // u + 0x7fff + lsb cannot exceed 2^32 for any non-NaN u, so the implicit
  // ToUint32 in >>> is a no-op rather than a wraparound.
  return ((u + 0x7fff + lsb) >>> 16) & 0xffff;
}

/**
 * Widen a block of bfloat16 into f32.
 *
 * Writes through a Uint32Array aliased onto the output's buffer, so the whole
 * loop is an integer shift and no float rounding can creep in.
 */
export function bf16ToF32Array(src, out) {
  const n = src.length;
  const dst = out ?? new Float32Array(n);
  const bits = new Uint32Array(dst.buffer, dst.byteOffset, n);
  for (let i = 0; i < n; i++) bits[i] = src[i] << 16;
  return dst;
}

// --------------------------------------------------------------- float16 --
// 1 sign, 5 exponent, 10 mantissa. Unlike bf16 this is not a bit slice of f32,
// so both directions need real work.

export function f16ToF32(bits) {
  const sign = (bits & 0x8000) << 16;
  const exp = (bits >>> 10) & 0x1f;
  const mant = bits & 0x3ff;

  if (exp === 0x1f) {
    // Inf or NaN: f16 bias 15 -> f32 bias 127, mantissa left-aligned.
    _u32[0] = sign | 0x7f800000 | (mant << 13);
  } else if (exp === 0) {
    if (mant === 0) {
      _u32[0] = sign; // signed zero
    } else {
      // Subnormal. Normalise by shifting the mantissa up until the implicit
      // bit appears, decrementing the exponent to match.
      let e = -1;
      let m = mant;
      do { m <<= 1; e++; } while ((m & 0x400) === 0);
      _u32[0] = sign | ((127 - 15 - e) << 23) | ((m & 0x3ff) << 13);
    }
  } else {
    _u32[0] = sign | ((exp - 15 + 127) << 23) | (mant << 13);
  }
  return _f32[0];
}

/** Round-to-nearest-even, with overflow to infinity and gradual underflow. */
export function f32ToF16(x) {
  _f32[0] = x;
  const u = _u32[0];
  const sign = (u >>> 16) & 0x8000;
  const exp = (u >>> 23) & 0xff;
  const mant = u & 0x7fffff;

  if (exp === 0xff) {
    // Preserve NaN-ness; a nonzero f32 mantissa must not round to infinity.
    return sign | 0x7c00 | (mant !== 0 ? (mant >>> 13) | 0x200 : 0);
  }

  const e = exp - 127 + 15; // rebias

  if (e >= 0x1f) return sign | 0x7c00;      // overflow -> inf
  if (e <= 0) {
    if (e < -10) return sign;               // rounds to zero
    // Subnormal: restore the implicit bit, then shift so the result lands in
    // the 10-bit field, rounding to nearest even on the way.
    const m = mant | 0x800000;
    const shift = 14 - e;                   // 14 .. 24
    const half = 1 << (shift - 1);
    const lsb = (m >>> shift) & 1;
    return sign | ((m + half - 1 + lsb) >>> shift);
  }

  const lsb = (mant >>> 13) & 1;
  const rounded = mant + 0x0fff + lsb;
  // Rounding can carry into the exponent; adding it in rather than masking
  // lets the carry propagate, and e < 0x1f above means the result stays finite
  // unless the carry itself pushes it to inf, which is correct.
  return sign | ((e << 10) + (rounded >>> 13));
}

export function f16ToF32Array(src, out) {
  const n = src.length;
  const dst = out ?? new Float32Array(n);
  for (let i = 0; i < n; i++) dst[i] = f16ToF32(src[i]);
  return dst;
}

export function f32ToF16Array(src, out) {
  const n = src.length;
  const dst = out ?? new Uint16Array(n);
  for (let i = 0; i < n; i++) dst[i] = f32ToF16(src[i]);
  return dst;
}
