import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { existsSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import { Safetensors } from '../src/core/safetensors.js';
import { FileSource, BufferSource } from '../src/core/source-node.js';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const MODEL = join(ROOT, 'weights', 'model.safetensors');
const DIGEST = join(ROOT, 'golden', 'tensor_digest.json');

// ---------------------------------------------------------------------------
// Header validation. These run against synthetic files, so they need no model
// and no network, and they are the tests that actually get exercised when
// something is wrong.
// ---------------------------------------------------------------------------

/** Build a safetensors file in memory from a header object and a data section. */
function build(header, data = new Uint8Array(0)) {
  const json = new TextEncoder().encode(JSON.stringify(header));
  const out = new Uint8Array(8 + json.length + data.length);
  new DataView(out.buffer).setBigUint64(0, BigInt(json.length), true);
  out.set(json, 8);
  out.set(data, 8 + json.length);
  return new BufferSource(out);
}

const ok = { t: { dtype: 'F32', shape: [2, 2], data_offsets: [0, 16] } };

async function rejects(source, match) {
  await assert.rejects(() => Safetensors.open(source), match);
}

test('opens a well-formed file and reports its tensors', async () => {
  const data = new Uint8Array(new Float32Array([1, 2, 3, 4]).buffer);
  const st = await Safetensors.open(build({ ...ok, __metadata__: { format: 'pt' } }, data));
  assert.equal(st.size, 1);
  assert.deepEqual(st.names(), ['t']);
  assert.deepEqual(st.metadata, { format: 'pt' });
  assert.deepEqual([...st.info('t').shape], [2, 2]);
  assert.equal(st.info('t').numel, 4);
  assert.deepEqual([...(await st.readF32('t'))], [1, 2, 3, 4]);
});

test('rejects a file too short to hold a header length', async () => {
  await rejects(new BufferSource(new Uint8Array(4)), /not a safetensors file/);
});

test('rejects a header length that runs past the end of the file', async () => {
  const src = build(ok, new Uint8Array(16));
  const bytes = await src.read(0, await src.size());
  new DataView(bytes.buffer).setBigUint64(0, 1n << 40n, true);
  await rejects(new BufferSource(bytes), /does not fit/);
});

test('rejects a header that is not JSON', async () => {
  const bad = new Uint8Array(8 + 3);
  new DataView(bad.buffer).setBigUint64(0, 3n, true);
  bad.set(new TextEncoder().encode('{{{'), 8);
  await rejects(new BufferSource(bad), /not valid UTF-8 JSON/);
});

test('rejects an unknown dtype rather than guessing an element size', async () => {
  await rejects(
    build({ t: { dtype: 'F4_SECRET', shape: [4], data_offsets: [0, 16] } }, new Uint8Array(16)),
    /unknown dtype/,
  );
});

test('rejects a shape whose element count disagrees with its byte range', async () => {
  // 2x2 F32 needs 16 bytes; the header claims 12.
  await rejects(
    build({ t: { dtype: 'F32', shape: [2, 2], data_offsets: [0, 12] } }, new Uint8Array(12)),
    /needs 16 bytes but the header reserves 12/,
  );
});

test('rejects a tensor that ends past the data section, ie a truncated download', async () => {
  await rejects(build(ok, new Uint8Array(8)), /data section is only 8 bytes/);
});

test('rejects negative and inverted offsets', async () => {
  await rejects(
    build({ t: { dtype: 'F32', shape: [1], data_offsets: [8, 4] } }, new Uint8Array(16)),
    /bad data_offsets/,
  );
});

test('rejects overlapping tensors, which would silently corrupt one another', async () => {
  await rejects(
    build({
      a: { dtype: 'F32', shape: [4], data_offsets: [0, 16] },
      b: { dtype: 'F32', shape: [4], data_offsets: [8, 24] },
    }, new Uint8Array(24)),
    /overlaps/,
  );
});

test('reads a tensor stored at a non-zero offset, not just the first one', async () => {
  const data = new Uint8Array(new Float32Array([9, 9, 1, 2, 3]).buffer);
  const st = await Safetensors.open(build({
    a: { dtype: 'F32', shape: [2], data_offsets: [0, 8] },
    b: { dtype: 'F32', shape: [3], data_offsets: [8, 20] },
  }, data));
  assert.deepEqual([...(await st.readF32('b'))], [1, 2, 3]);
});

test('widens bf16 tensors on read', async () => {
  // 0x3f80 is 1.0, 0xc000 is -2.0, 0x0000 is +0.
  const data = new Uint8Array(new Uint16Array([0x3f80, 0xc000, 0x0000]).buffer);
  const st = await Safetensors.open(build({
    t: { dtype: 'BF16', shape: [3], data_offsets: [0, 6] },
  }, data));
  assert.deepEqual([...(await st.readF32('t'))], [1, -2, 0]);
});

test('names a missing tensor in the error instead of returning undefined', async () => {
  const st = await Safetensors.open(build(ok, new Uint8Array(16)));
  assert.throws(() => st.info('nope'), /no tensor named "nope"/);
});

// ---------------------------------------------------------------------------
// The Phase 0 gate: every tensor in the real model, bit-for-bit against an
// independent NumPy parse (oracle/tensor_digest.py). Skipped rather than failed
// when the weights are absent, so a fresh clone still has a green suite.
// ---------------------------------------------------------------------------

const haveModel = existsSync(MODEL) && existsSync(DIGEST);

test('every tensor in the model matches the independent NumPy digest', {
  skip: haveModel ? false : 'run oracle/fetch_weights.py and oracle/tensor_digest.py first',
}, async () => {
  const golden = JSON.parse(readFileSync(DIGEST, 'utf-8'));
  const st = await Safetensors.open(await FileSource.open(MODEL));

  assert.equal(st.size, golden.tensor_count, 'tensor count');
  assert.deepEqual(st.names().sort(), Object.keys(golden.tensors).sort(), 'tensor names');

  for (const [name, want] of Object.entries(golden.tensors)) {
    const info = st.info(name);
    assert.equal(info.dtype, want.dtype, `${name} dtype`);
    assert.deepEqual([...info.shape], want.shape, `${name} shape`);

    const raw = await st.readRaw(name);
    assert.equal(
      createHash('sha256').update(raw).digest('hex'), want.raw_sha256,
      `${name}: raw bytes differ, so the offsets are wrong`,
    );

    const wide = await st.readF32(name);
    assert.equal(wide.length, info.numel, `${name} element count`);
    assert.equal(
      createHash('sha256').update(new Uint8Array(wide.buffer, wide.byteOffset, wide.byteLength))
        .digest('hex'),
      want.f32_sha256,
      `${name}: widened f32 differs from the NumPy widening`,
    );
  }

  await st.close();
});
