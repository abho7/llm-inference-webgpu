// A safetensors reader.
//
// The container is deliberately simple: a little-endian u64 header length, a
// JSON header mapping tensor names to {dtype, shape, data_offsets}, then the
// raw tensor bytes back to back. Nothing here decompresses or executes; the
// only real work is validating that the header describes the file it came in.
//
// Reads are lazy and go through a ByteSource, because the model is 988 MB and
// this machine has under 2 GB of free physical memory. Materialising the whole
// file and then widening it to f32 would need ~3 GB and page hard, so tensors
// are pulled from disk as the layers that need them run.

import { bf16ToF32Array, f16ToF32Array } from './dtype.js';

/** Bytes per element, for the dtypes safetensors defines. */
const DTYPE_SIZE = {
  BOOL: 1, U8: 1, I8: 1, F8_E4M3: 1, F8_E5M2: 1,
  I16: 2, U16: 2, F16: 2, BF16: 2,
  I32: 4, U32: 4, F32: 4,
  I64: 8, U64: 8, F64: 8,
};

export class TensorInfo {
  constructor(name, dtype, shape, begin, end) {
    this.name = name;
    this.dtype = dtype;
    this.shape = Object.freeze(shape);
    this.begin = begin;   // offset from the start of the data section
    this.end = end;
    Object.freeze(this);
  }
  get numel() { return this.shape.reduce((a, b) => a * b, 1); }
  get byteLength() { return this.end - this.begin; }
}

export class Safetensors {
  #source;
  #tensors;      // Map<string, TensorInfo>
  #dataStart;    // byte offset of the data section within the file

  constructor(source, tensors, dataStart, metadata) {
    this.#source = source;
    this.#tensors = tensors;
    this.#dataStart = dataStart;
    this.metadata = metadata;
  }

  /**
   * Parse the header of `source` and validate it against the file's real size.
   *
   * Every check here corresponds to a way the file can be wrong: a truncated
   * download, a shape that disagrees with its byte range, two tensors claiming
   * the same bytes. Failing loudly at open time is much cheaper than debugging
   * a model that produces plausible-looking garbage.
   */
  static async open(source) {
    const fileSize = await source.size();
    if (fileSize < 8) throw new Error(`not a safetensors file: ${fileSize} bytes`);

    const lenBytes = await source.read(0, 8);
    const headerLen = Number(
      new DataView(lenBytes.buffer, lenBytes.byteOffset, 8).getBigUint64(0, true),
    );
    if (!Number.isSafeInteger(headerLen) || headerLen <= 0 || 8 + headerLen > fileSize) {
      throw new Error(`header length ${headerLen} does not fit in a ${fileSize}-byte file`);
    }

    const headerBytes = await source.read(8, headerLen);
    let header;
    try {
      header = JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(headerBytes));
    } catch (cause) {
      throw new Error(`header is not valid UTF-8 JSON: ${cause.message}`, { cause });
    }
    if (header === null || typeof header !== 'object' || Array.isArray(header)) {
      throw new Error('header must be a JSON object');
    }

    const metadata = header.__metadata__ ?? null;
    delete header.__metadata__;

    const dataStart = 8 + headerLen;
    const dataSize = fileSize - dataStart;
    const tensors = new Map();

    for (const [name, entry] of Object.entries(header)) {
      const dtype = entry?.dtype;
      const shape = entry?.shape;
      const offsets = entry?.data_offsets;
      const elemSize = DTYPE_SIZE[dtype];
      if (elemSize === undefined) throw new Error(`${name}: unknown dtype ${JSON.stringify(dtype)}`);
      if (!Array.isArray(shape) || !shape.every((d) => Number.isSafeInteger(d) && d >= 0)) {
        throw new Error(`${name}: shape must be an array of non-negative integers`);
      }
      if (!Array.isArray(offsets) || offsets.length !== 2) {
        throw new Error(`${name}: data_offsets must be a pair`);
      }
      const begin = offsets[0];
      const end = offsets[1];
      if (!Number.isSafeInteger(begin) || !Number.isSafeInteger(end) || begin < 0 || end < begin) {
        throw new Error(`${name}: bad data_offsets [${begin}, ${end}]`);
      }
      if (end > dataSize) {
        throw new Error(`${name}: ends at ${end} but the data section is only ${dataSize} bytes`);
      }
      const numel = shape.reduce((a, b) => a * b, 1);
      if (end - begin !== numel * elemSize) {
        throw new Error(
          `${name}: shape [${shape}] of ${dtype} needs ${numel * elemSize} bytes ` +
          `but the header reserves ${end - begin}`,
        );
      }
      tensors.set(name, new TensorInfo(name, dtype, shape, begin, end));
    }

    // Overlapping ranges would let one tensor silently corrupt another.
    const ranges = [...tensors.values()].sort((a, b) => a.begin - b.begin);
    for (let i = 1; i < ranges.length; i++) {
      if (ranges[i].begin < ranges[i - 1].end) {
        throw new Error(`${ranges[i].name} overlaps ${ranges[i - 1].name}`);
      }
    }

    return new Safetensors(source, tensors, dataStart, metadata);
  }

  get size() { return this.#tensors.size; }
  names() { return [...this.#tensors.keys()]; }
  has(name) { return this.#tensors.has(name); }

  info(name) {
    const t = this.#tensors.get(name);
    if (!t) throw new Error(`no tensor named ${JSON.stringify(name)}`);
    return t;
  }

  /** The tensor's bytes, exactly as stored. */
  async readRaw(name) {
    const t = this.info(name);
    return this.#source.read(this.#dataStart + t.begin, t.byteLength);
  }

  /**
   * A byte range within one tensor, offset from that tensor's own start.
   *
   * The point of this is the tied output projection: the embedding matrix is
   * 272 MB of bfloat16 and 544 MB widened, so the logit computation walks it in
   * row blocks instead of ever holding it. Bounds are checked against the
   * tensor rather than the file, so a slice cannot silently read a neighbour.
   */
  async readRawRange(name, offset, length) {
    const t = this.info(name);
    if (offset < 0 || length < 0 || offset + length > t.byteLength) {
      throw new RangeError(
        `${name}: range ${offset}+${length} outside a ${t.byteLength}-byte tensor`,
      );
    }
    return this.#source.read(this.#dataStart + t.begin + offset, length);
  }

  /**
   * The tensor widened to f32, whatever it was stored as.
   *
   * Callers can pass `out` to reuse a buffer, which matters in the decode loop
   * where the alternative is allocating megabytes per token.
   */
  async readF32(name, out) {
    const t = this.info(name);
    const bytes = await this.readRaw(name);
    // A source may hand back a view at any offset, and typed arrays other than
    // Uint8Array require natural alignment, so copy when the offset is unlucky.
    const aligned = (bytes.byteOffset % 8 === 0) ? bytes : new Uint8Array(bytes);

    switch (t.dtype) {
      case 'F32': {
        const src = new Float32Array(aligned.buffer, aligned.byteOffset, t.numel);
        if (!out) return src.slice();
        out.set(src);
        return out;
      }
      case 'BF16':
        return bf16ToF32Array(new Uint16Array(aligned.buffer, aligned.byteOffset, t.numel), out);
      case 'F16':
        return f16ToF32Array(new Uint16Array(aligned.buffer, aligned.byteOffset, t.numel), out);
      case 'F64': {
        const src = new Float64Array(aligned.buffer, aligned.byteOffset, t.numel);
        const dst = out ?? new Float32Array(t.numel);
        for (let i = 0; i < t.numel; i++) dst[i] = src[i];
        return dst;
      }
      default:
        throw new Error(`${name}: cannot widen ${t.dtype} to f32`);
    }
  }

  async close() {
    if (this.#source.close) await this.#source.close();
  }
}
