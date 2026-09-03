// A ByteSource backed by a file descriptor.
//
// Deliberately not fs.readFileSync: the point of the interface is that no more
// of the model is resident than the caller has actually asked for.

import { open } from 'node:fs/promises';

export class FileSource {
  #handle;
  #size;

  constructor(handle, size) {
    this.#handle = handle;
    this.#size = size;
  }

  static async open(path) {
    const handle = await open(path, 'r');
    const stat = await handle.stat();
    return new FileSource(handle, stat.size);
  }

  async size() { return this.#size; }

  async read(offset, length) {
    if (offset < 0 || length < 0 || offset + length > this.#size) {
      throw new RangeError(`read ${offset}+${length} outside a ${this.#size}-byte file`);
    }
    const out = new Uint8Array(length);
    let done = 0;
    // One read() is not obliged to return everything asked for.
    while (done < length) {
      const r = await this.#handle.read(out, done, length - done, offset + done);
      if (r.bytesRead === 0) throw new Error(`unexpected EOF at ${offset + done}`);
      done += r.bytesRead;
    }
    return out;
  }

  async close() { await this.#handle.close(); }
}

/** A ByteSource over bytes already in memory. Used by the tests. */
export class BufferSource {
  #bytes;
  constructor(bytes) { this.#bytes = bytes; }
  async size() { return this.#bytes.byteLength; }
  async read(offset, length) {
    if (offset < 0 || length < 0 || offset + length > this.#bytes.byteLength) {
      throw new RangeError(`read ${offset}+${length} outside ${this.#bytes.byteLength} bytes`);
    }
    return this.#bytes.subarray(offset, offset + length);
  }
}
