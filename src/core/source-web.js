// A ByteSource backed by HTTP range requests.
//
// The browser counterpart of FileSource. Same interface, so the safetensors
// reader, the config parser and the tokenizer are shared verbatim between Node
// and the browser rather than reimplemented -- which also means the per-layer
// GPU comparison runs against exactly the weights the CPU reference used, not
// against a second copy that might have been loaded differently.

export class FetchSource {
  #url;
  #size;

  constructor(url, size) {
    this.#url = url;
    this.#size = size;
  }

  /**
   * Learn the resource's length, and refuse to continue if the server will not
   * serve ranges -- silently downloading 988 MB is worse than failing.
   */
  static async open(url) {
    const head = await fetch(url, { method: 'HEAD' });
    if (!head.ok) throw new Error(`HEAD ${url}: ${head.status} ${head.statusText}`);

    const length = head.headers.get('content-length');
    if (length === null) throw new Error(`${url}: no content-length`);
    if ((head.headers.get('accept-ranges') ?? '').toLowerCase() !== 'bytes') {
      throw new Error(`${url}: server does not advertise byte ranges`);
    }
    return new FetchSource(url, Number(length));
  }

  async size() { return this.#size; }

  async read(offset, length) {
    if (offset < 0 || length < 0 || offset + length > this.#size) {
      throw new RangeError(`read ${offset}+${length} outside a ${this.#size}-byte resource`);
    }
    if (length === 0) return new Uint8Array(0);

    const end = offset + length - 1;
    const response = await fetch(this.#url, { headers: { Range: `bytes=${offset}-${end}` } });
    if (response.status !== 206) {
      // A 200 here means the server ignored the range and is sending the whole
      // file. Reading past what was asked for would appear to work while
      // quietly transferring the entire model, so treat it as an error.
      throw new Error(
        `${this.#url}: expected 206 for bytes=${offset}-${end}, got ${response.status}`,
      );
    }
    const bytes = new Uint8Array(await response.arrayBuffer());
    if (bytes.byteLength !== length) {
      throw new Error(
        `${this.#url}: asked for ${length} bytes at ${offset}, received ${bytes.byteLength}`,
      );
    }
    return bytes;
  }
}

/**
 * A ByteSource over an ArrayBuffer already in memory.
 *
 * Used when the whole file has been fetched once and is being sliced locally,
 * which is faster than a request per tensor when memory allows it.
 */
export class MemorySource {
  #bytes;
  constructor(bytes) { this.#bytes = new Uint8Array(bytes); }
  async size() { return this.#bytes.byteLength; }
  async read(offset, length) {
    if (offset < 0 || length < 0 || offset + length > this.#bytes.byteLength) {
      throw new RangeError(`read ${offset}+${length} outside ${this.#bytes.byteLength} bytes`);
    }
    return this.#bytes.subarray(offset, offset + length);
  }
}
