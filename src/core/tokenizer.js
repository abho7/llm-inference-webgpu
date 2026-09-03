// A byte-level BPE tokenizer, built from tokenizer.json.
//
// Four stages, in the order tokenizer.json declares them:
//
//   1. added tokens are cut out of the raw text first, because they are marked
//      normalized:false and so must be matched before anything rewrites them
//   2. NFC normalisation
//   3. pre-tokenisation: a GPT-4-style regex split, then the GPT-2 byte-level
//      alphabet, which maps each UTF-8 byte to a printable code point so that
//      BPE only ever sees characters
//   4. BPE merges, ranked
//
// Stage 2 delegates to String.prototype.normalize. NFC is a large Unicode table
// rather than an algorithm, and the reference tokenizer delegates it too (to
// Rust's unicode-normalization); reimplementing the table would be copying
// data, not writing code. Everything else here is ours.

/**
 * The GPT-2 byte-level alphabet.
 *
 * BPE needs to work on characters, but the input is bytes, and most byte values
 * are not printable characters. So the 188 bytes that already are printable map
 * to themselves, and the other 68 are lifted into an unused block starting at
 * U+0100. That is why a leading space appears as U+0120 in the vocabulary.
 */
export function byteLevelAlphabet() {
  const printable = new Set();
  for (let b = 0x21; b <= 0x7e; b++) printable.add(b); // '!' through '~'
  for (let b = 0xa1; b <= 0xac; b++) printable.add(b);
  for (let b = 0xae; b <= 0xff; b++) printable.add(b);

  const byteToChar = new Array(256);
  const charToByte = new Map();
  let next = 0;
  for (let b = 0; b < 256; b++) {
    const code = printable.has(b) ? b : 0x100 + next++;
    const ch = String.fromCharCode(code);
    byteToChar[b] = ch;
    charToByte.set(ch, b);
  }
  return { byteToChar, charToByte };
}

const { byteToChar: BYTE_TO_CHAR, charToByte: CHAR_TO_BYTE } = byteLevelAlphabet();

/**
 * The pre-tokenisation split, translated from the pattern in tokenizer.json.
 *
 * Two things had to change, and both are places where a faithful-looking
 * translation would be wrong:
 *
 * The original opens with an inline case-insensitive group, (?i:'s|'t|...),
 * which JavaScript has no syntax for, so the seven contractions are written out
 * in both cases instead.
 *
 * The original's \s is Rust's, which means \p{White_Space}. JavaScript's \s is
 * a different set: it also matches U+FEFF, the byte order mark. Writing \s here
 * would silently retokenize any text containing a BOM, which the corpus test
 * caught on 16 cases. Spelling the property out keeps the two in step.
 */
const NOT_SPACE = '\\P{White_Space}';
const SPACE = '\\p{White_Space}';
const SPLIT_PATTERN = [
  "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])",
  '[^\\r\\n\\p{L}\\p{N}]?\\p{L}+',
  '\\p{N}',
  ` ?[^${SPACE}\\p{L}\\p{N}]+[\\r\\n]*`,
  `${SPACE}*[\\r\\n]+`,
  `${SPACE}+(?!${NOT_SPACE})`,
  `${SPACE}+`,
].join('|');

const escapeRegExp = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

const UTF8_ENCODER = new TextEncoder();
// ignoreBOM keeps a leading U+FEFF in the output. Without it TextDecoder
// silently swallows the byte order mark, so decoding text that legitimately
// begins with one returns something shorter than what was encoded.
const UTF8_DECODER = new TextDecoder('utf-8', { ignoreBOM: true });

export class Tokenizer {
  #vocab;          // Map<token string, id>
  #idToToken;      // token string indexed by id
  #ranks;          // Map<"left right", merge rank>
  #cache;          // Map<byte-level piece, id[]>
  #splitRe;
  #addedRe;        // null when the model declares no added tokens
  #added;          // Map<content, {id, special}>
  #specialIds;

  constructor({ vocab, merges, addedTokens = [], normalizer = 'NFC' }) {
    this.#vocab = vocab;
    this.#ranks = merges;
    this.#cache = new Map();
    this.normalizer = normalizer;
    this.#splitRe = new RegExp(SPLIT_PATTERN, 'gu');

    this.#added = new Map();
    this.#specialIds = new Set();
    let maxId = -1;
    for (const id of vocab.values()) maxId = Math.max(maxId, id);
    for (const t of addedTokens) {
      this.#added.set(t.content, { id: t.id, special: t.special === true });
      if (t.special === true) this.#specialIds.add(t.id);
      maxId = Math.max(maxId, t.id);
    }

    this.#idToToken = new Array(maxId + 1).fill(null);
    for (const [token, id] of vocab) this.#idToToken[id] = token;
    for (const t of addedTokens) this.#idToToken[t.id] = t.content;

    // Longest content first, so a token that is a prefix of another cannot win
    // the match. This is the leftmost-longest rule the reference implementation
    // gets from Aho-Corasick.
    const contents = [...this.#added.keys()].sort((a, b) => b.length - a.length);
    this.#addedRe = contents.length
      ? new RegExp(contents.map(escapeRegExp).join('|'), 'g')
      : null;
  }

  static fromJSON(text) {
    const spec = JSON.parse(text);
    const model = spec.model ?? {};
    if (model.type !== 'BPE') {
      throw new Error(`tokenizer: only BPE is implemented, got ${JSON.stringify(model.type)}`);
    }
    // Each of these would change the output, so refuse rather than ignore.
    if (model.dropout) throw new Error('tokenizer: BPE dropout is not implemented');
    if (model.byte_fallback) throw new Error('tokenizer: byte_fallback is not implemented');
    if (model.continuing_subword_prefix) {
      throw new Error('tokenizer: continuing_subword_prefix is not implemented');
    }
    if (model.end_of_word_suffix) {
      throw new Error('tokenizer: end_of_word_suffix is not implemented');
    }
    if (model.ignore_merges) throw new Error('tokenizer: ignore_merges is not implemented');

    const normType = spec.normalizer ? spec.normalizer.type : null;
    if (normType !== null && normType !== 'NFC') {
      throw new Error(`tokenizer: only the NFC normaliser is implemented, got ${normType}`);
    }

    const vocab = new Map(Object.entries(model.vocab));

    const ranks = new Map();
    model.merges.forEach((merge, rank) => {
      // Merges are stored as "left right". The byte-level alphabet maps the
      // space byte to a printable character, so a real space here is always the
      // separator and never part of either side.
      const pair = Array.isArray(merge) ? merge : merge.split(' ');
      if (pair.length !== 2) {
        throw new Error(`tokenizer: malformed merge ${JSON.stringify(merge)}`);
      }
      ranks.set(`${pair[0]} ${pair[1]}`, rank);
    });

    return new Tokenizer({
      vocab,
      merges: ranks,
      addedTokens: spec.added_tokens ?? [],
      normalizer: normType,
    });
  }

  get vocabSize() { return this.#idToToken.length; }
  isSpecial(id) { return this.#specialIds.has(id); }
  idToToken(id) { return this.#idToToken[id] ?? null; }

  tokenToId(token) {
    const fromVocab = this.#vocab.get(token);
    if (fromVocab !== undefined) return fromVocab;
    const fromAdded = this.#added.get(token);
    return fromAdded ? fromAdded.id : null;
  }

  /** Encode text to token ids. */
  encode(text, { addedTokens = true } = {}) {
    const out = [];
    if (!addedTokens || !this.#addedRe) {
      this.#encodePlain(text, out);
      return out;
    }
    // Added tokens are matched against the raw text: they are all declared
    // normalized:false, so normalising first could destroy what we are
    // looking for.
    this.#addedRe.lastIndex = 0;
    let cursor = 0;
    for (let m = this.#addedRe.exec(text); m !== null; m = this.#addedRe.exec(text)) {
      if (m.index > cursor) this.#encodePlain(text.slice(cursor, m.index), out);
      out.push(this.#added.get(m[0]).id);
      cursor = m.index + m[0].length;
    }
    if (cursor < text.length) this.#encodePlain(text.slice(cursor), out);
    return out;
  }

  #encodePlain(text, out) {
    if (text.length === 0) return;
    const normalized = this.normalizer === 'NFC' ? text.normalize('NFC') : text;

    // The split is declared with behavior "Isolated", meaning the matches
    // themselves are the pieces. The pattern covers essentially every
    // character, but any gap between matches is still a piece and is kept
    // rather than dropped.
    this.#splitRe.lastIndex = 0;
    let cursor = 0;
    for (let m = this.#splitRe.exec(normalized); m !== null; m = this.#splitRe.exec(normalized)) {
      if (m.index > cursor) this.#emit(normalized.slice(cursor, m.index), out);
      this.#emit(m[0], out);
      cursor = m.index + m[0].length;
      if (m[0].length === 0) this.#splitRe.lastIndex++; // never happens, but do not spin
    }
    if (cursor < normalized.length) this.#emit(normalized.slice(cursor), out);
  }

  #emit(piece, out) {
    if (piece.length === 0) return;
    const bytes = UTF8_ENCODER.encode(piece);
    let encoded = '';
    for (let i = 0; i < bytes.length; i++) encoded += BYTE_TO_CHAR[bytes[i]];
    const ids = this.#bpe(encoded);
    for (let i = 0; i < ids.length; i++) out.push(ids[i]);
  }

  /**
   * Merge one byte-level piece into token ids.
   *
   * Repeatedly finds the lowest-ranked adjacent pair and merges every
   * occurrence of it. Every single byte-level character is in the vocabulary by
   * construction, so this cannot produce an unknown token.
   */
  #bpe(word) {
    const hit = this.#cache.get(word);
    if (hit !== undefined) return hit;

    let parts = Array.from(word);
    while (parts.length > 1) {
      let bestRank = Infinity;
      let bestAt = -1;
      for (let i = 0; i < parts.length - 1; i++) {
        const rank = this.#ranks.get(`${parts[i]} ${parts[i + 1]}`);
        if (rank !== undefined && rank < bestRank) {
          bestRank = rank;
          bestAt = i;
        }
      }
      if (bestAt < 0) break;

      const left = parts[bestAt];
      const right = parts[bestAt + 1];
      const merged = [];
      for (let i = 0; i < parts.length;) {
        if (i < parts.length - 1 && parts[i] === left && parts[i + 1] === right) {
          merged.push(left + right);
          i += 2;
        } else {
          merged.push(parts[i]);
          i += 1;
        }
      }
      parts = merged;
    }

    const ids = parts.map((token) => {
      const id = this.#vocab.get(token);
      if (id === undefined) {
        throw new Error(`tokenizer: ${JSON.stringify(token)} is not in the vocabulary`);
      }
      return id;
    });
    // The cache is unbounded on purpose: pre-tokens are short and the set of
    // distinct ones in any real workload is small.
    this.#cache.set(word, ids);
    return ids;
  }

  /**
   * Decode ids back to text.
   *
   * Byte-level decoding has to happen over the whole sequence at once rather
   * than per token: a multi-byte character is routinely split across two
   * tokens, and decoding each one separately would turn it into replacement
   * characters.
   */
  decode(ids, { skipSpecialTokens = false } = {}) {
    const bytes = [];
    for (const id of ids) {
      if (skipSpecialTokens && this.#specialIds.has(id)) continue;
      const token = this.#idToToken[id];
      if (token === undefined || token === null) {
        throw new Error(`tokenizer: id ${id} is not in the vocabulary`);
      }
      for (const ch of token) {
        const b = CHAR_TO_BYTE.get(ch);
        if (b === undefined) {
          // Added tokens are stored as literal text rather than byte-level, so
          // anything outside the alphabet passes through as UTF-8.
          for (const raw of UTF8_ENCODER.encode(ch)) bytes.push(raw);
        } else {
          bytes.push(b);
        }
      }
    }
    return UTF8_DECODER.decode(new Uint8Array(bytes));
  }
}
