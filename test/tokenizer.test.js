import test from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import { Tokenizer, byteLevelAlphabet } from '../src/core/tokenizer.js';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const TOKENIZER_JSON = join(ROOT, 'weights', 'tokenizer.json');
const GOLDEN = join(ROOT, 'golden', 'tokenizer_cases.json');

// ---------------------------------------------------------------------------
// The byte-level alphabet, exhaustively. It is a bijection between 256 byte
// values and 256 code points; if it is not, some byte silently becomes another
// byte and the damage shows up as garbled text much later.
// ---------------------------------------------------------------------------

test('the byte-level alphabet is a bijection over all 256 bytes', () => {
  const { byteToChar, charToByte } = byteLevelAlphabet();
  assert.equal(byteToChar.length, 256);
  assert.equal(charToByte.size, 256, 'two bytes must not share a character');
  for (let b = 0; b < 256; b++) {
    const ch = byteToChar[b];
    assert.equal(typeof ch, 'string');
    assert.equal(ch.length, 1, `byte ${b} must map to one UTF-16 code unit`);
    assert.equal(charToByte.get(ch), b, `byte ${b} must map back to itself`);
  }
});

test('the byte-level alphabet uses only printable characters', () => {
  // The whole point of the mapping is that BPE never sees a control character,
  // because the vocabulary is stored as JSON text.
  const { byteToChar } = byteLevelAlphabet();
  for (let b = 0; b < 256; b++) {
    const code = byteToChar[b].charCodeAt(0);
    assert.ok(code > 0x20, `byte ${b} maps to U+${code.toString(16)}, which is not printable`);
    assert.notEqual(code, 0x7f, `byte ${b} maps to DEL`);
  }
});

test('the byte-level alphabet places a space at U+0120, as the vocabulary expects', () => {
  // Not an arbitrary check: leading-space tokens are the single most common
  // shape in the vocabulary, so getting this wrong breaks almost every word.
  const { byteToChar } = byteLevelAlphabet();
  assert.equal(byteToChar[0x20].charCodeAt(0), 0x120);
  assert.equal(byteToChar[0x0a].charCodeAt(0), 0x10a, 'newline');
  assert.equal(byteToChar[0x41], 'A', 'printable ASCII maps to itself');
});

// ---------------------------------------------------------------------------
// Configuration handling. These use hand-built specs, so they need no weights.
// ---------------------------------------------------------------------------

const minimalSpec = (overrides = {}) => JSON.stringify({
  normalizer: { type: 'NFC' },
  added_tokens: [],
  model: {
    type: 'BPE',
    vocab: { a: 0, b: 1, ab: 2 },
    merges: ['a b'],
    ...overrides,
  },
});

test('encodes with a hand-built vocabulary', () => {
  const tok = Tokenizer.fromJSON(minimalSpec());
  assert.deepEqual(tok.encode('ab'), [2], 'the merge should fire');
  assert.deepEqual(tok.encode('ba'), [1, 0], 'no merge in the other order');
});

test('refuses tokenizer features it does not implement, rather than ignoring them', () => {
  const cases = [
    [{ type: 'WordPiece' }, /only BPE is implemented/],
    [{ dropout: 0.1 }, /dropout/],
    [{ byte_fallback: true }, /byte_fallback/],
    [{ continuing_subword_prefix: '##' }, /continuing_subword_prefix/],
    [{ end_of_word_suffix: '</w>' }, /end_of_word_suffix/],
    [{ ignore_merges: true }, /ignore_merges/],
  ];
  for (const [override, match] of cases) {
    assert.throws(() => Tokenizer.fromJSON(minimalSpec(override)), match,
      `should refuse ${JSON.stringify(override)}`);
  }
  assert.throws(
    () => Tokenizer.fromJSON(JSON.stringify({
      normalizer: { type: 'NFKC' },
      model: { type: 'BPE', vocab: {}, merges: [] },
    })),
    /only the NFC normaliser/,
  );
});

test('rejects a malformed merge rather than half-applying it', () => {
  assert.throws(
    () => Tokenizer.fromJSON(minimalSpec({ merges: ['a b c'] })),
    /malformed merge/,
  );
});

// ---------------------------------------------------------------------------
// The Phase 0 gate: exact agreement with the reference Rust tokenizer over the
// whole corpus.
// ---------------------------------------------------------------------------

const haveGolden = existsSync(TOKENIZER_JSON) && existsSync(GOLDEN);
const skip = haveGolden
  ? false
  : 'run oracle/fetch_weights.py, oracle/make_corpus.py and oracle/tokenize_corpus.py first';

test('matches the reference tokenizer on every corpus case', { skip }, async (t) => {
  const tok = Tokenizer.fromJSON(readFileSync(TOKENIZER_JSON, 'utf-8'));
  const golden = JSON.parse(readFileSync(GOLDEN, 'utf-8'));

  const mismatches = [];
  for (const { text, ids } of golden.cases) {
    const got = tok.encode(text);
    if (got.length !== ids.length || got.some((v, i) => v !== ids[i])) {
      mismatches.push({ text, want: ids, got });
    }
  }

  if (mismatches.length) {
    // Report a few concretely: with 3098 cases, "one of them differs" is not
    // an actionable failure message.
    const shown = mismatches.slice(0, 5).map(({ text, want, got }) =>
      `  ${JSON.stringify(text.slice(0, 60))}\n` +
      `    reference: [${want.slice(0, 16)}]\n` +
      `    ours:      [${got.slice(0, 16)}]`).join('\n');
    assert.fail(`${mismatches.length} of ${golden.cases.length} cases differ:\n${shown}`);
  }
  t.diagnostic(`${golden.case_count} cases, ${golden.token_count} tokens, all identical`);
});

test('decodes exactly as the reference decodes, including where it loses text', { skip }, () => {
  // 279 corpus cases do not survive a round trip, because NFC rewrites the
  // text before it is ever tokenized. That is the reference's behaviour, not a
  // defect, so the assertion is that we match its output rather than that we
  // recover the input.
  const tok = Tokenizer.fromJSON(readFileSync(TOKENIZER_JSON, 'utf-8'));
  const golden = JSON.parse(readFileSync(GOLDEN, 'utf-8'));

  let lossy = 0;
  for (const { text, ids, decoded } of golden.cases) {
    assert.equal(tok.decode(ids), decoded, `decode of ${JSON.stringify(text.slice(0, 60))}`);
    if (decoded !== text) lossy++;
  }
  assert.equal(lossy, golden.round_trip_failures, 'lossy-case count should match the reference');
});

test('round-trips every corpus case that is already NFC', { skip }, () => {
  // The complement of the test above: where normalisation has nothing to do,
  // encode followed by decode must be the identity.
  const tok = Tokenizer.fromJSON(readFileSync(TOKENIZER_JSON, 'utf-8'));
  const golden = JSON.parse(readFileSync(GOLDEN, 'utf-8'));

  let checked = 0;
  for (const { text } of golden.cases) {
    if (text.normalize('NFC') !== text) continue;
    assert.equal(tok.decode(tok.encode(text)), text, `round trip of ${JSON.stringify(text.slice(0, 60))}`);
    checked++;
  }
  assert.ok(checked > 2500, `expected most cases to be NFC-stable, only ${checked} were`);
});

test('every id the tokenizer can emit is within the model vocabulary', { skip }, () => {
  // The embedding matrix has 151936 rows but the tokenizer only knows 151665
  // tokens; the gap is padding. An id outside the matrix would read past the
  // end of the embedding table, so this is a real bound, not a formality.
  const tok = Tokenizer.fromJSON(readFileSync(TOKENIZER_JSON, 'utf-8'));
  const golden = JSON.parse(readFileSync(GOLDEN, 'utf-8'));
  const EMBEDDING_ROWS = 151936;

  assert.ok(tok.vocabSize <= EMBEDDING_ROWS,
    `tokenizer vocab ${tok.vocabSize} exceeds ${EMBEDDING_ROWS} embedding rows`);
  for (const { ids } of golden.cases) {
    for (const id of ids) {
      assert.ok(Number.isInteger(id) && id >= 0 && id < EMBEDDING_ROWS, `id ${id} out of range`);
    }
  }
});

test('treats the chat markup as single special tokens', { skip }, () => {
  const tok = Tokenizer.fromJSON(readFileSync(TOKENIZER_JSON, 'utf-8'));
  assert.deepEqual(tok.encode('<|im_start|>'), [151644]);
  assert.deepEqual(tok.encode('<|im_end|>'), [151645]);
  assert.equal(tok.isSpecial(151644), true);
  assert.equal(tok.isSpecial(872), false, 'an ordinary token is not special');

  // A near-miss must not be treated as special.
  assert.ok(tok.encode('<|im_start').length > 1);

  // And skipping specials must drop them from the text, not leave a hole.
  const ids = tok.encode('<|im_start|>user\nHi<|im_end|>');
  assert.equal(tok.decode(ids, { skipSpecialTokens: true }), 'user\nHi');
});

test('encoding with added tokens disabled falls back to plain BPE', { skip }, () => {
  const tok = Tokenizer.fromJSON(readFileSync(TOKENIZER_JSON, 'utf-8'));
  const plain = tok.encode('<|im_start|>', { addedTokens: false });
  assert.ok(plain.length > 1, 'should be split into ordinary tokens');
  assert.equal(tok.decode(plain), '<|im_start|>');
});
