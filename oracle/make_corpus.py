"""Generate test/corpus.json, the text the tokenizer is checked against.

Deterministic and committed, so the golden ids in golden/tokenizer_cases.json
stay meaningful. Non-ASCII text is written as escapes rather than literals so
the file survives tools with opinions about encodings.

The curated cases target the places a byte-level BPE tokenizer actually goes
wrong, rather than being a large pile of English:

  * NFC normalisation, where the same text has two encodings
  * the contraction group, which the reference spells with an inline
    case-insensitive flag that JavaScript has no syntax for
  * whitespace runs, which the split pattern handles with three interacting
    alternatives
  * multi-byte characters, which BPE routinely splits mid-character
  * the 22 added tokens, which are matched before normalisation
  * characters where Rust and JavaScript disagree about what counts as
    whitespace, which is a real portability hazard rather than a hypothetical

Those are then backed by seeded random strings drawn across scripts, which is
where two BPE implementations really diverge: the merge order on a piece nobody
thought to write down by hand.
"""
import io
import json
import pathlib
import random

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "test" / "corpus.json"

PROSE = [
    "The capital of France is Paris.",
    "Hello, world!",
    "In 1994, 3.14159 apples cost $12.50 (a 27% increase).",
    "She said: “it's fine” — and it wasn't.",
    "The quick brown fox jumps over the lazy dog.",
    "a" * 200,
    "supercalifragilisticexpialidocious antidisestablishmentarianism",
]

# Every contraction the split pattern special-cases, in several casings, since
# the original uses an inline (?i:...) group that had to be rewritten by hand.
CONTRACTIONS = []
for _base in ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"]:
    for _variant in [_base, _base.upper(), _base[0] + _base[1].upper() + _base[2:]]:
        CONTRACTIONS.append("it" + _variant + " here")
        CONTRACTIONS.append("don" + _variant)

WHITESPACE = [
    "  leading",
    "trailing   ",
    "a  b   c    d",
    "line\nbreak",
    "crlf\r\nending",
    "tabs\tand\t\tmore",
    "\n\n\n",
    "   \n   ",
    "\t \t \n\r\n \t",
    "word \n word",
    " ",
    "",
]

# NFC has real work to do here: these are the same text composed and decomposed,
# and the tokenizer must produce identical ids for both spellings.
NORMALISATION = [
    "café",                    # composed e-acute
    "café",                   # e + combining acute
    "Ångström",           # composed
    "Ångström",         # decomposed
    "ẛ̣",                 # an edge case from the Unicode annex
    "ＡＢＣ",           # fullwidth latin, which NFC leaves alone
]

MULTIBYTE = [
    "你好世界",                          # Chinese
    "こんにちは",                    # Japanese
    "안녕하세요",                    # Korean
    "مرحبا بالعالم",
    "Привет",              # Cyrillic
    "שלום",                          # Hebrew
    "\U0001f600\U0001f680\U0001f9ea",                    # emoji, 4 UTF-8 bytes each
    "\U0001f469‍\U0001f4bb",                        # ZWJ sequence
    "\U0001f1ef\U0001f1f5",                              # regional indicator pair
    "नमस्ते",              # Devanagari with a virama
    "\U0001d54a\U0001d556\U0001d55b",                    # astral-plane maths letters
]

CODE = [
    "def f(x):\n    return x ** 2  # square\n",
    "const x = {a: 1, b: [2, 3]};",
    "SELECT * FROM t WHERE id = 42;",
    "<div class=\"x\">&amp;</div>",
    "https://example.com/a?b=c&d=e#f",
    "/* */ // ### --- === !== <=> |> :: ->",
    "\\n\\t\\\\ \\u0041",
]

# Added tokens are matched against the raw text before normalisation, so they
# get their own cases, including ones that abut ordinary text and near-misses
# that must NOT be treated as special.
SPECIALS = [
    "<|im_start|>",
    "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n",
    "<|endoftext|>",
    "text<|im_end|>text",
    "<|im_start|><|im_end|>",
    "<|fim_prefix|>a<|fim_suffix|>b<|fim_middle|>",
    "<|repo_name|>x<|file_sep|>y",
    "not a real <|token|> here",
    "<|im_start",
    "|im_end|>",
]

# Characters JavaScript and Rust classify differently. U+FEFF is the one that
# matters: the JavaScript specification counts it as whitespace, the Unicode
# White_Space property does not, and the split pattern leans on \s.
DISPUTED = [
    "﻿leading byte order mark",
    "a﻿b",
    "  line separator",
    "  paragraph separator",
    "᠎ mongolian vowel separator",
    "​ zero width space",
    "　 ideographic space",
    " next line",
    " vertical tab and form feed",
    "  no-break space",
]

DIGITS = [
    "0123456789",
    "1 2 3",
    "3.14 1e-9 0x1F 1_000_000",
    "١٢٣",           # Arabic-Indic digits
    "ⅠⅡⅢ",           # Roman numerals: \p{N} but not digits
    "½ ⅓",                # vulgar fractions
]

EDGE = [
    "\x00",
    "\x00\x01\x02",
    "\x7f",
    "a\x00b",
    "ÿþ",
    "?" * 500,
    "the " * 300,
]

RANDOM_ALPHABETS = [
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
    " \t\n\r",
    "0123456789",
    "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~",
    "àéîõüçñßåøæ",
    "你好世界漢字東京",
    "こんにちはカタカナ",
    "абвгдеёжз",
    "العربية",
    "\U0001f600\U0001f680\U0001f9ea\U0001f469‍\U0001f4bb❤️",
    "̧́̈̃",     # combining marks, for NFC to work on
    "﻿  　",     # the disputed-whitespace set
]


def random_cases(n, seed=20260902):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        # Mix two or three alphabets per case, so pieces straddle script
        # boundaries the way real text does.
        alphabets = rng.sample(RANDOM_ALPHABETS, rng.randint(2, 3))
        pool = "".join(alphabets)
        out.append("".join(rng.choice(pool) for _ in range(rng.randint(1, 60))))
    return out


def main():
    corpus = []
    seen = set()
    for group in (PROSE, CONTRACTIONS, WHITESPACE, NORMALISATION, MULTIBYTE,
                  CODE, SPECIALS, DISPUTED, DIGITS, EDGE, random_cases(3000)):
        for case in group:
            if case not in seen:
                seen.add(case)
                corpus.append(case)

    OUT.parent.mkdir(exist_ok=True)
    with io.open(OUT, "w", encoding="utf-8", newline="\n") as f:
        json.dump(corpus, f, ensure_ascii=True, indent=1)
        f.write("\n")

    total = sum(len(c.encode("utf-8")) for c in corpus)
    distinct = len({b for c in corpus for b in c.encode("utf-8")})
    print("%d cases, %s bytes, %d distinct byte values -> %s"
          % (len(corpus), format(total, ","), distinct, OUT.relative_to(ROOT)))


if __name__ == "__main__":
    main()
