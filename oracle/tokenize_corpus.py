"""Tokenize test/corpus.json with the reference implementation.

The other half of the Phase 0 gate. `tokenizers` is the Rust library Qwen's
tokenizer.json was written for, so it is the definition of correct here, and it
shares no code with src/core/tokenizer.js. Agreement on every case means the
hand-translated split pattern, the byte-level alphabet, the merge ranking and
the added-token handling are all right -- and disagreement says exactly which
case to look at.
"""
import io
import json
import pathlib

from tokenizers import Tokenizer

ROOT = pathlib.Path(__file__).resolve().parent.parent
CORPUS = ROOT / "test" / "corpus.json"
OUT = ROOT / "golden" / "tokenizer_cases.json"
TOKENIZER = ROOT / "weights" / "tokenizer.json"


def main():
    tok = Tokenizer.from_file(str(TOKENIZER))
    corpus = json.loads(io.open(CORPUS, encoding="utf-8").read())

    cases = []
    round_trip_failures = []
    for text in corpus:
        ids = tok.encode(text, add_special_tokens=False).ids
        decoded = tok.decode(ids, skip_special_tokens=False)
        cases.append({"text": text, "ids": ids, "decoded": decoded})
        if decoded != text:
            round_trip_failures.append(text)

    doc = {
        "source": "Qwen/Qwen2.5-0.5B-Instruct tokenizer.json",
        "produced_by": "oracle/tokenize_corpus.py (huggingface tokenizers, Rust)",
        "tokenizers_version": _version(),
        "case_count": len(cases),
        "token_count": sum(len(c["ids"]) for c in cases),
        # Recorded rather than asserted. Byte-level BPE is not required to
        # round-trip text that normalisation rewrites, so where the reference
        # does not round-trip, ours is not expected to either -- but it must
        # fail on exactly the same cases.
        "round_trip_failures": len(round_trip_failures),
        "cases": cases,
    }

    OUT.parent.mkdir(exist_ok=True)
    with io.open(OUT, "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, ensure_ascii=True, indent=1)
        f.write("\n")

    print("%d cases, %s tokens -> %s"
          % (len(cases), format(doc["token_count"], ","), OUT.relative_to(ROOT)))
    print("cases the reference itself does not round-trip: %d" % len(round_trip_failures))
    for text in round_trip_failures[:5]:
        print("   %r" % (text[:60],))


def _version():
    try:
        import tokenizers
        return tokenizers.__version__
    except Exception:
        return "unknown"


if __name__ == "__main__":
    main()
