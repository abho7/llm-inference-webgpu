"""Download the Qwen2.5-0.5B-Instruct weights and tokenizer into weights/.

These are the only files this project needs from Hugging Face. Everything that
reads them -- the safetensors parser, the bf16 conversion, the tokenizer -- is
written from scratch in src/core.
"""
import sys, pathlib, shutil
from huggingface_hub import hf_hub_download

REPO = "Qwen/Qwen2.5-0.5B-Instruct"
FILES = ["model.safetensors", "tokenizer.json", "tokenizer_config.json",
         "config.json", "generation_config.json", "vocab.json", "merges.txt"]

dest = pathlib.Path(__file__).resolve().parent.parent / "weights"
dest.mkdir(exist_ok=True)

for name in FILES:
    try:
        p = hf_hub_download(REPO, name)
    except Exception as e:
        print(f"skip {name}: {type(e).__name__}", flush=True)
        continue
    out = dest / name
    shutil.copyfile(p, out)
    print(f"{name:26s} {out.stat().st_size/1e6:9.1f} MB", flush=True)
