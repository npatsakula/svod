"""Generate the BGE goldens for the XLM-RoBERTa encoder-head parity tests.

Writes two fixtures under the BGE-M3 data dir (`SVOD_BGEM3` env var or
`../data/bgem3` relative to this crate):

`golden_dense.safetensors` — `BAAI/bge-m3`'s dense vectors:

  input_ids           (B, L)  int64    the texts, padded to --max-seq
  attention_mask      (B, L)  int64    1 = real token, 0 = pad
  texts_json          (n,)    uint8    the texts, as UTF-8 JSON
  expected_embedding  (B, D)  float32  the `<s>` state, L2-normalized — the
                                       checkpoint's sentence-transformers
                                       pooling (CLS, then Normalize)

`golden_reranker.safetensors` — `BAAI/bge-reranker-v2-m3`'s scores:

  input_ids        (B, L)  int64    the (query, passage) pairs, padded to
                                    --max-seq, one of them truncated
  attention_mask   (B, L)  int64
  pairs_json       (n,)    uint8    the pairs, as UTF-8 JSON
  expected_logits  (B, 1)  float32  raw logits (no sigmoid)

Pairs are encoded as the model card does, `tokenizer(pairs, truncation=True,
max_length=...)`: LongestFirst, which cuts the longer text first. FlagEmbedding's
`FlagReranker` caps the query and then cuts the passage only; that path is not
reproduced here.

The `*_bf16.{cpu,cuda}_{sdpa,eager}` tensors are the same outputs from the
checkpoint loaded in bf16 along four equivalent paths, SDPA and eager attention
on the CPU and on CUDA (upcast to f32): PyTorch's own bf16 drift from the f32
golden, the yardstick the bf16 head tests bound Svod's by. The CUDA paths need a
CUDA device.

Usage:
    uv run --project model/scripts model/scripts/generate_bge_golden.py
    SVOD_BGEM3=/path uv run --project model/scripts model/scripts/generate_bge_golden.py
"""

import argparse
import json
import os
from pathlib import Path

import safetensors.torch
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

# Revisions the parity tests pin.
DENSE_REPO = ("BAAI/bge-m3", "5617a9f61b028005a4858fdac845db406aefb181")
RERANKER_REPO = ("BAAI/bge-reranker-v2-m3", "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e")

# Different lengths, so padding is load-bearing, and more than one script.
TEXTS = [
    "What is BGE M3?",
    "BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.",
    "Привет, мир! Это проверка многоязычных эмбеддингов.",
]

PANDA = (
    "The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, "
    "is a bear species endemic to China."
)
# The last passage runs past --max-seq, so LongestFirst cuts it.
PAIRS = [
    ("what is panda?", "hi"),
    ("what is panda?", PANDA),
    ("what is panda?", " ".join([PANDA] * 4)),
]


def bf16_devices() -> tuple[str, ...]:
    """Where the bf16 yardstick runs: the CPU, and CUDA, whose paths the bf16
    head tests require — without a CUDA device the goldens still serve the f32
    parity tests."""
    if torch.cuda.is_available():
        return ("cpu", "cuda")
    print("warning: no CUDA device; the bf16 head tests will refuse these goldens' CPU-only bf16 paths")
    return ("cpu",)


def utf8_json(value) -> torch.Tensor:
    return torch.tensor(list(json.dumps(value, ensure_ascii=False).encode("utf-8")), dtype=torch.uint8)


def dense(model, input_ids, attention_mask) -> torch.Tensor:
    hidden = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    return F.normalize(hidden[:, 0].to(torch.float32), dim=-1)


def logits(model, input_ids, attention_mask) -> torch.Tensor:
    return model(input_ids=input_ids, attention_mask=attention_mask).logits.to(torch.float32)


def golden(auto, run, repo: tuple[str, str], encoded, key: str) -> dict[str, torch.Tensor]:
    """`run`'s f32 output under `key`, and its four bf16 paths."""
    name, revision = repo
    input_ids, attention_mask = encoded["input_ids"], encoded["attention_mask"]
    model = auto.from_pretrained(name, revision=revision, dtype=torch.float32).eval()
    with torch.no_grad():
        out = {key: run(model, input_ids, attention_mask)}
    for device in bf16_devices():
        for impl in ("sdpa", "eager"):
            bf16 = auto.from_pretrained(name, revision=revision, dtype=torch.bfloat16, attn_implementation=impl)
            bf16 = bf16.eval().to(device)
            with torch.no_grad():
                out[f"{key}_bf16.{device}_{impl}"] = run(bf16, input_ids.to(device), attention_mask.to(device)).cpu()
    out["input_ids"] = input_ids.to(torch.int64)
    out["attention_mask"] = attention_mask.to(torch.int64)
    return out


def save(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    safetensors.torch.save_file({k: v.contiguous() for k, v in tensors.items()}, str(path))
    print(f"Wrote {path} ({len(tensors)} tensors)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-seq", type=int, default=64, help="pad/truncate target length")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir or os.environ.get("SVOD_BGEM3", Path(__file__).resolve().parent.parent / "../data/bgem3"))
    out_dir.mkdir(parents=True, exist_ok=True)
    pad = {"padding": "max_length", "max_length": args.max_seq, "truncation": True, "return_tensors": "pt"}

    tokenizer = AutoTokenizer.from_pretrained(DENSE_REPO[0], revision=DENSE_REPO[1])
    encoded = tokenizer(TEXTS, **pad)
    tensors = golden(AutoModel, dense, DENSE_REPO, encoded, "expected_embedding")
    tensors["texts_json"] = utf8_json(TEXTS)
    save(out_dir / "golden_dense.safetensors", tensors)
    print(f"  real tokens per text: {encoded['attention_mask'].sum(dim=-1).tolist()}")

    tokenizer = AutoTokenizer.from_pretrained(RERANKER_REPO[0], revision=RERANKER_REPO[1])
    encoded = tokenizer([q for q, _ in PAIRS], [p for _, p in PAIRS], **pad)
    tensors = golden(AutoModelForSequenceClassification, logits, RERANKER_REPO, encoded, "expected_logits")
    tensors["pairs_json"] = utf8_json(PAIRS)
    save(out_dir / "golden_reranker.safetensors", tensors)
    print(f"  real tokens per pair: {encoded['attention_mask'].sum(dim=-1).tolist()}")
    print(f"  logits: {tensors['expected_logits'].flatten().tolist()}")


if __name__ == "__main__":
    main()
