# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "numpy", "safetensors", "transformers", "huggingface_hub"]
# ///
"""Generate ModernBERT parity fixtures for the Rust `svod-model` port.

Downloads `answerdotai/ModernBERT-base`, runs the backbone (no MLM head) on a
fixed `input_ids` sequence, and dumps a `golden.safetensors` storing:

  - `input_ids`        (T,)  int64 — the exact token ids fed to the model
  - `input_ids_shape`  (2,)  int64 — (batch, seq_len)
  - `last_hidden_state` (B, T, D) f32 — `model(...).last_hidden_state`
  - `mlm_logits`       (B, T, V) f32 — `AutoModelForMaskedLM(...).logits`

The weights themselves are left as-is (`model.safetensors` is fetched by the
Rust test via the same HF repo, so no copy is needed here).

`--long` writes `golden_long_<length>.safetensors` instead: two right-padded rows of
`--long-length` tokens (one full, one short), long enough for the local layers'
sliding window to matter, with the backbone's `last_hidden_state` in f32 and in
bf16 (upcast) — the bf16 run is PyTorch's own drift from f32, the yardstick for
a 16-bit port.

Usage:
  uv run scripts/convert_modernbert.py            # writes ../data/modernbert/golden.safetensors
  uv run scripts/convert_modernbert.py --large    # ModernBERT-large
  uv run scripts/convert_modernbert.py --out path/to/golden.safetensors
  uv run scripts/convert_modernbert.py --long     # writes ../data/modernbert/golden_long_500.safetensors
  uv run scripts/convert_modernbert.py --long --long-length 512

Run the Rust parity test with the local fixture:
  SVOD_MODERNBERT=$PWD/data/modernbert \
      cargo test -p svod-model --lib modernbert::parity -- --ignored
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer

DEFAULT_PROMPT = "The quick brown fox jumps over the lazy dog."
LONG_TEXT = (
    "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of "
    "foolishness, it was the epoch of belief, it was the epoch of incredulity, it was the season of Light, "
    "it was the season of Darkness, it was the spring of hope, it was the winter of despair, we had "
    "everything before us, we had nothing before us, we were all going direct to Heaven, we were all going "
    "direct the other way. "
)
SHORT_TEXT = "Call me Ishmael. Some years ago, never mind how long precisely, having little or no money in my purse."
HUB_BASE = "answerdotai/ModernBERT-base"
HUB_LARGE = "answerdotai/ModernBERT-large"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--large", action="store_true", help="use ModernBERT-large (default: base)")
    p.add_argument("--prompt", default=DEFAULT_PROMPT, help="text to tokenize for the golden forward")
    p.add_argument("--out", type=Path, default=None, help="output golden.safetensors path")
    p.add_argument("--max-length", type=int, default=32, help="tokenizer max_length / padding target")
    p.add_argument("--long", action="store_true", help="write the long two-row f32 + bf16 fixture")
    p.add_argument("--long-length", type=int, default=500, help="the long fixture's padded length")
    args = p.parse_args()

    hub = HUB_LARGE if args.large else HUB_BASE
    name = f"golden_long_{args.long_length}.safetensors" if args.long else "golden.safetensors"
    out = args.out or (Path(__file__).resolve().parent.parent / "data" / "modernbert" / name)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.long:
        long_fixture(hub, args.long_length, out)
        return

    tok = AutoTokenizer.from_pretrained(hub)
    model = AutoModel.from_pretrained(hub, torch_dtype=torch.float32)
    model.eval()

    enc = tok(args.prompt, return_tensors="pt", padding="max_length", max_length=args.max_length, truncation=True)
    input_ids = enc["input_ids"]  # (1, T)
    attn = enc["attention_mask"]  # (1, T)

    with torch.no_grad():
        out_t = model(input_ids=input_ids, attention_mask=attn).last_hidden_state  # (1, T, D)

    # MLM head: `AutoModelForMaskedLM` shares the backbone weights and adds the
    # `head.dense` / `head.norm` / tied-decoder path producing (B, T, V) logits.
    mlm = AutoModelForMaskedLM.from_pretrained(hub, torch_dtype=torch.float32)
    mlm.eval()
    with torch.no_grad():
        logits_t = mlm(input_ids=input_ids, attention_mask=attn).logits  # (1, T, V)

    ids_np = input_ids.squeeze(0).to(torch.int64).numpy()
    attn_np = attn.squeeze(0).to(torch.int64).numpy()  # (T,) 1=real, 0=pad
    hidden_np = out_t.squeeze(0).to(torch.float32).numpy()  # (T, D)
    logits_np = logits_t.squeeze(0).to(torch.float32).numpy()  # (T, V)

    # The Rust test reads input_ids_shape to recover (B, T).
    shape_np = np.array(input_ids.shape, dtype=np.int64)

    save_file(
        {
            "input_ids": ids_np,
            "attention_mask": attn_np,
            "input_ids_shape": shape_np,
            "last_hidden_state": hidden_np,
            "mlm_logits": logits_np,
        },
        str(out),
    )
    print(f"wrote {out}")
    print(
        f"  input_ids {tuple(ids_np.shape)}  attention_mask {tuple(attn_np.shape)}  "
        f"hidden {tuple(hidden_np.shape)}  logits {tuple(logits_np.shape)}  ({hub})"
    )


def long_fixture(hub: str, length: int, out: Path) -> None:
    """Two right-padded rows — one filling `length`, one short — through the
    backbone in f32 and in bf16, both on the CPU."""
    tok = AutoTokenizer.from_pretrained(hub)
    texts = [LONG_TEXT * (length // 60 + 1), SHORT_TEXT]
    enc = tok(texts, return_tensors="pt", padding="max_length", max_length=length, truncation=True)
    hidden = {}
    for dtype in (torch.float32, torch.bfloat16):
        model = AutoModel.from_pretrained(hub, torch_dtype=dtype, attn_implementation="sdpa").eval()
        with torch.no_grad():
            hidden[dtype] = model(**enc).last_hidden_state.to(torch.float32).numpy()
    save_file(
        {
            "input_ids": enc["input_ids"].to(torch.int64).numpy(),
            "attention_mask": enc["attention_mask"].to(torch.int64).numpy(),
            "last_hidden_state": hidden[torch.float32],
            "last_hidden_state_bf16": hidden[torch.bfloat16],
        },
        str(out),
    )
    lens = enc["attention_mask"].sum(-1).tolist()
    print(f"wrote {out}\n  input_ids {tuple(enc['input_ids'].shape)} real tokens per row {lens}  ({hub})")


if __name__ == "__main__":
    main()
