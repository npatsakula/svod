"""Generate the ModernBERT token-classification golden fixture for the parity tests.

Produces `golden_token.safetensors` under the ModernBERT data dir
(`SVOD_MODERNBERT` env var or `../data/modernbert` relative to this crate) with:

  input_ids        (B, L)          int64    tokenized sentences (batch = 2)
  attention_mask   (B, L)          int64    1 = real token, 0 = pad
  expected_logits  (B, L, n_lab)   float32  per-token logits from ModernBertForTokenClassification

The logits feed `token_logits_match_pytorch` (token-classification parity);
produced by HuggingFace `transformers` so Svod's f32 output is compared against
an independent reference. Padding is load-bearing: the mask keeps pad tokens out
of real-token representations.

Usage:
    uv run scripts/generate_modernbert_token_golden.py
    SVOD_MODERNBERT=/path uv run scripts/generate_modernbert_token_golden.py
    uv run scripts/generate_modernbert_token_golden.py --repo sanketrai/modernbert-base-conll2003-english-ner --max-seq 64

The `*_bf16.cpu_{sdpa,eager}` tensors are the same outputs from the checkpoint
loaded in bf16 along two equivalent attention paths (upcast to f32): PyTorch's
own bf16 drift from the f32 golden, the yardstick the bf16 head test bounds
Svod's by.
"""

import argparse
import os
from pathlib import Path

import torch
import safetensors.torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

# Two sentences of different lengths so padding is load-bearing and the batch
# dimension is exercised; NER-rich content (people, places, orgs) so the
# prediction surface is non-degenerate.
SAMPLE_SENTENCES = [
    "Barack Obama was born in Hawaii and later served as President of the United States.",
    "Apple Inc. is headquartered in Cupertino, California, near San Francisco.",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default="sanketrai/modernbert-base-conll2003-english-ner")
    parser.add_argument("--max-seq", type=int, default=64, help="pad/truncate target length (keeps the mask load-bearing)")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir or os.environ.get("SVOD_MODERNBERT", Path(__file__).resolve().parent.parent / "../data/modernbert"))
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.repo)
    # Force f32: the parity tests run Svod in f32, so the golden must too.
    # Without this a checkpoint whose config defaults to bfloat16 (this NER
    # fine-tune is saved in bf16) would load/run in bf16 and produce
    # bf16-rounded logits that Svod's f32 output can never match (~0.1 drift).
    model = AutoModelForTokenClassification.from_pretrained(args.repo, torch_dtype=torch.float32).to(device).eval()

    enc = tokenizer(
        SAMPLE_SENTENCES,
        padding="max_length",
        max_length=args.max_seq,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits  # (B, L, num_labels) float32

    num_labels = logits.shape[-1]
    golden = {
        "input_ids": input_ids.to(torch.int64).cpu(),
        "attention_mask": attention_mask.to(torch.int64).cpu(),
        "expected_logits": logits.to(torch.float32).cpu(),
    }
    for impl in ("sdpa", "eager"):
        bf16 = AutoModelForTokenClassification.from_pretrained(args.repo, torch_dtype=torch.bfloat16, attn_implementation=impl).eval()
        with torch.no_grad():
            bf16_logits = bf16(input_ids=input_ids, attention_mask=attention_mask).logits
        golden[f"expected_logits_bf16.cpu_{impl}"] = bf16_logits.to(torch.float32).cpu()
    golden_path = out_dir / "golden_token.safetensors"
    safetensors.torch.save_file({k: v.contiguous() for k, v in golden.items()}, str(golden_path))
    real = attention_mask.sum(dim=-1).tolist()
    print(f"Wrote {golden_path} ({len(golden)} tensors)")
    print(f"  batch={len(SAMPLE_SENTENCES)} seq_len={args.max_seq} num_labels={num_labels}")
    label_map = getattr(model.config, "id2label", {})
    for i, (s, r) in enumerate(zip(SAMPLE_SENTENCES, real)):
        print(f"  [{i}] real_tokens={r} pad_tokens={args.max_seq - r}  \"{s[:60]}...\"")
        preds = logits[i, :r].argmax(dim=-1).tolist()
        names = [label_map.get(p, p) for p in preds]
        print(f"      first-token-labels={names[:8]}{'...' if len(names) > 8 else ''}")
    print("Note: logits are NOT softmaxed — the parity test compares raw logits at real-token positions.")


if __name__ == "__main__":
    main()
