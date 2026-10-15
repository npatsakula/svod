"""Generate the ModernBERT golden fixture for the parity tests.

Produces `golden.safetensors` under the ModernBERT data dir
(`SVOD_MODERNBERT` env var or `../data/modernbert` relative to this crate) with:

  input_ids          (L,)    int64   tokenized sample (batch = 1)
  attention_mask     (L,)    int64   1 = real token, 0 = pad
  input_ids_shape    (2,)    int64   (batch, seq_len)
  last_hidden_state  (L, D)  float32 backbone output (real tokens only matter)
  mlm_logits         (L, V)  float32 MLM head output (vocab logits per token)
  expected_embedding (D,)    float32 masked mean-pool + L2-normalize

The first four feed `last_hidden_state_matches_pytorch` (backbone parity);
`mlm_logits` feeds `mlm_logits_match_pytorch` (MLM-head parity);
`expected_embedding` feeds `embeddings_match_pytorch` (embed-pipeline parity).
All are produced by HuggingFace `transformers` so Svod's f32 output is compared
against an independent reference.

`--long` writes `golden_long_<length>.safetensors` instead: two right-padded rows of
`--long-length` tokens (one full, one short), long enough for the local layers'
sliding window to matter, with the backbone's `last_hidden_state` in f32 (on the
CPU) and, as `last_hidden_state_bf16.<path>` (upcast), in bf16 along six
equivalent paths — SDPA and eager attention on the CPU and on CUDA, and on CUDA
each row also alone. Their drift from f32 is PyTorch's own, and their envelope is
the yardstick for a 16-bit port: they differ by rounding alone, yet on the full
row one moves the drift up to ~1.6× another's. Needs a CUDA device.

Usage:
    uv run scripts/generate_modernbert_golden.py
    SVOD_MODERNBERT=/path uv run scripts/generate_modernbert_golden.py
    uv run scripts/generate_modernbert_golden.py --repo answerdotai/ModernBERT-base --max-seq 32
    uv run scripts/generate_modernbert_golden.py --long                    # golden_long_500.safetensors
    uv run scripts/generate_modernbert_golden.py --long --long-length 512

The `*_bf16.{cpu,cuda}_{sdpa,eager}` tensors are the same embedding from the
checkpoint loaded in bf16 along four equivalent paths, SDPA and eager attention
on the CPU and on CUDA (upcast to f32): PyTorch's own bf16 drift from the f32
golden, the yardstick the bf16 head test bounds Svod's by. The CUDA paths need a
CUDA device.
"""

import argparse
import os
from pathlib import Path

import torch
import safetensors.torch
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer

# Pure masked mean-pool + L2-normalize (the canonical sentence-embedding recipe,
# matching Svod's pool_embed). Svod guards the denominator/norm with EPS=1e-12;
# that is negligible against the 1e-3 parity bound, so we omit it here on
# purpose so the reference stays independent of Svod's exact numerics.

LONG_TEXT = (
    "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of "
    "foolishness, it was the epoch of belief, it was the epoch of incredulity, it was the season of Light, "
    "it was the season of Darkness, it was the spring of hope, it was the winter of despair, we had "
    "everything before us, we had nothing before us, we were all going direct to Heaven, we were all going "
    "direct the other way. "
)
SHORT_TEXT = "Call me Ishmael. Some years ago, never mind how long precisely, having little or no money in my purse."


def bf16_devices() -> tuple[str, ...]:
    """Where the bf16 yardstick runs: the CPU, and CUDA, whose paths the bf16
    head test requires — without a CUDA device the golden still serves the f32
    parity tests."""
    if torch.cuda.is_available():
        return ("cpu", "cuda")
    print("warning: no CUDA device; the bf16 head test will refuse this golden's CPU-only bf16 paths")
    return ("cpu",)


def pool_embed(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Masked mean-pool + L2-normalize: `(B, L, D)` → `(B, D)`."""
    mask_f = attention_mask.unsqueeze(-1).to(hidden.dtype)
    mean = (hidden * mask_f).sum(dim=1) / mask_f.sum(dim=1)
    return mean / torch.linalg.vector_norm(mean, ord=2, dim=-1, keepdim=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default="answerdotai/ModernBERT-base")
    parser.add_argument("--revision", default="8949b909ec900327062f0ebf497f51aef5e6f0c8", help="hub revision; the default is the one the parity tests pin")
    parser.add_argument("--text", default="The quick brown fox jumps over the lazy dog near the quiet riverbank.")
    parser.add_argument("--max-seq", type=int, default=32, help="pad/truncate target length (keeps the mask load-bearing)")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--long", action="store_true", help="write the long two-row f32 + bf16-paths fixture")
    parser.add_argument("--long-length", type=int, default=500, help="the long fixture's padded length")
    args = parser.parse_args()

    out_dir = Path(args.out_dir or os.environ.get("SVOD_MODERNBERT", Path(__file__).resolve().parent.parent / "../data/modernbert"))
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.long:
        long_fixture(args.repo, args.revision, args.long_length, out_dir / f"golden_long_{args.long_length}.safetensors")
        return

    device = "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.repo, revision=args.revision)
    # Force f32: the parity tests run Svod in f32, so the golden must too.
    # Without this a checkpoint whose config defaults to bfloat16 (e.g. a
    # fine-tune saved in bf16) would load/run in bf16 and produce bf16-rounded
    # logits that Svod's f32 output can never match (divergence ~0.1).
    model = AutoModel.from_pretrained(args.repo, revision=args.revision, torch_dtype=torch.float32).to(device).eval()
    # Same f32 rationale as the backbone: the MLM head's matmul/layernorm/GELU
    # accumulate in f32, and Svod's parity test runs in f32 too.
    mlm = AutoModelForMaskedLM.from_pretrained(args.repo, revision=args.revision, torch_dtype=torch.float32).to(device).eval()

    enc = tokenizer(args.text, padding="max_length", max_length=args.max_seq, truncation=True, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = out.last_hidden_state  # (1, L, D) float32
        mlm_logits = mlm(input_ids=input_ids, attention_mask=attention_mask).logits  # (1, L, V) float32

    embedding = pool_embed(last_hidden_state, attention_mask)

    golden = {
        "input_ids": input_ids.squeeze(0).to(torch.int64).cpu(),
        "attention_mask": attention_mask.squeeze(0).to(torch.int64).cpu(),
        "input_ids_shape": torch.tensor([1, args.max_seq], dtype=torch.int64),
        "last_hidden_state": last_hidden_state.squeeze(0).to(torch.float32).cpu(),
        "mlm_logits": mlm_logits.squeeze(0).to(torch.float32).cpu(),
        "expected_embedding": embedding.squeeze(0).to(torch.float32).cpu(),
    }
    for bf16_device in bf16_devices():
        for impl in ("sdpa", "eager"):
            bf16 = AutoModel.from_pretrained(args.repo, revision=args.revision, torch_dtype=torch.bfloat16, attn_implementation=impl)
            bf16 = bf16.eval().to(bf16_device)
            ids, mask = input_ids.to(bf16_device), attention_mask.to(bf16_device)
            with torch.no_grad():
                hidden = bf16(input_ids=ids, attention_mask=mask).last_hidden_state
            # Pooled in f32, as Svod pools: only the backbone runs in bf16.
            golden[f"expected_embedding_bf16.{bf16_device}_{impl}"] = pool_embed(hidden.float(), mask).squeeze(0).cpu()
    golden_path = out_dir / "golden.safetensors"
    safetensors.torch.save_file({k: v.contiguous() for k, v in golden.items()}, str(golden_path))
    real = int(attention_mask.sum())
    print(f"Wrote {golden_path} ({len(golden)} tensors)")
    print(f"  seq_len={args.max_seq} hidden={last_hidden_state.shape[-1]} real_tokens={real} pad_tokens={args.max_seq - real}")


def long_fixture(repo: str, revision: str, length: int, out: Path) -> None:
    """Two right-padded rows — one filling `length`, one short — through the
    backbone in f32 on the CPU, and in bf16 along PyTorch's equivalent paths."""
    if not torch.cuda.is_available():
        raise SystemExit("--long needs a CUDA device: PyTorch's GPU bf16 paths are part of the yardstick")
    tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision)
    texts = [LONG_TEXT * (length // 60 + 1), SHORT_TEXT]
    enc = tokenizer(texts, return_tensors="pt", padding="max_length", max_length=length, truncation=True)
    ids, mask = enc["input_ids"], enc["attention_mask"]

    def forward(dtype: torch.dtype, impl: str, device: str, alone: bool = False) -> torch.Tensor:
        """`last_hidden_state` as f32; `alone` runs each row as its own batch,
        a row without padding unmasked (as a single-row caller passes it)."""
        model = AutoModel.from_pretrained(repo, revision=revision, torch_dtype=dtype, attn_implementation=impl).eval().to(device)
        with torch.no_grad():
            if alone:
                rows = []
                for r in range(ids.shape[0]):
                    m = mask[r : r + 1]
                    kwargs = {} if bool(m.all()) else {"attention_mask": m.to(device)}
                    rows.append(model(input_ids=ids[r : r + 1].to(device), **kwargs).last_hidden_state)
                hidden = torch.cat(rows)
            else:
                hidden = model(input_ids=ids.to(device), attention_mask=mask.to(device)).last_hidden_state
        return hidden.to(torch.float32).cpu()

    tensors = {
        "input_ids": ids.to(torch.int64),
        "attention_mask": mask.to(torch.int64),
        "last_hidden_state": forward(torch.float32, "sdpa", "cpu"),
    }
    for device in ("cpu", "cuda"):
        for impl in ("sdpa", "eager"):
            tensors[f"last_hidden_state_bf16.{device}_{impl}"] = forward(torch.bfloat16, impl, device)
            if device == "cuda":
                tensors[f"last_hidden_state_bf16.{device}_{impl}_alone"] = forward(torch.bfloat16, impl, device, True)
    safetensors.torch.save_file({k: v.contiguous() for k, v in tensors.items()}, str(out))
    lens = mask.sum(-1).tolist()
    paths = [k.split(".", 1)[1] for k in tensors if k.startswith("last_hidden_state_bf16.")]
    print(f"Wrote {out}\n  input_ids {tuple(ids.shape)} real tokens per row {lens}  ({repo})\n  bf16 paths {paths}")


if __name__ == "__main__":
    main()
