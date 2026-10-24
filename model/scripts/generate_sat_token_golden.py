"""Generate the SaT (XLM-RoBERTa token classifier) golden fixtures for the parity tests.

Produces, under the XLM-R data dir (`SVOD_XLM_ROBERTA` env var or
`../data/xlm_roberta` relative to this crate):

  golden_sat.safetensors      three texts in three languages, padded to --max-seq
  golden_sat_512.safetensors  a full 512-token row (the longest the 514-row
                              position table numbers) and a shorter one padded to it

each with:

  input_ids        (B, L)     int64    tokenized texts, padded with pad_token_id = 1
  attention_mask   (B, L)     int64    1 = real token, 0 = pad
  expected_logits  (B, L, 1)  float32  per-token newline logits

and

  golden_sat_windows.safetensors  one long text stitched from overlapping
                                  windows by wtpsplit's own `extract`:

  text                     (bytes,)  uint8    the text, UTF-8
  input_ids                (N,)      int64    its tokens, no specials
  stride                   (1,)      int64    tokens between window starts
  expected_logits_uniform  (N, 1)    float32  per-token logits, windows averaged
  expected_logits_hat      (N, 1)    float32  the same, `hat`-weighted

`extract` accumulates in float16, whose rounding of a sum of up to eight
logits would swamp an f32 comparison; the expected logits come from `extract`
with its numpy float16 read as float32, the windowing and weighting its own.
The script checks they stay within float16 rounding of `extract` unpatched.

`segment-any-text/sat-3l-sm` is wtpsplit's `SubwordXLMForTokenClassification`
(`model_type: xlm-token`). With `lookahead: null` its forward is HF's
`XLMRobertaForTokenClassification`, which loads the checkpoint key for key, so
the reference needs no wtpsplit. The repo ships no tokenizer: SaT uses
`FacebookAI/xlm-roberta-base`'s.

The `*_bf16.{cpu,cuda}_{sdpa,eager}` tensors are the same logits from the
checkpoint loaded in bf16 along four equivalent paths, SDPA and eager attention
on the CPU and on CUDA (upcast to f32): PyTorch's own bf16 drift from the f32
golden, the yardstick the bf16 test bounds Svod's by. The CUDA paths need a
CUDA device.

Usage:
    uv run scripts/generate_sat_token_golden.py
    SVOD_XLM_ROBERTA=/path uv run scripts/generate_sat_token_golden.py
"""

import argparse
import contextlib
import importlib
import json
import os
from pathlib import Path

import numpy as np
import safetensors.torch
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, XLMRobertaConfig, XLMRobertaForTokenClassification

# The module: the package re-exports its `extract` function under the same name.
wtp_extract = importlib.import_module("wtpsplit.extract")

# Unpunctuated and punctuated text in three scripts, of different lengths so
# padding is load-bearing.
SAMPLE_TEXTS = [
    "This is a test This is another test",
    "Привет мир это первое предложение а это уже второе и оно немного длиннее первого",
    "Der Zug kam pünktlich an. Wir stiegen aus und gingen in die Stadt. 東京は日本の首都です。人口がとても多い。",
]

# Repeated into the long golden's full row, past 512 tokens before truncation.
LONG_TEXT = " ".join(
    [
        "The committee met on Tuesday to review the budget and it decided to postpone the vote until next month",
        "Комиссия собралась во вторник чтобы рассмотреть бюджет но голосование отложили на следующий месяц",
        "Le comité s'est réuni mardi. Il a décidé de reporter le vote au mois prochain.",
        "委员会周二开会审议预算。投票推迟到下个月。",
    ]
    * 12
)

# Long enough for a dozen overlapping windows at wtpsplit's default stride. The
# emoji tokenize as a lone "▁" sharing a byte span with the emoji after it.
WINDOWS_TEXT = " ".join(
    [
        "We met at noon 👍🏽 and talked for hours about the plan it was late when we left",
        "Мы встретились в полдень 🙂 и долго обсуждали план. Разошлись уже поздно",
        "Nous avons parlé du projet pendant des heures. Il était tard quand nous sommes partis 🚆",
        "我们中午见面，讨论了很久。离开时已经很晚了。",
    ]
    * 14
)


def bf16_devices() -> tuple[str, ...]:
    """Where the bf16 yardstick runs: the CPU, and CUDA, whose paths the bf16
    test requires — without a CUDA device the golden still serves the f32
    parity tests."""
    if torch.cuda.is_available():
        return ("cpu", "cuda")
    print("warning: no CUDA device; the bf16 test will refuse this golden's CPU-only bf16 paths")
    return ("cpu",)


def load(repo: str, revision: str, config: XLMRobertaConfig, dtype: torch.dtype, **kwargs) -> XLMRobertaForTokenClassification:
    return XLMRobertaForTokenClassification.from_pretrained(repo, revision=revision, config=config, dtype=dtype, **kwargs).eval()


def write_golden(path: Path, args, config: XLMRobertaConfig, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> None:
    model = load(args.repo, args.revision, config, torch.float32)
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    golden = {
        "input_ids": input_ids.to(torch.int64),
        "attention_mask": attention_mask.to(torch.int64),
        "expected_logits": logits.to(torch.float32),
    }
    for device in bf16_devices():
        for impl in ("sdpa", "eager"):
            bf16 = load(args.repo, args.revision, config, torch.bfloat16, attn_implementation=impl).to(device)
            with torch.no_grad():
                out = bf16(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device)).logits
            golden[f"expected_logits_bf16.{device}_{impl}"] = out.to(torch.float32).cpu()
    safetensors.torch.save_file({k: v.contiguous() for k, v in golden.items()}, str(path))
    real = attention_mask.sum(dim=-1).tolist()
    probs = torch.sigmoid(logits[..., 0])
    print(f"Wrote {path} ({len(golden)} tensors), shape {tuple(input_ids.shape)}, real tokens {real}")
    for row, r in enumerate(real):
        print(f"  [{row}] tokens past the 0.25 newline threshold: {(probs[row, :r] > 0.25).nonzero().flatten().tolist()}")


class _Float32Numpy:
    """numpy with `float16` read as `float32`."""

    def __getattr__(self, name):
        return np.float32 if name == "float16" else getattr(np, name)


@contextlib.contextmanager
def float32_accumulators():
    """Run wtpsplit's `extract` with its float16 sums, weights and result widened."""
    original, wtp_extract.np = wtp_extract.np, _Float32Numpy()
    try:
        yield
    finally:
        wtp_extract.np = original


def write_windows_golden(path: Path, args, config: XLMRobertaConfig, tokenizer, stride: int) -> None:
    model = wtp_extract.PyTorchWrapper(load(args.repo, args.revision, config, torch.float32))
    ids = tokenizer(WINDOWS_TEXT, add_special_tokens=False)["input_ids"]
    assert len(ids) > 2 * 510, "the text no longer spans several windows"
    golden = {
        "text": torch.tensor(list(WINDOWS_TEXT.encode()), dtype=torch.uint8),
        "input_ids": torch.tensor(ids, dtype=torch.int64),
        "stride": torch.tensor([stride], dtype=torch.int64),
    }
    for weighting in ("uniform", "hat"):
        def stitch():
            kwargs = dict(stride=stride, max_block_size=512, batch_size=8, weighting=weighting, tokenizer=tokenizer)
            return wtp_extract.extract([WINDOWS_TEXT], model, **kwargs)[0][0].astype(np.float32)

        native = stitch()
        with float32_accumulators():
            widened = stitch()
        drift = float(np.abs(native - widened).max())
        print(f"  {weighting}: extract's float16 sums drift {drift:.3e} from float32")
        assert drift < 5e-2, "the widened extract is no longer extract"
        golden[f"expected_logits_{weighting}"] = torch.from_numpy(widened)
    safetensors.torch.save_file(golden, str(path))
    print(f"Wrote {path}: {len(ids)} tokens, stride {stride}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default="segment-any-text/sat-3l-sm")
    parser.add_argument("--revision", default="137da054051ad9f1eac42025f758db4ac9f22535", help="hub revision; the default is the one the parity tests pin")
    parser.add_argument("--tokenizer", default="FacebookAI/xlm-roberta-base")
    parser.add_argument("--tokenizer-revision", default="e73636d4f797dec63c3081bb6ed5c7b0bb3f2089")
    parser.add_argument("--max-seq", type=int, default=64, help="pad/truncate target of the short golden")
    parser.add_argument("--stride", type=int, default=64, help="window stride of the windows golden (wtpsplit's default)")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir or os.environ.get("SVOD_XLM_ROBERTA", Path(__file__).resolve().parent.parent / "../data/xlm_roberta"))
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = json.loads(Path(hf_hub_download(args.repo, "config.json", revision=args.revision)).read_text())
    assert raw.get("lookahead") is None and raw.get("lookahead_split_layers") is None, "a banded SaT is not plain XLM-R"
    config = XLMRobertaConfig(**{k: v for k, v in raw.items() if k not in ("model_type", "architectures", "_name_or_path")})
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, revision=args.tokenizer_revision)
    assert tokenizer.pad_token_id == config.pad_token_id == 1

    enc = tokenizer(SAMPLE_TEXTS, padding="max_length", max_length=args.max_seq, truncation=True, return_tensors="pt")
    write_golden(out_dir / "golden_sat.safetensors", args, config, enc["input_ids"], enc["attention_mask"])

    longest = config.max_position_embeddings - config.pad_token_id - 1
    enc = tokenizer([LONG_TEXT, SAMPLE_TEXTS[1]], padding="max_length", max_length=longest, truncation=True, return_tensors="pt")
    assert enc["attention_mask"][0].sum() == longest, "the long text no longer fills the row"
    write_golden(out_dir / f"golden_sat_{longest}.safetensors", args, config, enc["input_ids"], enc["attention_mask"])

    write_windows_golden(out_dir / "golden_sat_windows.safetensors", args, config, tokenizer, args.stride)


if __name__ == "__main__":
    main()
