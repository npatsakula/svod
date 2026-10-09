#!/usr/bin/env python3
"""Check that a published safetensors file is the conversion this repo expects.

`scripts/convert_yolo.py` uploads `data/yolo/model.safetensors` to a HuggingFace
repo so the parity test can fetch weights without PyTorch. This proves the
uploaded artifact is that same file: byte-for-byte, and -- if a local copy of
the generated file is available -- tensor-for-tensor as well.

Usage:
  # against a local download (also the pre-upload check)
  python scripts/verify_yolo_weights.py path/to/model.safetensors

  # against the Hub (after uploading)
  python scripts/verify_yolo_weights.py --hub <owner>/<repo>
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tempfile
import urllib.request
from pathlib import Path

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parent.parent
GENERATED = ROOT / "data" / "yolo" / "model.safetensors"
EXPECTED_SHA256 = "74094bff9e372ab4e971fe55649acca3c84e46daf4d522a3071bc0b20e342922"

# Tensors whose shapes pin the two scale-sensitive decisions. Layer 2's shallow
# C3k2 nests a `C3k` at M/L/X, so its 1x1 split convs (48, 96, 1, 1) are what a
# plain `Bottleneck` would get wrong as (48, 96, 3, 3); layer 6 takes an
# explicit `c3k: True` from the YAML at every scale; the head bias gives nc.
SPOT_CHECKS = {
    "model.2.m.0.cv1.conv.weight": (48, 96, 1, 1),
    "model.2.m.0.cv2.conv.weight": (48, 96, 1, 1),
    "model.2.m.0.m.0.cv1.conv.weight": (48, 48, 3, 3),
    "model.6.m.0.cv1.conv.weight": (192, 384, 1, 1),
    "model.23.one2one_cv3.0.2.bias": (80,),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_hub(repo_id: str) -> Path:
    """Download `model.safetensors` into a throwaway temp directory.

    Not the working directory: a CWD-local download would litter 236 MB
    wherever the script runs from -- or clobber `data/yolo/`'s pristine
    local conversion when run from there, silently disarming the
    tensor-for-tensor comparison below.
    """
    url = f"https://huggingface.co/{repo_id}/resolve/main/model.safetensors"
    dst = Path(tempfile.mkdtemp(prefix="svod-yolo-")) / "model.safetensors"
    print(f"downloading {url} ...")
    with urllib.request.urlopen(url, timeout=600) as src, dst.open("wb") as out:
        while chunk := src.read(1 << 20):
            out.write(chunk)
    return dst


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", nargs="?", type=Path, help="local safetensors file")
    ap.add_argument("--hub", metavar="OWNER/REPO", help="fetch from the HuggingFace Hub")
    args = ap.parse_args()
    if bool(args.path) == bool(args.hub):
        ap.error("give exactly one of PATH or --hub")

    path = fetch_hub(args.hub) if args.hub else args.path
    if not path.is_file():
        print(f"FAIL: {path} does not exist", file=sys.stderr)
        return 1

    digest = sha256(path)
    print(f"file   : {path}  ({path.stat().st_size:,} bytes)")
    print(f"sha256 : {digest}")
    if digest != EXPECTED_SHA256:
        print(f"FAIL: digest does not match the expected conversion {EXPECTED_SHA256}", file=sys.stderr)
        print("      Re-run scripts/convert_yolo.py; a stale or partial upload looks like this.", file=sys.stderr)
        return 1
    print("ok     : byte-identical to the generated conversion")

    with safe_open(path, framework="pt") as f:
        keys = list(f.keys())
        print(f"ok     : {len(keys)} tensors readable")
        for name, want in SPOT_CHECKS.items():
            got = tuple(f.get_slice(name).get_shape())
            if got != want:
                print(f"FAIL: {name} shape {got} != {want}", file=sys.stderr)
                return 1
            print(f"ok     : {name} {got}")

        if GENERATED.is_file() and GENERATED.resolve() != path.resolve():
            local = safe_open(GENERATED, framework="pt")
            missing = sorted(set(local.keys()) - set(keys))
            if missing:
                print(f"FAIL: {len(missing)} tensors missing vs {GENERATED}: {missing[:5]}", file=sys.stderr)
                return 1
            for name in keys:
                if not torch.equal(f.get_tensor(name), local.get_tensor(name)):
                    print(f"FAIL: tensor {name} differs from {GENERATED}", file=sys.stderr)
                    return 1
            print(f"ok     : every tensor equal to {GENERATED}")

    print("\nPASS: the published weights are this repo's conversion of the checkpoint.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
