#!/usr/bin/env python3
"""Generate the PyTorch fixtures for the YOLO26 parity test.

The test input is committed (`model/src/test/data/yolo/bus640.png`, git-lfs);
everything else is generated here, into `data/yolo/`:

  - model.safetensors    the Ultralytics checkpoint, fp32, PyTorch key names
  - golden.safetensors
      images        [1, 3, 640, 640] f32 -- the committed image, [0, 1]
      images_shape  [4] i64
      output        [1, 84, 8400] f32 -- what Yolo26Detect::forward returns

Upstream is AGPL-3.0, so nothing here is committed: the checkpoint is fetched
from a pinned URL and digest, and the converted weights are published to the
HuggingFace repo `parity.rs` names, which lets the test fetch weights without
PyTorch. Publishing an update means uploading the regenerated
`data/yolo/model.safetensors` and checking it first:

  pip install ultralytics==8.4.153 safetensors torch pillow
  python scripts/convert_yolo.py
  python scripts/verify_yolo_weights.py data/yolo/model.safetensors
  # upload data/yolo/model.safetensors to the Hub repo, then re-verify via --hub
  cargo test -p svod-model --lib yolo::parity -- --ignored
"""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file

REPO = "Ultralytics/YOLO26"
ASSET = "yolo26x.pt"
ASSET_SHA256 = "9fdd44a31c504547ffb81d2c6d9e6dac3493c8eaa8b0398d3f43bae6c7003e92"
ASSET_URL = f"https://huggingface.co/{REPO}/resolve/main/{ASSET}"

ROOT = Path(__file__).resolve().parent.parent
IMAGE = ROOT / "model" / "src" / "test" / "data" / "yolo" / "bus640.png"
OUTPUT_DIR = ROOT / "data" / "yolo"


def fixture_image() -> torch.Tensor:
    """The committed image as the `[1, 3, 640, 640]` CHW tensor forward wants.

    Decoding is `uint8 / 255`, which is exact in both languages -- no resampling
    happens on either side, so PyTorch and Rust see identical bits.
    """
    rgb = np.asarray(Image.open(IMAGE).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).contiguous()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint() -> Path:
    """Download the pinned checkpoint, or reuse a verified local copy."""
    dst = OUTPUT_DIR / ASSET
    if dst.exists() and sha256(dst) == ASSET_SHA256:
        return dst
    print(f"downloading {ASSET_URL} ...")
    dst.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(ASSET_URL, timeout=300) as src, dst.open("wb") as out:
        shutil.copyfileobj(src, out)
    digest = sha256(dst)
    if digest != ASSET_SHA256:
        raise RuntimeError(f"{dst} digest {digest} != pinned {ASSET_SHA256}")
    return dst


def raw_predictions(net: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    """The decoded `[B, 4 + nc, A]` tensor, before top-k selection.

    YOLO26's head is end-to-end, so calling the model returns `(y, aux)`: the
    raw `one2many`/`one2one` branches in `aux`, and the head's own decode of
    one branch in `y`. `Yolo26Detect::forward` stops earlier than that, so
    take the `one2one` branch `aux` carries and decode it the way it does.

    The head's own `_inference` is deliberately not used: it decodes through
    `decode_bboxes(...)`, which passes `xywh and not end2end and not xyxy`
    down to `dist2bbox` -- and `end2end` is a property that reads `False` on
    an unfused model (`xyxy` is `False` by default), so it yields
    centre/width/height while svod yields corners. Decoding explicitly with
    `xywh=False` keeps both sides on xyxy.
    """
    with torch.no_grad():
        out = net(images)
    if not (isinstance(out, tuple) and isinstance(out[1], dict) and "one2one" in out[1]):
        raise RuntimeError(
            f"Expected an end-to-end head returning (preds, {{'one2one': ...}}), got {type(out)}. "
            "Is this checkpoint really a YOLO26?"
        )
    detect = net.model[-1]
    one2one = out[1]["one2one"]
    with torch.no_grad():
        # Mirrors svod's Detect::forward: dist2bbox over the anchors, times the
        # strides, concatenated with the sigmoid'd class scores.
        dbox = detect.decode_bboxes(detect.dfl(one2one["boxes"]), detect.anchors.unsqueeze(0), xywh=False)
        return torch.cat((dbox * detect.strides, one2one["scores"].sigmoid()), 1)


def main() -> None:
    from ultralytics import YOLO

    net = YOLO(str(checkpoint())).model.float().eval()

    weights = {k: v.contiguous() for k, v in net.state_dict().items() if v.is_floating_point()}
    save_file(weights, str(OUTPUT_DIR / "model.safetensors"))
    print(f"saved {len(weights)} tensors -> {OUTPUT_DIR / 'model.safetensors'}")

    images = fixture_image()
    output = raw_predictions(net, images)
    print(f"output shape: {tuple(output.shape)}  max score: {output[:, 4:].max():.6f}")

    save_file(
        {
            "images": images,
            "images_shape": torch.tensor(list(images.shape), dtype=torch.int64),
            "output": output.contiguous(),
        },
        str(OUTPUT_DIR / "golden.safetensors"),
    )
    print(f"saved {OUTPUT_DIR / 'golden.safetensors'}")


if __name__ == "__main__":
    main()
