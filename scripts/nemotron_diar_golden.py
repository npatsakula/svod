# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch",
#   "numpy",
#   "soundfile",
#   "librosa",
#   "safetensors",
#   "huggingface_hub",
#   "transformers @ https://github.com/huggingface/transformers/archive/e598fbad926d80bc356c0c4d96532030e3dfdb62.tar.gz",
# ]
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# ///
"""Golden generator for the svod port of nvidia/Nemotron-3-Diarization.

Runs the HF reference (`Nemotron3DiarizationForAudioFrameClassification`, fp32, CPU) over a clip with the
chunked AOSC speaker-cache loop re-implemented step by step from the HF modules, so every intermediate of the
selected steps can be dumped. The loop is checked bit-for-bit against the official HF `forward`.

Usage:
  uv run scripts/nemotron_diar_golden.py                       # audio_1.wav: long (full) + short (first 20 s)
  uv run scripts/nemotron_diar_golden.py --check-session       # also replay the streaming profile through the
                                                               # processor's per-chunk mel (streaming session API)
  uv run scripts/nemotron_diar_golden.py --nemo-modules <NeMo>/nemo/collections/asr/modules/sortformer_modules.py
                                                               # also replay the cache logic with NeMo's
                                                               # `SortformerModules.streaming_update`

Profiles (encoder frames of 80 ms):
  offline   : spkcache 264, fifo 40,  chunk 340, rc 40, update period 300, lc 0
  streaming : spkcache 264, fifo 264, chunk 9,   rc 4,  update period 222, lc 0 (model-card 1.04 s)

Outputs, in --out (default data/nemotron_diar/):
  golden_<clip>_<profile>.safetensors, segments_<clip>_<profile>.rttm

Per-step tensors (prefix `step{i}.`), N = n_cache + n_fifo + n_chunk + n_rc encoder frames:
  embeds        [N, 512]   encoder input: [spkcache | fifo | chunk | right context], pre input-LayerNorm
  lens          [4]  i64   n_cache, n_fifo, n_chunk, n_rc (lc is always 0)
  mask          [N]  i64   valid encoder frames (cache/fifo always 1; chunk from mel attention_mask[::8])
  mel           [128, M]   mel slice of chunk + rc, M = min((end+rc)*8, T) - start*8 (zero-padded to x8 by stacking)
  logits        [8N, 8]    classifier logits for the whole sequence at 10 ms (row r = encoder frame r // 8)
  probs         [8N, 8]    sigmoid(logits)
  pooled_probs  [N, 8]     avg_pool(8) of probs, times mask: what the cache update scores
  chunk_logits  [8*n_chunk, 8]  emitted rows: logits[8*(n_cache+n_fifo) : 8*(n_cache+n_fifo+n_chunk)]
  state         [5]  i64   after update: n_cache, n_fifo, is_compressed, n_popped, compressed_this_step
  compress_scores [n_cache + n_popped, 8]  compressing steps only: the scores the cache ranks by (after both
                boosts, before the silence slots); exact ties among them are broken arbitrarily by torch.topk
  spkcache      [n_cache', 512], spkcache_probs [n_cache', 8]   cache after the update
  fifo          [n_fifo', 512],  fifo_probs     [n_fifo', 8]    fifo after the update; fifo_probs are this step's
                pooled probs of those frames (NeMo `fifo_preds`); the next step re-estimates them, unused by HF
  step0 only: after_input_ln, after_layer0, encoder_out (final LayerNorm, before proj) [N, 512],
              after_proj [N, 192], upsampled [8N, 192]
Global: audio [L], mel [128, T] (T = L//160 + 1, frames >= mel_len are zero), mel_len, attention_mask [T],
  silence_embeds [512], logits/probs [T, 8], steps [S, 6] i64 =
  (chunk_start, chunk_end, rc_end) in encoder frames, (mel_start, mel_emit_end, mel_input_end) in mel frames.
"""

import argparse
import importlib.util
import json
import logging
import math
import sys
import types
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from safetensors.torch import save_file
from transformers import AutoProcessor, Nemotron3DiarizationForAudioFrameClassification
from transformers.models.nemotron3_diarization.modeling_nemotron3_diarization import (
    Nemotron3DiarizationSpeakerCache,
)

MODEL_ID = "nvidia/Nemotron-3-Diarization"
SR = 16000
PROFILES = {
    "offline": dict(spkcache=264, fifo=40, chunk=340, rc=40, update_period=300),
    "streaming": dict(spkcache=264, fifo=264, chunk=9, rc=4, update_period=222),
}


def set_profile(model, p):
    cfg = model.config
    cfg.chunk_length, cfg.chunk_right_context = p["chunk"], p["rc"]
    cfg.fifo_length, cfg.speaker_cache_update_period = p["fifo"], p["update_period"]
    assert cfg.streaming_config.speaker_cache_length == p["spkcache"]


class Capture:
    """Forward hooks recording the step-0 encoder intermediates."""

    def __init__(self, model):
        tower = model.model.audio_tower
        self.out = {}
        self.enabled = False
        hooks = {
            "after_input_ln": tower.input_layer_norm,
            "after_layer0": tower.layers[0],
            "encoder_out": tower.layer_norm,
            "after_proj": model.model.proj,
            "upsampled": model.model.upsampler,
        }
        for name, module in hooks.items():
            module.register_forward_hook(self._hook(name))

    def _hook(self, name):
        def fn(_module, _inputs, output):
            if self.enabled:
                self.out[name] = (output[0] if isinstance(output, tuple) else output)[0].detach().clone()

        return fn


@torch.no_grad()
def run_chunked(model, feats, mask, p, capture):
    """Mirror of the HF forward (offline mode with profile `p`), recording every step."""
    cache = Nemotron3DiarizationSpeakerCache(
        model.config.streaming_config, fifo_length=p["fifo"], speaker_cache_update_period=p["update_period"]
    )
    # Record the scores a compression ranks by (the last boost's output), so a
    # port can tell a different selection from an exact tie torch broke its way.
    boost = cache._boost_scores

    def recording_boost(scores, num_boosted, **kwargs):
        recording_boost.last = boost(scores, num_boosted, **kwargs)
        return recording_boost.last

    cache._boost_scores = recording_boost
    sub = model.config.audio_config.subsampling_factor
    num_frames = feats.shape[1]
    embeds = model.model.audio_tower.embedder(feats)
    num_embeds = embeds.shape[1]
    embed_mask = mask[:, ::sub].bool()

    steps, logits_out = [], []
    for i, start in enumerate(range(0, num_embeds, p["chunk"])):
        end = min(start + p["chunk"], num_embeds)
        rc_end = min(end + p["rc"], num_embeds)
        chunk_embeds = embeds[:, start:rc_end]
        cached = cache.get_embeds(chunk_embeds)
        n_cache, n_fifo = cache.num_cache_frames, cache.num_fifo_frames
        seq = torch.cat([cached, chunk_embeds], dim=1)
        step_mask = torch.cat([embed_mask.new_ones(1, cached.shape[1]), embed_mask[:, start:rc_end]], dim=1)
        position_ids = torch.arange(seq.shape[1])[None]

        capture.enabled = i == 0
        hidden = model.model(inputs_embeds=seq, attention_mask=step_mask, position_ids=position_ids).last_hidden_state
        capture.enabled = False
        logits = model.classifier(hidden)
        pooled = cache._pool_probs(logits, step_mask)
        n_popped = cache._num_popped_frames(n_fifo + end - start)
        compressed_now = n_popped > 0 and n_cache + n_popped > cache.speaker_cache_length
        cache.update(seq, logits, model.silence_embeds, end - start, mask=step_mask)

        lo, hi = (n_cache + n_fifo) * sub, (n_cache + n_fifo + end - start) * sub
        logits_out.append(logits[:, lo:hi])
        fifo_lo = n_cache + n_popped
        steps.append(
            dict(
                bounds=(start, end, rc_end, start * sub, min(end * sub, num_frames), min(rc_end * sub, num_frames)),
                embeds=seq[0],
                lens=(n_cache, n_fifo, end - start, rc_end - end),
                mask=step_mask[0],
                mel=feats[0, start * sub : min(rc_end * sub, num_frames)].T,
                logits=logits[0],
                pooled_probs=pooled[0],
                chunk_logits=logits[0, lo:hi],
                state=(
                    cache.num_cache_frames,
                    cache.num_fifo_frames,
                    int(cache.is_compressed),
                    n_popped,
                    int(compressed_now),
                ),
                spkcache=cache.embeds[0, : cache.num_cache_frames].clone(),
                spkcache_probs=cache.probs[0, : cache.num_cache_frames].clone(),
                fifo=cache.fifo[0, : cache.num_fifo_frames].clone(),
                fifo_probs=pooled[0, fifo_lo : fifo_lo + cache.num_fifo_frames].clone(),
                extra={**(dict(capture.out) if i == 0 else {}),
                       **({"compress_scores": recording_boost.last[0].clone()} if compressed_now else {})},
            )
        )
    logits = torch.cat(logits_out, dim=1)[:, :num_frames]
    return logits[0], steps


@torch.no_grad()
def run_session(model, processor, audio):
    """Streaming-profile logits via the processor's per-chunk session API (per-chunk mel), and the number of
    leading frames emitted before its last chunk (the last chunk absorbs the whole tail, unlike the fixed chunking)."""
    fe = processor.feature_extractor
    per_step, cache, out = processor.num_mel_frames_per_step, None, []
    first = processor.num_samples_first_audio_chunk
    pos_mel = 0
    while True:
        is_first = pos_mel == 0
        a0 = 0 if is_first else processor.audio_chunk_start(pos_mel)
        n = first if is_first else processor.num_samples_per_audio_chunk
        is_last = a0 + n > len(audio)
        chunk = audio[a0:] if is_last else audio[a0 : a0 + n]
        inputs = processor(
            chunk, sampling_rate=fe.sampling_rate, is_streaming=True,
            is_first_audio_chunk=is_first, is_last_audio_chunk=is_last, return_tensors="pt",
        )
        kwargs = {"num_lookahead_frames": int(inputs["num_lookahead_frames"])} if "num_lookahead_frames" in inputs else {}
        res = model(
            input_features=inputs["input_features"], attention_mask=inputs["attention_mask"],
            speaker_cache=cache, **kwargs,
        )
        cache = res.speaker_cache
        out.append(res.logits[0])
        pos_mel += per_step
        if is_last:
            break
    return torch.cat(out), pos_mel - per_step


def load_nemo_modules(path):
    """Import NeMo's sortformer_modules.py with its three nemo imports stubbed out."""
    for name in ["nemo", "nemo.core", "nemo.core.classes", "nemo.core.classes.exportable",
                 "nemo.core.classes.module", "nemo.utils"]:
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["nemo.core.classes.exportable"].Exportable = object
    sys.modules["nemo.core.classes.module"].NeuralModule = torch.nn.Module
    sys.modules["nemo.utils"].logging = logging.getLogger("nemo")
    spec = importlib.util.spec_from_file_location("nemo_sortformer_modules", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@torch.no_grad()
def run_nemo_cache(model, feats, mask, p, nemo):
    """NeMo `forward_streaming` (sync, high-res) with HF networks and NeMo's cache logic."""
    sc = model.config.streaming_config
    sm = nemo.SortformerModules(
        num_spks=8, fc_d_model=512, tf_d_model=192, subsampling_factor=8, spkcache_len=p["spkcache"],
        fifo_len=p["fifo"], chunk_len=p["chunk"], spkcache_update_period=p["update_period"],
        chunk_left_context=0, chunk_right_context=p["rc"],
        spkcache_sil_frames_per_spk=sc.speaker_cache_silence_frames_per_speaker,
        pred_score_threshold=sc.prediction_score_threshold, scores_boost_latest=sc.latest_frames_score_boost,
        strong_boost_rate=sc.strong_boost_rate, weak_boost_rate=sc.weak_boost_rate,
        min_pos_scores_rate=sc.min_positive_scores_rate, use_learnable_sil_emb=True, upsample_factor=8,
    ).eval()
    sm.learnable_sil_emb.data.copy_(model.silence_embeds)
    state = sm.init_streaming_state(batch_size=1)
    embeds = model.model.audio_tower.embedder(feats)
    embed_mask = mask[:, ::8].bool()
    out = []
    for start in range(0, embeds.shape[1], p["chunk"]):
        end = min(start + p["chunk"], embeds.shape[1])
        rc_end = min(end + p["rc"], embeds.shape[1])
        chunk = embeds[:, start:rc_end]
        seq = torch.cat([state.spkcache, state.fifo, chunk], dim=1)
        n_prev = state.spkcache.shape[1] + state.fifo.shape[1]
        step_mask = torch.cat([embed_mask.new_ones(1, n_prev), embed_mask[:, start:rc_end]], dim=1)
        hidden = model.model(inputs_embeds=seq, attention_mask=step_mask,
                             position_ids=torch.arange(seq.shape[1])[None]).last_hidden_state
        hi_res = model.classifier(hidden).sigmoid()
        preds = sm.downsample_preds(hi_res, 8) * step_mask[..., None]
        state, _ = sm.streaming_update(state, chunk, preds, lc=0, rc=rc_end - end)
        out.append(hi_res[0, n_prev * 8 : (n_prev + end - start) * 8])
    return torch.cat(out)[: feats.shape[1]]


def write_rttm(path, probs, uri, threshold=0.5, frame=0.01):
    active = (probs > threshold).int()
    pad = torch.zeros(1, active.shape[1], dtype=torch.int32)
    changes = torch.cat([pad, active, pad]).diff(dim=0)
    segs = []
    for spk in range(active.shape[1]):
        starts = (changes[:, spk] == 1).nonzero()[:, 0].tolist()
        ends = (changes[:, spk] == -1).nonzero()[:, 0].tolist()
        segs += [(s * frame, e * frame, spk) for s, e in zip(starts, ends)]
    segs.sort()
    with open(path, "w") as f:
        for s, e, spk in segs:
            f.write(f"SPEAKER {uri} 1 {s:.2f} {e - s:.2f} <NA> <NA> speaker_{spk} <NA> <NA>\n")
    return segs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    root = Path(__file__).resolve().parents[1]
    ap.add_argument("--audio", type=Path, default=root / "audio_1.wav")
    ap.add_argument("--short-seconds", type=float, default=20.0)
    ap.add_argument("--out", type=Path, default=root / "data/nemotron_diar")
    ap.add_argument("--check-session", action="store_true")
    ap.add_argument("--nemo-modules", type=Path)
    args = ap.parse_args()

    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.out.mkdir(parents=True, exist_ok=True)

    model = Nemotron3DiarizationForAudioFrameClassification.from_pretrained(
        MODEL_ID, dtype=torch.float32, attn_implementation="eager"
    ).eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    capture = Capture(model)
    nemo = load_nemo_modules(args.nemo_modules) if args.nemo_modules else None

    audio, sr = sf.read(str(args.audio), dtype="float32", always_2d=True)
    assert sr == SR, f"expected {SR} Hz, got {sr}"
    audio = audio[:, 0]
    clips = {"long": audio, "short": audio[: int(args.short_seconds * SR)]}

    for clip, wav in clips.items():
        inputs = processor(wav, sampling_rate=SR, return_tensors="pt")
        feats, mask = inputs["input_features"], inputs["attention_mask"]
        for prof, p in PROFILES.items():
            set_profile(model, p)
            logits, steps = run_chunked(model, feats, mask, p, capture)
            with torch.no_grad():
                ref = model(input_features=feats, attention_mask=mask).logits[0]
            diff_official = (logits - ref).abs().max().item()
            assert diff_official == 0.0, f"loop diverges from the HF forward: {diff_official}"
            probs = logits.sigmoid()

            compress = [i for i, s in enumerate(steps) if s["state"][4]]
            first = compress[0] if compress else None
            pick = sorted({0, len(steps) - 1} | ({first, min(first + 1, len(steps) - 1)} if compress else set()))

            t = {
                "audio": torch.from_numpy(np.ascontiguousarray(wav)),
                "mel": feats[0].T.contiguous(),
                "mel_len": mask.sum(-1).to(torch.int64),
                "attention_mask": mask[0].to(torch.int64),
                "silence_embeds": model.silence_embeds.detach().clone(),
                "logits": logits,
                "probs": probs,
                "steps": torch.tensor([s["bounds"] for s in steps], dtype=torch.int64),
            }
            for i in pick:
                s = steps[i]
                for k in ("embeds", "mask", "mel", "logits", "pooled_probs", "chunk_logits",
                          "spkcache", "spkcache_probs", "fifo", "fifo_probs"):
                    v = s[k]
                    t[f"step{i}.{k}"] = v.to(torch.int64) if k == "mask" else v
                t[f"step{i}.probs"] = s["logits"].sigmoid()
                t[f"step{i}.lens"] = torch.tensor(s["lens"], dtype=torch.int64)
                t[f"step{i}.state"] = torch.tensor(s["state"], dtype=torch.int64)
                for k, v in s["extra"].items():
                    t[f"step{i}.{k}"] = v
            t = {k: v.detach().clone().contiguous() for k, v in t.items()}

            checks = {"loop_vs_hf_forward": diff_official}
            if args.check_session and prof == "streaming":
                sess, n = run_session(model, processor, wav)
                checks["session_prefix_max_prob_diff"] = (sess[:n].sigmoid() - probs[:n]).abs().max().item()
            if nemo is not None:
                nemo_probs = run_nemo_cache(model, feats, mask, p, nemo)
                checks["nemo_cache_max_prob_diff"] = (nemo_probs - probs).abs().max().item()

            active = [k for k in range(8) if (probs[:, k] > 0.5).any()]
            meta = {
                "clip": clip,
                "profile": prof,
                "source_audio": args.audio.name,
                "params": json.dumps(p),
                "num_steps": str(len(steps)),
                "saved_steps": json.dumps(pick),
                "compression_steps": json.dumps(compress),
                "active_speakers": json.dumps(active),
                "checks": json.dumps(checks),
                "hf_model": MODEL_ID,
                "transformers_commit": "e598fbad926d80bc356c0c4d96532030e3dfdb62",
            }
            save_file(t, str(args.out / f"golden_{clip}_{prof}.safetensors"), metadata=meta)
            segs = write_rttm(args.out / f"segments_{clip}_{prof}.rttm", probs, f"{args.audio.stem}_{clip}")
            print(
                f"{clip}/{prof}: {len(wav) / SR:.1f}s T={feats.shape[1]} steps={len(steps)} saved={pick} "
                f"compress={compress[:8]}{'...' if len(compress) > 8 else ''} ({len(compress)} total) "
                f"speakers={active} segments={len(segs)} checks={checks}"
            )


if __name__ == "__main__":
    main()
