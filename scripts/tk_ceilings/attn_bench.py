"""Attention-forward ceilings at tk_bench's shapes and flop formula (4·B·H·T²·D, halved when causal).

Backends (bf16 in, bf16 out, scale 1/sqrt(d)):
  fa3     FlashAttention-3 hopper `flash_attn_interface.flash_attn_func`, q/k/v [B,T,H,D] (tk's layout)
  tkhazy  ThunderKittens mha_h100 `_C.mha_forward`, q/k/v [B,H,T,D]; also writes L [B,H,T] f32
  sdpa-*  torch scaled_dot_product_attention under one SDPA backend, [B,H,T,D]
Inputs are made contiguous in the backend's layout before timing; no transpose is timed.

Timing: a 0.3 s warmup, then N back-to-back calls under torch.profiler; a call's time is the summed
device duration of the kernels it launches (CUPTI stamps, like tk's tuner). Reported: min and mean
over calls, plus the event-timed mean of back-to-back calls (host gaps included).
Line format: ref|<backend>|bf16|b{B}h{H}t{T}d{D}{c}|<kernels>|min_ns|TFLOP/s|mean_ns|TFLOP/s|evt_ns|TFLOP/s|maxerr
"""
import argparse, json, math, os, sys, tempfile, time

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def backends(names):
    out = {}
    for n in names:
        if n == "fa3":
            from flash_attn_interface import flash_attn_func
            out[n] = ("bthd", lambda q, k, v, c: flash_attn_func(q, k, v, causal=c))
        elif n == "fa2":
            from flash_attn import flash_attn_func as fa2
            out[n] = ("bthd", lambda q, k, v, c: fa2(q, k, v, causal=c))
        elif n == "tkhazy":
            sys.path.insert(0, os.environ["TK_ATTN_DIR"])
            import _C as tk
            out[n] = ("bhtd", lambda q, k, v, c: tk.mha_forward(q, k, v, c)[0])
        elif n.startswith("sdpa-"):
            be = {"sdpa-flash": SDPBackend.FLASH_ATTENTION, "sdpa-cudnn": SDPBackend.CUDNN_ATTENTION,
                  "sdpa-efficient": SDPBackend.EFFICIENT_ATTENTION}[n]

            def f(q, k, v, c, be=be):
                with sdpa_kernel([be]):
                    return F.scaled_dot_product_attention(q, k, v, is_causal=c)
            out[n] = ("bhtd", f)
        else:
            raise SystemExit(f"unknown backend {n}")
    return out


def kernel_times(fn, calls):
    """Per-call summed kernel durations (ns) over `calls` back-to-back calls."""
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        fn()
        torch.cuda.synchronize()
    with tempfile.NamedTemporaryFile(suffix=".json") as f:
        p.export_chrome_trace(f.name)
        one = [e for e in json.load(open(f.name))["traceEvents"] if e.get("cat") == "kernel"]
    per_call = len(one)
    names = sorted({e["name"][:60] for e in one})
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(calls):
            fn()
        torch.cuda.synchronize()
    with tempfile.NamedTemporaryFile(suffix=".json") as f:
        p.export_chrome_trace(f.name)
        ks = sorted((e for e in json.load(open(f.name))["traceEvents"] if e.get("cat") == "kernel"), key=lambda e: e["ts"])
    assert len(ks) == per_call * calls, (len(ks), per_call, calls)
    times = [sum(e["dur"] for e in ks[i:i + per_call]) * 1e3 for i in range(0, len(ks), per_call)]
    return times, per_call, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", default="fa3,tkhazy,sdpa-flash,sdpa-cudnn")
    ap.add_argument("--shapes", default="4x8x2048x64,4x8x2048x128")  # BxHxTxD
    ap.add_argument("--calls", type=int, default=50)
    a = ap.parse_args()
    torch.manual_seed(0)
    bes = backends(a.backends.split(","))
    for s in a.shapes.split(","):
        b, h, t, d = map(int, s.split("x"))
        for causal in (False, True):
            q, k, v = (torch.randn(b, t, h, d, device="cuda", dtype=torch.bfloat16) for _ in range(3))
            with sdpa_kernel([SDPBackend.MATH]):
                ref = F.scaled_dot_product_attention(*(x.transpose(1, 2).float() for x in (q, k, v)),
                                                     is_causal=causal).transpose(1, 2)
            flops = 4 * b * h * t * t * d / (2 if causal else 1)
            tag = f"b{b}h{h}t{t}d{d}{'c' if causal else ''}"
            for name, (layout, f) in bes.items():
                args = (q, k, v) if layout == "bthd" else tuple(x.transpose(1, 2).contiguous() for x in (q, k, v))
                try:
                    o = f(*args, causal)
                except Exception as e:  # noqa: BLE001
                    print(f"ref|{name}|bf16|{tag}|failed: {type(e).__name__}: {str(e)[:120]}", flush=True)
                    continue
                o = o if layout == "bthd" else o.transpose(1, 2)
                err = (o.float() - ref).abs().max().item()
                call = lambda: f(*args, causal)
                end = time.time() + 0.3
                while time.time() < end:
                    call()
                torch.cuda.synchronize()
                times, per_call, names = kernel_times(call, a.calls)
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record()
                for _ in range(a.calls):
                    call()
                e1.record()
                torch.cuda.synchronize()
                evt = e0.elapsed_time(e1) * 1e6 / a.calls
                mn, mean = min(times), sum(times) / len(times)
                print(f"ref|{name}|bf16|{tag}|{per_call}k:{';'.join(names)}|{mn:.0f}|{flops / mn / 1e3:.1f}|"
                      f"{mean:.0f}|{flops / mean / 1e3:.1f}|{evt:.0f}|{flops / evt / 1e3:.1f}|{err:.3g}", flush=True)
            del q, k, v, ref
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
