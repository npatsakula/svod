# svod-model

High-level inference for pretrained deep learning models on top of
`svod-tensor`. Each model is a pure-Rust port of an upstream checkpoint,
fetched from HuggingFace Hub at runtime and executed through JIT-compiled
plans.

## Common infrastructure

| Module | Role |
|---|---|
| `jit` | `jit_wrapper!`-generated wrappers, `JitRecurrent<J>`, `InputSpec`, `JitError`. Build-once / run-many execution. See [JIT Graphs](../website/docs/architecture/jit-graphs.md). |
| `audio` | Log-mel spectrogram, `Splitter` trait for long-form chunking (`FireRedVadSplitter`, `SileroVadSplitter`, `FixedLengthSplitter`). |
| `state` | Safetensors loading (single-file and sharded), dtype casting, and the origin scopes that name a module's weights. The `StateDict` / `Module` contract itself lives in `svod_tensor::nn`. |
| `blocks` | Shared `conv2d`, `batchnorm2d`, `BasicBlock`, `Bottleneck`, `ResidualStage` reused by every ResNet-shaped backbone. timm/torchvision key convention. |
| `wavlm` | WavLM speech-representation backbone (Conv1d feature extractor + gated rel-pos Transformer, per-layer pruning) consumed by `diarizen`. |
| `xlm_roberta` | XLM-RoBERTa text encoder backbone (absolute position embeddings, post-norm Transformer) consumed by `bgem3`. |
| `qwen3` | Qwen3 decoder-only LLM backbone (causal attention, GQA, per-head Q/K RMSNorm, RoPE, SwiGLU) consumed by embedding models. |
| `sentencepiece` | Minimal SentencePiece `.model` protobuf loader (vocab piece extraction). |

## Models

| Name | Domain | Module | Upstream | HuggingFace |
|---|---|---|---|---|
| GigaAM v3 (CTC + RN-T) | Speech | `gigaam` | [salute-developers/GigaAM](https://github.com/salute-developers/GigaAM) | [`vpermilp/GigaAM-v3`](https://huggingface.co/vpermilp/GigaAM-v3) |
| FireRedVAD (batch + streaming) | Voice activity | `firered_vad` | [FireRedTeam/FireRedVAD](https://github.com/FireRedTeam/FireRedVAD) | [`vpermilp/firered_vad`](https://huggingface.co/vpermilp/firered_vad) |
| Silero VAD 16k | Voice activity | `silero_vad` | [snakers4/silero-vad](https://github.com/snakers4/silero-vad) | [`vpermilp/silero-vad`](https://huggingface.co/vpermilp/silero-vad) |
| GTCRN (offline / streaming) | Speech enhancement | `gtcrn` | [Xiaobin-Rong/gtcrn](https://github.com/Xiaobin-Rong/gtcrn) | [`vpermilp/gtcrn`](https://huggingface.co/vpermilp/gtcrn) |
| DiariZen segmentation (WavLM + Conformer) | Speaker diarization | `diarizen` | [BUT-FIT/DiariZen](https://github.com/BUTSpeechFIT/DiariZen) | [`BUT-FIT/diarizen-wavlm-large-s80-md-v2`](https://huggingface.co/BUT-FIT/diarizen-wavlm-large-s80-md-v2) |
| ModernBERT (base / large) | Text embeddings, fill-mask (MLM) | `modernbert` | [Answer.AI ModernBERT](https://github.com/AnswerDotAI/ModernBERT) | [`answerdotai/ModernBERT-base`](https://huggingface.co/answerdotai/ModernBERT-base) |
| BGE-M3 (dense / sparse / ColBERT) | Text embeddings, retrieval | `bgem3` | [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) (XLM-RoBERTa-large) | [`BAAI/bge-m3`](https://huggingface.co/BAAI/bge-m3) |
| BGE-reranker-v2-m3 | Cross-encoder reranking | `bgem3` | [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) | [`BAAI/bge-reranker-v2-m3`](https://huggingface.co/BAAI/bge-reranker-v2-m3) |
| Qwen3-Embedding-0.6B | Text embeddings, retrieval | `qwen3` | [Qwen team](https://huggingface.co/Qwen) (Qwen3 LLM) | [`Qwen/Qwen3-Embedding-0.6B`](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) |
| Qwen3-Reranker-0.6B | Cross-encoder reranking | `qwen3` | [Qwen team](https://huggingface.co/Qwen) (Qwen3 LLM) | [`Qwen/Qwen3-Reranker-0.6B`](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B) |
| ResNet (18 / 34 / 50 / 101 / 152) | Vision | `resnet` | [He et al. 2015](https://arxiv.org/abs/1512.03385) | [`timm/resnet*.a1_in1k`](https://huggingface.co/timm) |
| WeSpeaker ResNet34 | Speaker embedding | `wespeaker` | [wenet-e2e/wespeaker](https://github.com/wenet-e2e/wespeaker) | [`pyannote/wespeaker-voxceleb-resnet34-LM`](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM) |
| YOLO v26 (detect / cls / seg / obb / pose / depth / semseg) | Vision | `yolo` | [Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26) | [`ultralytics/yolo26n`](https://huggingface.co/ultralytics/yolo26n) |
| Whisper (tiny / base / small / medium / large-v3 / turbo) | Speech recognition | `whisper` | [OpenAI Whisper](https://github.com/openai/whisper) | [`openai/whisper-tiny`](https://huggingface.co/openai/whisper-tiny) |

## Examples

```bash
cargo run -p svod-model --release --example gigaam_infer -- audio.wav [--rnnt] [--timestamps] [--profile]
SVOD_ORIGIN=1 cargo run -p svod-model --release --example gigaam_infer -- audio.wav --profile --origin-depth 3 --profile-json profile.json  # per-layer profile
cargo run -p svod-model --release --example vad_bench -- audio.wav            # Silero vs FireRedVAD timing
cargo run -p svod-model --release --example vad_stream -- audio.wav           # streaming VAD events (simulated mic)
cargo run -p svod-model --release --example resnet_classify -- --hub --image dog.bin --side 224
cargo run -p svod-model --release --example wespeaker_parity -- --hub --data reference.npz
cargo run -p svod-model --release --example yolo_detect -- --hub
cargo run -p svod-model --release --example yolo_detect -- --hub --scale small --image photo.bin --side 640
cargo run -p svod-model --release --example whisper_infer -- audio.wav [--size tiny|base|small] [--profile]
cargo run -p svod-model --release --example whisper_infer -- audio.wav --language auto --timestamps
cargo run -p svod-model --release --example gtcrn_enhance -- --hub --in noisy.wav --out enhanced.wav
cargo run -p svod-model --release --example qwen3_embed -- --texts texts.txt
```
