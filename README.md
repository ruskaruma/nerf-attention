# NeRF-Attention: SIREN KV Cache Compression

Can we replace KV cache *storage* (IO-bound) with KV cache *generation* via
implicit neural representations (compute-bound), exploiting the arithmetic
intensity gap on modern GPUs?

See [CONTEXT.md](CONTEXT.md) for the full research context.

## Quick Start

```bash
uv sync
uv run quickstart          # GPU (auto-detected)
uv run quickstart --cpu    # CPU only
```

Runs the full pipeline on synthetic data in under 2 minutes. No model download needed.

## Full Experiment (Llama 3.1-8B)

```bash
uv sync --extra llm

uv run extract --model meta-llama/Llama-3.1-8B --seq_len 2048
uv run analyze --kv_dir results/kv_cache
uv run fit --kv_dir results/kv_cache --epochs 5000
uv run evaluate --siren_dir results/fits --kv_dir results/kv_cache
```

Or as modules:

```bash
uv run python -m nerf_attention.extract --model meta-llama/Llama-3.1-8B
uv run python -m nerf_attention.analyze
uv run python -m nerf_attention.fit --epochs 5000
uv run python -m nerf_attention.evaluate
```

## Hardware

- RTX 4060 (8GB VRAM) — sufficient for all experiments
- ~16GB system RAM
- ~10GB disk for model + cached tensors

## Key Metrics

- **Cosine Similarity**: Reconstruction quality per position (target: >0.95)
- **Compression Ratio**: Raw KV size / SIREN parameters (target: >10x at 2K+ tokens)
- **Forward Pass Latency**: SIREN inference vs HBM read time
