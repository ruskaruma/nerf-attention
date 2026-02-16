"""Full pipeline on synthetic data. No model download needed."""

import argparse
from pathlib import Path

import torch

from nerf_attention import (
    extract_kv_cache_synthetic,
    analyze_kv_cache,
    fit_kv_cache,
    load_results,
    plot_pareto_frontier,
    generate_summary_figure,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', action='store_true', help='Force CPU mode')
    args = parser.parse_args()

    device = 'cpu' if args.cpu or not torch.cuda.is_available() else 'cuda'
    print(f"Device: {device}\n")

    kv_dir = Path('results/kv_cache_quick')
    analysis_dir = Path('results/analysis_quick')
    fits_dir = Path('results/fits_quick')
    figures_dir = Path('results/figures_quick')

    print("=" * 60)
    print("STEP 1: Generate synthetic KV cache")
    print("=" * 60)
    extract_kv_cache_synthetic(
        seq_len=512, num_layers=4, num_kv_heads=4, head_dim=128,
        output_dir=kv_dir,
    )

    print("\n" + "=" * 60)
    print("STEP 2: Analyze KV structure")
    print("=" * 60)
    analyze_kv_cache(kv_dir=kv_dir, output_dir=analysis_dir)

    print("\n" + "=" * 60)
    print("STEP 3: Fit SIRENs (quick mode)")
    print("=" * 60)
    fit_kv_cache(
        kv_dir=kv_dir, output_dir=fits_dir,
        epochs=2000, device=device, quick=True,
    )

    print("\n" + "=" * 60)
    print("STEP 4: Evaluate and plot")
    print("=" * 60)
    figures_dir.mkdir(parents=True, exist_ok=True)
    results = load_results(fits_dir)
    plot_pareto_frontier(results, figures_dir)
    generate_summary_figure(results, figures_dir)

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)
    print(f"\nResults in: {figures_dir}/")
    print(f"\nNext: Run on REAL Llama KV cache:")
    print(f"  uv run python -m nerf_attention.extract --model meta-llama/Llama-3.1-8B")
    print(f"  uv run python -m nerf_attention.analyze")
    print(f"  uv run python -m nerf_attention.fit --epochs 5000")
    print(f"  uv run python -m nerf_attention.evaluate")


if __name__ == '__main__':
    main()
