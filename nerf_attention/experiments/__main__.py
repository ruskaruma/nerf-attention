"""CLI for running follow-up experiments.

Usage:
    uv run python -m nerf_attention.experiments scaling
    uv run python -m nerf_attention.experiments multi_prompt
    uv run python -m nerf_attention.experiments svd
    uv run python -m nerf_attention.experiments layer_profile
    uv run python -m nerf_attention.experiments all
"""

import argparse
import json
from pathlib import Path

import torch

from nerf_attention.experiments.scaling import (
    run_scaling_experiment, plot_scaling_crossover, plot_scaling_quality,
    run_full_layer_profile, plot_full_layer_profile,
)
from nerf_attention.experiments.multi_prompt import run_multi_prompt_experiment, plot_multi_prompt
from nerf_attention.experiments.svd import run_svd_experiment, plot_siren_vs_svd
from nerf_attention.experiments.summary import generate_final_summary


def main() -> None:
    parser = argparse.ArgumentParser(description='Run follow-up experiments')
    parser.add_argument('experiment', choices=['scaling', 'multi_prompt', 'svd', 'layer_profile', 'all'])
    parser.add_argument('--model', type=str, default='unsloth/Llama-3.1-8B')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--kv_dir', type=str, default='results/kv_cache')
    parser.add_argument('--siren_dir', type=str, default='results/fits')
    args = parser.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'

    figures_dir = Path('results/figures')
    figures_dir.mkdir(parents=True, exist_ok=True)

    if args.experiment in ('scaling', 'all'):
        print("\n" + "=" * 60)
        print("EXPERIMENT 1: Sequence Length Scaling")
        print("=" * 60)
        scaling = run_scaling_experiment(
            model_name=args.model,
            seq_lengths=[512, 1024, 2048, 4096, 8192],
            base_dir=Path('results/scaling'),
            device=args.device,
            epochs=args.epochs,
        )
        plot_scaling_crossover(scaling, figures_dir)
        plot_scaling_quality(scaling, figures_dir)

    if args.experiment in ('multi_prompt', 'all'):
        print("\n" + "=" * 60)
        print("EXPERIMENT 2: Multi-Prompt Robustness")
        print("=" * 60)
        prompts = run_multi_prompt_experiment(
            model_name=args.model,
            base_dir=Path('results/multi_prompt'),
            device=args.device,
            epochs=args.epochs,
        )
        plot_multi_prompt(prompts, figures_dir)

    if args.experiment in ('svd', 'all'):
        print("\n" + "=" * 60)
        print("EXPERIMENT 3: SVD Baseline Comparison")
        print("=" * 60)
        svd = run_svd_experiment(
            kv_dir=Path(args.kv_dir),
            base_dir=Path('results/svd'),
        )
        siren = json.loads((Path(args.siren_dir) / 'fit_results.json').read_text())
        plot_siren_vs_svd(siren, svd, figures_dir)

    if args.experiment in ('layer_profile', 'all'):
        print("\n" + "=" * 60)
        print("EXPERIMENT 4: Full Layer Profile (32 layers)")
        print("=" * 60)
        layer_results = run_full_layer_profile(
            kv_dir=Path(args.kv_dir),
            output_dir=Path('results/layer_profile'),
            device=args.device,
            epochs=args.epochs,
        )
        plot_full_layer_profile(layer_results, figures_dir)

    if args.experiment == 'all':
        print("\n" + "=" * 60)
        print("FINAL SUMMARY")
        print("=" * 60)
        scaling_data = json.loads((Path('results/scaling') / 'scaling_results.json').read_text())
        scaling_data = {int(k): v for k, v in scaling_data.items()}
        prompt_data = json.loads((Path('results/multi_prompt') / 'multi_prompt_results.json').read_text())
        svd_data = json.loads((Path('results/svd') / 'svd_results.json').read_text())
        siren_data = json.loads((Path(args.siren_dir) / 'fit_results.json').read_text())
        generate_final_summary(scaling_data, prompt_data, siren_data, svd_data, figures_dir)


if __name__ == '__main__':
    main()
