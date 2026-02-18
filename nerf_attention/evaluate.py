"""Evaluation and publication-quality figure generation."""

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import torch

from nerf_attention.siren import SIREN
from nerf_attention.types import SIRENConfig


CONFIG_COLORS = {
    'tiny': '#e74c3c', 'small': '#e67e22', 'medium': '#2ecc71',
    'large': '#3498db', 'deep': '#9b59b6', 'hifreq': '#1abc9c', 'lofreq': '#f1c40f',
}
CONFIG_MARKERS = {
    'tiny': 'v', 'small': 's', 'medium': 'o',
    'large': 'D', 'deep': '^', 'hifreq': 'P', 'lofreq': 'X',
}


def load_results(siren_dir: Path) -> list[dict]:
    with open(Path(siren_dir) / 'fit_results.json') as f:
        return json.load(f)


def _load_model_from_checkpoint(checkpoint: dict, device: str) -> SIREN:
    cfg = checkpoint['config']
    config = SIRENConfig(
        hidden_features=cfg['hidden_features'],
        hidden_layers=cfg['hidden_layers'],
        omega_0=cfg['omega_0'],
        name=cfg.get('name', 'medium'),
    )
    model = SIREN(config, out_features=cfg['out_features']).to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    return model


def plot_pareto_frontier(
    results: list[dict], output_dir: Path, svd_results: list[dict] | None = None,
) -> None:
    output_dir = Path(output_dir)
    fig, ax = plt.subplots(figsize=(10, 7))

    for cn in sorted(set(r['config_name'] for r in results)):
        cr = [r for r in results if r['config_name'] == cn]
        ax.scatter(
            [r['compression_ratio'] for r in cr],
            [r['final_cosine_mean'] for r in cr],
            c=CONFIG_COLORS.get(cn, '#95a5a6'),
            marker=CONFIG_MARKERS.get(cn, 'o'),
            s=80, alpha=0.7, label=f'SIREN {cn}', edgecolors='black', linewidth=0.5,
        )

    if svd_results:
        svd_keys = [r for r in svd_results if r['kv_type'] == 'key']
        svd_vals = [r for r in svd_results if r['kv_type'] == 'value']
        if svd_keys:
            ax.scatter([r['actual_compression'] for r in svd_keys],
                       [r['final_cosine_mean'] for r in svd_keys],
                       c='black', marker='D', s=100, alpha=0.8, label='SVD (keys)',
                       edgecolors='black', linewidth=0.5, zorder=6)
        if svd_vals:
            ax.scatter([r['actual_compression'] for r in svd_vals],
                       [r['final_cosine_mean'] for r in svd_vals],
                       c='gray', marker='D', s=100, alpha=0.8, label='SVD (values)',
                       edgecolors='black', linewidth=0.5, zorder=6)

    ax.axhline(y=0.95, color='green', linestyle='--', alpha=0.4, label='0.95 target')
    ax.axhline(y=0.90, color='orange', linestyle='--', alpha=0.4, label='0.90 minimum')
    ax.set(xlabel='Compression Ratio (x)', ylabel='Cosine Similarity',
           title='SIREN vs SVD: Compression-Fidelity Tradeoff')
    ax.set_xscale('log')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'pareto_frontier.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: pareto_frontier.png")


def plot_keys_vs_values(results: list[dict], output_dir: Path) -> None:
    output_dir = Path(output_dir)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    medium = [r for r in results if r['config_name'] == 'medium']
    keys = [r for r in medium if r['kv_type'] == 'key']
    vals = [r for r in medium if r['kv_type'] == 'value']

    ax = axes[0]
    if keys:
        ax.scatter([r['layer'] for r in keys], [r['final_cosine_mean'] for r in keys],
                   c='blue', marker='o', s=60, label='Keys', alpha=0.7)
    if vals:
        ax.scatter([r['layer'] for r in vals], [r['final_cosine_mean'] for r in vals],
                   c='red', marker='s', s=60, label='Values', alpha=0.7)
    ax.set(xlabel='Layer Index', ylabel='Cosine Similarity',
           title='Reconstruction Quality by Layer (Medium SIREN)')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    if keys and vals:
        ax.hist([r['final_cosine_mean'] for r in keys], bins=15, alpha=0.5, label='Keys', color='blue')
        ax.hist([r['final_cosine_mean'] for r in vals], bins=15, alpha=0.5, label='Values', color='red')
    ax.set(xlabel='Cosine Similarity', ylabel='Count', title='Distribution of Reconstruction Quality')
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'keys_vs_values.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: keys_vs_values.png")


def plot_per_position_error(
    siren_dir: Path, kv_dir: Path, output_dir: Path, device: str = 'cpu',
) -> None:
    siren_dir, kv_dir, output_dir = Path(siren_dir), Path(kv_dir), Path(output_dir)
    model_files = sorted(siren_dir.glob('*medium_model.pt'))
    if not model_files:
        print("  No medium models found, skipping per-position plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Per-Position Reconstruction Error', fontsize=14)

    for idx, model_file in enumerate(model_files[:4]):
        ax = axes[idx // 2, idx % 2]
        checkpoint = torch.load(model_file, map_location=device, weights_only=True)

        model = _load_model_from_checkpoint(checkpoint, device)
        metrics = checkpoint['metrics']

        kv_data = torch.load(
            kv_dir / f"layer_{metrics['layer']:02d}.pt",
            map_location=device, weights_only=True,
        )
        original = kv_data['keys' if metrics['kv_type'] == 'key' else 'values'][metrics['head']]

        seq_len = original.shape[0]
        positions = torch.linspace(0, 1, seq_len).unsqueeze(1).to(device)

        with torch.no_grad():
            pred = model(positions) * checkpoint['target_std'].to(device) + checkpoint['target_mean'].to(device)
            per_pos_cos = torch.nn.functional.cosine_similarity(pred, original, dim=1).cpu().numpy()

        ax.plot(range(seq_len), per_pos_cos, alpha=0.5, linewidth=0.5)
        window = min(50, seq_len // 10)
        if window > 1:
            rolling = np.convolve(per_pos_cos, np.ones(window) / window, mode='valid')
            ax.plot(range(window // 2, window // 2 + len(rolling)), rolling,
                    color='red', linewidth=2, label=f'Rolling avg (w={window})')

        ax.set(xlabel='Token Position', ylabel='Cosine Similarity',
               title=f"L{metrics['layer']} H{metrics['head']} {metrics['kv_type']}")
        ax.set_ylim(bottom=max(0, per_pos_cos.min() - 0.05))
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'per_position_error.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: per_position_error.png")


def profile_latency(siren_dir: Path, output_dir: Path, device: str = 'cuda') -> None:
    """SIREN forward pass vs theoretical HBM read time (RTX 4060 / H100)."""
    siren_dir, output_dir = Path(siren_dir), Path(output_dir)
    model_files = sorted(siren_dir.glob('*_model.pt'))
    if not model_files:
        print("  No models found for latency profiling")
        return

    results = []
    for model_file in model_files[:8]:
        checkpoint = torch.load(model_file, map_location=device, weights_only=True)
        metrics = checkpoint['metrics']

        model = _load_model_from_checkpoint(checkpoint, device)
        positions = torch.linspace(0, 1, metrics['seq_len']).unsqueeze(1).to(device)

        # Warmup
        for _ in range(10):
            with torch.no_grad():
                model(positions)
        if device == 'cuda':
            torch.cuda.synchronize()

        num_runs = 100
        start = time.perf_counter()
        for _ in range(num_runs):
            with torch.no_grad():
                model(positions)
        if device == 'cuda':
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / num_runs

        raw_bytes = metrics['raw_size_bytes']
        result = {
            'name': metrics['name'],
            'config': metrics['config_name'],
            'siren_time_ms': elapsed * 1000,
            'hbm_time_4060_ms': raw_bytes / 272e9 * 1000,
            'hbm_time_h100_ms': raw_bytes / 3350e9 * 1000,
            'speedup_vs_4060': (raw_bytes / 272e9) / max(elapsed, 1e-10),
            'speedup_vs_h100': (raw_bytes / 3350e9) / max(elapsed, 1e-10),
            'num_params': sum(p.numel() for p in model.parameters()),
        }
        results.append(result)
        print(f"  {metrics['name']}: SIREN={elapsed*1000:.3f}ms | "
              f"HBM(4060)={result['hbm_time_4060_ms']:.3f}ms | "
              f"HBM(H100)={result['hbm_time_h100_ms']:.3f}ms")

    if results:
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(results))
        width = 0.25
        ax.bar(x - width, [r['siren_time_ms'] for r in results], width,
               label='SIREN Forward', color='#3498db')
        ax.bar(x, [r['hbm_time_4060_ms'] for r in results], width,
               label='HBM (RTX 4060)', color='#e74c3c')
        ax.bar(x + width, [r['hbm_time_h100_ms'] for r in results], width,
               label='HBM (H100)', color='#2ecc71')
        ax.set(ylabel='Time (ms)', title='SIREN Inference vs Memory Read Latency')
        ax.set_xticks(x)
        ax.set_xticklabels([r['name'] for r in results], rotation=45, ha='right', fontsize=8)
        ax.set_yscale('log')
        ax.legend(); ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(output_dir / 'latency_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: latency_comparison.png")

    with open(output_dir / 'latency_results.json', 'w') as f:
        json.dump(results, f, indent=2)


def generate_summary_figure(results: list[dict], output_dir: Path) -> None:
    """6-panel figure combining key results."""
    output_dir = Path(output_dir)
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)
    fig.suptitle('NeRF-Attention: SIREN Compression of LLM KV Cache',
                 fontsize=16, fontweight='bold', y=1.02)

    # Pareto
    ax = fig.add_subplot(gs[0, 0])
    for cn in sorted(set(r['config_name'] for r in results)):
        cr = [r for r in results if r['config_name'] == cn]
        ax.scatter([r['compression_ratio'] for r in cr],
                   [r['final_cosine_mean'] for r in cr], s=40, alpha=0.7, label=cn)
    ax.set(xlabel='Compression Ratio (x)', ylabel='Cosine Similarity', title='Compression vs Fidelity')
    ax.axhline(y=0.95, color='green', linestyle='--', alpha=0.3)
    ax.legend(fontsize=7); ax.grid(True, alpha=0.2)

    # Keys vs Values
    ax = fig.add_subplot(gs[0, 1])
    medium = [r for r in results if r['config_name'] == 'medium']
    k_cos = [r['final_cosine_mean'] for r in medium if r['kv_type'] == 'key']
    v_cos = [r['final_cosine_mean'] for r in medium if r['kv_type'] == 'value']
    if k_cos and v_cos:
        ax.boxplot([k_cos, v_cos], labels=['Keys', 'Values'])
        ax.set(ylabel='Cosine Similarity', title='Keys vs Values')
        ax.grid(True, alpha=0.2)

    # Layer variation
    ax = fig.add_subplot(gs[0, 2])
    layer_data: dict[int, list[float]] = {}
    for r in medium:
        layer_data.setdefault(r['layer'], []).append(r['final_cosine_mean'])
    if layer_data:
        ls = sorted(layer_data.keys())
        ax.errorbar(ls, [np.mean(layer_data[l]) for l in ls],
                    yerr=[np.std(layer_data[l]) for l in ls], fmt='o-', capsize=3)
        ax.set(xlabel='Layer Index', ylabel='Avg Cosine Similarity', title='Compressibility by Layer')
        ax.grid(True, alpha=0.2)

    # K/V split per architecture
    ax = fig.add_subplot(gs[1, 0])
    config_k: dict[str, list[float]] = {}
    config_v: dict[str, list[float]] = {}
    for r in results:
        cn = r['config_name']
        if r['kv_type'] == 'key':
            config_k.setdefault(cn, []).append(r['final_cosine_mean'])
        else:
            config_v.setdefault(cn, []).append(r['final_cosine_mean'])
    cfgs = sorted(set(list(config_k.keys()) + list(config_v.keys())))
    if cfgs:
        x = np.arange(len(cfgs))
        width = 0.35
        ax.bar(x - width/2, [np.mean(config_k.get(c, [0])) for c in cfgs],
               width, label='Keys', color='#3498db', alpha=0.8)
        ax.bar(x + width/2, [np.mean(config_v.get(c, [0])) for c in cfgs],
               width, label='Values', color='#e74c3c', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(cfgs, fontsize=7, rotation=45, ha='right')
        ax.set(ylabel='Avg CosSim', title='K/V Gap by Architecture')
        ax.legend(fontsize=7); ax.grid(True, alpha=0.2, axis='y')

    # Architecture comparison
    ax = fig.add_subplot(gs[1, 1])
    config_stats: dict[str, list[float]] = {}
    for r in results:
        config_stats.setdefault(r['config_name'], []).append(r['final_cosine_mean'])
    configs = sorted(config_stats.keys())
    ax.barh(range(len(configs)), [np.mean(config_stats[c]) for c in configs],
            color='steelblue', alpha=0.7)
    ax.set_yticks(range(len(configs)))
    ax.set_yticklabels(configs)
    ax.set(xlabel='Avg Cosine Similarity', title='Architecture Comparison')
    ax.grid(True, alpha=0.2, axis='x')

    # Key findings text
    ax = fig.add_subplot(gs[1, 2])
    ax.axis('off')
    best = max(results, key=lambda r: r['final_cosine_mean'])
    best_compress = max(results, key=lambda r: r['compression_ratio']
                        if r['final_cosine_mean'] > 0.9 else 0)
    text = (f"Key Findings\n{'─'*30}\n\n"
            f"Best fidelity:\n  CosSim={best['final_cosine_mean']:.4f}\n"
            f"  {best['config_name']}, {best['compression_ratio']:.1f}x\n\n"
            f"Best compression (>0.9):\n  {best_compress['compression_ratio']:.1f}x\n"
            f"  CosSim={best_compress['final_cosine_mean']:.4f}\n\n"
            f"Experiments: {len(results)}")
    ax.text(0.1, 0.9, text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.savefig(output_dir / 'summary_figure.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: summary_figure.png")


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate SIREN compression')
    parser.add_argument('--kv_dir', type=str, default='results/kv_cache')
    parser.add_argument('--siren_dir', type=str, default='results/fits')
    parser.add_argument('--output_dir', type=str, default='results/figures')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'

    print("Loading results...")
    results = load_results(Path(args.siren_dir))

    print("\nGenerating plots...")
    plot_pareto_frontier(results, output_dir)
    plot_keys_vs_values(results, output_dir)
    plot_per_position_error(Path(args.siren_dir), Path(args.kv_dir), output_dir, device=args.device)
    generate_summary_figure(results, output_dir)

    print("\nProfiling latency...")
    profile_latency(Path(args.siren_dir), output_dir, device=args.device)

    print(f"\nAll figures saved to {output_dir}/")


if __name__ == '__main__':
    main()
