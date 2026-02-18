"""Experiment 3: SVD baseline comparison.

Truncated SVD at matched compression ratios to compare against SIREN.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from nerf_attention.types import KVMetadata


def run_svd_experiment(
    kv_dir: Path,
    base_dir: Path,
    target_compressions: list[float] | None = None,
) -> list[dict]:
    """Truncated SVD at matched compression ratios for comparison with SIREN."""
    kv_dir, base_dir = Path(kv_dir), Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    if target_compressions is None:
        target_compressions = [2.0, 4.0, 8.0, 16.0]

    with open(kv_dir / 'metadata.json') as f:
        metadata = KVMetadata.from_dict(json.load(f))

    layers_to_fit = sorted({0, metadata.num_layers // 2, metadata.num_layers - 1})
    all_results: list[dict] = []

    for layer_idx in layers_to_fit:
        filepath = kv_dir / f'layer_{layer_idx:02d}.pt'
        if not filepath.exists():
            continue
        data = torch.load(filepath, map_location='cpu', weights_only=True)

        for head_idx in range(min(metadata.num_kv_heads, 4)):
            for kv_type, tensor in [('key', data['keys'][head_idx]), ('value', data['values'][head_idx])]:
                seq_len, d_head = tensor.shape
                raw_bytes = seq_len * d_head * 4

                for target_cr in target_compressions:
                    # svd_bytes = (seq_len * rank + rank + rank * d_head) * 4
                    rank = max(1, int(raw_bytes / (target_cr * 4 * (seq_len + 1 + d_head))))
                    rank = min(rank, min(seq_len, d_head))

                    U, S, Vt = torch.linalg.svd(tensor, full_matrices=False)
                    reconstructed = U[:, :rank] @ torch.diag(S[:rank]) @ Vt[:rank, :]

                    svd_bytes = (seq_len * rank + rank + rank * d_head) * 4
                    cos_sim = F.cosine_similarity(reconstructed, tensor, dim=1)

                    all_results.append({
                        'name': f'L{layer_idx}_H{head_idx}_{kv_type}_svd_r{rank}',
                        'method': 'svd',
                        'layer': layer_idx,
                        'head': head_idx,
                        'kv_type': kv_type,
                        'rank': rank,
                        'target_compression': target_cr,
                        'actual_compression': float(raw_bytes / svd_bytes),
                        'final_cosine_mean': float(cos_sim.mean().item()),
                        'final_cosine_min': float(cos_sim.min().item()),
                        'final_cosine_std': float(cos_sim.std().item()),
                        'raw_size_bytes': raw_bytes,
                        'svd_size_bytes': svd_bytes,
                        'seq_len': seq_len,
                        'd_head': d_head,
                    })

                print(f"  L{layer_idx}_H{head_idx}_{kv_type}: "
                      + " | ".join(f"r{r['rank']}={r['final_cosine_mean']:.4f}@{r['actual_compression']:.1f}x"
                                   for r in all_results if r['name'].startswith(f'L{layer_idx}_H{head_idx}_{kv_type}')))

    with open(base_dir / 'svd_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    _print_summary(all_results, target_compressions)
    return all_results


def _print_summary(all_results: list[dict], target_compressions: list[float]) -> None:
    key_r = [r for r in all_results if r['kv_type'] == 'key']
    val_r = [r for r in all_results if r['kv_type'] == 'value']
    print(f"\nSVD Summary:")
    for tc in target_compressions:
        kr = [r for r in key_r if r['target_compression'] == tc]
        vr = [r for r in val_r if r['target_compression'] == tc]
        if kr:
            print(f"  {tc:.0f}x: keys CosSim={np.mean([r['final_cosine_mean'] for r in kr]):.4f}, "
                  f"values CosSim={np.mean([r['final_cosine_mean'] for r in vr]):.4f}")


def plot_siren_vs_svd(
    siren_results: list[dict],
    svd_results: list[dict],
    output_dir: Path,
) -> None:
    """Pareto frontier: SIREN points + SVD black diamonds."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from nerf_attention.evaluate import CONFIG_COLORS, CONFIG_MARKERS

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    ax = axes[0]
    for cn in sorted(set(r['config_name'] for r in siren_results)):
        cr = [r for r in siren_results if r['config_name'] == cn]
        ax.scatter(
            [r['compression_ratio'] for r in cr],
            [r['final_cosine_mean'] for r in cr],
            c=CONFIG_COLORS.get(cn, '#95a5a6'),
            marker=CONFIG_MARKERS.get(cn, 'o'),
            s=60, alpha=0.5, label=f'SIREN {cn}', edgecolors='black', linewidth=0.3,
        )
    ax.scatter(
        [r['actual_compression'] for r in svd_results],
        [r['final_cosine_mean'] for r in svd_results],
        c='black', marker='D', s=80, alpha=0.7, label='SVD', edgecolors='black', linewidth=0.5,
    )
    ax.axhline(y=0.95, color='green', linestyle='--', alpha=0.3)
    ax.axhline(y=0.90, color='orange', linestyle='--', alpha=0.3)
    ax.set_xscale('log')
    ax.set(xlabel='Compression Ratio (x)', ylabel='Cosine Similarity',
           title='SIREN vs SVD: Fidelity vs Compression')
    ax.legend(fontsize=7, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    siren_keys = [r for r in siren_results if r['kv_type'] == 'key']
    svd_keys = [r for r in svd_results if r['kv_type'] == 'key']
    svd_vals = [r for r in svd_results if r['kv_type'] == 'value']
    if siren_keys:
        ax.scatter([r['compression_ratio'] for r in siren_keys],
                   [r['final_cosine_mean'] for r in siren_keys],
                   c='#3498db', s=60, alpha=0.5, label='SIREN (keys)', edgecolors='black', linewidth=0.3)
    if svd_keys:
        ax.scatter([r['actual_compression'] for r in svd_keys],
                   [r['final_cosine_mean'] for r in svd_keys],
                   c='black', marker='D', s=80, alpha=0.7, label='SVD (keys)')
    if svd_vals:
        ax.scatter([r['actual_compression'] for r in svd_vals],
                   [r['final_cosine_mean'] for r in svd_vals],
                   c='red', marker='D', s=80, alpha=0.7, label='SVD (values)')
    ax.axhline(y=0.95, color='green', linestyle='--', alpha=0.3)
    ax.set_xscale('log')
    ax.set(xlabel='Compression Ratio (x)', ylabel='Cosine Similarity', title='Keys: SIREN vs SVD')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'siren_vs_svd.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/siren_vs_svd.png")
