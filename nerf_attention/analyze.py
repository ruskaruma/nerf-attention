"""KV cache structure analysis — determines if SIRENs should work before fitting.

Measures autocorrelation, spectral energy concentration, and effective rank
per layer/head, separately for keys and values.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from nerf_attention.types import AnalysisResult, KVMetadata, LayerSummary


def _autocorrelation(signal: np.ndarray, max_lag: int = 50) -> np.ndarray:
    n = len(signal)
    signal = signal - signal.mean()
    var = (signal ** 2).sum()
    if var < 1e-10:
        return np.zeros(max_lag + 1)

    autocorr = np.zeros(max_lag + 1)
    for lag in range(min(max_lag + 1, n)):
        autocorr[lag] = (signal[:n - lag] * signal[lag:]).sum() / var
    return autocorr


def _spectral_energy(signal: np.ndarray) -> dict[str, float]:
    windowed = (signal - signal.mean()) * np.hanning(len(signal))
    spectrum = np.abs(np.fft.rfft(windowed))
    total = (spectrum ** 2).sum()
    if total < 1e-10:
        return {'top_5pct': 1.0, 'top_10pct': 1.0, 'top_25pct': 1.0, 'top_50pct': 1.0}

    n_freqs = len(spectrum)
    return {
        f'top_{int(pct*100)}pct': float((spectrum[:max(1, int(n_freqs * pct))] ** 2).sum() / total)
        for pct in [0.05, 0.10, 0.25, 0.50]
    }


def _effective_rank(matrix: torch.Tensor, threshold: float = 0.99) -> dict[str, float]:
    _, S, _ = torch.linalg.svd(matrix)
    total = S.sum()
    cumulative = torch.cumsum(S, dim=0)
    rank = (cumulative < threshold * total).sum().item() + 1
    return {
        'effective_rank_99': rank,
        'full_rank': len(S),
        'rank_ratio': rank / len(S),
        'top_sv_fraction': (S[0] / total).item(),
        'top_10_sv_fraction': (S[:10].sum() / total).item() if len(S) >= 10 else 1.0,
    }


def _analyze_tensor(tensor: torch.Tensor, name: str, max_lag: int = 50) -> dict:
    seq_len, d_head = tensor.shape
    dims_to_sample = min(d_head, 16)
    dim_indices = range(0, d_head, max(1, d_head // dims_to_sample))

    autocorrs = np.array([_autocorrelation(tensor[:, d].numpy(), max_lag) for d in dim_indices])
    mean_autocorr = autocorrs.mean(axis=0)
    lag1 = float(mean_autocorr[1]) if len(mean_autocorr) > 1 else 0.0

    all_ratios = [_spectral_energy(tensor[:, d].numpy()) for d in dim_indices]
    avg_energy = {k: float(np.mean([r[k] for r in all_ratios])) for k in all_ratios[0]}

    return {
        'name': name,
        'shape': list(tensor.shape),
        'lag1_autocorrelation': lag1,
        'mean_autocorrelation': mean_autocorr.tolist(),
        'spectral_energy': avg_energy,
        'rank': _effective_rank(tensor),
    }


def _select_layers(num_layers: int) -> list[int]:
    return sorted({0, num_layers // 4, num_layers // 2, 3 * num_layers // 4, num_layers - 1})


def _feasibility_label(val: float, good: float = 0.5, bad: float = 0.2) -> str:
    if val > good:
        return 'GOOD'
    if val > bad:
        return 'CONCERNING'
    return 'BAD'


def analyze_kv_cache(kv_dir: Path, output_dir: Path) -> AnalysisResult:
    """Run structure analysis across sampled layers and heads."""
    kv_dir, output_dir = Path(kv_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(kv_dir / 'metadata.json') as f:
        metadata = KVMetadata.from_dict(json.load(f))

    print(f"Analyzing KV cache: {metadata.num_layers} layers x {metadata.num_kv_heads} heads")
    print(f"Sequence length: {metadata.seq_len}, Head dim: {metadata.head_dim}")

    all_results: list[dict] = []
    layer_summaries: list[LayerSummary] = []
    layers_to_analyze = _select_layers(metadata.num_layers)

    for layer_idx in layers_to_analyze:
        filepath = kv_dir / f'layer_{layer_idx:02d}.pt'
        if not filepath.exists():
            print(f"  Skipping layer {layer_idx} (not found)")
            continue

        data = torch.load(filepath, map_location='cpu', weights_only=True)
        keys, values = data['keys'], data['values']

        ac_k, ac_v, en_k, en_v, rk_k, rk_v = [], [], [], [], [], []

        for head_idx in range(min(metadata.num_kv_heads, 4)):
            k_result = _analyze_tensor(keys[head_idx], f'L{layer_idx}_H{head_idx}_K')
            v_result = _analyze_tensor(values[head_idx], f'L{layer_idx}_H{head_idx}_V')
            all_results.extend([k_result, v_result])

            ac_k.append(k_result['lag1_autocorrelation'])
            ac_v.append(v_result['lag1_autocorrelation'])
            en_k.append(k_result['spectral_energy']['top_10pct'])
            en_v.append(v_result['spectral_energy']['top_10pct'])
            rk_k.append(k_result['rank']['rank_ratio'])
            rk_v.append(v_result['rank']['rank_ratio'])

        summary = LayerSummary(
            layer=layer_idx,
            avg_autocorr_k=float(np.mean(ac_k)),
            avg_autocorr_v=float(np.mean(ac_v)),
            avg_energy_10pct_k=float(np.mean(en_k)),
            avg_energy_10pct_v=float(np.mean(en_v)),
            avg_rank_ratio_k=float(np.mean(rk_k)),
            avg_rank_ratio_v=float(np.mean(rk_v)),
        )
        layer_summaries.append(summary)

        print(f"\n  Layer {layer_idx}:")
        print(f"    Keys   - Autocorr: {summary.avg_autocorr_k:.3f} | "
              f"Spectral: {summary.avg_energy_10pct_k:.3f} | "
              f"Rank: {summary.avg_rank_ratio_k:.3f}")
        print(f"    Values - Autocorr: {summary.avg_autocorr_v:.3f} | "
              f"Spectral: {summary.avg_energy_10pct_v:.3f} | "
              f"Rank: {summary.avg_rank_ratio_v:.3f}")

    avg_ac_k = float(np.mean([s.avg_autocorr_k for s in layer_summaries]))
    avg_ac_v = float(np.mean([s.avg_autocorr_v for s in layer_summaries]))
    avg_en_k = float(np.mean([s.avg_energy_10pct_k for s in layer_summaries]))
    avg_en_v = float(np.mean([s.avg_energy_10pct_v for s in layer_summaries]))

    print(f"\n{'='*60}")
    print("SIREN FEASIBILITY ASSESSMENT")
    print(f"{'='*60}")
    print(f"\nAutocorrelation (lag-1):")
    print(f"  Keys:   {avg_ac_k:.3f}  {_feasibility_label(avg_ac_k)} (>0.5)")
    print(f"  Values: {avg_ac_v:.3f}  {_feasibility_label(avg_ac_v)} (>0.5)")
    print(f"\nSpectral concentration (energy in lowest 10% frequencies):")
    print(f"  Keys:   {avg_en_k:.3f}  {_feasibility_label(avg_en_k)} (>0.5)")
    print(f"  Values: {avg_en_v:.3f}  {_feasibility_label(avg_en_v)} (>0.5)")

    if avg_ac_k > 0.5 and avg_en_k > 0.5:
        print(f"\nOverall prediction:")
        print(f"  PROMISING: KV cache has significant structure. SIREN should compress well.")
    elif avg_ac_k > 0.2 or avg_en_k > 0.3:
        print(f"\nOverall prediction:")
        print(f"  MIXED: Some structure. SIREN may work partially.")
    else:
        print(f"\nOverall prediction:")
        print(f"  CHALLENGING: Noisy/unstructured. Document why it fails.")

    _plot_analysis(all_results, layer_summaries, output_dir)

    result = AnalysisResult(
        metadata=metadata,
        layer_summaries=layer_summaries,
        avg_autocorr_keys=avg_ac_k,
        avg_autocorr_values=avg_ac_v,
        avg_spectral_keys=avg_en_k,
        avg_spectral_values=avg_en_v,
    )

    results_data = {
        'metadata': metadata.to_dict(),
        'layer_summaries': [
            {
                'layer': s.layer,
                'avg_autocorr_k': s.avg_autocorr_k,
                'avg_autocorr_v': s.avg_autocorr_v,
                'avg_energy_10pct_k': s.avg_energy_10pct_k,
                'avg_energy_10pct_v': s.avg_energy_10pct_v,
                'avg_rank_ratio_k': s.avg_rank_ratio_k,
                'avg_rank_ratio_v': s.avg_rank_ratio_v,
            }
            for s in layer_summaries
        ],
        'assessment': {
            'avg_autocorr_keys': avg_ac_k,
            'avg_autocorr_values': avg_ac_v,
            'avg_spectral_keys': avg_en_k,
            'avg_spectral_values': avg_en_v,
        },
    }
    with open(output_dir / 'analysis_results.json', 'w') as f:
        json.dump(results_data, f, indent=2)

    print(f"\nResults saved to {output_dir}/")
    return result


def _plot_analysis(
    all_results: list[dict],
    layer_summaries: list[LayerSummary],
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('KV Cache Structure Analysis: SIREN Feasibility', fontsize=14, fontweight='bold')

    layers = [s.layer for s in layer_summaries]

    ax = axes[0, 0]
    ax.plot(layers, [s.avg_autocorr_k for s in layer_summaries], 'bo-', label='Keys', markersize=8)
    ax.plot(layers, [s.avg_autocorr_v for s in layer_summaries], 'rs-', label='Values', markersize=8)
    ax.axhline(y=0.5, color='green', linestyle='--', alpha=0.5, label='Good threshold')
    ax.axhline(y=0.2, color='red', linestyle='--', alpha=0.5, label='Bad threshold')
    ax.set(xlabel='Layer Index', ylabel='Lag-1 Autocorrelation', title='Temporal Correlation by Layer')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(layers, [s.avg_energy_10pct_k for s in layer_summaries], 'bo-', label='Keys', markersize=8)
    ax.plot(layers, [s.avg_energy_10pct_v for s in layer_summaries], 'rs-', label='Values', markersize=8)
    ax.axhline(y=0.5, color='green', linestyle='--', alpha=0.5, label='Good threshold')
    ax.set(xlabel='Layer Index', ylabel='Energy in Low 10% Frequencies', title='Spectral Concentration by Layer')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    for result in all_results[:4]:
        ac = result['mean_autocorrelation']
        ax.plot(range(len(ac)), ac, label=result['name'], alpha=0.7)
    ax.set(xlabel='Lag (tokens)', ylabel='Autocorrelation', title='Autocorrelation Decay')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(layers, [s.avg_rank_ratio_k for s in layer_summaries], 'bo-', label='Keys', markersize=8)
    ax.plot(layers, [s.avg_rank_ratio_v for s in layer_summaries], 'rs-', label='Values', markersize=8)
    ax.set(xlabel='Layer Index', ylabel='Effective Rank / Full Rank', title='Effective Dimensionality by Layer')
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'kv_structure_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved plot: {output_dir}/kv_structure_analysis.png")


def main() -> None:
    parser = argparse.ArgumentParser(description='Analyze KV cache structure')
    parser.add_argument('--kv_dir', type=str, default='results/kv_cache')
    parser.add_argument('--output_dir', type=str, default='results/analysis')
    args = parser.parse_args()
    analyze_kv_cache(Path(args.kv_dir), Path(args.output_dir))


if __name__ == '__main__':
    main()
