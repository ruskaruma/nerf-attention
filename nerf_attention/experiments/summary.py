"""Final combined 6-panel summary figure."""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np


def generate_final_summary(
    scaling_results: dict[int, dict] | None,
    prompt_results: dict[str, dict] | None,
    siren_results: list[dict] | None,
    svd_results: list[dict] | None,
    output_dir: Path,
    head_dim: int = 128,
    layer_profile: list[dict] | None = None,
) -> None:
    """6-panel summary: Pareto, K/V boxplot, layer profile, scaling, multi-prompt, findings."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Try loading layer profile from disk if not passed
    if layer_profile is None:
        lp_path = Path('results/layer_profile/full_layer_profile.json')
        if lp_path.exists():
            layer_profile = json.loads(lp_path.read_text())

    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)
    fig.suptitle('NeRF-Attention: SIREN Compression of LLM KV Cache',
                 fontsize=16, fontweight='bold', y=1.02)

    # [0,0] SIREN vs SVD Pareto
    ax = fig.add_subplot(gs[0, 0])
    if siren_results:
        from nerf_attention.evaluate import CONFIG_COLORS, CONFIG_MARKERS
        for cn in sorted(set(r['config_name'] for r in siren_results)):
            cr = [r for r in siren_results if r['config_name'] == cn]
            ax.scatter([r['compression_ratio'] for r in cr],
                       [r['final_cosine_mean'] for r in cr],
                       c=CONFIG_COLORS.get(cn, '#95a5a6'),
                       marker=CONFIG_MARKERS.get(cn, 'o'),
                       s=40, alpha=0.6, label=f'SIREN {cn}',
                       edgecolors='black', linewidth=0.3)
    if svd_results:
        svd_k = [r for r in svd_results if r['kv_type'] == 'key']
        svd_v = [r for r in svd_results if r['kv_type'] == 'value']
        if svd_k:
            ax.scatter([r['actual_compression'] for r in svd_k],
                       [r['final_cosine_mean'] for r in svd_k],
                       c='black', marker='D', s=60, alpha=0.8, label='SVD keys', zorder=6)
        if svd_v:
            ax.scatter([r['actual_compression'] for r in svd_v],
                       [r['final_cosine_mean'] for r in svd_v],
                       c='gray', marker='D', s=60, alpha=0.8, label='SVD values', zorder=6)
    ax.axhline(y=0.95, color='green', linestyle='--', alpha=0.3)
    ax.axhline(y=0.90, color='orange', linestyle='--', alpha=0.3)
    ax.set_xscale('log')
    ax.set(xlabel='Compression Ratio (x)', ylabel='Cosine Similarity',
           title='SVD Dominates at Every Ratio')
    ax.legend(fontsize=6, ncol=2); ax.grid(True, alpha=0.2)

    # [0,1] Keys vs Values boxplot (from 280 baseline)
    ax = fig.add_subplot(gs[0, 1])
    if siren_results:
        medium = [r for r in siren_results if r.get('config_name') == 'medium']
        k_cos = [r['final_cosine_mean'] for r in medium if r['kv_type'] == 'key']
        v_cos = [r['final_cosine_mean'] for r in medium if r['kv_type'] == 'value']
        if k_cos and v_cos:
            bp = ax.boxplot([k_cos, v_cos], labels=['Keys', 'Values'],
                            patch_artist=True, widths=0.5)
            bp['boxes'][0].set_facecolor('#3498db')
            bp['boxes'][0].set_alpha(0.6)
            bp['boxes'][1].set_facecolor('#e74c3c')
            bp['boxes'][1].set_alpha(0.6)
            ax.set(ylabel='Cosine Similarity',
                   title=f'K/V Asymmetry (Keys={np.mean(k_cos):.3f}, Values={np.mean(v_cos):.3f})')
            ax.grid(True, alpha=0.2, axis='y')
    if not ax.has_data():
        ax.text(0.5, 0.5, 'No baseline data', ha='center', va='center', transform=ax.transAxes)

    # [0,2] Full 32-layer profile
    ax = fig.add_subplot(gs[0, 2])
    if layer_profile:
        lp_keys = [r for r in layer_profile if r['kv_type'] == 'key']
        lp_vals = [r for r in layer_profile if r['kv_type'] == 'value']
        k_layers = [r['layer'] for r in lp_keys]
        k_cos = [r['final_cosine_mean'] for r in lp_keys]
        v_layers = [r['layer'] for r in lp_vals]
        v_cos = [r['final_cosine_mean'] for r in lp_vals]
        ax.plot(k_layers, k_cos, 'o-', color='#3498db', label='Keys', markersize=4, linewidth=1.2)
        ax.plot(v_layers, v_cos, 's-', color='#e74c3c', label='Values', markersize=4, linewidth=1.2)
        ax.fill_between(k_layers, k_cos, v_cos, alpha=0.08, color='gray')
        # Mark key dips (local minima)
        k_arr = np.array(k_cos)
        for i in range(1, len(k_arr) - 1):
            if k_arr[i] < k_arr[i-1] and k_arr[i] < k_arr[i+1]:
                ax.annotate(f'L{k_layers[i]}', xy=(k_layers[i], k_arr[i]), fontsize=7, color='#3498db',
                            xytext=(k_layers[i] + 1, k_arr[i] - 0.02),
                            arrowprops=dict(arrowstyle='->', color='#3498db', alpha=0.6, lw=0.8))
        ax.set(xlabel='Layer', ylabel='CosSim',
               title='32-Layer Profile: Non-Monotonic Structure')
        ax.set_ylim(0.4, 1.0)
        ax.legend(fontsize=7); ax.grid(True, alpha=0.2)
    else:
        ax.text(0.5, 0.5, 'No layer profile data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Layer Profile')

    # [1,0] Scaling: quality vs sequence length
    ax = fig.add_subplot(gs[1, 0])
    if scaling_results:
        seq_lens = sorted(scaling_results.keys())
        ax.plot(seq_lens, [scaling_results[s]['avg_cossim_keys'] for s in seq_lens],
                'o-', color='#3498db', label='Keys', markersize=6, linewidth=1.5)
        ax.plot(seq_lens, [scaling_results[s]['avg_cossim_values'] for s in seq_lens],
                's-', color='#e74c3c', label='Values', markersize=6, linewidth=1.5)
        ax.axhline(y=0.95, color='green', linestyle='--', alpha=0.3)
        ax.set_xscale('log')
        ax.set(xlabel='Sequence Length', ylabel='CosSim',
               title='Quality Degrades with Length')
        ax.legend(fontsize=7); ax.grid(True, alpha=0.2)
    else:
        ax.text(0.5, 0.5, 'No scaling data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Scaling Quality')

    # [1,1] Multi-prompt grouped bars
    ax = fig.add_subplot(gs[1, 1])
    if prompt_results:
        names = list(prompt_results.keys())
        x = np.arange(len(names))
        width = 0.35
        ax.bar(x - width/2, [prompt_results[n]['avg_cossim_keys'] for n in names],
               width, label='Keys', color='#3498db', alpha=0.8,
               edgecolor='black', linewidth=0.5)
        ax.bar(x + width/2, [prompt_results[n]['avg_cossim_values'] for n in names],
               width, label='Values', color='#e74c3c', alpha=0.8,
               edgecolor='black', linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([n.capitalize() for n in names], fontsize=8)
        k_spread = max(prompt_results[n]['avg_cossim_keys'] for n in names) - \
                   min(prompt_results[n]['avg_cossim_keys'] for n in names)
        ax.set(ylabel='CosSim',
               title=f'Content Invariant (keys spread={k_spread:.3f})')
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7); ax.grid(True, alpha=0.2, axis='y')
    else:
        ax.text(0.5, 0.5, 'No prompt data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Multi-Prompt')

    # [1,2] Key findings text
    ax = fig.add_subplot(gs[1, 2])
    ax.axis('off')

    total_experiments = len(siren_results or [])
    if layer_profile:
        total_experiments += len(layer_profile)
    if svd_results:
        total_experiments += len(svd_results)
    if scaling_results:
        total_experiments += sum(s.get('num_experiments', 0) for s in scaling_results.values())

    findings = ["NeRF-Attention: Key Findings", "=" * 32, ""]

    if siren_results:
        k_avg = np.mean([r['final_cosine_mean'] for r in siren_results if r['kv_type'] == 'key'])
        v_avg = np.mean([r['final_cosine_mean'] for r in siren_results if r['kv_type'] == 'value'])
        findings += [f"Keys avg:   {k_avg:.4f} CosSim",
                     f"Values avg: {v_avg:.4f} CosSim",
                     f"K/V gap: architectural, not content", ""]

    if svd_results and siren_results:
        svd_k2 = [r for r in svd_results if r['kv_type'] == 'key' and r.get('target_compression') == 2]
        siren_k = [r for r in siren_results if r['kv_type'] == 'key' and r.get('config_name') == 'medium']
        if svd_k2 and siren_k:
            svd_q = np.mean([r['final_cosine_mean'] for r in svd_k2])
            sir_q = np.mean([r['final_cosine_mean'] for r in siren_k])
            sir_ratio = np.mean([r['compression_ratio'] for r in siren_k])
            ratio_label = f"{sir_ratio:.1f}x"
            if sir_ratio < 1.0:
                ratio_label += " = expansion"
            findings += [f"SVD 2x keys: {svd_q:.2f} CosSim",
                         f"SIREN keys:  {sir_q:.2f} ({ratio_label})",
                         "  SVD wins with zero training", ""]

    if prompt_results:
        k_vals = [prompt_results[n]['avg_cossim_keys'] for n in prompt_results]
        findings += [f"Cross-content spread: {max(k_vals)-min(k_vals):.3f}",
                     "  Structure is architectural", ""]

    if scaling_results:
        seq_lens = sorted(scaling_results.keys())
        ratios = [scaling_results[s]['siren_time_ms'] / scaling_results[s]['hbm_4060_ms']
                  for s in seq_lens]
        findings += ["Both latencies scale with seq len",
                     f"  SIREN {min(ratios):.0f}-{max(ratios):.0f}x slower than HBM",
                     "  No crossover at practical lengths", ""]

    findings += [f"Total experiments: {total_experiments}",
                 "Conclusion: negative result,",
                 "  characterization contribution"]

    ax.text(0.05, 0.95, '\n'.join(findings), transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.savefig(output_dir / 'final_summary.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/final_summary.png")
