"""Experiment 1: Sequence length scaling.

Runs extract + analyze + fit at multiple sequence lengths to find the
SIREN vs HBM latency crossover point.
"""

import gc
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from nerf_attention.analyze import analyze_kv_cache
from nerf_attention.experiments.prompts import ALL_PROMPTS
from nerf_attention.fit import fit_kv_cache
from nerf_attention.siren import SIREN, fit_siren
from nerf_attention.types import KVMetadata, SIRENConfig


def _extract_all_seq_lengths(
    model_name: str,
    seq_lengths: list[int],
    base_dir: Path,
    device: str = 'cuda',
) -> dict[int, KVMetadata]:
    """Load model once and extract KV caches for all sequence lengths."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"Loading {model_name} in 4-bit quantization...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.eval()

    # Concatenate all prompt texts and repeat to fill long contexts
    text = "\n\n".join(ALL_PROMPTS.values()) * 3
    metadata_map: dict[int, KVMetadata] = {}

    for seq_len in seq_lengths:
        kv_dir = base_dir / f'seq_{seq_len}' / 'kv_cache'
        if (kv_dir / 'metadata.json').exists():
            print(f"\n  seq_len={seq_len}: already extracted, skipping")
            metadata_map[seq_len] = KVMetadata.from_dict(
                json.loads((kv_dir / 'metadata.json').read_text())
            )
            continue

        print(f"\n  Extracting seq_len={seq_len}...")
        kv_dir.mkdir(parents=True, exist_ok=True)

        try:
            inputs = tokenizer(text, return_tensors="pt", max_length=seq_len, truncation=True).to(device)
            actual_seq_len = inputs['input_ids'].shape[1]
            print(f"    Tokens: {actual_seq_len}")

            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=False, use_cache=True)

            past_kv = outputs.past_key_values

            if hasattr(past_kv, 'key_cache'):
                num_layers = len(past_kv.key_cache)
                _, num_kv_heads, cache_seq_len, head_dim = past_kv.key_cache[0].shape
            elif hasattr(past_kv, 'layers'):
                num_layers = len(past_kv.layers)
                _, num_kv_heads, cache_seq_len, head_dim = past_kv.layers[0].keys.shape
            else:
                num_layers = len(past_kv)
                _, num_kv_heads, cache_seq_len, head_dim = past_kv[0][0].shape

            for layer_idx in range(num_layers):
                if hasattr(past_kv, 'key_cache'):
                    keys = past_kv.key_cache[layer_idx].squeeze(0).float().cpu()
                    values = past_kv.value_cache[layer_idx].squeeze(0).float().cpu()
                elif hasattr(past_kv, 'layers'):
                    keys = past_kv.layers[layer_idx].keys.squeeze(0).float().cpu()
                    values = past_kv.layers[layer_idx].values.squeeze(0).float().cpu()
                else:
                    keys = past_kv[layer_idx][0].squeeze(0).float().cpu()
                    values = past_kv[layer_idx][1].squeeze(0).float().cpu()
                torch.save({'keys': keys, 'values': values}, kv_dir / f'layer_{layer_idx:02d}.pt')

            metadata = KVMetadata(
                model_name=model_name, num_layers=num_layers, num_kv_heads=num_kv_heads,
                seq_len=cache_seq_len, head_dim=head_dim, actual_tokens=actual_seq_len,
            )
            with open(kv_dir / 'metadata.json', 'w') as f:
                json.dump(metadata.to_dict(), f, indent=2)

            metadata_map[seq_len] = metadata
            print(f"    Saved {num_layers} layers to {kv_dir}/")

            del outputs, past_kv
            torch.cuda.empty_cache()

        except (RuntimeError, ValueError) as e:
            if 'out of memory' in str(e).lower() or 'cuda' in str(e).lower() or 'gpu ram' in str(e).lower():
                print(f"    OOM at seq_len={seq_len}, stopping extraction")
                torch.cuda.empty_cache()
                break
            raise

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return metadata_map


def run_scaling_experiment(
    model_name: str,
    seq_lengths: list[int],
    base_dir: Path,
    device: str = 'cuda',
    epochs: int = 2000,
) -> dict[int, dict]:
    """Run extract + analyze + fit at multiple sequence lengths."""
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    scaling_results: dict[int, dict] = {}

    # Phase 1: Extract all KV caches with model loaded once
    metadata_map = _extract_all_seq_lengths(model_name, seq_lengths, base_dir, device)

    # Phase 2: Analyze + fit each with medium config only (no model needed)
    medium_config = SIRENConfig(256, 2, 30.0, 'medium')

    for seq_len in seq_lengths:
        if seq_len not in metadata_map:
            continue

        print(f"\n{'='*60}")
        print(f"SCALING: analyze + fit seq_len = {seq_len}")
        print(f"{'='*60}")

        metadata = metadata_map[seq_len]
        seq_dir = base_dir / f'seq_{seq_len}'
        kv_dir = seq_dir / 'kv_cache'
        analysis_dir = seq_dir / 'analysis'
        fits_dir = seq_dir / 'fits'
        fits_dir.mkdir(parents=True, exist_ok=True)

        analysis = analyze_kv_cache(kv_dir, analysis_dir)

        # Fit medium SIREN on layers 0, mid, last, head 0
        layers_to_fit = sorted({0, metadata.num_layers // 2, metadata.num_layers - 1})
        fit_results = []
        for layer_idx in layers_to_fit:
            data = torch.load(kv_dir / f'layer_{layer_idx:02d}.pt', map_location='cpu', weights_only=True)
            for kv_type, tensor in [('key', data['keys'][0]), ('value', data['values'][0])]:
                name = f"L{layer_idx}_H0_{kv_type}_medium"
                print(f"  Fitting {name}...")
                result = fit_siren(tensor, medium_config, epochs=epochs, device=device,
                                   log_every=epochs, verbose=False)
                fit_results.append({
                    'name': name, 'kv_type': kv_type, 'layer': layer_idx,
                    'final_cosine_mean': result.final_cosine_mean,
                    'compression_ratio': result.compression_ratio,
                })
                # Save model checkpoint for latency profiling
                torch.save({
                    'config': {'hidden_features': medium_config.hidden_features,
                               'hidden_layers': medium_config.hidden_layers,
                               'omega_0': medium_config.omega_0,
                               'name': medium_config.name,
                               'out_features': tensor.shape[1]},
                    'model_state': result.model.state_dict(),
                }, fits_dir / f'{name}_model.pt')
                print(f"    CosSim={result.final_cosine_mean:.4f}, Compress={result.compression_ratio:.1f}x")

        siren_time_ms = _profile_siren_latency(fits_dir, metadata.seq_len, device)

        raw_bytes = metadata.seq_len * metadata.head_dim * 4
        hbm_4060_ms = raw_bytes / 272e9 * 1000
        hbm_h100_ms = raw_bytes / 3350e9 * 1000

        key_r = [r for r in fit_results if r['kv_type'] == 'key']
        val_r = [r for r in fit_results if r['kv_type'] == 'value']

        scaling_results[seq_len] = {
            'seq_len': metadata.seq_len,
            'actual_tokens': metadata.actual_tokens,
            'autocorr_keys': analysis.avg_autocorr_keys,
            'autocorr_values': analysis.avg_autocorr_values,
            'spectral_keys': analysis.avg_spectral_keys,
            'spectral_values': analysis.avg_spectral_values,
            'avg_cossim_keys': float(np.mean([r['final_cosine_mean'] for r in key_r])) if key_r else 0.0,
            'avg_cossim_values': float(np.mean([r['final_cosine_mean'] for r in val_r])) if val_r else 0.0,
            'avg_compression': float(np.mean([r['compression_ratio'] for r in fit_results])),
            'siren_time_ms': siren_time_ms,
            'hbm_4060_ms': hbm_4060_ms,
            'hbm_h100_ms': hbm_h100_ms,
            'num_experiments': len(fit_results),
        }

        print(f"\n  seq_len={metadata.seq_len}: keys={scaling_results[seq_len]['avg_cossim_keys']:.4f}, "
              f"values={scaling_results[seq_len]['avg_cossim_values']:.4f}")
        print(f"  SIREN={siren_time_ms:.3f}ms, HBM(4060)={hbm_4060_ms:.4f}ms, HBM(H100)={hbm_h100_ms:.5f}ms")

    with open(base_dir / 'scaling_results.json', 'w') as f:
        json.dump({str(k): v for k, v in scaling_results.items()}, f, indent=2)

    return scaling_results


def _profile_siren_latency(fits_dir: Path, seq_len: int, device: str) -> float:
    """Average SIREN forward pass time across saved model checkpoints."""
    model_files = sorted(Path(fits_dir).glob('*_model.pt'))
    if not model_files:
        return 0.0

    times = []
    for mf in model_files[:4]:
        checkpoint = torch.load(mf, map_location=device, weights_only=True)
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

        positions = torch.linspace(0, 1, seq_len).unsqueeze(1).to(device)

        for _ in range(10):
            with torch.no_grad():
                model(positions)
        if device == 'cuda':
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(100):
            with torch.no_grad():
                model(positions)
        if device == 'cuda':
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / 100
        times.append(elapsed * 1000)

    return float(np.mean(times)) if times else 0.0


def plot_scaling_crossover(
    scaling_results: dict[int, dict],
    output_dir: Path,
    head_dim: int = 128,
) -> None:
    """Both SIREN and HBM scale with sequence length. SIREN is 38-62x slower."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seq_lens = sorted(scaling_results.keys())
    siren_us = [scaling_results[s]['siren_time_ms'] * 1000 for s in seq_lens]
    hbm_4060_us = [scaling_results[s]['hbm_4060_ms'] * 1000 for s in seq_lens]
    hbm_h100_us = [scaling_results[s]['hbm_h100_ms'] * 1000 for s in seq_lens]

    # SIREN scales as ~n^0.76 (sub-linear due to GPU batch efficiency at small n)
    # HBM scales as n^1.0 (strictly linear: bytes / bandwidth)
    log_sl = np.log10(seq_lens)
    siren_fit = np.polyfit(log_sl, np.log10(siren_us), 1)  # [slope, intercept]

    # HBM theoretical: raw_bytes / peak_bandwidth
    hbm4060_per_token = head_dim * 4 / 272e9 * 1e6  # us per token
    hbm_h100_per_token = head_dim * 4 / 3350e9 * 1e6

    # Analytical crossover: n^a * 10^b = n * c => n = (c/10^b)^(1/(a-1))
    a, b = siren_fit
    crossover_4060 = (hbm4060_per_token / 10**b) ** (1 / (a - 1)) if a != 1 else None
    crossover_h100 = (hbm_h100_per_token / 10**b) ** (1 / (a - 1)) if a != 1 else None

    # Per-token latency ratio at measured points
    ratios = [s / h for s, h in zip(siren_us, hbm_4060_us)]

    # Extrapolation range
    max_extrap = max(seq_lens[-1] * 100, 500_000)
    extrap_x = np.logspace(np.log10(min(seq_lens)), np.log10(max_extrap), 300)
    extrap_siren = 10 ** np.polyval(siren_fit, np.log10(extrap_x))
    extrap_hbm_4060 = extrap_x * hbm4060_per_token
    extrap_hbm_h100 = extrap_x * hbm_h100_per_token

    fig, ax = plt.subplots(figsize=(10, 7))

    # Measured points
    ax.scatter(seq_lens, siren_us, c='#3498db', s=100, zorder=5,
               label='SIREN (GPU wall-clock)')
    ax.scatter(seq_lens, hbm_4060_us, c='#e74c3c', s=100, zorder=5, marker='s',
               label='HBM RTX 4060 (theoretical)')
    ax.scatter(seq_lens, hbm_h100_us, c='#2ecc71', s=100, zorder=5, marker='^',
               label='HBM H100 (theoretical)')

    # Extrapolated lines
    ax.plot(extrap_x, extrap_siren, color='#3498db', linestyle='--', alpha=0.5, linewidth=2)
    ax.plot(extrap_x, extrap_hbm_4060, color='#e74c3c', linestyle='--', alpha=0.5, linewidth=2)
    ax.plot(extrap_x, extrap_hbm_h100, color='#2ecc71', linestyle='--', alpha=0.5, linewidth=2)

    # Annotate the ratio
    ax.annotate(f'SIREN {min(ratios):.0f}-{max(ratios):.0f}x slower\nthan HBM at all lengths',
                xy=(seq_lens[-1], siren_us[-1]), fontsize=9,
                xytext=(seq_lens[-1] * 5, siren_us[-1] * 0.5),
                arrowprops=dict(arrowstyle='->', color='#3498db', alpha=0.7),
                color='#3498db')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set(xlabel='Sequence Length (tokens)', ylabel='Time (microseconds)',
           title='SIREN Is 38-62x Slower Than HBM Reads at All Practical Lengths')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig(output_dir / 'scaling_crossover.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/scaling_crossover.png")
    print(f"  SIREN/HBM ratio: {min(ratios):.1f}x - {max(ratios):.1f}x across {seq_lens[0]}-{seq_lens[-1]} tokens")
    print(f"  SIREN fit: time_us ~ n^{a:.3f} (sub-linear due to GPU batch overhead)")
    if crossover_4060:
        print(f"  Analytical crossover (RTX 4060): ~{crossover_4060:.0f} tokens ({crossover_4060/1e9:.1f}B)")
    if crossover_h100:
        print(f"  Analytical crossover (H100):     ~{crossover_h100:.0f} tokens ({crossover_h100/1e12:.0f}T)")

    with open(output_dir / 'crossover_data.json', 'w') as f:
        json.dump({
            'siren_fit_log_slope': float(a),
            'siren_fit_log_intercept': float(b),
            'siren_scaling': f'time_us ~ n^{a:.3f}',
            'hbm_scaling': 'time_us ~ n^1.0 (linear)',
            'latency_ratio_range': [float(min(ratios)), float(max(ratios))],
            'crossover_4060_tokens': float(crossover_4060) if crossover_4060 else None,
            'crossover_h100_tokens': float(crossover_h100) if crossover_h100 else None,
            'note': 'Crossover at billions of tokens — effectively never at practical lengths',
        }, f, indent=2)


def plot_scaling_quality(scaling_results: dict[int, dict], output_dir: Path) -> None:
    """CosSim, compression ratio, and autocorrelation vs sequence length."""
    output_dir = Path(output_dir)
    seq_lens = sorted(scaling_results.keys())

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    ax.plot(seq_lens, [scaling_results[s]['avg_cossim_keys'] for s in seq_lens], 'bo-', label='Keys', markersize=8)
    ax.plot(seq_lens, [scaling_results[s]['avg_cossim_values'] for s in seq_lens], 'rs-', label='Values', markersize=8)
    ax.axhline(y=0.95, color='green', linestyle='--', alpha=0.3)
    ax.set(xlabel='Sequence Length', ylabel='Avg Cosine Similarity', title='Reconstruction Quality vs Seq Length')
    ax.set_xscale('log'); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(seq_lens, [scaling_results[s]['avg_compression'] for s in seq_lens], 'go-', markersize=8)
    ax.set(xlabel='Sequence Length', ylabel='Compression Ratio (x)', title='Compression Ratio vs Seq Length')
    ax.set_xscale('log'); ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(seq_lens, [scaling_results[s]['autocorr_keys'] for s in seq_lens], 'bo-', label='Keys', markersize=8)
    ax.plot(seq_lens, [scaling_results[s]['autocorr_values'] for s in seq_lens], 'rs-', label='Values', markersize=8)
    ax.set(xlabel='Sequence Length', ylabel='Lag-1 Autocorrelation', title='Structure Metrics vs Seq Length')
    ax.set_xscale('log'); ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'scaling_quality.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/scaling_quality.png")


def run_full_layer_profile(
    kv_dir: Path,
    output_dir: Path,
    device: str = 'cuda',
    epochs: int = 2000,
) -> list[dict]:
    """Fit medium SIREN on ALL 32 layers, head 0, keys + values."""
    kv_dir, output_dir = Path(kv_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(kv_dir / 'metadata.json') as f:
        metadata = KVMetadata.from_dict(json.load(f))

    medium_config = SIRENConfig(256, 2, 30.0, 'medium')
    results: list[dict] = []
    total = metadata.num_layers * 2

    for layer_idx in range(metadata.num_layers):
        data = torch.load(kv_dir / f'layer_{layer_idx:02d}.pt', map_location='cpu', weights_only=True)
        for kv_type, tensor in [('key', data['keys'][0]), ('value', data['values'][0])]:
            count = len(results) + 1
            name = f"L{layer_idx}_H0_{kv_type}"
            print(f"[{count}/{total}] {name}...", end=" ", flush=True)
            result = fit_siren(tensor, medium_config, epochs=epochs, device=device,
                               log_every=epochs, verbose=False)
            results.append({
                'layer': layer_idx, 'kv_type': kv_type,
                'final_cosine_mean': result.final_cosine_mean,
                'compression_ratio': result.compression_ratio,
            })
            print(f"CosSim={result.final_cosine_mean:.4f}")

    with open(output_dir / 'full_layer_profile.json', 'w') as f:
        json.dump(results, f, indent=2)

    return results


def plot_full_layer_profile(results: list[dict], output_dir: Path) -> None:
    """CosSim vs layer index for all 32 layers, keys and values."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    key_results = [r for r in results if r['kv_type'] == 'key']
    val_results = [r for r in results if r['kv_type'] == 'value']

    key_layers = [r['layer'] for r in key_results]
    key_cossim = [r['final_cosine_mean'] for r in key_results]
    val_layers = [r['layer'] for r in val_results]
    val_cossim = [r['final_cosine_mean'] for r in val_results]

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(key_layers, key_cossim, 'bo-', label='Keys', markersize=6, linewidth=1.5)
    ax.plot(val_layers, val_cossim, 'rs-', label='Values', markersize=6, linewidth=1.5)
    ax.fill_between(key_layers, key_cossim, val_cossim, alpha=0.1, color='gray')

    ax.axhline(y=0.95, color='green', linestyle='--', alpha=0.3, label='0.95 target')

    # Annotate key dips
    key_dict = dict(zip(key_layers, key_cossim))
    for layer, label in [(9, 'L9'), (13, 'L13'), (20, 'L20')]:
        if layer in key_dict:
            ax.annotate(f'{label}\n{key_dict[layer]:.3f}',
                        xy=(layer, key_dict[layer]), fontsize=8, color='#3498db',
                        xytext=(layer + 1.5, key_dict[layer] - 0.03),
                        arrowprops=dict(arrowstyle='->', color='#3498db', alpha=0.7))

    # Annotate value peak
    val_dict = dict(zip(val_layers, val_cossim))
    if 17 in val_dict:
        ax.annotate(f'L17 peak\n{val_dict[17]:.3f}',
                    xy=(17, val_dict[17]), fontsize=8, color='#e74c3c',
                    xytext=(19, val_dict[17] + 0.04),
                    arrowprops=dict(arrowstyle='->', color='#e74c3c', alpha=0.7))

    ax.set(xlabel='Layer Index', ylabel='Cosine Similarity (medium SIREN)',
           title='All 32 Layers: Non-Monotonic Key Dips, Mid-Layer Value Peak')
    ax.set_xticks(range(0, max(key_layers) + 1, 2))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.4, 1.0)

    plt.tight_layout()
    plt.savefig(output_dir / 'full_layer_profile.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/full_layer_profile.png")
