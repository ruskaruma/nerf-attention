"""Experiment 2: Multi-prompt robustness.

Tests whether K/V asymmetry holds across different content types.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from nerf_attention.analyze import analyze_kv_cache
from nerf_attention.experiments.prompts import ALL_PROMPTS
from nerf_attention.siren import fit_siren
from nerf_attention.types import KVMetadata, SIRENConfig


def run_multi_prompt_experiment(
    model_name: str,
    base_dir: Path,
    device: str = 'cuda',
    epochs: int = 2000,
    seq_len: int = 2048,
) -> dict[str, dict]:
    """Extract KV cache for 4 content types, fit medium SIREN, compare K/V."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

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

    prompt_results: dict[str, dict] = {}
    medium_config = SIRENConfig(256, 2, 30.0, 'medium')

    for prompt_name, prompt_text in ALL_PROMPTS.items():
        print(f"\n{'='*60}")
        print(f"PROMPT: {prompt_name}")
        print(f"{'='*60}")

        prompt_dir = base_dir / prompt_name
        kv_dir = prompt_dir / 'kv_cache'
        analysis_dir = prompt_dir / 'analysis'
        kv_dir.mkdir(parents=True, exist_ok=True)

        inputs = tokenizer(prompt_text, return_tensors="pt", max_length=seq_len, truncation=True).to(device)
        actual_tokens = inputs['input_ids'].shape[1]
        print(f"  Tokens: {actual_tokens}")

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=False, use_cache=True)

        past_kv = outputs.past_key_values
        num_layers, num_kv_heads, cache_seq_len, head_dim = _extract_kv_shape(past_kv)

        for layer_idx in range(num_layers):
            keys, values = _get_layer_kv(past_kv, layer_idx)
            torch.save({'keys': keys, 'values': values}, kv_dir / f'layer_{layer_idx:02d}.pt')

        metadata = KVMetadata(
            model_name=model_name, num_layers=num_layers, num_kv_heads=num_kv_heads,
            seq_len=cache_seq_len, head_dim=head_dim, actual_tokens=actual_tokens,
        )
        with open(kv_dir / 'metadata.json', 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)

        del outputs
        torch.cuda.empty_cache()

        analysis = analyze_kv_cache(kv_dir, analysis_dir)

        # Fit medium SIREN on sampled layers
        layers_to_fit = sorted({0, num_layers // 2, num_layers - 1})
        key_cossims, val_cossims = [], []

        for layer_idx in layers_to_fit:
            data = torch.load(kv_dir / f'layer_{layer_idx:02d}.pt', map_location='cpu', weights_only=True)
            for head_idx in range(min(num_kv_heads, 2)):
                for kv_type, tensor in [('key', data['keys'][head_idx]), ('value', data['values'][head_idx])]:
                    name = f"{prompt_name}_L{layer_idx}_H{head_idx}_{kv_type}"
                    print(f"  Fitting {name}...")
                    result = fit_siren(tensor, medium_config, epochs=epochs, device=device,
                                       log_every=epochs, verbose=False)
                    (key_cossims if kv_type == 'key' else val_cossims).append(result.final_cosine_mean)
                    print(f"    CosSim={result.final_cosine_mean:.4f}, Compress={result.compression_ratio:.1f}x")

        prompt_results[prompt_name] = {
            'actual_tokens': actual_tokens,
            'autocorr_keys': analysis.avg_autocorr_keys,
            'autocorr_values': analysis.avg_autocorr_values,
            'spectral_keys': analysis.avg_spectral_keys,
            'spectral_values': analysis.avg_spectral_values,
            'avg_cossim_keys': float(np.mean(key_cossims)),
            'avg_cossim_values': float(np.mean(val_cossims)),
            'std_cossim_keys': float(np.std(key_cossims)),
            'std_cossim_values': float(np.std(val_cossims)),
        }

        print(f"\n  {prompt_name}: keys={prompt_results[prompt_name]['avg_cossim_keys']:.4f}, "
              f"values={prompt_results[prompt_name]['avg_cossim_values']:.4f}")

    del model
    torch.cuda.empty_cache()

    with open(base_dir / 'multi_prompt_results.json', 'w') as f:
        json.dump(prompt_results, f, indent=2)

    _print_summary_table(prompt_results)
    return prompt_results


def _extract_kv_shape(past_kv) -> tuple[int, int, int, int]:
    """Return (num_layers, num_kv_heads, seq_len, head_dim) from any cache format."""
    if hasattr(past_kv, 'layers'):
        n = len(past_kv.layers)
        _, h, s, d = past_kv.layers[0].keys.shape
    elif hasattr(past_kv, 'key_cache'):
        n = len(past_kv.key_cache)
        _, h, s, d = past_kv.key_cache[0].shape
    else:
        n = len(past_kv)
        _, h, s, d = past_kv[0][0].shape
    return n, h, s, d


def _get_layer_kv(past_kv, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract (keys, values) for one layer, handling all cache formats."""
    if hasattr(past_kv, 'layers'):
        keys = past_kv.layers[layer_idx].keys.squeeze(0).float().cpu()
        values = past_kv.layers[layer_idx].values.squeeze(0).float().cpu()
    elif hasattr(past_kv, 'key_cache'):
        keys = past_kv.key_cache[layer_idx].squeeze(0).float().cpu()
        values = past_kv.value_cache[layer_idx].squeeze(0).float().cpu()
    else:
        keys = past_kv[layer_idx][0].squeeze(0).float().cpu()
        values = past_kv[layer_idx][1].squeeze(0).float().cpu()
    return keys, values


def _print_summary_table(prompt_results: dict[str, dict]) -> None:
    print(f"\n{'='*80}")
    print(f"{'Prompt':<16} {'K AutoCorr':>11} {'V AutoCorr':>11} {'K CosSim':>10} {'V CosSim':>10}")
    print(f"{'-'*16} {'-'*11} {'-'*11} {'-'*10} {'-'*10}")
    for name, r in prompt_results.items():
        print(f"{name:<16} {r['autocorr_keys']:>11.3f} {r['autocorr_values']:>11.3f} "
              f"{r['avg_cossim_keys']:>10.4f} {r['avg_cossim_values']:>10.4f}")


def plot_multi_prompt(prompt_results: dict[str, dict], output_dir: Path) -> None:
    """Grouped bar chart: 4 prompts, keys vs values CosSim + autocorrelation."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    names = list(prompt_results.keys())
    x = np.arange(len(names))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.bar(x - width/2, [prompt_results[n]['avg_cossim_keys'] for n in names],
           width, yerr=[prompt_results[n]['std_cossim_keys'] for n in names],
           label='Keys', color='#3498db', capsize=5, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.bar(x + width/2, [prompt_results[n]['avg_cossim_values'] for n in names],
           width, yerr=[prompt_results[n]['std_cossim_values'] for n in names],
           label='Values', color='#e74c3c', capsize=5, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.axhline(y=0.95, color='green', linestyle='--', alpha=0.3, label='0.95 target')
    ax.set_xticks(x)
    ax.set_xticklabels([n.capitalize() for n in names])
    ax.set(ylabel='Avg Cosine Similarity', title='SIREN Reconstruction by Content Type')
    ax.legend(); ax.grid(True, alpha=0.3, axis='y'); ax.set_ylim(0, 1.05)

    ax = axes[1]
    ax.bar(x - width/2, [prompt_results[n]['autocorr_keys'] for n in names],
           width, label='Keys', color='#3498db', alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.bar(x + width/2, [prompt_results[n]['autocorr_values'] for n in names],
           width, label='Values', color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([n.capitalize() for n in names])
    ax.set(ylabel='Lag-1 Autocorrelation', title='KV Structure by Content Type')
    ax.legend(); ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'multi_prompt_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/multi_prompt_comparison.png")
