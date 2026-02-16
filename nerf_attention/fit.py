"""Fit SIREN networks to KV cache tensors across architecture configurations."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from nerf_attention.siren import SIREN, fit_siren
from nerf_attention.types import (
    CONFIGS_FULL,
    CONFIGS_QUICK,
    FitResult,
    KVMetadata,
    SIRENConfig,
)


def fit_kv_cache(
    kv_dir: Path,
    output_dir: Path,
    epochs: int = 5000,
    device: str = 'cuda',
    quick: bool = False,
) -> list[dict]:
    """Fit SIRENs to extracted KV cache and record metrics."""
    kv_dir, output_dir = Path(kv_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(kv_dir / 'metadata.json') as f:
        metadata = KVMetadata.from_dict(json.load(f))

    print(f"KV Cache: {metadata.num_layers} layers x {metadata.num_kv_heads} heads")
    print(f"Per tensor: ({metadata.seq_len}, {metadata.head_dim}) = "
          f"{metadata.seq_len * metadata.head_dim * 4 / 1024:.1f} KB")
    print(f"Device: {device}, Epochs: {epochs}")

    if quick:
        layers_to_fit = [0, metadata.num_layers // 2, metadata.num_layers - 1]
        heads_per_layer = 1
        configs = CONFIGS_QUICK
    else:
        layers_to_fit = [0, metadata.num_layers // 4, metadata.num_layers // 2,
                         3 * metadata.num_layers // 4, metadata.num_layers - 1]
        heads_per_layer = min(metadata.num_kv_heads, 4)
        configs = CONFIGS_FULL

    layers_to_fit = sorted(set(l for l in layers_to_fit if l < metadata.num_layers))
    total = len(layers_to_fit) * heads_per_layer * 2 * len(configs)
    all_results: list[dict] = []
    exp_count = 0

    for layer_idx in layers_to_fit:
        filepath = kv_dir / f'layer_{layer_idx:02d}.pt'
        if not filepath.exists():
            print(f"  Skipping layer {layer_idx} (not found)")
            continue

        data = torch.load(filepath, map_location='cpu', weights_only=True)
        keys, values = data['keys'], data['values']

        for head_idx in range(heads_per_layer):
            for kv_type, tensor in [('key', keys[head_idx]), ('value', values[head_idx])]:
                for config in configs:
                    exp_count += 1
                    name = f"L{layer_idx}_H{head_idx}_{kv_type}_{config.name}"
                    print(f"\n[{exp_count}/{total}] {name}")

                    result = fit_siren(
                        tensor, config,
                        epochs=epochs,
                        device=device,
                        log_every=max(epochs // 5, 100),
                        verbose=True,
                    )

                    record = _result_to_record(name, layer_idx, head_idx, kv_type, result)
                    all_results.append(record)

                    if config.name == 'medium':
                        _save_model(output_dir, name, result, record)

                    print(f"  -> CosSim: {result.final_cosine_mean:.4f} | "
                          f"Compress: {result.compression_ratio:.1f}x | "
                          f"Time: {result.train_time_seconds:.1f}s")

    with open(output_dir / 'fit_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    _print_summary(all_results, layers_to_fit)
    return all_results


def _result_to_record(
    name: str, layer: int, head: int, kv_type: str, result: FitResult,
) -> dict:
    return {
        'name': name,
        'layer': layer,
        'head': head,
        'kv_type': kv_type,
        'config_name': result.config.name,
        'hidden_features': result.config.hidden_features,
        'hidden_layers': result.config.hidden_layers,
        'omega_0': result.config.omega_0,
        'final_mse': result.final_mse,
        'final_cosine_mean': result.final_cosine_mean,
        'final_cosine_min': result.final_cosine_min,
        'final_cosine_std': result.final_cosine_std,
        'compression_ratio': result.compression_ratio,
        'raw_size_bytes': result.raw_size_bytes,
        'siren_size_bytes': result.siren_size_bytes,
        'train_time_seconds': result.train_time_seconds,
        'num_parameters': result.num_parameters,
        'seq_len': result.seq_len,
        'd_head': result.d_head,
    }


def _save_model(output_dir: Path, name: str, result: FitResult, record: dict) -> None:
    torch.save(
        {
            'model_state': result.model.state_dict(),
            'config': {
                'hidden_features': result.config.hidden_features,
                'hidden_layers': result.config.hidden_layers,
                'omega_0': result.config.omega_0,
                'name': result.config.name,
                'out_features': result.d_head,
            },
            'target_mean': result.target_mean,
            'target_std': result.target_std,
            'metrics': record,
        },
        output_dir / f'{name}_model.pt',
    )


def _print_summary(all_results: list[dict], layers_to_fit: list[int]) -> None:
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"{'Name':<35} {'CosSim':>8} {'MSE':>10} {'Compress':>10} {'Time':>8}")
    print(f"{'-'*35} {'-'*8} {'-'*10} {'-'*10} {'-'*8}")

    for r in sorted(all_results, key=lambda x: x['final_cosine_mean'], reverse=True):
        print(f"{r['name']:<35} {r['final_cosine_mean']:>8.4f} "
              f"{r['final_mse']:>10.6f} {r['compression_ratio']:>9.1f}x "
              f"{r['train_time_seconds']:>7.1f}s")

    print(f"\n{'='*80}")
    print("KEY FINDINGS")
    print(f"{'='*80}")

    for cn in sorted(set(r['config_name'] for r in all_results)):
        cr = [r for r in all_results if r['config_name'] == cn]
        print(f"  {cn:<10}: avg CosSim={np.mean([r['final_cosine_mean'] for r in cr]):.4f}, "
              f"avg Compression={np.mean([r['compression_ratio'] for r in cr]):.1f}x")

    key_r = [r for r in all_results if r['kv_type'] == 'key']
    val_r = [r for r in all_results if r['kv_type'] == 'value']
    if key_r and val_r:
        k_avg = np.mean([r['final_cosine_mean'] for r in key_r])
        v_avg = np.mean([r['final_cosine_mean'] for r in val_r])
        print(f"\n  Keys avg CosSim:   {k_avg:.4f}")
        print(f"  Values avg CosSim: {v_avg:.4f}")
        diff = v_avg - k_avg
        if diff > 0.01:
            print("  -> Values compress better (smoother signal)")
        elif diff < -0.01:
            print("  -> Keys compress better (stronger positional structure)")
        else:
            print("  -> Similar compressibility")

    for layer_idx in layers_to_fit:
        lr = [r for r in all_results if r['layer'] == layer_idx and r['config_name'] == 'medium']
        if lr:
            print(f"  Layer {layer_idx:2d} (medium): "
                  f"avg CosSim={np.mean([r['final_cosine_mean'] for r in lr]):.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description='Fit SIRENs to KV cache')
    parser.add_argument('--kv_dir', type=str, default='results/kv_cache')
    parser.add_argument('--output_dir', type=str, default='results/fits')
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'

    fit_kv_cache(Path(args.kv_dir), Path(args.output_dir), args.epochs, args.device, args.quick)


if __name__ == '__main__':
    main()
