"""KV cache extraction from Llama 3.1-8B or synthetic generation."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from nerf_attention.types import KVMetadata


def get_sample_text() -> str:
    """Mixed content (narrative + code + technical) to exercise diverse attention patterns."""
    return """
The architecture of modern neural networks draws heavily from biological
inspiration, yet diverges in fundamental ways. While biological neurons
communicate through sparse, asynchronous electrical impulses, artificial
neurons process dense, synchronous floating-point vectors. This distinction
becomes critical when we consider memory: biological brains store information
in synaptic weights distributed across billions of connections, while
artificial systems rely on explicit key-value caches that grow linearly
with context length.

Consider a transformer processing a legal document. Each paragraph builds
upon definitions established in the preamble. The attention mechanism must
maintain precise references to clause 3.1(a) when interpreting clause 7.2(b),
even if thousands of tokens separate them. This creates sparse, long-range
attention patterns that look nothing like the smooth, local correlations
found in natural images.

def transformer_attention(query, key, value, mask=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    attn_weights = F.softmax(scores, dim=-1)
    return torch.matmul(attn_weights, value), attn_weights

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size = x.size(0)
        Q = self.W_q(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        attn_output, attn_weights = transformer_attention(Q, K, V, mask)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.W_o(attn_output)

The memory bandwidth problem becomes acute at scale. An NVIDIA H100 provides
approximately 3.35 TB/s of HBM bandwidth but nearly 1000 TFLOPS of compute.
This means the GPU needs roughly 300 floating-point operations per byte loaded
to keep its cores busy. Standard attention has an arithmetic intensity of
approximately 2 (one multiply-add per element loaded), making it catastrophically
memory-bound.

For a 100K token context with Llama 3.1-70B, the KV cache occupies approximately:
- 80 layers x 8 KV heads x 2 (K+V) x 100000 tokens x 128 dims x 2 bytes (fp16)
- = 32.768 GB of KV cache
- At 3.35 TB/s, reading this takes ~9.8ms per decode step
- But the actual compute (attention + FFN) takes <1ms
- The GPU is idle >90% of the time

Recent approaches to this problem include quantization (KIVI, QAQ), token eviction
(H2O, StreamingLLM), and learned compression (Titans, neural memory). Each trades
fidelity for efficiency in different ways. The question we ask is different: can we
replace discrete storage with a continuous function that generates KV vectors on
demand?

import numpy as np
from scipy.fft import fft, fftfreq

def analyze_kv_spectrum(kv_cache):
    seq_len, d_head = kv_cache.shape
    spectra = []
    for dim in range(d_head):
        signal = kv_cache[:, dim]
        spectrum = np.abs(fft(signal))
        freqs = fftfreq(seq_len)
        spectra.append((freqs, spectrum))
    return spectra
""" * 3


def extract_kv_cache(
    model_name: str,
    seq_len: int,
    output_dir: Path,
    device: str = 'cuda',
) -> KVMetadata:
    """Load quantized LLM, run inference, save per-layer KV tensors."""
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

    text = get_sample_text()
    inputs = tokenizer(text, return_tensors="pt", max_length=seq_len, truncation=True).to(device)
    actual_seq_len = inputs['input_ids'].shape[1]
    print(f"Sequence length: {actual_seq_len} tokens")

    print("Running inference...")
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=False, use_cache=True)

    past_kv = outputs.past_key_values

    # Handle DynamicCache (transformers v5+) vs legacy tuple format
    if hasattr(past_kv, 'layers'):
        num_layers = len(past_kv.layers)
        _, num_kv_heads, cache_seq_len, head_dim = past_kv.layers[0].keys.shape
    elif hasattr(past_kv, 'key_cache'):
        num_layers = len(past_kv.key_cache)
        _, num_kv_heads, cache_seq_len, head_dim = past_kv.key_cache[0].shape
    else:
        num_layers = len(past_kv)
        _, num_kv_heads, cache_seq_len, head_dim = past_kv[0][0].shape

    print(f"KV cache: {num_layers} layers, {num_kv_heads} heads, "
          f"seq_len={cache_seq_len}, head_dim={head_dim}")

    output_dir.mkdir(parents=True, exist_ok=True)

    for layer_idx in range(num_layers):
        if hasattr(past_kv, 'layers'):
            keys = past_kv.layers[layer_idx].keys.squeeze(0).float().cpu()
            values = past_kv.layers[layer_idx].values.squeeze(0).float().cpu()
        elif hasattr(past_kv, 'key_cache'):
            keys = past_kv.key_cache[layer_idx].squeeze(0).float().cpu()
            values = past_kv.value_cache[layer_idx].squeeze(0).float().cpu()
        else:
            keys = past_kv[layer_idx][0].squeeze(0).float().cpu()
            values = past_kv[layer_idx][1].squeeze(0).float().cpu()
        torch.save(
            {'keys': keys, 'values': values},
            output_dir / f'layer_{layer_idx:02d}.pt',
        )

    metadata = KVMetadata(
        model_name=model_name,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        seq_len=cache_seq_len,
        head_dim=head_dim,
        actual_tokens=actual_seq_len,
    )
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata.to_dict(), f, indent=2)

    print(f"Saved {num_layers} layers to {output_dir}/")

    del model, outputs, past_kv
    torch.cuda.empty_cache()
    return metadata


def extract_kv_cache_synthetic(
    seq_len: int = 2048,
    num_layers: int = 32,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    output_dir: Path = Path('results/kv_cache_synthetic'),
) -> KVMetadata:
    """Generate synthetic KV cache with tunable structure.

    Structure per dimension: low-freq base + mid-freq sentence patterns +
    sparse attention spikes + noise. Layer depth increases sharpness.
    Values are smoother than keys (matching real KV cache properties).
    """
    print(f"Generating synthetic KV cache...")
    print(f"  {num_layers} layers, {num_kv_heads} heads, seq_len={seq_len}, head_dim={head_dim}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    t = torch.linspace(0, 1, seq_len)

    for layer_idx in range(num_layers):
        keys_all, values_all = [], []
        layer_sharpness = 1.0 + 2.0 * (layer_idx / max(num_layers - 1, 1))

        for head_idx in range(num_kv_heads):
            rng = np.random.RandomState(layer_idx * num_kv_heads + head_idx)
            k_vecs = torch.zeros(seq_len, head_dim)
            v_vecs = torch.zeros(seq_len, head_dim)

            for d in range(head_dim):
                freq1, freq2 = rng.uniform(1, 5), rng.uniform(3, 10)
                base = (0.5 * np.sin(2 * np.pi * freq1 * t.numpy()) +
                        0.3 * np.cos(2 * np.pi * freq2 * t.numpy()))

                freq_mid = rng.uniform(10, 30)
                mid = 0.2 * np.sin(2 * np.pi * freq_mid * t.numpy() + rng.uniform(0, 2 * np.pi))

                spikes = np.zeros(seq_len)
                for _ in range(int(3 * layer_sharpness)):
                    pos = rng.randint(0, seq_len)
                    width = rng.randint(1, max(2, int(5 / layer_sharpness)))
                    amp = rng.uniform(0.5, 2.0)
                    for offset in range(-width, width + 1):
                        if 0 <= pos + offset < seq_len:
                            spikes[pos + offset] += amp * np.exp(
                                -0.5 * (offset / max(1, width / 2)) ** 2
                            )

                noise = rng.randn(seq_len) * 0.1
                k_vecs[:, d] = torch.tensor(base + mid + spikes + noise, dtype=torch.float32)

                v_base = 0.6 * np.sin(2 * np.pi * rng.uniform(1, 8) * t.numpy())
                v_vecs[:, d] = torch.tensor(v_base + rng.randn(seq_len) * 0.15, dtype=torch.float32)

            keys_all.append(k_vecs.unsqueeze(0))
            values_all.append(v_vecs.unsqueeze(0))

        keys = torch.cat(keys_all, dim=0)
        values = torch.cat(values_all, dim=0)
        torch.save(
            {'keys': keys, 'values': values},
            output_dir / f'layer_{layer_idx:02d}.pt',
        )

    metadata = KVMetadata(
        model_name='synthetic',
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        seq_len=seq_len,
        head_dim=head_dim,
        actual_tokens=seq_len,
    )
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata.to_dict(), f, indent=2)

    total_mb = num_layers * num_kv_heads * seq_len * head_dim * 2 * 4 / 1024 / 1024
    print(f"Saved to {output_dir}/ ({total_mb:.1f} MB)")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description='Extract KV cache')
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.1-8B')
    parser.add_argument('--seq_len', type=int, default=2048)
    parser.add_argument('--output_dir', type=str, default='results/kv_cache')
    parser.add_argument('--synthetic', action='store_true')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    if args.synthetic:
        extract_kv_cache_synthetic(
            seq_len=args.seq_len,
            output_dir=Path(args.output_dir + '_synthetic'),
        )
    else:
        extract_kv_cache(args.model, args.seq_len, Path(args.output_dir), args.device)


if __name__ == '__main__':
    main()
