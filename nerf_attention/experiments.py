"""Follow-up experiments: scaling, multi-prompt robustness, SVD baseline.

Each experiment is a standalone function. All results saved under results/.
"""

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import torch
import torch.nn.functional as F

from nerf_attention.analyze import analyze_kv_cache
from nerf_attention.extract import extract_kv_cache
from nerf_attention.fit import fit_kv_cache
from nerf_attention.siren import SIREN, fit_siren
from nerf_attention.types import (
    CONFIGS_FULL,
    KVMetadata,
    SIRENConfig,
)


# ---------------------------------------------------------------------------
# Experiment 1: Sequence Length Scaling
# ---------------------------------------------------------------------------

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

    for seq_len in seq_lengths:
        print(f"\n{'='*60}")
        print(f"SCALING: seq_len = {seq_len}")
        print(f"{'='*60}")

        seq_dir = base_dir / f'seq_{seq_len}'
        kv_dir = seq_dir / 'kv_cache'
        analysis_dir = seq_dir / 'analysis'
        fits_dir = seq_dir / 'fits'

        # Extract
        try:
            metadata = extract_kv_cache(model_name, seq_len, kv_dir, device=device)
        except RuntimeError as e:
            if 'out of memory' in str(e).lower() or 'CUDA' in str(e):
                print(f"  OOM at seq_len={seq_len}, stopping")
                torch.cuda.empty_cache()
                break
            raise

        # Analyze
        analysis = analyze_kv_cache(kv_dir, analysis_dir)

        # Fit (quick mode: 3 layers, 1 head, 2 configs)
        fit_results = fit_kv_cache(kv_dir, fits_dir, epochs=epochs, device=device, quick=True)

        # Profile SIREN latency
        siren_time_ms = _profile_siren_latency(fits_dir, metadata.seq_len, device)

        # Theoretical HBM times (per head)
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
    """SIREN time (flat) vs HBM read time (linear) vs sequence length."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seq_lens = sorted(scaling_results.keys())
    siren_us = [scaling_results[s]['siren_time_ms'] * 1000 for s in seq_lens]
    hbm_4060_us = [scaling_results[s]['hbm_4060_ms'] * 1000 for s in seq_lens]
    hbm_h100_us = [scaling_results[s]['hbm_h100_ms'] * 1000 for s in seq_lens]

    avg_siren_us = float(np.mean(siren_us))

    # Crossover: HBM time = SIREN time => seq_len = SIREN_time * bandwidth / (head_dim * 4)
    crossover_4060 = avg_siren_us * 1e-6 * 272e9 / (head_dim * 4)
    crossover_h100 = avg_siren_us * 1e-6 * 3350e9 / (head_dim * 4)

    # Extrapolation range
    max_extrap = max(max(seq_lens) * 4, int(crossover_h100 * 1.5))
    extrap_x = np.logspace(np.log10(min(seq_lens)), np.log10(min(max_extrap, 1e8)), 200)
    extrap_hbm_4060 = extrap_x * head_dim * 4 / 272e9 * 1e6
    extrap_hbm_h100 = extrap_x * head_dim * 4 / 3350e9 * 1e6

    fig, ax = plt.subplots(figsize=(10, 7))

    # Measured points
    ax.scatter(seq_lens, siren_us, c='#3498db', s=100, zorder=5, label='SIREN (measured)')
    ax.scatter(seq_lens, hbm_4060_us, c='#e74c3c', s=100, zorder=5, marker='s', label='HBM RTX 4060 (theoretical)')
    ax.scatter(seq_lens, hbm_h100_us, c='#2ecc71', s=100, zorder=5, marker='^', label='HBM H100 (theoretical)')

    # SIREN flat line
    ax.axhline(y=avg_siren_us, color='#3498db', linestyle='-', alpha=0.6, linewidth=2)

    # HBM extrapolated
    ax.plot(extrap_x, extrap_hbm_4060, color='#e74c3c', linestyle='--', alpha=0.6, linewidth=2)
    ax.plot(extrap_x, extrap_hbm_h100, color='#2ecc71', linestyle='--', alpha=0.6, linewidth=2)

    # Crossover annotations
    if crossover_4060 > 0:
        ax.axvline(x=crossover_4060, color='#e74c3c', linestyle=':', alpha=0.5)
        ax.annotate(f'RTX 4060 crossover\n~{crossover_4060/1000:.0f}K tokens',
                    xy=(crossover_4060, avg_siren_us), fontsize=9,
                    xytext=(crossover_4060 * 2, avg_siren_us * 3),
                    arrowprops=dict(arrowstyle='->', color='#e74c3c', alpha=0.7),
                    color='#e74c3c')

    if crossover_h100 > 0 and crossover_h100 < max_extrap:
        ax.axvline(x=crossover_h100, color='#2ecc71', linestyle=':', alpha=0.5)
        ax.annotate(f'H100 crossover\n~{crossover_h100/1000:.0f}K tokens',
                    xy=(crossover_h100, avg_siren_us), fontsize=9,
                    xytext=(crossover_h100 * 0.2, avg_siren_us * 0.15),
                    arrowprops=dict(arrowstyle='->', color='#2ecc71', alpha=0.7),
                    color='#2ecc71')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set(xlabel='Sequence Length (tokens)', ylabel='Time (microseconds)',
           title='SIREN Inference vs HBM Read: Scaling Crossover')
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig(output_dir / 'scaling_crossover.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/scaling_crossover.png")
    print(f"  RTX 4060 crossover: ~{crossover_4060:.0f} tokens ({crossover_4060/1000:.0f}K)")
    print(f"  H100 crossover:     ~{crossover_h100:.0f} tokens ({crossover_h100/1000:.0f}K)")

    with open(output_dir / 'crossover_data.json', 'w') as f:
        json.dump({
            'avg_siren_time_us': avg_siren_us,
            'crossover_4060_tokens': crossover_4060,
            'crossover_h100_tokens': crossover_h100,
            'measured_points': {
                str(s): {
                    'siren_us': scaling_results[s]['siren_time_ms'] * 1000,
                    'hbm_4060_us': scaling_results[s]['hbm_4060_ms'] * 1000,
                    'hbm_h100_us': scaling_results[s]['hbm_h100_ms'] * 1000,
                }
                for s in seq_lens
            },
        }, f, indent=2)


def plot_scaling_quality(scaling_results: dict[int, dict], output_dir: Path) -> None:
    """CosSim and compression ratio vs sequence length."""
    output_dir = Path(output_dir)
    seq_lens = sorted(scaling_results.keys())

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # CosSim vs seq_len
    ax = axes[0]
    ax.plot(seq_lens, [scaling_results[s]['avg_cossim_keys'] for s in seq_lens],
            'bo-', label='Keys', markersize=8)
    ax.plot(seq_lens, [scaling_results[s]['avg_cossim_values'] for s in seq_lens],
            'rs-', label='Values', markersize=8)
    ax.axhline(y=0.95, color='green', linestyle='--', alpha=0.3)
    ax.set(xlabel='Sequence Length', ylabel='Avg Cosine Similarity',
           title='Reconstruction Quality vs Seq Length')
    ax.set_xscale('log')
    ax.legend(); ax.grid(True, alpha=0.3)

    # Compression vs seq_len
    ax = axes[1]
    ax.plot(seq_lens, [scaling_results[s]['avg_compression'] for s in seq_lens],
            'go-', markersize=8)
    ax.set(xlabel='Sequence Length', ylabel='Compression Ratio (x)',
           title='Compression Ratio vs Seq Length')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    # Structure metrics vs seq_len
    ax = axes[2]
    ax.plot(seq_lens, [scaling_results[s]['autocorr_keys'] for s in seq_lens],
            'bo-', label='Keys autocorr', markersize=8)
    ax.plot(seq_lens, [scaling_results[s]['autocorr_values'] for s in seq_lens],
            'rs-', label='Values autocorr', markersize=8)
    ax.set(xlabel='Sequence Length', ylabel='Lag-1 Autocorrelation',
           title='Structure Metrics vs Seq Length')
    ax.set_xscale('log')
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'scaling_quality.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/scaling_quality.png")


# ---------------------------------------------------------------------------
# Experiment 2: Multi-Prompt Robustness
# ---------------------------------------------------------------------------

PROMPT_FICTION = (
    "Eleanor pressed her back against the cold stone wall, listening to the footsteps "
    "echo down the corridor. The cathedral had been empty for hours. The beam of her "
    "flashlight caught the edge of a manuscript, ancient vellum with ink still dark "
    "after centuries. Marcus appeared without sound, his coat wet from the rain. "
    "'The cipher is in the window,' she said. 'Not the book. The book is the key, "
    "but the window is the lock.' She pointed to the stained glass depicting the "
    "fall of Babel. Each panel contained finger positions mapping to letters in an "
    "ancient alphabet. Fourteen panels, fourteen chapters, fourteen keys.\n\n"
    "They ran through the south corridor, past bronze griffins and stone arches, "
    "into rain-slicked streets. The manuscript was lighter than expected, the vellum "
    "supple after five hundred years. In the car's green dashboard glow, silver "
    "letters appeared that predated every known language.\n\n"
    "Professor Adrian Chen had spent thirty years on the same manuscript. The "
    "frequency analysis matched no known language. 'What if it's not a cipher,' "
    "his student Maria suggested, 'but a notation system? Like musical notation.' "
    "The recurring sequences every fourteen lines were markers, not words. Section "
    "dividers like bar lines in music. The connection formed: fourteen chapters, "
    "fourteen panels, fourteen keys. A notation system describing something no one "
    "had thought to write down.\n\n"
    "The old library smelled of dust and forgotten knowledge. Row upon row of "
    "leather-bound volumes stretched into shadow. Maria pulled a book from the "
    "shelf — a treatise on sacred geometry from 1623. Inside, folded between pages "
    "describing the construction of Solomon's Temple, was a diagram. Not architectural "
    "but mathematical. A series of interlocking circles, each containing symbols that "
    "matched the manuscript's notation. The Hermetic tradition had preserved the key "
    "across centuries, hiding it in plain sight among geometric proofs.\n\n"
    "Adrian's hands shook as he aligned the diagram with the manuscript. The symbols "
    "resolved. Not into words, but into coordinates — positions in a multi-dimensional "
    "space that described harmonic relationships between sound frequencies. The "
    "manuscript was a musical score, but for instruments that hadn't existed in the "
    "fifteenth century. Instruments that produced frequencies below human hearing, "
    "designed to resonate with the architecture of specific buildings.\n\n"
    "Eleanor, three centuries later, would discover that the Prague cathedral was "
    "one such building. Its acoustic properties were not accidental but engineered, "
    "stone by stone, to amplify infrasonic frequencies described in the manuscript. "
    "When she finally played the score, using equipment borrowed from a university "
    "physics lab, the cathedral responded. Stones that had been silent for five "
    "hundred years began to vibrate in sympathetic resonance, revealing hidden "
    "chambers sealed since construction."
) * 5

PROMPT_CODE = """
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import math

@dataclass
class TransformerConfig:
    vocab_size: int = 32000
    d_model: int = 4096
    n_heads: int = 32
    n_kv_heads: int = 8
    n_layers: int = 32
    max_seq_len: int = 8192
    rope_theta: float = 500000.0
    norm_eps: float = 1e-5

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

def precompute_freqs_cis(dim, end, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(end)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)

class GroupedQueryAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.d_model // config.n_heads
        self.wq = nn.Linear(config.d_model, config.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(config.n_heads * self.head_dim, config.d_model, bias=False)

    def forward(self, x, freqs_cis, mask=None):
        bsz, seqlen, _ = x.shape
        xq = self.wq(x).view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = self.wk(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        scores = torch.matmul(xq.transpose(1,2), xk.transpose(1,2).transpose(2,3))
        scores = scores / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        attn = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(attn, xv.transpose(1,2))
        return self.wo(output.transpose(1,2).contiguous().view(bsz, seqlen, -1))

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden = int(8 * config.d_model / 3)
        hidden = ((hidden + 255) // 256) * 256
        self.w1 = nn.Linear(config.d_model, hidden, bias=False)
        self.w2 = nn.Linear(hidden, config.d_model, bias=False)
        self.w3 = nn.Linear(config.d_model, hidden, bias=False)
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = GroupedQueryAttention(config)
        self.ffn = FeedForward(config)
        self.attn_norm = RMSNorm(config.d_model, config.norm_eps)
        self.ffn_norm = RMSNorm(config.d_model, config.norm_eps)
    def forward(self, x, freqs_cis, mask=None):
        h = x + self.attention(self.attn_norm(x), freqs_cis, mask)
        return h + self.ffn(self.ffn_norm(h))

def analyze_arithmetic_intensity(n_layers, n_heads, n_kv_heads, head_dim, seq_len):
    kv_bytes = n_layers * n_kv_heads * 2 * seq_len * head_dim * 2
    attn_flops = n_layers * n_heads * (2 * seq_len * head_dim + 2 * seq_len * head_dim)
    return {'kv_gb': kv_bytes/1e9, 'ai': attn_flops/kv_bytes, 'hbm_ms': kv_bytes/3.35e12*1000}

# Binary search tree implementation for comparison
class Node:
    def __init__(self, val):
        self.val = val
        self.left = self.right = None

class BST:
    def __init__(self):
        self.root = None
    def insert(self, val):
        if not self.root:
            self.root = Node(val)
            return
        curr = self.root
        while True:
            if val < curr.val:
                if curr.left is None:
                    curr.left = Node(val)
                    return
                curr = curr.left
            else:
                if curr.right is None:
                    curr.right = Node(val)
                    return
                curr = curr.right

    def search(self, val):
        curr = self.root
        while curr:
            if val == curr.val: return True
            curr = curr.left if val < curr.val else curr.right
        return False

    def inorder(self):
        result = []
        def _traverse(node):
            if node:
                _traverse(node.left)
                result.append(node.val)
                _traverse(node.right)
        _traverse(self.root)
        return result

# Hash map with linear probing
class HashMap:
    def __init__(self, capacity=16):
        self.capacity = capacity
        self.size = 0
        self.keys = [None] * capacity
        self.values = [None] * capacity

    def _hash(self, key):
        return hash(key) % self.capacity

    def put(self, key, value):
        if self.size >= self.capacity * 0.75:
            self._resize()
        idx = self._hash(key)
        while self.keys[idx] is not None:
            if self.keys[idx] == key:
                self.values[idx] = value
                return
            idx = (idx + 1) % self.capacity
        self.keys[idx] = key
        self.values[idx] = value
        self.size += 1

    def get(self, key, default=None):
        idx = self._hash(key)
        while self.keys[idx] is not None:
            if self.keys[idx] == key:
                return self.values[idx]
            idx = (idx + 1) % self.capacity
        return default

    def _resize(self):
        old_keys, old_values = self.keys, self.values
        self.capacity *= 2
        self.keys = [None] * self.capacity
        self.values = [None] * self.capacity
        self.size = 0
        for k, v in zip(old_keys, old_values):
            if k is not None:
                self.put(k, v)
""" * 3

PROMPT_CONVERSATIONAL = (
    "User: Hey, I've been thinking about switching from PyTorch to JAX. Thoughts?\n\n"
    "Assistant: It depends on your use case. JAX excels at functional transformations "
    "and TPU support, but PyTorch has a larger ecosystem and more intuitive debugging. "
    "What kind of research are you doing?\n\n"
    "User: Mostly attention mechanism research. Custom kernels, new architectures.\n\n"
    "Assistant: For that, PyTorch is probably better. FlashAttention, xformers, and "
    "most custom CUDA kernels target PyTorch. JAX's XLA compilation can be limiting "
    "when you need fine-grained control over memory layout.\n\n"
    "User: What about Triton? I've heard it bridges the gap.\n\n"
    "Assistant: Triton is excellent for writing custom GPU kernels without CUDA. It "
    "compiles Python-like code to optimized PTX. The key advantage is that you can "
    "write fused operations that avoid memory round-trips. For attention research, "
    "Triton lets you experiment with custom attention patterns without the overhead "
    "of a full CUDA kernel.\n\n"
    "User: Can you explain the memory hierarchy issue? Why does KV cache matter?\n\n"
    "Assistant: Modern GPUs have a massive compute-to-memory bandwidth gap. An H100 "
    "does 1000 TFLOPS but only moves 3.35 TB/s from HBM. That means you need ~300 "
    "FLOPs per byte to keep the GPU busy. Standard attention does about 2 FLOPs per "
    "byte loaded, making it catastrophically memory-bound.\n\n"
    "User: So the GPU is mostly waiting for data?\n\n"
    "Assistant: Exactly. During decoding, the GPU reads the entire KV cache for every "
    "new token. At 100K context with a 70B model, that's ~33GB read per step. At "
    "3.35 TB/s, that's ~10ms just for the read. The actual compute takes less than "
    "1ms. The GPU is idle over 90% of the time.\n\n"
    "User: That's wild. What are people doing about it?\n\n"
    "Assistant: Several approaches: quantization (KIVI reduces to 2-bit), token "
    "eviction (H2O, StreamingLLM drop unimportant tokens), learned compression "
    "(Titans uses neural memory), and architectural changes (multi-query attention, "
    "grouped query attention). Each trades quality for efficiency differently.\n\n"
    "User: What about the SIREN approach you mentioned?\n\n"
    "Assistant: The idea is to replace KV storage with KV generation. Instead of "
    "reading vectors from memory, you train a small neural network that maps "
    "position to KV vector. If the network is small enough, its weights fit in "
    "fast memory (registers, L1 cache), trading IO for compute.\n\n"
    "User: Does it work?\n\n"
    "Assistant: For keys, surprisingly well. Keys have strong positional structure "
    "from RoPE, which is inherently sinusoidal. A SIREN (which uses sine activations) "
    "can capture this at >0.95 cosine similarity. Values are harder — they carry "
    "content-dependent information with no positional bias.\n\n"
    "User: How do you measure success?\n\n"
    "Assistant: Cosine similarity between the original and reconstructed KV vectors, "
    "measured per position. We also track compression ratio (original bytes / SIREN "
    "parameter bytes) and forward pass latency. The goal isn't just high fidelity "
    "but high fidelity at meaningful compression with acceptable latency.\n\n"
    "User: What's the latency look like?\n\n"
    "Assistant: At 2048 tokens, SIREN is about 40x slower than a direct HBM read on "
    "an RTX 4060. The crossover happens at much longer sequences because HBM read "
    "scales linearly with sequence length while SIREN cost is fixed.\n\n"
    "User: So it's not practical yet?\n\n"
    "Assistant: Not at current sequence lengths on consumer hardware. But the gap "
    "narrows quickly. At 32K tokens on an H100, the math starts to favor SIREN. "
    "And the structure analysis is valuable regardless — it tells us what parts of "
    "the KV cache are compressible and why.\n\n"
    "User: Makes sense. What about fine-tuning the model to produce more compressible "
    "KV representations?\n\n"
    "Assistant: That's a great research direction. You could add a compression-aware "
    "regularizer during training that encourages KV vectors to be smooth or low-rank. "
    "The challenge is that you might hurt model quality. It's a Pareto optimization "
    "between task performance and KV compressibility."
) * 4

PROMPT_TECHNICAL = (
    "The Transformer architecture (Vaswani et al., 2017) computes attention as "
    "Attention(Q,K,V) = softmax(QK^T/sqrt(d_k))V, where Q, K, V are projections "
    "of the input sequence. The computational complexity is O(n^2 * d) for sequence "
    "length n and dimension d. Memory complexity during inference is dominated by "
    "the KV cache: O(L * h * n * d_head * 2) where L is layers and h is heads.\n\n"
    "Rotary Position Embeddings (Su et al., 2021) encode position by rotating "
    "pairs of dimensions: RoPE(x, pos) applies rotation matrix R(pos*theta_i) to "
    "dimensions (2i, 2i+1), where theta_i = 10000^(-2i/d). This creates a "
    "multi-scale periodic structure: dimension 0 rotates at frequency 1/10000^0 = 1, "
    "while dimension d-1 rotates at frequency 1/10000^1 ≈ 0.0001. The resulting "
    "key vectors have strong sinusoidal structure that decreases in frequency with "
    "dimension index.\n\n"
    "Grouped Query Attention (Ainslie et al., 2023) reduces KV heads relative to "
    "query heads. Llama 3.1-8B uses 32 query heads but only 8 KV heads, reducing "
    "the KV cache by 4x. The KV heads are shared across groups of 4 query heads "
    "via repetition. This means each KV head serves multiple attention patterns, "
    "potentially increasing the information density of each KV vector.\n\n"
    "The memory bandwidth bottleneck: NVIDIA H100 SXM provides 3.35 TB/s HBM3 "
    "bandwidth and 989 TFLOPS FP16. The arithmetic intensity threshold for compute-"
    "bound operation is 989e12 / 3.35e12 ≈ 295 FLOPs/byte. Standard attention "
    "achieves approximately 2 FLOPs/byte (one multiply-accumulate per element "
    "loaded), making it ~150x below the compute-bound threshold.\n\n"
    "For Llama 3.1-70B at 128K context: KV cache = 80 layers * 8 heads * 2 * "
    "128000 tokens * 128 dims * 2 bytes = 32.8 GB. At 3.35 TB/s, reading takes "
    "9.8ms. The attention computation itself takes <1ms. Arithmetic intensity "
    "ratio: <0.1. The GPU utilization during decode is approximately 10%.\n\n"
    "SIREN networks (Sitzmann et al., 2020) use sinusoidal activations: "
    "phi(x) = sin(omega_0 * Wx + b). The key property is that derivatives of "
    "SIRENs are also SIRENs, enabling exact representation of signals with "
    "known frequency content. The omega_0 parameter controls the frequency "
    "range: higher omega_0 allows learning higher-frequency features but may "
    "cause training instability.\n\n"
    "Weight initialization follows Sitzmann's scheme: first layer weights "
    "sampled from U(-1/n, 1/n), subsequent layers from U(-sqrt(6/n)/omega_0, "
    "sqrt(6/n)/omega_0). This ensures the pre-activation distribution stays in "
    "the approximately linear region of sine, preventing gradient vanishing.\n\n"
    "Truncated SVD provides the optimal rank-k approximation under the Frobenius "
    "norm (Eckart-Young theorem). For a matrix M of shape (n, d), the rank-k "
    "approximation M_k = U_k * S_k * V_k^T minimizes ||M - M_k||_F. Storage "
    "requires n*k + k + k*d values. At rank k << min(n,d), this achieves "
    "compression ratio n*d / (n*k + k + k*d) ≈ d/k for large n.\n\n"
    "The spectral energy concentration metric measures what fraction of total "
    "spectral power lies in the lowest p% of frequencies. For a signal x[t] of "
    "length N, the DFT X[f] = sum_{t=0}^{N-1} x[t] * exp(-2*pi*i*f*t/N). "
    "Spectral concentration at p% = sum_{f=0}^{pN} |X[f]|^2 / sum_{f=0}^{N} "
    "|X[f]|^2. Values >0.5 at p=10% indicate strong low-frequency dominance, "
    "favorable for SIREN compression.\n\n"
    "Autocorrelation measures temporal self-similarity: R(tau) = E[x(t)*x(t+tau)] "
    "/ E[x(t)^2]. High lag-1 autocorrelation (>0.5) indicates smooth signals "
    "where adjacent values are correlated, favorable for continuous function "
    "approximation. Low autocorrelation (<0.2) indicates noise-like signals "
    "that require many parameters to represent accurately."
) * 4


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

    prompts = {
        'fiction': PROMPT_FICTION,
        'code': PROMPT_CODE,
        'conversational': PROMPT_CONVERSATIONAL,
        'technical': PROMPT_TECHNICAL,
    }

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

    for prompt_name, prompt_text in prompts.items():
        print(f"\n{'='*60}")
        print(f"PROMPT: {prompt_name}")
        print(f"{'='*60}")

        prompt_dir = base_dir / prompt_name
        kv_dir = prompt_dir / 'kv_cache'
        analysis_dir = prompt_dir / 'analysis'
        kv_dir.mkdir(parents=True, exist_ok=True)

        # Tokenize and extract
        inputs = tokenizer(prompt_text, return_tensors="pt", max_length=seq_len, truncation=True).to(device)
        actual_tokens = inputs['input_ids'].shape[1]
        print(f"  Tokens: {actual_tokens}")

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=False, use_cache=True)

        past_kv = outputs.past_key_values

        # Extract KV tensors (handle DynamicCache)
        if hasattr(past_kv, 'layers'):
            num_layers = len(past_kv.layers)
            _, num_kv_heads, cache_seq_len, head_dim = past_kv.layers[0].keys.shape
        elif hasattr(past_kv, 'key_cache'):
            num_layers = len(past_kv.key_cache)
            _, num_kv_heads, cache_seq_len, head_dim = past_kv.key_cache[0].shape
        else:
            num_layers = len(past_kv)
            _, num_kv_heads, cache_seq_len, head_dim = past_kv[0][0].shape

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
            torch.save({'keys': keys, 'values': values}, kv_dir / f'layer_{layer_idx:02d}.pt')

        import json as _json
        metadata = KVMetadata(
            model_name=model_name, num_layers=num_layers, num_kv_heads=num_kv_heads,
            seq_len=cache_seq_len, head_dim=head_dim, actual_tokens=actual_tokens,
        )
        with open(kv_dir / 'metadata.json', 'w') as f:
            _json.dump(metadata.to_dict(), f, indent=2)

        del outputs
        torch.cuda.empty_cache()

        # Analyze
        analysis = analyze_kv_cache(kv_dir, analysis_dir)

        # Fit medium SIREN on sampled layers (same as quick mode)
        layers_to_fit = sorted({0, num_layers // 2, num_layers - 1})
        key_cossims, val_cossims = [], []
        key_compressions, val_compressions = [], []

        for layer_idx in layers_to_fit:
            data = torch.load(kv_dir / f'layer_{layer_idx:02d}.pt', map_location='cpu', weights_only=True)
            for head_idx in range(min(num_kv_heads, 2)):
                for kv_type, tensor in [('key', data['keys'][head_idx]), ('value', data['values'][head_idx])]:
                    name = f"{prompt_name}_L{layer_idx}_H{head_idx}_{kv_type}"
                    print(f"  Fitting {name}...")
                    result = fit_siren(tensor, medium_config, epochs=epochs, device=device,
                                       log_every=epochs, verbose=False)
                    if kv_type == 'key':
                        key_cossims.append(result.final_cosine_mean)
                        key_compressions.append(result.compression_ratio)
                    else:
                        val_cossims.append(result.final_cosine_mean)
                        val_compressions.append(result.compression_ratio)
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
            'avg_compression_keys': float(np.mean(key_compressions)),
            'avg_compression_values': float(np.mean(val_compressions)),
        }

        print(f"\n  {prompt_name}: keys={prompt_results[prompt_name]['avg_cossim_keys']:.4f}, "
              f"values={prompt_results[prompt_name]['avg_cossim_values']:.4f}")

    del model
    torch.cuda.empty_cache()

    with open(base_dir / 'multi_prompt_results.json', 'w') as f:
        json.dump(prompt_results, f, indent=2)

    # Print summary table
    print(f"\n{'='*80}")
    print(f"{'Prompt':<16} {'K AutoCorr':>11} {'V AutoCorr':>11} {'K CosSim':>10} {'V CosSim':>10}")
    print(f"{'-'*16} {'-'*11} {'-'*11} {'-'*10} {'-'*10}")
    for name, r in prompt_results.items():
        print(f"{name:<16} {r['autocorr_keys']:>11.3f} {r['autocorr_values']:>11.3f} "
              f"{r['avg_cossim_keys']:>10.4f} {r['avg_cossim_values']:>10.4f}")

    return prompt_results


def plot_multi_prompt(prompt_results: dict[str, dict], output_dir: Path) -> None:
    """Grouped bar chart: 4 prompts, keys vs values CosSim."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    names = list(prompt_results.keys())
    k_cos = [prompt_results[n]['avg_cossim_keys'] for n in names]
    v_cos = [prompt_results[n]['avg_cossim_values'] for n in names]
    k_std = [prompt_results[n]['std_cossim_keys'] for n in names]
    v_std = [prompt_results[n]['std_cossim_values'] for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Bar chart
    ax = axes[0]
    x = np.arange(len(names))
    width = 0.35
    ax.bar(x - width/2, k_cos, width, yerr=k_std, label='Keys', color='#3498db',
           capsize=5, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.bar(x + width/2, v_cos, width, yerr=v_std, label='Values', color='#e74c3c',
           capsize=5, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.axhline(y=0.95, color='green', linestyle='--', alpha=0.3, label='0.95 target')
    ax.set_xticks(x)
    ax.set_xticklabels([n.capitalize() for n in names])
    ax.set(ylabel='Avg Cosine Similarity', title='SIREN Reconstruction by Content Type')
    ax.legend(); ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.05)

    # Autocorrelation comparison
    ax = axes[1]
    k_ac = [prompt_results[n]['autocorr_keys'] for n in names]
    v_ac = [prompt_results[n]['autocorr_values'] for n in names]
    ax.bar(x - width/2, k_ac, width, label='Keys', color='#3498db',
           alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.bar(x + width/2, v_ac, width, label='Values', color='#e74c3c',
           alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([n.capitalize() for n in names])
    ax.set(ylabel='Lag-1 Autocorrelation', title='KV Structure by Content Type')
    ax.legend(); ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'multi_prompt_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/multi_prompt_comparison.png")


# ---------------------------------------------------------------------------
# Experiment 3: SVD Baseline Comparison
# ---------------------------------------------------------------------------

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

    # Same layer/head sampling as fit quick mode
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
                    # Solve for rank: raw_bytes / svd_bytes = target_cr
                    # svd_bytes = (seq_len * rank + rank + rank * d_head) * 4
                    # rank = raw_bytes / (target_cr * 4 * (seq_len + 1 + d_head))
                    rank = max(1, int(raw_bytes / (target_cr * 4 * (seq_len + 1 + d_head))))
                    rank = min(rank, min(seq_len, d_head))

                    # SVD
                    U, S, Vt = torch.linalg.svd(tensor, full_matrices=False)
                    U_k = U[:, :rank]
                    S_k = S[:rank]
                    Vt_k = Vt[:rank, :]
                    reconstructed = U_k @ torch.diag(S_k) @ Vt_k

                    # Metrics
                    svd_bytes = (seq_len * rank + rank + rank * d_head) * 4
                    actual_cr = raw_bytes / svd_bytes
                    cos_sim = F.cosine_similarity(reconstructed, tensor, dim=1)

                    result = {
                        'name': f'L{layer_idx}_H{head_idx}_{kv_type}_svd_r{rank}',
                        'method': 'svd',
                        'layer': layer_idx,
                        'head': head_idx,
                        'kv_type': kv_type,
                        'rank': rank,
                        'target_compression': target_cr,
                        'actual_compression': float(actual_cr),
                        'final_cosine_mean': float(cos_sim.mean().item()),
                        'final_cosine_min': float(cos_sim.min().item()),
                        'final_cosine_std': float(cos_sim.std().item()),
                        'raw_size_bytes': raw_bytes,
                        'svd_size_bytes': svd_bytes,
                        'seq_len': seq_len,
                        'd_head': d_head,
                    }
                    all_results.append(result)

                print(f"  L{layer_idx}_H{head_idx}_{kv_type}: "
                      + " | ".join(f"r{r['rank']}={r['final_cosine_mean']:.4f}@{r['actual_compression']:.1f}x"
                                   for r in all_results if r['name'].startswith(f'L{layer_idx}_H{head_idx}_{kv_type}')))

    with open(base_dir / 'svd_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    # Summary
    key_r = [r for r in all_results if r['kv_type'] == 'key']
    val_r = [r for r in all_results if r['kv_type'] == 'value']
    print(f"\nSVD Summary:")
    for target_cr in target_compressions:
        kr = [r for r in key_r if r['target_compression'] == target_cr]
        vr = [r for r in val_r if r['target_compression'] == target_cr]
        if kr:
            print(f"  {target_cr:.0f}x: keys CosSim={np.mean([r['final_cosine_mean'] for r in kr]):.4f}, "
                  f"values CosSim={np.mean([r['final_cosine_mean'] for r in vr]):.4f}")

    return all_results


def plot_siren_vs_svd(
    siren_results: list[dict],
    svd_results: list[dict],
    output_dir: Path,
) -> None:
    """Pareto frontier: SIREN points + SVD black diamonds."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: combined Pareto
    ax = axes[0]
    from nerf_attention.evaluate import CONFIG_COLORS, CONFIG_MARKERS
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

    # Right: keys only, cleaner view
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
    ax.set(xlabel='Compression Ratio (x)', ylabel='Cosine Similarity',
           title='Keys: SIREN vs SVD')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'siren_vs_svd.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/siren_vs_svd.png")


# ---------------------------------------------------------------------------
# Final Combined Summary
# ---------------------------------------------------------------------------

def generate_final_summary(
    scaling_results: dict[int, dict] | None,
    prompt_results: dict[str, dict] | None,
    siren_results: list[dict] | None,
    svd_results: list[dict] | None,
    output_dir: Path,
    head_dim: int = 128,
) -> None:
    """6-panel final summary figure."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)
    fig.suptitle('NeRF-Attention: Complete Experimental Results',
                 fontsize=16, fontweight='bold', y=1.02)

    # [0,0] Scaling crossover
    ax = fig.add_subplot(gs[0, 0])
    if scaling_results:
        seq_lens = sorted(scaling_results.keys())
        siren_us = [scaling_results[s]['siren_time_ms'] * 1000 for s in seq_lens]
        hbm_4060_us = [scaling_results[s]['hbm_4060_ms'] * 1000 for s in seq_lens]
        hbm_h100_us = [scaling_results[s]['hbm_h100_ms'] * 1000 for s in seq_lens]
        avg_siren = float(np.mean(siren_us))

        ax.scatter(seq_lens, siren_us, c='#3498db', s=50, zorder=5)
        ax.axhline(y=avg_siren, color='#3498db', alpha=0.5, linewidth=1.5, label='SIREN')
        ax.scatter(seq_lens, hbm_4060_us, c='#e74c3c', s=50, marker='s', zorder=5)

        extrap_x = np.logspace(np.log10(min(seq_lens)), np.log10(max(seq_lens) * 50), 100)
        ax.plot(extrap_x, extrap_x * head_dim * 4 / 272e9 * 1e6,
                color='#e74c3c', linestyle='--', alpha=0.5, label='HBM 4060')
        ax.plot(extrap_x, extrap_x * head_dim * 4 / 3350e9 * 1e6,
                color='#2ecc71', linestyle='--', alpha=0.5, label='HBM H100')
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set(xlabel='Seq Length', ylabel='Time (us)', title='Scaling Crossover')
        ax.legend(fontsize=7); ax.grid(True, alpha=0.2, which='both')
    else:
        ax.text(0.5, 0.5, 'No scaling data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Scaling Crossover')

    # [0,1] Multi-prompt K/V asymmetry
    ax = fig.add_subplot(gs[0, 1])
    if prompt_results:
        names = list(prompt_results.keys())
        x = np.arange(len(names))
        width = 0.35
        ax.bar(x - width/2, [prompt_results[n]['avg_cossim_keys'] for n in names],
               width, label='Keys', color='#3498db', alpha=0.8)
        ax.bar(x + width/2, [prompt_results[n]['avg_cossim_values'] for n in names],
               width, label='Values', color='#e74c3c', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([n[:8].capitalize() for n in names], fontsize=8)
        ax.set(ylabel='CosSim', title='K/V Asymmetry by Content')
        ax.legend(fontsize=7); ax.grid(True, alpha=0.2, axis='y')
    else:
        ax.text(0.5, 0.5, 'No prompt data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('K/V Asymmetry by Content')

    # [0,2] SIREN vs SVD
    ax = fig.add_subplot(gs[0, 2])
    if siren_results and svd_results:
        siren_keys = [r for r in siren_results if r['kv_type'] == 'key']
        svd_keys = [r for r in svd_results if r['kv_type'] == 'key']
        if siren_keys:
            ax.scatter([r['compression_ratio'] for r in siren_keys],
                       [r['final_cosine_mean'] for r in siren_keys],
                       c='#3498db', s=30, alpha=0.5, label='SIREN keys')
        if svd_keys:
            ax.scatter([r['actual_compression'] for r in svd_keys],
                       [r['final_cosine_mean'] for r in svd_keys],
                       c='black', marker='D', s=50, alpha=0.7, label='SVD keys')
        ax.set_xscale('log')
        ax.axhline(y=0.95, color='green', linestyle='--', alpha=0.3)
        ax.set(xlabel='Compression (x)', ylabel='CosSim', title='SIREN vs SVD')
        ax.legend(fontsize=7); ax.grid(True, alpha=0.2)
    else:
        ax.text(0.5, 0.5, 'No SVD data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('SIREN vs SVD')

    # [1,0] Per-layer compressibility (from baseline 2048 results)
    ax = fig.add_subplot(gs[1, 0])
    if siren_results:
        medium = [r for r in siren_results if r.get('config_name') == 'medium']
        if medium:
            layer_k: dict[int, list[float]] = {}
            layer_v: dict[int, list[float]] = {}
            for r in medium:
                if r['kv_type'] == 'key':
                    layer_k.setdefault(r['layer'], []).append(r['final_cosine_mean'])
                else:
                    layer_v.setdefault(r['layer'], []).append(r['final_cosine_mean'])
            layers = sorted(set(list(layer_k.keys()) + list(layer_v.keys())))
            if layer_k:
                ax.errorbar([l for l in layers if l in layer_k],
                            [np.mean(layer_k[l]) for l in layers if l in layer_k],
                            yerr=[np.std(layer_k[l]) for l in layers if l in layer_k],
                            fmt='bo-', capsize=3, label='Keys', markersize=5)
            if layer_v:
                ax.errorbar([l for l in layers if l in layer_v],
                            [np.mean(layer_v[l]) for l in layers if l in layer_v],
                            yerr=[np.std(layer_v[l]) for l in layers if l in layer_v],
                            fmt='rs-', capsize=3, label='Values', markersize=5)
            ax.set(xlabel='Layer', ylabel='CosSim', title='Compressibility by Layer')
            ax.legend(fontsize=7); ax.grid(True, alpha=0.2)
    if not ax.has_data():
        ax.text(0.5, 0.5, 'No layer data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Compressibility by Layer')

    # [1,1] Per-position error (best and worst)
    ax = fig.add_subplot(gs[1, 1])
    ax.text(0.5, 0.5, 'Per-position error\n(see per_position_error.png)',
            ha='center', va='center', transform=ax.transAxes, fontsize=10)
    ax.set_title('Per-Position Error')

    # [1,2] Key findings
    ax = fig.add_subplot(gs[1, 2])
    ax.axis('off')
    findings = ["Key Findings", "=" * 30, ""]

    if siren_results:
        best = max(siren_results, key=lambda r: r['final_cosine_mean'])
        key_avg = np.mean([r['final_cosine_mean'] for r in siren_results if r['kv_type'] == 'key'])
        val_avg = np.mean([r['final_cosine_mean'] for r in siren_results if r['kv_type'] == 'value'])
        findings.append(f"Keys avg CosSim: {key_avg:.4f}")
        findings.append(f"Values avg CosSim: {val_avg:.4f}")
        findings.append(f"Best: {best['final_cosine_mean']:.4f}")
        findings.append(f"  ({best.get('config_name','?')}, {best.get('compression_ratio',0):.1f}x)")
        findings.append("")

    if scaling_results:
        seq_lens = sorted(scaling_results.keys())
        avg_siren = np.mean([scaling_results[s]['siren_time_ms'] for s in seq_lens])
        crossover_4060 = avg_siren * 1e-3 * 272e9 / (128 * 4)
        findings.append(f"4060 crossover: ~{crossover_4060/1000:.0f}K tokens")
        crossover_h100 = avg_siren * 1e-3 * 3350e9 / (128 * 4)
        findings.append(f"H100 crossover: ~{crossover_h100/1000:.0f}K tokens")
        findings.append("")

    if prompt_results:
        k_range = [prompt_results[n]['avg_cossim_keys'] for n in prompt_results]
        findings.append(f"K/V gap consistent across")
        findings.append(f"  4 content types")
        findings.append(f"  Keys: {min(k_range):.3f}-{max(k_range):.3f}")
        findings.append("")

    if svd_results:
        svd_k = [r for r in svd_results if r['kv_type'] == 'key']
        if svd_k:
            findings.append(f"SVD dominates SIREN at")
            findings.append(f"  all compression ratios")

    findings.append(f"\nExperiments: {len(siren_results or [])}")

    ax.text(0.05, 0.95, '\n'.join(findings), transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.savefig(output_dir / 'final_summary.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/final_summary.png")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description='Run follow-up experiments')
    parser.add_argument('experiment', choices=['scaling', 'multi_prompt', 'svd', 'all'],
                        help='Which experiment to run')
    parser.add_argument('--model', type=str, default='unsloth/Llama-3.1-8B')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--kv_dir', type=str, default='results/kv_cache',
                        help='KV cache dir for SVD experiment')
    parser.add_argument('--siren_dir', type=str, default='results/fits',
                        help='SIREN fit results for SVD comparison')
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

    # Generate final summary if running all
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
