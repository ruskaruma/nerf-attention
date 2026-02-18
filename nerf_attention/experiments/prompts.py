"""Prompt texts for multi-prompt robustness experiment."""

FICTION = (
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
    "resolved into coordinates — positions in a multi-dimensional space describing "
    "harmonic relationships between sound frequencies. The manuscript was a musical "
    "score for instruments that hadn't existed in the fifteenth century. Instruments "
    "producing frequencies below human hearing, designed to resonate with the "
    "architecture of specific buildings. Eleanor, three centuries later, would "
    "discover that the Prague cathedral was one such building."
) * 5

CODE = """
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
    return torch.polar(torch.ones_like(torch.outer(t, freqs)), torch.outer(t, freqs))

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
        if mask is not None: scores = scores + mask
        attn = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(attn, xv.transpose(1,2))
        return self.wo(output.transpose(1,2).contiguous().view(bsz, seqlen, -1))

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden = ((int(8 * config.d_model / 3) + 255) // 256) * 256
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

class Node:
    def __init__(self, val):
        self.val, self.left, self.right = val, None, None

class BST:
    def __init__(self): self.root = None
    def insert(self, val):
        if not self.root: self.root = Node(val); return
        curr = self.root
        while True:
            if val < curr.val:
                if not curr.left: curr.left = Node(val); return
                curr = curr.left
            else:
                if not curr.right: curr.right = Node(val); return
                curr = curr.right
    def search(self, val):
        curr = self.root
        while curr:
            if val == curr.val: return True
            curr = curr.left if val < curr.val else curr.right
        return False

class HashMap:
    def __init__(self, cap=16):
        self.cap, self.sz = cap, 0
        self.keys, self.vals = [None]*cap, [None]*cap
    def _hash(self, k): return hash(k) % self.cap
    def put(self, k, v):
        if self.sz >= self.cap * 0.75: self._resize()
        i = self._hash(k)
        while self.keys[i] is not None:
            if self.keys[i] == k: self.vals[i] = v; return
            i = (i+1) % self.cap
        self.keys[i], self.vals[i], self.sz = k, v, self.sz+1
    def get(self, k, d=None):
        i = self._hash(k)
        while self.keys[i] is not None:
            if self.keys[i] == k: return self.vals[i]
            i = (i+1) % self.cap
        return d
""" * 3

CONVERSATIONAL = (
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
    "compiles Python-like code to optimized PTX. For attention research, Triton lets "
    "you experiment with custom attention patterns without full CUDA overhead.\n\n"
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
    "User: What are people doing about it?\n\n"
    "Assistant: Several approaches: quantization (KIVI reduces to 2-bit), token "
    "eviction (H2O, StreamingLLM), learned compression (Titans), and architectural "
    "changes (grouped query attention). Each trades quality for efficiency.\n\n"
    "User: What about the SIREN approach?\n\n"
    "Assistant: Replace KV storage with KV generation. Train a small neural network "
    "that maps position to KV vector. If it's small enough, weights fit in fast "
    "memory, trading IO for compute.\n\n"
    "User: Does it work?\n\n"
    "Assistant: For keys, surprisingly well. RoPE gives them sinusoidal structure "
    "that SIRENs capture at >0.95 cosine similarity. Values are harder — they carry "
    "content-dependent information with no positional bias.\n\n"
    "User: What's the latency like?\n\n"
    "Assistant: At 2048 tokens, SIREN is about 40x slower than HBM read on RTX 4060. "
    "The crossover happens at much longer sequences.\n\n"
    "User: So not practical yet?\n\n"
    "Assistant: Not at current sequence lengths. But the gap narrows quickly — at "
    "32K+ tokens on H100, the math starts to favor SIREN. The structure analysis "
    "is valuable regardless.\n\n"
    "User: What about fine-tuning for more compressible KV representations?\n\n"
    "Assistant: Great research direction. Add a compression-aware regularizer during "
    "training. The challenge is balancing task performance with KV compressibility."
) * 4

TECHNICAL = (
    "The Transformer architecture (Vaswani et al., 2017) computes attention as "
    "Attention(Q,K,V) = softmax(QK^T/sqrt(d_k))V. Computational complexity is "
    "O(n^2 * d) for sequence length n. Memory complexity during inference is "
    "dominated by the KV cache: O(L * h * n * d_head * 2).\n\n"
    "Rotary Position Embeddings (Su et al., 2021) encode position by rotating "
    "dimension pairs: RoPE(x, pos) applies rotation R(pos*theta_i) to dims "
    "(2i, 2i+1), where theta_i = 10000^(-2i/d). This creates multi-scale "
    "periodic structure: dim 0 rotates at frequency 1, dim d-1 at ~0.0001.\n\n"
    "Grouped Query Attention (Ainslie et al., 2023) reduces KV heads. Llama 3.1-8B "
    "uses 32 query heads but only 8 KV heads, reducing cache by 4x.\n\n"
    "Memory bandwidth bottleneck: H100 SXM provides 3.35 TB/s HBM3 and 989 TFLOPS "
    "FP16. Arithmetic intensity threshold: 989e12/3.35e12 ≈ 295 FLOPs/byte. "
    "Standard attention achieves ~2 FLOPs/byte, 150x below compute-bound.\n\n"
    "For Llama 3.1-70B at 128K context: KV cache = 80*8*2*128000*128*2 = 32.8 GB. "
    "At 3.35 TB/s, reading takes 9.8ms. Compute takes <1ms. GPU utilization ~10%.\n\n"
    "SIREN networks (Sitzmann et al., 2020) use sinusoidal activations: "
    "phi(x) = sin(omega_0 * Wx + b). Derivatives of SIRENs are also SIRENs. "
    "omega_0 controls frequency range. Weight initialization: first layer from "
    "U(-1/n, 1/n), subsequent from U(-sqrt(6/n)/omega_0, sqrt(6/n)/omega_0).\n\n"
    "Truncated SVD provides optimal rank-k approximation (Eckart-Young theorem). "
    "For matrix M of shape (n,d), M_k = U_k*S_k*V_k^T minimizes ||M-M_k||_F. "
    "Storage: n*k + k + k*d values. Compression ratio ≈ d/k for large n.\n\n"
    "Spectral energy concentration: fraction of power in lowest p% frequencies. "
    "Values >0.5 at p=10% indicate strong low-frequency dominance, favorable for "
    "SIREN. Autocorrelation R(tau) = E[x(t)*x(t+tau)]/E[x(t)^2]. High lag-1 "
    "(>0.5) indicates smooth signals favorable for continuous approximation."
) * 4

ALL_PROMPTS = {
    'fiction': FICTION,
    'code': CODE,
    'conversational': CONVERSATIONAL,
    'technical': TECHNICAL,
}
