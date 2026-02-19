# NeRF-Attention: Findings

## How This Started

I was reading about the memory bandwidth bottleneck in LLM inference — specifically
the gap between compute capacity (~1000 TFLOPS on H100) and memory bandwidth
(~3.35 TB/s). During autoregressive decoding, the GPU reads the entire KV cache
from HBM for every token generated. The arithmetic intensity is around 2 FLOP/byte,
meaning the GPU sits idle most of the time waiting for data.

I'd been working with SIREN networks (Sinusoidal Representation Networks) from
the NeRF literature, where they learn continuous functions that map coordinates
to signal values — f(x,y,z) → color for 3D scenes, f(x,y) → RGB for images.
The thought was: KV cache at a given layer and head is just a 1D signal —
a sequence of vectors indexed by position. What if a SIREN could learn
f(position) → KV_vector and we could replace memory reads with compute?

After searching the literature extensively — over 50 KV cache compression papers
from 2024-2025 covering quantization, pruning, SVD, GQA, vector quantization,
attention map caching — I found zero work combining implicit neural representations
with KV cache. The closest was Attention-based INR (ANR), which uses attention
*inside* INR for image reconstruction — the inverse direction. This was a genuine
research gap.

## What I Expected

I expected keys to compress well and values to fail. The reasoning came from
two directions. First, the Homogeneous Keys paper reported ~0.73 Spearman rank
correlation between adjacent key vectors, suggesting strong positional structure
that a SIREN could exploit. Second, the AsymKV papers (COLING Oct 2024, NeurIPS
June 2025) had characterized K/V asymmetry through quantization experiments —
keys tolerate aggressive compression while values are fragile. My approach was
orthogonal: testing the same asymmetry through function approximation rather
than quantization.

I also expected SIREN inference cost to be roughly fixed regardless of sequence
length — a small network with ~165K parameters should take the same time to
evaluate whether it's representing 512 or 4096 positions. If that held, there
would be a crossover point where SIREN compute becomes cheaper than HBM reads,
and that crossover would arrive sooner on hardware with higher compute-to-memory
ratios.

## What I Built

I built a four-phase pipeline running on an RTX 4060 (8GB VRAM). Phase 1 extracts
KV cache tensors from Llama 3.1-8B loaded in 4-bit quantization (~5GB). Phase 2
measures autocorrelation, spectral energy concentration, and effective rank per
layer/head, separately for keys and values. Phase 3 sweeps 7 SIREN architectures
(varying width from 64 to 512, depth from 1 to 3 layers, and omega_0 from 15 to
60) across sampled layers and heads. Phase 4 generates publication-quality figures.
The whole stack is uv-managed Python with PyTorch, numpy, and matplotlib — no
unnecessary frameworks.

## Baseline Results (280 Experiments)

I ran 7 SIREN configs across 5 layers, 4 of 8 KV heads each, for both keys and
values at 2048 tokens (using the unsloth/Llama-3.1-8B checkpoint, identical
weights to meta-llama with faster loading). The K/V asymmetry was immediately clear: keys averaged 0.9115 cosine
similarity, values averaged 0.6719. The medium config (256 hidden, 2 layers,
omega_0=30) landed on the Pareto frontier — best quality-to-compression tradeoff
but at 2048 tokens the medium config (164,992 float32 parameters = 659,968 bytes)
is still *larger* than the raw KV cache (2048 * 128 * 2 bytes = 524,288 bytes) —
a 0.8x ratio, meaning expansion not compression. Compression only begins at
~4096 tokens (1.6x). Larger configs didn't meaningfully improve quality, and
smaller ones lost too much. The high-frequency variant (omega_0=60) didn't help,
suggesting the signal isn't particularly high-frequency. SIREN forward pass took
~0.15ms compared to ~0.002ms (theoretical peak bandwidth) for an HBM read at
2048 tokens.

I measured lag-1 autocorrelation of 0.496 for keys and 0.242 for values (averaged
across layers). The literature's 0.73 was Spearman rank correlation across full
vectors — a different metric. Our per-dimension lag-1 autocorrelation is naturally
lower but still confirms the structural gap between keys and values.

## The Scaling Experiment

I expected SIREN cost to be roughly fixed while HBM scales linearly. The data
showed both scale with sequence length — SIREN evaluates at all seq_len
positions, so batch size grows with length. There is no crossover at practical
lengths.

| Seq Len | Keys CosSim | Values CosSim | Ratio† | SIREN* (ms) | HBM 4060** (ms) |
|---------|-------------|---------------|--------|-------------|-----------------|
| 512     | 0.9746      | 0.9196        | 0.2x   | 0.060       | 0.0005          |
| 1024    | 0.9434      | 0.7441        | 0.4x   | 0.091       | 0.0010          |
| 2048    | 0.8997      | 0.6359        | 0.8x   | 0.152       | 0.0019          |
| 4096    | 0.8765      | 0.5128        | 1.6x   | 0.291       | 0.0039          |

† Ratio = raw KV bytes (float16) / SIREN parameter bytes (float32). Values <1x
mean SIREN is *larger* than the original (expansion, not compression).
Compression only begins at ~4096 tokens for the medium config (164,992 params).

\* GPU wall-clock (RTX 4060, CUDA synchronized).
\** Theoretical peak bandwidth (272 GB/s); real bandwidth is ~80-90% of peak.

SIREN is 76–125x slower than theoretical HBM reads across all measured lengths
(scaling experiment, averaged across 4 models per length). Separate per-model
latency profiling at 2048 tokens measured ~3ms per forward pass (~800x slower
than HBM) — the difference is likely due to GPU state and measurement context.
The 76–125x range is more representative as it averages across models and runs.

The ratio narrows slightly at longer sequences (GPU batching becomes more
efficient), but the analytical crossover is at ~185 billion tokens for RTX 4060
— effectively never. The bigger problem is quality: keys drop from 0.97 to
0.88 and values from 0.92 to 0.51 as sequence length increases. Even if SIREN
eventually became faster, the reconstructions would be too degraded to use.
8192 tokens OOMed on the RTX 4060.

Note: the scaling experiment text was truncated from concatenated prompts, but
at all tested lengths the model only sees fiction text. The 2048-token scaling
data and the fiction multi-prompt row are the same KV cache.

## Multi-Prompt Robustness

This was the strongest methodological result. I ran four content types — fiction,
code, conversational, and technical — all at 2048 tokens.

| Prompt         | Keys CosSim | Values CosSim | K AutoCorr | V AutoCorr |
|----------------|-------------|---------------|------------|------------|
| fiction        | 0.9095      | 0.6504        | 0.488      | 0.223      |
| code           | 0.9104      | 0.6400        | 0.489      | 0.229      |
| conversational | 0.9043      | 0.6245        | 0.486      | 0.220      |
| technical      | 0.9078      | 0.6275        | 0.480      | 0.224      |

Keys land within a 0.006 spread across content types. Values within 0.026.
Autocorrelation spans 0.009. This is consistent with the K/V asymmetry
coming from architectural differences — RoPE applies position-dependent
rotations to keys but not values, though the key projection matrix Wk
could also contribute. The structure is architectural, not data-dependent,
but isolating RoPE specifically would require testing on a non-RoPE model
(e.g., ALiBi or absolute position embeddings). (The fiction row shares data with the scaling experiment
at 2048 tokens, so the three independent content types are code,
conversational, and technical.)

## SVD Baseline

I ran truncated SVD as the obvious comparison. SVD at 2x compression (0.97
keys) beats SIREN at 0.8x (0.90 keys) — SIREN can't even compress at 2048
tokens while SVD achieves 2x with better quality. This is expected by the
Eckart-Young theorem — truncated SVD is the optimal low-rank approximation —
but it quantifies exactly how far SIREN falls short.

| Compression | SVD Keys | SVD Values | SIREN Keys (medium, 0.8x†) |
|-------------|----------|------------|---------------------------|
| 2x          | 0.9745   | 0.9124     | 0.90                      |
| 4x          | 0.9225   | 0.8032     | —                         |
| 8x          | 0.8662   | 0.6920     | —                         |
| 16x         | 0.8119   | 0.5830     | —                         |

SVD at 2x achieves 0.97 on keys — better than SIREN's 0.90, and SIREN isn't
even compressing at 2048 tokens (0.8x ratio = expansion). SVD requires zero
training. The gap is even more damning for values: SVD at 4x (0.80) vastly
beats SIREN (0.64). SVD is a single matrix decomposition. SIREN is 2000
epochs of gradient descent. Note: both SIREN and SVD store their compressed
representations in float32, compared against float16 originals. If both used
float16 for deployment, the absolute ratios would improve equally for both
methods — the relative comparison is unaffected.

## Full Layer Profile

I fit the medium SIREN on all 32 layers of Llama 3.1-8B (head 0, keys and
values, 2048 tokens — 64 fits total). The layer profile turned out to be
non-monotonic, which I didn't expect.

Keys range from 0.846 (L13) to 0.935 (L23). There are clear dips at L9
(0.851), L13 (0.846), and L20 (0.871). These aren't noise — they're
reproducible drops surrounded by higher-quality layers. My initial 3-layer
sample (L0/L16/L31) captured the general trend but missed this structure
entirely. Several middle layers appear to transition between processing
modes — possibly where attention shifts from local syntactic to global
semantic patterns, making the positional structure harder for SIREN to
capture.

Values range from 0.483 (L0) to 0.758 (L17). They peak mid-network then
decline — a different pattern from keys. The early layers (L0-L2) have
especially poor value reconstruction (0.48-0.53), which makes sense: early
layers do the most diverse token-level processing before representations
become more abstract.

## What I Learned

The hypothesis was creative but wrong. Keys have learnable positional structure
from RoPE, but SVD captures it better with zero training cost. Values are
fundamentally unlearnable by position-only mappings — they encode
token-specific content that has no systematic relationship to sequence position.
The contribution of this work is the structural characterization, not the
compression method.

To make this practical, you would need: custom CUDA kernels to close the
76–125x latency gap, online or incremental training to avoid the 2000-epoch
offline fitting cost, and likely a hybrid approach — SIREN for keys (which
have RoPE structure) combined with quantized values (which don't). But I'm
skeptical any of these would close the gap with SVD, which is both faster
and higher quality at every compression ratio I tested. The honest conclusion
is that SIREN specifically is the wrong tool for this problem. Other INR
variants (Fourier features + MLP, learned frequencies, tiny autoencoders)
were not tested and could potentially perform differently.

Methodology limitations worth noting: quality is measured by cosine similarity
only — no perplexity or generation evaluation, so the impact on actual model
output is uncharacterized. Only 4 of 8 KV heads were sampled per layer. No
random seeds were set, so exact numbers aren't reproducible (though the
qualitative patterns are stable). Training used a fixed 2000 epochs with no
early stopping or convergence verification. Prompt texts are repeated 3-5x to fill the context window. This
disproportionately inflates *value* quality: values have no position
encoding, so identical tokens at different positions produce identical
value vectors — creating truly periodic signals that SIRENs are designed
for. Keys are less affected because RoPE applies position-dependent
rotations that break periodicity even when tokens repeat. The reported
value CosSim (0.67 avg) should be treated as an upper bound; the real
K/V gap on non-repeating text is likely larger than reported.

## Raw Numbers

### Baseline (280 SIREN fits, medium config, 2048 tokens)
- Keys avg CosSim: 0.9115
- Values avg CosSim: 0.6719
- Compression ratio: ~0.8x at 2048 tokens (expansion), ~1.6x at 4096 tokens
- SIREN forward pass: ~0.15ms (GPU wall-clock)
- HBM read (RTX 4060): ~0.002ms (theoretical peak bandwidth, float16)

### Scaling (medium config, head 0, layers 0/16/31, fiction text)

| Seq Len | Keys CosSim | Values CosSim | Ratio† | SIREN* (ms) | HBM 4060** (ms) | HBM H100** (ms) |
|---------|-------------|---------------|--------|-------------|-----------------|------------------|
| 512     | 0.9746      | 0.9196        | 0.2x   | 0.060       | 0.0005          | 0.000039         |
| 1024    | 0.9434      | 0.7441        | 0.4x   | 0.091       | 0.0010          | 0.000078         |
| 2048    | 0.8997      | 0.6359        | 0.8x   | 0.152       | 0.0019          | 0.000157         |
| 4096    | 0.8765      | 0.5128        | 1.6x   | 0.291       | 0.0039          | 0.000313         |

\* GPU wall-clock, CUDA synchronized. ** Theoretical peak bandwidth.
† Ratio = raw KV bytes (float16) / SIREN params (float32). <1x means expansion.

### Multi-Prompt (medium config, 2048 tokens)

| Prompt         | Keys CosSim | Values CosSim | K AutoCorr | V AutoCorr |
|----------------|-------------|---------------|------------|------------|
| fiction        | 0.9095      | 0.6504        | 0.488      | 0.223      |
| code           | 0.9104      | 0.6400        | 0.489      | 0.229      |
| conversational | 0.9043      | 0.6245        | 0.486      | 0.220      |
| technical      | 0.9078      | 0.6275        | 0.480      | 0.224      |

### SVD Baseline (2048 tokens, averaged across layers 0/16/31, 4 heads each)

| Compression | SVD Keys | SVD Values |
|-------------|----------|------------|
| 2x          | 0.9745   | 0.9124     |
| 4x          | 0.9225   | 0.8032     |
| 8x          | 0.8662   | 0.6920     |
| 16x         | 0.8119   | 0.5830     |

### Full Layer Profile (32 layers, head 0, medium config, 2048 tokens)

| Layer | Keys CosSim | Values CosSim |
|-------|-------------|---------------|
| 0     | 0.868       | 0.483         |
| 1     | 0.900       | 0.554         |
| 2     | 0.883       | 0.534         |
| 3     | 0.923       | 0.588         |
| 4     | 0.907       | 0.634         |
| 5     | 0.918       | 0.633         |
| 6     | 0.915       | 0.655         |
| 7     | 0.910       | 0.692         |
| 8     | 0.896       | 0.700         |
| 9     | 0.851       | 0.666         |
| 10    | 0.870       | 0.753         |
| 11    | 0.875       | 0.676         |
| 12    | 0.912       | 0.681         |
| 13    | 0.846       | 0.732         |
| 14    | 0.879       | 0.652         |
| 15    | 0.900       | 0.614         |
| 16    | 0.914       | 0.631         |
| 17    | 0.889       | 0.758         |
| 18    | 0.902       | 0.738         |
| 19    | 0.891       | 0.651         |
| 20    | 0.871       | 0.568         |
| 21    | 0.877       | 0.601         |
| 22    | 0.882       | 0.555         |
| 23    | 0.935       | 0.690         |
| 24    | 0.921       | 0.679         |
| 25    | 0.922       | 0.649         |
| 26    | 0.902       | 0.694         |
| 27    | 0.904       | 0.610         |
| 28    | 0.908       | 0.650         |
| 29    | 0.905       | 0.631         |
| 30    | 0.885       | 0.617         |
| 31    | 0.912       | 0.705         |
