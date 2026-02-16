"""Shared dataclasses for the nerf-attention pipeline."""

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn


@dataclass(frozen=True)
class SIRENConfig:
    hidden_features: int = 256
    hidden_layers: int = 2
    omega_0: float = 30.0
    name: str = 'medium'


@dataclass
class FitResult:
    model: nn.Module
    config: SIRENConfig
    target_mean: torch.Tensor
    target_std: torch.Tensor
    losses: list[float]
    final_mse: float
    final_cosine_mean: float
    final_cosine_min: float
    final_cosine_std: float
    per_pos_mse: np.ndarray
    cosine_sims: np.ndarray
    compression_ratio: float
    raw_size_bytes: int
    siren_size_bytes: int
    train_time_seconds: float
    seq_len: int
    d_head: int
    num_parameters: int


@dataclass
class KVMetadata:
    model_name: str
    num_layers: int
    num_kv_heads: int
    seq_len: int
    head_dim: int
    actual_tokens: int
    dtype: str = 'float32'

    def to_dict(self) -> dict:
        return {
            'model_name': self.model_name,
            'num_layers': self.num_layers,
            'num_kv_heads': self.num_kv_heads,
            'seq_len': self.seq_len,
            'head_dim': self.head_dim,
            'actual_tokens': self.actual_tokens,
            'dtype': self.dtype,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'KVMetadata':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class LayerSummary:
    layer: int
    avg_autocorr_k: float
    avg_autocorr_v: float
    avg_energy_10pct_k: float
    avg_energy_10pct_v: float
    avg_rank_ratio_k: float
    avg_rank_ratio_v: float


@dataclass
class AnalysisResult:
    metadata: KVMetadata
    layer_summaries: list[LayerSummary]
    avg_autocorr_keys: float
    avg_autocorr_values: float
    avg_spectral_keys: float
    avg_spectral_values: float


CONFIGS_QUICK: list[SIRENConfig] = [
    SIRENConfig(128, 1, 30.0, 'small'),
    SIRENConfig(256, 2, 30.0, 'medium'),
]

CONFIGS_FULL: list[SIRENConfig] = [
    SIRENConfig(64,  1, 30.0, 'tiny'),
    SIRENConfig(128, 1, 30.0, 'small'),
    SIRENConfig(256, 2, 30.0, 'medium'),
    SIRENConfig(512, 2, 30.0, 'large'),
    SIRENConfig(256, 3, 30.0, 'deep'),
    SIRENConfig(256, 2, 60.0, 'hifreq'),
    SIRENConfig(256, 2, 15.0, 'lofreq'),
]
