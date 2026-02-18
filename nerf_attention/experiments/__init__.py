"""Follow-up experiments: scaling, multi-prompt robustness, SVD baseline."""

from nerf_attention.experiments.scaling import (
    run_scaling_experiment, plot_scaling_crossover, plot_scaling_quality,
    run_full_layer_profile, plot_full_layer_profile,
)
from nerf_attention.experiments.multi_prompt import run_multi_prompt_experiment, plot_multi_prompt
from nerf_attention.experiments.svd import run_svd_experiment, plot_siren_vs_svd
from nerf_attention.experiments.summary import generate_final_summary
