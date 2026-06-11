"""Tournament-model utilities for Allesina-Levine style experiments."""

from .generation import (
    factor_win_counts,
    fixed_rps_winner,
    payoff_from_probabilities,
    payoff_from_winner,
    posterior_mean_probability_matrix,
    sample_beta_probability_matrix,
    sample_fixed_tournament,
    win_probability_matrix,
)
from .lp import solve_zero_sum_equilibrium

__all__ = [
    "factor_win_counts",
    "fixed_rps_winner",
    "payoff_from_probabilities",
    "payoff_from_winner",
    "posterior_mean_probability_matrix",
    "sample_beta_probability_matrix",
    "sample_fixed_tournament",
    "solve_zero_sum_equilibrium",
    "win_probability_matrix",
]
