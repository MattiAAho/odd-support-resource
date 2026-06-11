"""Small smoke/demo for the standalone tournament module."""

from __future__ import annotations

import numpy as np

from tournament.generation import (
    fixed_rps_winner,
    independent_factor_ranks,
    payoff_from_probabilities,
    payoff_from_winner,
    sample_fixed_tournament,
    win_probability_matrix,
)
from tournament.kernels import (
    simulate_bernoulli_tournament,
    simulate_factor_resampling_tournament,
    simulate_fixed_tournament,
)
from tournament.lp import solve_zero_sum_equilibrium


def main() -> None:
    n_individuals = 25_000
    n_steps = 200_000
    sample_every = 1_000
    burn_in = 50_000

    print("RPS fixed tournament")
    winner = fixed_rps_winner()
    payoff = payoff_from_winner(winner)
    eq, value = solve_zero_sum_equilibrium(payoff)
    _, freqs, avg = simulate_fixed_tournament(
        winner, n_individuals, n_steps, sample_every, seed=42, burn_in=burn_in
    )
    print(f"  LP equilibrium: {np.round(eq, 4)}  value={value:.3g}")
    print(f"  time-average:   {np.round(avg, 4)}")
    print(f"  final:          {np.round(freqs[-1], 4)}")

    print("\nIndependent limiting-factor tournament")
    rng = np.random.default_rng(123)
    ranks = independent_factor_ranks(n_species=7, n_factors=5, rng=rng)
    p_win = win_probability_matrix(ranks)
    fixed = sample_fixed_tournament(p_win, rng=rng)

    eq_fixed, value_fixed = solve_zero_sum_equilibrium(payoff_from_winner(fixed))
    _, _, avg_fixed = simulate_fixed_tournament(
        fixed, n_individuals, n_steps, sample_every, seed=1, burn_in=burn_in
    )

    eq_dynamic, value_dynamic = solve_zero_sum_equilibrium(payoff_from_probabilities(p_win))
    _, _, avg_bern = simulate_bernoulli_tournament(
        p_win, n_individuals, n_steps, sample_every, seed=2, burn_in=burn_in
    )
    _, _, avg_factor = simulate_factor_resampling_tournament(
        ranks, n_individuals, n_steps, sample_every, seed=3, burn_in=burn_in
    )

    print(f"  fixed LP:       {np.round(eq_fixed, 4)}  value={value_fixed:.3g}")
    print(f"  fixed sim avg:  {np.round(avg_fixed, 4)}")
    print(f"  dynamic LP:     {np.round(eq_dynamic, 4)}  value={value_dynamic:.3g}")
    print(f"  bernoulli avg:  {np.round(avg_bern, 4)}")
    print(f"  factor avg:     {np.round(avg_factor, 4)}")


if __name__ == "__main__":
    main()
