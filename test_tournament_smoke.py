"""Smoke tests for the standalone tournament module."""

import numpy as np

from tournament.generation import fixed_rps_winner, payoff_from_winner
from tournament.kernels import simulate_fixed_tournament, simulate_fixed_tournament_counts
from tournament.lp import solve_zero_sum_equilibrium


def main() -> None:
    winner = fixed_rps_winner()
    payoff = payoff_from_winner(winner)
    eq, value = solve_zero_sum_equilibrium(payoff)

    assert np.allclose(eq, np.ones(3) / 3), eq
    assert abs(value) < 1e-8, value

    _, _, avg = simulate_fixed_tournament(
        winner,
        n_individuals=5_000,
        n_steps=50_000,
        sample_every=1_000,
        seed=123,
        burn_in=10_000,
    )
    assert np.max(np.abs(avg - (np.ones(3) / 3))) < 0.08, avg

    _, _, avg_counts = simulate_fixed_tournament_counts(
        winner,
        n_individuals=5_000,
        n_steps=50_000,
        sample_every=1_000,
        seed=123,
        burn_in=10_000,
    )
    assert np.max(np.abs(avg_counts - (np.ones(3) / 3))) < 0.08, avg_counts
    assert np.max(np.abs(avg_counts - avg)) < 0.08, (avg, avg_counts)
    print("All tournament smoke tests passed.")


if __name__ == "__main__":
    main()
