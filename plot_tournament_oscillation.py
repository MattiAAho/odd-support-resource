"""A&L-style oscillation figure for a single fixed tournament run."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tournament.generation import (
    independent_factor_ranks,
    payoff_from_winner,
    sample_fixed_tournament,
    win_probability_matrix,
)
from tournament.kernels import simulate_fixed_tournament
from tournament.lp import solve_zero_sum_equilibrium


COLORS = [
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#ff7f0e",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def species_labels(n_species: int) -> list[str]:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return [letters[i] if i < len(letters) else f"S{i}" for i in range(n_species)]


def find_interesting_fixed_tournament(
    n_species: int,
    n_factors: int,
    min_support: int,
    max_tries: int,
    seed_base: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Search for a fixed tournament with nontrivial LP support."""

    for k in range(max_tries):
        seed = seed_base + k
        rng = np.random.default_rng(seed)
        ranks = independent_factor_ranks(n_species, n_factors, rng)
        p_win = win_probability_matrix(ranks)
        winner = sample_fixed_tournament(p_win, rng)
        eq, _ = solve_zero_sum_equilibrium(payoff_from_winner(winner))
        support = int(np.sum(eq > 1e-6))
        if support >= min_support and np.max(eq) < 0.95:
            return ranks, winner, eq, seed
    raise RuntimeError(f"no tournament found with support >= {min_support} in {max_tries} tries")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-species", type=int, default=7)
    parser.add_argument("--n-factors", type=int, default=3)
    parser.add_argument("--n-individuals", type=int, default=25_000)
    parser.add_argument("--n-steps", type=int, default=10_000_000)
    parser.add_argument("--sample-every", type=int, default=10_000)
    parser.add_argument("--burn-in", type=int, default=500_000)
    parser.add_argument("--min-support", type=int, default=3)
    parser.add_argument("--max-tries", type=int, default=5000)
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument("--out", default="figures/fig_tournament_al_oscillation.pdf")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    ranks, winner, eq, tournament_seed = find_interesting_fixed_tournament(
        n_species=args.n_species,
        n_factors=args.n_factors,
        min_support=args.min_support,
        max_tries=args.max_tries,
        seed_base=args.seed_base,
    )

    sim_seed = tournament_seed + 100_000
    times, freqs, avg = simulate_fixed_tournament(
        winner,
        n_individuals=args.n_individuals,
        n_steps=args.n_steps,
        sample_every=args.sample_every,
        seed=sim_seed,
        burn_in=args.burn_in,
    )

    labels = species_labels(args.n_species)
    support = int(np.sum(eq > 1e-6))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)

    ax = axes[0]
    for i in range(args.n_species):
        ax.plot(times, freqs[:, i], lw=1.3, color=COLORS[i % len(COLORS)], label=labels[i])
    ax.set_title("A&L-style fixed tournament simulation")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Species frequency")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.2)
    ax.legend(ncol=2, fontsize=8, frameon=False)

    ax = axes[1]
    s = np.arange(args.n_species)
    width = 0.38
    ax.bar(s - width / 2, eq, width=width, color="#222222", alpha=0.9, label="Exact LP")
    ax.bar(s + width / 2, avg, width=width, color="#7aa6d1", alpha=0.9, label="Time-average")
    ax.set_title("Exact LP vs simulated time-average")
    ax.set_xlabel("Species")
    ax.set_ylabel("Frequency")
    ax.set_xticks(s)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)

    fig.suptitle(
        f"Fixed tournament, n_species={args.n_species}, n_factors={args.n_factors}, "
        f"LP support={support}, tournament_seed={tournament_seed}",
        fontsize=12,
    )
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=160, bbox_inches="tight")

    print(f"Saved {out}")
    print(
        f"tournament_seed={tournament_seed} sim_seed={sim_seed} "
        f"support={support} avg_lp_half_l1={np.abs(avg - eq).sum() / 2.0:.4f}"
    )
    print("LP:", np.round(eq, 4))
    print("AVG:", np.round(avg, 4))


if __name__ == "__main__":
    main()
