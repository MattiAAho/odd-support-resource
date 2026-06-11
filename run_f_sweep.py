"""F sweep: how does LP support distribution change with the number of limiting factors?

At phi_B=0 (complete tournament), A&L guarantees odd LP support.  The distribution
of support sizes depends on F: F=1 gives a fully transitive tournament (one species
wins every factor comparison → total order → support=1 always); larger F makes win
probabilities more balanced, increasing the expected support size.

Transitivity fraction tau = 1 - n_cyclic_triples / C(N,3) is computed from the winner
matrix via trace(W^3)/3.  At F=1, tau=1.0.  At F→∞, tau converges to ~0.91, not 3/4.
The 3/4 limit applies to i.i.d. fair-coin tournaments; the factor-rank model retains
structural transitivity at large F because ranks aggregate (a species ranked high on
many factors still tends to beat low-ranked species), so the empirical limit is higher.

LP-only (no simulation).  Seed formula matches run_sparsity_sweep.py at phi_B=0,
so F∈{3,5,7} results are directly comparable with existing sparsity_sweep_v1.pkl.

Only odd F values are swept to avoid ties (p_ab = p_ba = 0.5) that would confound
the result with the phi_A neutrality mechanism.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

from tournament.generation import (
    independent_factor_ranks,
    payoff_from_probabilities,
    payoff_from_winner,
    sample_fixed_tournament,
    win_probability_matrix,
)
from tournament.lp import solve_zero_sum_equilibrium


def support_size(x: np.ndarray, tol: float = 1e-3) -> int:
    return int(np.sum(x > tol))


def transitivity_fraction(winner: np.ndarray) -> float:
    """tau = 1 - n_cyclic_triples / C(N,3).

    Uses the identity: sum_{ordered (a,b,c) distinct} W[a,b]*W[b,c]*W[c,a] = 3 * n_cyclic,
    where n_cyclic counts unordered cyclic triples.  That sum equals trace(W^3).
    """
    n = winner.shape[0]
    n_total = n * (n - 1) * (n - 2) // 6
    if n_total == 0:
        return 1.0
    W = winner.astype(np.float64)
    n_cyclic = round(np.trace(W @ W @ W) / 3.0)
    return 1.0 - n_cyclic / n_total


def winner_from_payoff_sign(payoff: np.ndarray) -> np.ndarray:
    """Boolean winner matrix from sign of payoff — used for Bernoulli transitivity."""
    return payoff > 0


def run_sweep(args: argparse.Namespace) -> list[dict]:
    records = []
    for n_species in args.species:
        for n_factors in args.factors:
            for rep in range(args.reps):
                seed = (
                    args.seed_base
                    + 1_000_000 * rep
                    + 10_000 * n_species
                    + 100 * n_factors
                )
                rng = np.random.default_rng(seed)

                ranks = independent_factor_ranks(n_species, n_factors, rng)
                p_win = win_probability_matrix(ranks)

                base = {
                    "n_species": n_species,
                    "n_factors": n_factors,
                    "rep": rep,
                    "rank_seed": seed,
                }

                # Fixed model: sample one tournament, compute LP + transitivity
                winner = sample_fixed_tournament(p_win, rng)
                K_fixed = payoff_from_winner(winner)
                lp_fixed, _ = solve_zero_sum_equilibrium(K_fixed)
                supp_fixed = support_size(lp_fixed)
                tau_fixed = transitivity_fraction(winner)

                records.append({
                    **base,
                    "model": "fixed",
                    "lp_support_size": supp_fixed,
                    "lp_support_odd": supp_fixed % 2 == 1,
                    "lp_support_frac": supp_fixed / n_species,
                    "transitivity": tau_fixed,
                })

                # Bernoulli model: use expected payoff K = p - p^T
                K_bern = payoff_from_probabilities(p_win)
                lp_bern, _ = solve_zero_sum_equilibrium(K_bern)
                supp_bern = support_size(lp_bern)
                # Transitivity of the Bernoulli "effective tournament" (sign of K)
                winner_bern = winner_from_payoff_sign(K_bern)
                tau_bern = transitivity_fraction(winner_bern)

                records.append({
                    **base,
                    "model": "bernoulli",
                    "lp_support_size": supp_bern,
                    "lp_support_odd": supp_bern % 2 == 1,
                    "lp_support_frac": supp_bern / n_species,
                    "transitivity": tau_bern,
                })

    return records


def print_summary(records: list[dict]) -> None:
    factor_vals = sorted(set(r["n_factors"] for r in records))
    species_vals = sorted(set(r["n_species"] for r in records))

    print("\n=== Mean LP support fraction (support/N) by F and N [bernoulli] ===")
    header = "  F  " + "".join(f"   N={N}" for N in species_vals)
    print(header)
    for F in factor_vals:
        row = f"  {F:2d} "
        for N in species_vals:
            sub = [r for r in records
                   if r["n_factors"] == F and r["n_species"] == N
                   and r["model"] == "bernoulli"]
            if sub:
                row += f"  {np.mean([r['lp_support_frac'] for r in sub]):.3f}"
            else:
                row += "    —  "
        print(row)

    print("\n=== Mean transitivity (tau) by F and N [bernoulli] ===")
    print(header)
    for F in factor_vals:
        row = f"  {F:2d} "
        for N in species_vals:
            sub = [r for r in records
                   if r["n_factors"] == F and r["n_species"] == N
                   and r["model"] == "bernoulli"]
            if sub:
                row += f"  {np.mean([r['transitivity'] for r in sub]):.3f}"
            else:
                row += "    —  "
        print(row)

    print("\n=== LP parity check (should be 1.000 everywhere) ===")
    for model in ("fixed", "bernoulli"):
        rates = [r["lp_support_odd"] for r in records if r["model"] == model]
        print(f"  {model:10s}  odd={np.mean(rates):.4f}  N={len(rates)}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--factors", nargs="+", type=int,
                   default=[1, 3, 5, 7, 11, 15, 21, 31])
    p.add_argument("--species", nargs="+", type=int, default=[7, 9, 11, 13])
    p.add_argument("--reps", type=int, default=500)
    p.add_argument("--seed-base", type=int, default=42)
    p.add_argument("--out", default="results/f_sweep_v1.pkl")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    n_recs = len(args.factors) * len(args.species) * args.reps * 2
    print(
        f"F sweep (LP-only): F={args.factors}, N={args.species}, "
        f"reps={args.reps} → {n_recs} records"
    )
    records = run_sweep(args)
    print_summary(records)

    with out.open("wb") as fh:
        pickle.dump(records, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\nSaved {out} ({len(records)} records)")


if __name__ == "__main__":
    main()
