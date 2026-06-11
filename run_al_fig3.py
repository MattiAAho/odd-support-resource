"""A&L (2011) Fig. 3 strict reproduction — LP support size vs F, three correlation models.

Reproduces all three panels of Allesina & Levine (2011) Fig. 3 using LP Nash equilibrium
support size in place of their LV-dynamics survivor count.

  Panel A — independent:          x_ij ~ U(0,1) iid (our standard f-factor model)
  Panel B — positive correlation:  x_ij = x_i(j-1) + U(0,1), cumulative per species
  Panel C — trade-off:             sum_k x_ik = 1, Dirichlet(1,...,1) per species

For each replicate: generate x values, compute p_ij, draw ONE fixed tournament (Bernoulli
winner per pair), solve LP, record support size.  This matches A&L's procedure of drawing
a tournament and then running dynamics to equilibrium.

A&L's key claim: fixed-tournament LP support converges to N/2 (iid random tournament
baseline) as F increases.  For model C even F, p_ij = 0.5 for all pairs, so the sampled
fixed tournament IS an iid random tournament, reaching N/2 immediately.

Also computes iid random tournament LP (F→∞ proxy) per N for reference lines.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

from tournament.generation import (
    autocorrelated_factor_values,
    independent_factor_ranks,
    payoff_from_winner,
    random_iid_payoff,
    sample_fixed_tournament,
    tradeoff_factor_values,
    win_probabilities_from_values,
    win_probability_matrix,
)
from tournament.lp import solve_zero_sum_equilibrium

MODEL_A = "independent"
MODEL_B = "positive_correlation"
MODEL_C = "tradeoff"
MODEL_IID = "iid"
MODELS = (MODEL_A, MODEL_B, MODEL_C)


def support_size(x: np.ndarray, tol: float = 1e-3) -> int:
    return int(np.sum(x > tol))


def make_p_win(model: str, n_species: int, n_factors: int,
               rng: np.random.Generator) -> np.ndarray:
    if model == MODEL_A:
        ranks = independent_factor_ranks(n_species, n_factors, rng)
        return win_probability_matrix(ranks)
    if model == MODEL_B:
        x = autocorrelated_factor_values(n_species, n_factors, rng)
        return win_probabilities_from_values(x)
    if model == MODEL_C:
        x = tradeoff_factor_values(n_species, n_factors, rng)
        return win_probabilities_from_values(x)
    raise ValueError(f"Unknown model: {model}")


def run_sweep(args: argparse.Namespace) -> list[dict]:
    records = []

    for n_species in args.species:
        for n_factors in args.factors:
            for m_idx, model in enumerate(MODELS):
                for rep in range(args.reps):
                    seed = (args.seed_base
                            + 1_000_000 * rep
                            + 10_000 * n_species
                            + 100 * n_factors
                            + m_idx)
                    rng = np.random.default_rng(seed)

                    p_win = make_p_win(model, n_species, n_factors, rng)
                    winner = sample_fixed_tournament(p_win, rng)
                    K = payoff_from_winner(winner)

                    try:
                        lp_x, _ = solve_zero_sum_equilibrium(K)
                        s = support_size(lp_x)
                        degen = bool(np.max(np.abs(K)) < 1e-10)
                    except RuntimeError:
                        s = 1
                        degen = True

                    records.append({
                        "n_species": n_species,
                        "n_factors": n_factors,
                        "model": model,
                        "rep": rep,
                        "lp_support_size": s,
                        "lp_support_frac": s / n_species,
                        "lp_support_odd": s % 2 == 1,
                        "lp_degenerate": degen,
                    })

        # iid random tournament: F→∞ proxy; seed slot n_factors=99 (not in args.factors)
        for rep in range(args.reps):
            seed = (args.seed_base
                    + 1_000_000 * rep
                    + 10_000 * n_species
                    + 99)
            rng = np.random.default_rng(seed)
            K = random_iid_payoff(n_species, rng)
            try:
                lp_x, _ = solve_zero_sum_equilibrium(K)
                s = support_size(lp_x)
            except RuntimeError:
                s = 1
            records.append({
                "n_species": n_species,
                "n_factors": -1,
                "model": MODEL_IID,
                "rep": rep,
                "lp_support_size": s,
                "lp_support_frac": s / n_species,
                "lp_support_odd": s % 2 == 1,
                "lp_degenerate": False,
            })

    return records


def print_summary(records: list[dict]) -> None:
    species_vals = sorted(set(r["n_species"] for r in records))
    factor_vals = sorted(f for f in set(r["n_factors"] for r in records) if f > 0)

    for model in (*MODELS, MODEL_IID):
        print(f"\n=== {model}: mean LP support size ===")
        header = "     " + "".join(f"  N={N:2d}" for N in species_vals)
        print(header)
        if model == MODEL_IID:
            row = "  iid"
            for N in species_vals:
                sub = [r for r in records if r["model"] == model and r["n_species"] == N]
                row += f"  {np.mean([r['lp_support_size'] for r in sub]):5.2f}"
            print(row)
        else:
            for F in factor_vals:
                row = f"  F={F:2d}"
                for N in species_vals:
                    sub = [r for r in records
                           if r["model"] == model and r["n_species"] == N
                           and r["n_factors"] == F]
                    row += f"  {np.mean([r['lp_support_size'] for r in sub]):5.2f}"
                print(row)

    print("\n=== Reference: N/2 (A&L iid theoretical expectation) ===")
    for N in species_vals:
        sub_iid = [r for r in records if r["model"] == MODEL_IID and r["n_species"] == N]
        iid_mean = np.mean([r["lp_support_size"] for r in sub_iid])
        print(f"  N={N}: theoretical N/2={N/2:.1f},  empirical iid LP={iid_mean:.2f}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--factors", nargs="+", type=int,
                   default=[1, 2, 3, 5, 7, 10, 15, 20])
    p.add_argument("--species", nargs="+", type=int, default=[10, 20, 30])
    p.add_argument("--reps", type=int, default=1000)
    p.add_argument("--seed-base", type=int, default=42)
    p.add_argument("--out", default="results/al_fig3_v1.pkl")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    n_recs = (len(args.factors) * len(args.species) * len(MODELS)
              + len(args.species)) * args.reps
    print(
        f"A&L Fig. 3 reproduction: N={args.species}, F={args.factors}, "
        f"models={MODELS}, reps={args.reps} → {n_recs} records"
    )
    records = run_sweep(args)
    print_summary(records)

    with out.open("wb") as fh:
        pickle.dump(records, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\nSaved {out} ({len(records)} records)")


if __name__ == "__main__":
    main()
