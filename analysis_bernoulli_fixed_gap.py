"""Bernoulli-vs-fixed LP parity gap: is K-matrix sampling noise the cause?

For each paired replicate (same rank_seed as run_sparsity_sweep.py), we compute:
  K_exp  = payoff_from_probabilities(p_sparse)  -- the Bernoulli LP reference
  K_samp = payoff_from_winner(winner)            -- the fixed-model LP matrix

Both LPs are solved; parity_disagree flags when they give different support parity.

Two K-distance metrics are computed per replicate:
  frob_norm         -- Frobenius norm of (K_samp - K_exp) / n_active_pairs
  sign_disagree_frac -- fraction of active pairs where sign(K_samp) != sign(K_exp)

The hypothesis under test: parity disagreement is explained by sampling noise in K.
If true, both distance metrics should be significantly larger in disagreement cases.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

from tournament.generation import (
    apply_sparsity,
    independent_factor_ranks,
    payoff_from_probabilities,
    payoff_from_winner,
    sample_fixed_tournament,
    win_probability_matrix,
)
from tournament.lp import solve_zero_sum_equilibrium


def support_size(x: np.ndarray, tol: float = 1e-3) -> int:
    return int(np.sum(x > tol))


def k_distances(
    K_samp: np.ndarray,
    K_exp: np.ndarray,
    interact: np.ndarray,
) -> tuple[float, float, int]:
    """Return (frob_norm, sign_disagree_frac, n_active_pairs)."""
    n = K_samp.shape[0]
    frob_raw = float(np.linalg.norm(K_samp - K_exp, "fro"))
    sign_count = 0
    n_active = 0
    for a in range(n):
        for b in range(a + 1, n):
            if interact[a, b]:
                n_active += 1
                if np.sign(K_samp[a, b]) != np.sign(K_exp[a, b]):
                    sign_count += 1
    frob_norm = frob_raw / max(n_active, 1)
    sign_frac = sign_count / max(n_active, 1)
    return frob_norm, sign_frac, n_active


def run_analysis(args: argparse.Namespace) -> list[dict]:
    records = []
    for phi_B in args.phi_b:
        for n_species in args.species:
            for n_factors in args.factors:
                for rep in range(args.reps):
                    seed = (
                        args.seed_base
                        + 1_000_000 * rep
                        + 10_000 * n_species
                        + 100 * n_factors
                        + int(round(phi_B * 1000))
                    )
                    rng = np.random.default_rng(seed)

                    # Reproduce exact tournament structure (same order as run_sparsity_sweep.py)
                    ranks = independent_factor_ranks(n_species, n_factors, rng)
                    p_win = win_probability_matrix(ranks)
                    p_sparse, interact = apply_sparsity(p_win, phi_B, rng)

                    # Bernoulli LP: uses expected K — no RNG consumed
                    K_exp = payoff_from_probabilities(p_sparse)
                    lp_exp, _ = solve_zero_sum_equilibrium(K_exp)
                    supp_exp = support_size(lp_exp)

                    # Fixed LP: sample winner (continues same RNG as original sweep)
                    winner = sample_fixed_tournament(p_sparse, rng)
                    K_samp = payoff_from_winner(winner)
                    lp_samp, _ = solve_zero_sum_equilibrium(K_samp)
                    supp_samp = support_size(lp_samp)

                    parity_disagree = (supp_samp % 2 == 1) != (supp_exp % 2 == 1)

                    frob_norm, sign_frac, n_active = k_distances(K_samp, K_exp, interact)

                    records.append({
                        "phi_B": phi_B,
                        "n_species": n_species,
                        "n_factors": n_factors,
                        "rep": rep,
                        "n_active_pairs": n_active,
                        "n_total_pairs": n_species * (n_species - 1) // 2,
                        "supp_fixed": supp_samp,
                        "supp_bernoulli": supp_exp,
                        "fixed_odd": supp_samp % 2 == 1,
                        "bernoulli_odd": supp_exp % 2 == 1,
                        "parity_disagree": parity_disagree,
                        "frob_norm": frob_norm,
                        "sign_disagree_frac": sign_frac,
                    })

    return records


def print_summary(records: list[dict]) -> None:
    from scipy.stats import pointbiserialr, mannwhitneyu

    print("\n=== Parity disagreement rate by phi_B ===")
    phi_b_vals = sorted(set(r["phi_B"] for r in records))
    for phi_B in phi_b_vals:
        sub = [r for r in records if r["phi_B"] == phi_B]
        rate = sum(r["parity_disagree"] for r in sub) / len(sub)
        print(f"  phi_B={phi_B:.2f}  disagree={rate:.3f}  N={len(sub)}")

    focus = [r for r in records if r["phi_B"] == 0.3]
    if not focus:
        return

    print("\n=== K-distance at phi_B=0.3: agree vs disagree ===")
    agree = [r for r in focus if not r["parity_disagree"]]
    disagree = [r for r in focus if r["parity_disagree"]]
    for col in ("frob_norm", "sign_disagree_frac"):
        v_agree = [r[col] for r in agree]
        v_dis = [r[col] for r in disagree]
        print(f"  {col:24s}  agree mean={np.mean(v_agree):.4f}  "
              f"disagree mean={np.mean(v_dis):.4f}  "
              f"N_dis={len(disagree)}")
        if len(disagree) >= 5:
            stat, p = mannwhitneyu(v_agree, v_dis, alternative="two-sided")
            r_pb, p_pb = pointbiserialr(
                [float(r["parity_disagree"]) for r in focus],
                [r[col] for r in focus],
            )
            print(f"  {'':24s}  MW p={p:.3e}  r_pb={r_pb:.3f} (p={p_pb:.3e})")

    print("\n=== Correlation across all phi_B values ===")
    for phi_B in phi_b_vals:
        sub = [r for r in records if r["phi_B"] == phi_B]
        n_dis = sum(r["parity_disagree"] for r in sub)
        if n_dis < 3:
            print(f"  phi_B={phi_B:.2f}  (too few disagreements: {n_dis})")
            continue
        for col in ("frob_norm", "sign_disagree_frac"):
            y = [float(r["parity_disagree"]) for r in sub]
            x = [r[col] for r in sub]
            r_pb, p_pb = pointbiserialr(y, x)
            print(f"  phi_B={phi_B:.2f}  {col:24s}  r_pb={r_pb:.3f}  p={p_pb:.3e}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--phi-b", nargs="+", type=float,
                   default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    p.add_argument("--species", nargs="+", type=int, default=[7])
    p.add_argument("--factors", nargs="+", type=int, default=[3, 5, 7])
    p.add_argument("--reps", type=int, default=200)
    p.add_argument("--seed-base", type=int, default=42)
    p.add_argument("--out", default="results/bernoulli_fixed_gap_v1.pkl")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    n_recs = len(args.phi_b) * len(args.species) * len(args.factors) * args.reps
    print(f"Computing K-matrix distances for {n_recs} replicates "
          f"(phi_B={args.phi_b}, S={args.species}, F={args.factors}, reps={args.reps})")
    records = run_analysis(args)
    print_summary(records)

    with out.open("wb") as fh:
        pickle.dump(records, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\nSaved {out} ({len(records)} records)")


if __name__ == "__main__":
    main()
