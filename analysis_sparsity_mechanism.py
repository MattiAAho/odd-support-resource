"""Mechanism analysis for the sparsity sweep U-shape.

Re-generates interaction graphs using the same seeds as run_sparsity_sweep.py
and computes connectivity and per-component LP support parity.

Key test: if each connected component has odd LP support, then global support
parity == parity of the number of components.  This would locate the U-shape
as a graph-fragmentation effect rather than isolation-driven monodominance.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components as sp_connected_components

from tournament.generation import (
    apply_sparsity,
    factor_win_counts,
    independent_factor_ranks,
    payoff_from_winner,
    sample_fixed_tournament,
    win_probability_matrix,
)
from tournament.lp import solve_zero_sum_equilibrium

SEED_BASE = 42
INPUT = "results/sparsity_sweep_v1.pkl"
OUTPUT = "results/sparsity_mechanism_v1.pkl"
LP_TOL = 1e-3


def reconstruct_seed(phi_B: float, n_species: int, n_factors: int, rep: int) -> int:
    return (
        SEED_BASE
        + 1_000_000 * rep
        + 10_000 * n_species
        + 100 * n_factors
        + int(round(phi_B * 1000))
    )


def get_components(interact: np.ndarray) -> tuple[int, np.ndarray]:
    """Connected components of the undirected interaction graph."""
    adj = csr_matrix(interact.astype(np.uint8))
    n_comp, labels = sp_connected_components(adj, directed=False)
    return n_comp, labels


def component_lp_support(payoff: np.ndarray, indices: np.ndarray) -> int:
    """LP support size for a sub-tournament defined by species indices."""
    if len(indices) == 1:
        return 1  # isolated species: trivially support = 1
    sub = payoff[np.ix_(indices, indices)]
    try:
        x, _ = solve_zero_sum_equilibrium(sub)
        return int(np.sum(x > LP_TOL))
    except Exception:
        return -1  # solver failure; mark for exclusion


def analyse_tournament(
    phi_B: float, n_species: int, n_factors: int, rep: int,
    lp_x_stored: np.ndarray,
) -> dict:
    seed = reconstruct_seed(phi_B, n_species, n_factors, rep)
    rng = np.random.default_rng(seed)

    ranks = independent_factor_ranks(n_species, n_factors, rng)
    p_win = win_probability_matrix(ranks)
    p_sparse, interact = apply_sparsity(p_win, phi_B, rng)
    winner = sample_fixed_tournament(p_sparse, rng)
    payoff = payoff_from_winner(winner)

    n_comp, labels = get_components(interact)

    # Per-component sizes and LP support
    comp_sizes = []
    comp_lp_sizes = []
    for c in range(n_comp):
        idx = np.where(labels == c)[0]
        comp_sizes.append(len(idx))
        comp_lp_sizes.append(component_lp_support(payoff, idx))

    comp_sizes = np.array(comp_sizes)
    comp_lp_sizes = np.array(comp_lp_sizes)
    valid = comp_lp_sizes >= 0

    global_lp_size = int(np.sum(lp_x_stored > LP_TOL))
    global_lp_odd = global_lp_size % 2 == 1

    # Test: sum of per-component support sizes
    per_comp_sum = int(comp_lp_sizes[valid].sum()) if valid.any() else -1

    # Prediction: n_comp % 2 == 1 → global odd (if each component is odd)
    all_comp_odd = bool(np.all(comp_lp_sizes[valid] % 2 == 1)) if valid.any() else False
    pred_odd_from_n_comp = (n_comp % 2 == 1)

    return {
        "phi_B": phi_B,
        "n_species": n_species,
        "n_factors": n_factors,
        "rep": rep,
        "n_components": n_comp,
        "n_isolated": int(np.sum(comp_sizes == 1)),
        "largest_comp_size": int(comp_sizes.max()),
        "comp_sizes": comp_sizes.tolist(),
        "comp_lp_sizes": comp_lp_sizes.tolist(),
        "global_lp_size": global_lp_size,
        "global_lp_odd": global_lp_odd,
        "all_comp_odd": all_comp_odd,
        "per_comp_lp_sum": per_comp_sum,
        "pred_odd_from_n_comp": pred_odd_from_n_comp,
        "pred_matches_global": pred_odd_from_n_comp == global_lp_odd,
    }


def main() -> None:
    print(f"Loading {INPUT} ...")
    with open(INPUT, "rb") as fh:
        records = pickle.load(fh)

    # Index fixed-model records by (phi_B, n_species, n_factors, rep)
    fixed_records = {
        (r["phi_B"], r["n_species"], r["n_factors"], r["rep"]): r
        for r in records if r["model"] == "fixed"
    }
    print(f"  {len(fixed_records)} fixed-model records")

    keys = sorted(fixed_records.keys())
    results = []
    for i, key in enumerate(keys):
        if (i + 1) % 500 == 0:
            print(f"  [{i+1}/{len(keys)}]")
        phi_B, ns, nf, rep = key
        r = fixed_records[key]
        res = analyse_tournament(phi_B, ns, nf, rep, np.array(r["lp_x"]))
        results.append(res)

    with open(OUTPUT, "wb") as fh:
        pickle.dump(results, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved {OUTPUT} ({len(results)} records)")

    # Quick summary
    import pandas as pd
    df = pd.DataFrame(results)
    print("\n=== Mean n_components by phi_B (pooled N and F) ===")
    print(df.groupby("phi_B")["n_components"].mean().round(2).to_string())
    print("\n=== Fraction where pred (n_comp odd) matches global LP odd ===")
    print(df.groupby("phi_B")["pred_matches_global"].mean().round(3).to_string())
    print("\n=== Fraction where all components have odd LP support ===")
    print(df.groupby("phi_B")["all_comp_odd"].mean().round(3).to_string())
    print("\n=== Mean n_isolated by phi_B ===")
    print(df.groupby("phi_B")["n_isolated"].mean().round(2).to_string())


if __name__ == "__main__":
    main()
