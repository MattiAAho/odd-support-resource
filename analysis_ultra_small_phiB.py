"""Analysis of ultra-small phi_B sweep.

Combines ultra_small_phiB_v1.pkl (phi_B in 0.01..0.09, reps=500) with the
existing sparsity_sweep_v1.pkl (phi_B in 0.0..0.7, reps=200) for N=7 to trace
the early LP parity decline from phi_B=0.

Key question: does parity break immediately at phi_B > 0 (analytic: yes, A&L
requires strict completeness) or is there a statistical soft threshold near one
missing pair where LP odd drops noticeably?

For N=7 (21 pairs), one missing pair corresponds to phi_B = 1/21 ≈ 0.048.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom

MODELS = ["fixed", "bernoulli", "beta_sample"]
MODEL_COLORS = {"fixed": "#1f77b4", "bernoulli": "#ff7f0e", "beta_sample": "#2ca02c"}
MODEL_LABELS = {"fixed": "Fixed", "bernoulli": "Bernoulli", "beta_sample": "Beta-sample"}
N_PAIRS_N7 = 21  # C(7,2)


def load(path: str) -> list[dict]:
    with open(path, "rb") as fh:
        return pickle.load(fh)


def aggregate(records: list[dict], model: str, n_species: int = 7) -> list[dict]:
    sub = [r for r in records if r["model"] == model and r["n_species"] == n_species]
    phi_vals = sorted(set(r["phi_B"] for r in sub))
    rows = []
    for phi in phi_vals:
        rs = [r for r in sub if abs(r["phi_B"] - phi) < 1e-9]
        n = len(rs)
        lp_odd = np.mean([r["lp_support_odd"] for r in rs])
        se = np.sqrt(lp_odd * (1 - lp_odd) / n)
        mean_active = np.mean([r["n_active_pairs"] for r in rs])
        rows.append({"phi_B": phi, "lp_odd": lp_odd, "se": se,
                     "n": n, "mean_active_pairs": mean_active,
                     "mean_zero_pairs": N_PAIRS_N7 - mean_active})
    return rows


def expected_zero_pairs(phi_B: float) -> float:
    return phi_B * N_PAIRS_N7


def main() -> None:
    # Load and merge
    fine = load("results/ultra_small_phiB_v1.pkl")
    coarse_path = Path("results/sparsity_sweep_v1.pkl")
    coarse = load(str(coarse_path)) if coarse_path.exists() else []

    # Keep phi_B=0.0 from coarse (fine sweep doesn't include it)
    coarse_phi0 = [r for r in coarse if abs(r["phi_B"]) < 1e-9 and r["n_species"] == 7]
    combined = fine + coarse_phi0

    print("=" * 70)
    print("Ultra-small phi_B: LP odd-support fraction near phi_B=0  (N=7)")
    print("=" * 70)

    for model in MODELS:
        rows = aggregate(combined, model)
        rows_fine = [r for r in rows if r["phi_B"] <= 0.10 + 1e-9]
        print(f"\n{MODEL_LABELS[model]}:")
        print(f"  {'phi_B':>6}  {'E[#zero pairs]':>15}  {'LP odd':>8}  {'±2SE':>7}  {'n':>5}")
        print(f"  {'-'*50}")
        for r in rows_fine:
            two_se = 2 * r["se"]
            print(f"  {r['phi_B']:>6.3f}  {r['mean_zero_pairs']:>15.2f}  "
                  f"{r['lp_odd']:>8.4f}  {two_se:>7.4f}  {r['n']:>5}")

    # Statistical test: is phi_B=0.01 LP odd significantly < 1.0?
    print("\n--- Immediate break test (is phi_B=0.01 significantly below 1.0?) ---")
    for model in MODELS:
        rows = aggregate(combined, model)
        r01 = next((r for r in rows if abs(r["phi_B"] - 0.01) < 1e-9), None)
        if r01 is None:
            continue
        n_odd = int(round(r01["lp_odd"] * r01["n"]))
        n_total = r01["n"]
        # One-sided binomial test: H0: p_odd = 1.0 (can't compute p-value at boundary)
        # Use: number of even-support cases and their significance
        n_even = n_total - n_odd
        p_value = binom.sf(n_odd - 1, n_total, 1.0) if n_odd < n_total else 1.0
        # More useful: at what phi_B does the first statistically significant drop occur?
        print(f"  {MODEL_LABELS[model]}: phi_B=0.01, LP_odd={r01['lp_odd']:.4f}, "
              f"n_even={n_even}/{n_total}, 2SE={2*r01['se']:.4f}")

    print("\n--- At what phi_B does LP_odd first drop below thresholds? ---")
    for threshold in [0.999, 0.99, 0.98, 0.95, 0.90]:
        print(f"\n  Threshold LP_odd < {threshold}:")
        for model in MODELS:
            rows = aggregate(combined, model)
            for r in rows:
                if r["lp_odd"] < threshold:
                    print(f"    {MODEL_LABELS[model]}: phi_B={r['phi_B']:.3f} "
                          f"(E[zero_pairs]={r['mean_zero_pairs']:.2f}, LP_odd={r['lp_odd']:.4f})")
                    break

    # Figure
    fig, ax0 = plt.subplots(1, 1, figsize=(7, 4.5))

    for model in MODELS:
        rows_all = aggregate(combined, model)
        rows_fine = [r for r in rows_all if r["phi_B"] <= 0.10 + 1e-9]

        phi_f = [r["phi_B"] for r in rows_fine]
        lp_f = [r["lp_odd"] for r in rows_fine]
        se_f = [r["se"] for r in rows_fine]

        c = MODEL_COLORS[model]
        lbl = MODEL_LABELS[model]

        ax0.errorbar(phi_f, lp_f, yerr=[2*s for s in se_f],
                     fmt="o-", color=c, label=lbl, capsize=3, markersize=5)

    # Mark one-missing-pair threshold (phi_B = 1/21)
    ax0.axvline(1 / N_PAIRS_N7, color="gray", lw=1, ls="--", alpha=0.6,
                label=f"1 missing pair (φ_B=1/21≈{1/N_PAIRS_N7:.3f})")
    ax0.axhline(1.0, color="k", lw=0.6, ls=":", alpha=0.4)
    ax0.set_ylabel("LP odd-support fraction")
    ax0.legend(fontsize=8)

    ax0.set_xlabel("φ_B (sparsity)")
    ax0.set_title("Ultra-small φ_B: near-zero region (N=7, reps=500)\nerror bars = ±2 SE")
    ax0.set_xlim(-0.002, 0.105)
    ax0.set_ylim(0.88, 1.02)

    fig.tight_layout()
    fig.savefig("figures/fig_ultra_small_phiB.pdf", bbox_inches="tight")
    fig.savefig("figures/fig_ultra_small_phiB.png", dpi=150, bbox_inches="tight")
    print("\nSaved figures/fig_ultra_small_phiB.pdf / .png")
    plt.close(fig)


if __name__ == "__main__":
    main()
