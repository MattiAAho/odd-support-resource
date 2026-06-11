"""Unification check: does phi_B vs phi_A agreement survive all selection rules?

Finding 3 (v2) showed that, under HiGHS, phi_B and phi_A give nearly identical
LP odd fractions at matched zero-pair density — the 'three-decimal-place
agreement' used as evidence for a shared zero-payoff mechanism.

This script asks whether the agreement holds for all four selection rules
(HiGHS, min-sup, max-sup, uniform).  If it does, the shared mechanism is
selection-rule independent.  If it breaks for some rules, the unification is
HiGHS-specific.

Data source: nonuniqueness_sweep_v1.pkl (phi_B and phi_A at N=7,9, reps=500).
The zero-pair density for each record is estimated as:
  E[zero pairs] = actual_phi * n_pairs   (phi_B: zeros; phi_A: neutrals)
Both create K_ij = 0 in the Bernoulli LP, so they should collapse.
"""

from __future__ import annotations

import argparse
import pickle
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RULES = ["HiGHS", "min-sup", "max-sup", "uniform"]
STYLES_PHI_B = {
    "HiGHS":   dict(color="C0", marker="o", ms=7, ls="-",  lw=1.8),
    "min-sup": dict(color="C2", marker="^", ms=7, ls="--", lw=1.4),
    "max-sup": dict(color="C3", marker="v", ms=7, ls=":",  lw=1.4),
    "uniform": dict(color="C1", marker="s", ms=6, ls="-.", lw=1.4),
}
STYLES_PHI_A = {
    "HiGHS":   dict(color="C0", marker="o", ms=7, ls="-",  lw=1.8, alpha=0.45, mfc="white"),
    "min-sup": dict(color="C2", marker="^", ms=7, ls="--", lw=1.4, alpha=0.45, mfc="white"),
    "max-sup": dict(color="C3", marker="v", ms=7, ls=":",  lw=1.4, alpha=0.45, mfc="white"),
    "uniform": dict(color="C1", marker="s", ms=6, ls="-.", lw=1.4, alpha=0.45, mfc="white"),
}


def parity_values(r: dict) -> dict[str, float]:
    all_sizes = r["all_sizes"]
    return {
        "HiGHS":   float(r["highs_sz"] % 2 == 1),
        "min-sup": float(min(all_sizes) % 2 == 1),
        "max-sup": float(max(all_sizes) % 2 == 1),
        "uniform": sum(sz % 2 == 1 for sz in all_sizes) / len(all_sizes),
    }


def aggregate_by_density(records: list[dict]) -> dict:
    """Aggregate mean parity per rule, keyed by (param_type, n_species, expected_zero_pairs)."""
    n_pairs_for = {n: n * (n - 1) // 2 for n in set(r["n_species"] for r in records)}
    buckets: dict = defaultdict(lambda: defaultdict(list))
    for r in records:
        n = r["n_species"]
        n_pairs = n_pairs_for[n]
        e_zeros = round(r["actual_phi"] * n_pairs, 4)
        key = (r["param_type"], n, e_zeros)
        pv = parity_values(r)
        for rule, val in pv.items():
            buckets[key][rule].append(val)
    return {k: {rule: float(np.mean(v)) for rule, v in rv.items()}
            for k, rv in buckets.items()}


def print_unification_table(agg: dict, n: int) -> None:
    """Print phi_B vs phi_A comparison at matched E[zero pairs]."""
    phi_b_densities = sorted(set(
        e for (pt, ns, e), _ in agg.items() if pt == "phi_B" and ns == n
    ))
    phi_a_densities = sorted(set(
        e for (pt, ns, e), _ in agg.items() if pt == "phi_A" and ns == n
    ))
    # Find matched points (within a small tolerance)
    tol = 0.5
    matches = []
    for db in phi_b_densities:
        for da in phi_a_densities:
            if abs(db - da) <= tol:
                matches.append((db, da))

    if not matches:
        print(f"  N={n}: no matched E[zero pairs] in sweep grid.")
        return

    print(f"\nN={n}  (phi_B vs phi_A at matched E[zero pairs])")
    print(f"  {'E[z]':>6}  " + "  ".join(f"{r+' B':>11}  {r+' A':>11}  {'Δ':>6}" for r in RULES))
    print("  " + "-" * (6 + 2 + len(RULES) * 32))
    for db, da in sorted(matches):
        row_b = agg.get(("phi_B", n, db), {})
        row_a = agg.get(("phi_A", n, da), {})
        vals = []
        for r in RULES:
            vb = row_b.get(r, float("nan"))
            va = row_a.get(r, float("nan"))
            delta = vb - va
            vals.append(f"{vb:>11.3f}  {va:>11.3f}  {delta:>+6.3f}")
        print(f"  {db:>6.2f}  " + "  ".join(vals))


def plot_unification(agg: dict, species: list[int], out_path: str) -> None:
    fig, axes = plt.subplots(1, len(species), figsize=(6 * len(species), 5), sharey=True)
    if len(species) == 1:
        axes = [axes]

    for ax, n in zip(axes, species):
        # phi_B series (solid markers)
        phi_b_pts = sorted(
            (e, vals) for (pt, ns, e), vals in agg.items()
            if pt == "phi_B" and ns == n
        )
        # phi_A series (hollow markers)
        phi_a_pts = sorted(
            (e, vals) for (pt, ns, e), vals in agg.items()
            if pt == "phi_A" and ns == n
        )

        for rule in RULES:
            if phi_b_pts:
                xs = [e for e, _ in phi_b_pts]
                ys = [v.get(rule, np.nan) for _, v in phi_b_pts]
                label_b = rule if n == species[0] else None
                ax.plot(xs, ys, label=label_b, **STYLES_PHI_B[rule])
            if phi_a_pts:
                xs = [e for e, _ in phi_a_pts]
                ys = [v.get(rule, np.nan) for _, v in phi_a_pts]
                ax.plot(xs, ys, **STYLES_PHI_A[rule])

        ax.axhline(1.0, color="grey", lw=0.6, ls="--", alpha=0.5)
        ax.set_xlabel("E[zero pairs in K]", fontsize=11)
        ax.set_title(f"N={n}: unification check", fontsize=11)
        ax.set_ylim(-0.05, 1.08)

    axes[0].set_ylabel("LP odd-support fraction", fontsize=10)

    # Legend — phi_B (solid) vs phi_A (hollow)
    from matplotlib.lines import Line2D
    legend_rules = [
        Line2D([0], [0], **{**STYLES_PHI_B[r], "label": r}) for r in RULES
    ]
    legend_axis = [
        Line2D([0], [0], color="k", lw=1.5, ls="-", label="φ_B (solid)"),
        Line2D([0], [0], color="k", lw=1.5, ls="-", alpha=0.45, label="φ_A (hollow)"),
    ]
    fig.legend(handles=legend_rules + legend_axis,
               loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.08))

    fig.suptitle("Unification: φ_B and φ_A at matched zero-pair density", fontsize=12, y=1.02)
    fig.tight_layout()
    import os
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_path}")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pkl", default="results/nonuniqueness_sweep_v1.pkl")
    p.add_argument("--out", default="figures/fig_unification_check.pdf")
    args = p.parse_args()

    with open(args.pkl, "rb") as f:
        records = pickle.load(f)
    print(f"Loaded {len(records)} records.")

    agg = aggregate_by_density(records)
    species = sorted(set(r["n_species"] for r in records))

    print("\n=== Unification check: phi_B vs phi_A at matched E[zero pairs] ===")
    print("(Rule B = phi_B series, Rule A = phi_A series, Δ = B - A)")
    print("Agreement = Δ close to 0 across all rules => mechanism is rule-independent")
    for n in species:
        print_unification_table(agg, n)

    plot_unification(agg, species, args.out)

    # Verdict
    print("\n=== Verdict ===")
    for n in species:
        n_pairs = n * (n - 1) // 2
        phi_b_agg = {e: v for (pt, ns, e), v in agg.items() if pt == "phi_B" and ns == n}
        phi_a_agg = {e: v for (pt, ns, e), v in agg.items() if pt == "phi_A" and ns == n}
        deltas: dict[str, list[float]] = defaultdict(list)
        for e_b, vb in phi_b_agg.items():
            for e_a, va in phi_a_agg.items():
                if abs(e_b - e_a) < 0.5:
                    for rule in RULES:
                        if rule in vb and rule in va:
                            deltas[rule].append(abs(vb[rule] - va[rule]))
        print(f"\nN={n} mean |Δ| at matched density points:")
        for rule in RULES:
            if deltas[rule]:
                print(f"  {rule:<12}: {np.mean(deltas[rule]):.4f}  (max {max(deltas[rule]):.4f})")


if __name__ == "__main__":
    main()
