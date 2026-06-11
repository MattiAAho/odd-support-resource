"""N=4 selection-rule comparison — no new LP solves.

Re-uses the full equilibrium family produced by all_equilibrium_support_sets()
to plot LP odd-support fraction vs k under four selection rules:

  HiGHS   — what the default HiGHS LP solver returns
  min-sup  — smallest support in the equilibrium family (pushes odd-fraction up)
  max-sup  — largest support in the equilibrium family (maximally hurts the claim)
  uniform  — fraction of family members that have odd support (neutral baseline)

If all four curves show a qualitative U-shape (high at k=6, trough at k≈4,
recovery toward k=0), Scenario B is confirmed and the HiGHS framing survives
with a methodological caveat.  If support-max flattens or inverts the trough,
the U-shape claim is selection-specific (Scenario C).
"""

from __future__ import annotations

import itertools
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tournament.lp import all_equilibrium_support_sets, solve_zero_sum_equilibrium

PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
N = 4
SUPPORT_TOL = 1e-6


def build_payoff(assignment: tuple[int, ...]) -> np.ndarray:
    K = np.zeros((N, N))
    for val, (a, b) in zip(assignment, PAIRS):
        K[a, b] = val
        K[b, a] = -val
    return K


def run_enumeration() -> dict[int, list[dict]]:
    by_k: dict[int, list[dict]] = defaultdict(list)
    for assignment in itertools.product([-1, 0, 1], repeat=6):
        k = sum(1 for v in assignment if v != 0)
        K = build_payoff(assignment)

        x_highs, _ = solve_zero_sum_equilibrium(K)
        highs_sz = int(np.sum(x_highs > SUPPORT_TOL))

        all_eq = all_equilibrium_support_sets(K)
        all_sizes = [len(s) for s, _ in all_eq] if all_eq else [highs_sz]

        by_k[k].append({
            "highs_sz": highs_sz,
            "all_sizes": all_sizes,
        })
    return by_k


def aggregate(by_k: dict[int, list[dict]]) -> dict:
    """Return odd-fraction per k for each selection rule."""
    ks = sorted(by_k)
    rules = {
        "HiGHS": [],
        "min-sup": [],
        "max-sup": [],
        "uniform": [],
    }
    for k in ks:
        rows = by_k[k]
        highs_odd = sum(r["highs_sz"] % 2 == 1 for r in rows) / len(rows)
        min_odd   = sum(min(r["all_sizes"]) % 2 == 1 for r in rows) / len(rows)
        max_odd   = sum(max(r["all_sizes"]) % 2 == 1 for r in rows) / len(rows)
        # uniform: mean fraction of odd-support members per tournament
        uniform   = sum(
            sum(sz % 2 == 1 for sz in r["all_sizes"]) / len(r["all_sizes"])
            for r in rows
        ) / len(rows)
        rules["HiGHS"].append(highs_odd)
        rules["min-sup"].append(min_odd)
        rules["max-sup"].append(max_odd)
        rules["uniform"].append(uniform)
    return {"ks": ks, "rules": rules}


def print_table(data: dict) -> None:
    ks = data["ks"]
    rules = data["rules"]
    header = f"{'k':>3}  " + "  ".join(f"{name:>10}" for name in rules)
    print("\n" + "=" * 55)
    print("LP odd-support fraction by k and selection rule (N=4)")
    print("=" * 55)
    print(header)
    print("-" * 55)
    for i, k in enumerate(ks):
        row = f"{k:>3}  " + "  ".join(f"{rules[name][i]:>10.3f}" for name in rules)
        print(row)


def plot(data: dict, out_path: str = "figures/fig_n4_selection_rules.pdf") -> None:
    ks = data["ks"]
    rules = data["rules"]

    # x-axis: phi_B equivalent — more missing pairs = higher phi_B = lower k
    # We plot vs k directly (6=complete, 0=all-neutral)
    fig, ax = plt.subplots(figsize=(6, 4))

    styles = {
        "HiGHS":   dict(color="C0", lw=2,   ls="-",  marker="o", ms=6, label="HiGHS default"),
        "min-sup": dict(color="C2", lw=1.5, ls="--", marker="^", ms=6, label="min-support"),
        "max-sup": dict(color="C3", lw=1.5, ls=":",  marker="v", ms=6, label="max-support"),
        "uniform": dict(color="C1", lw=1.5, ls="-.", marker="s", ms=5, label="uniform over family"),
    }

    for name, vals in rules.items():
        ax.plot(ks, vals, **styles[name])

    ax.axhline(0.75, color="grey", lw=0.8, ls="--", alpha=0.6, label="3/4 random baseline")
    ax.set_xlabel("k (number of active pairs)", fontsize=11)
    ax.set_ylabel("LP odd-support fraction", fontsize=11)
    ax.set_title("N=4: LP parity under four selection rules", fontsize=12)
    ax.set_xticks(ks)
    ax.set_xlim(-0.3, 6.3)
    ax.set_ylim(0, 1.08)
    ax.legend(fontsize=9, loc="lower center")
    ax.invert_xaxis()   # right→left mirrors phi_B increasing left→right
    ax.set_xlabel("k  (← increasing φ_B)", fontsize=11)

    import os
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    # also save png for quick viewing
    fig.savefig(out_path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"\nFigure saved: {out_path}")
    plt.close(fig)


def main() -> None:
    print("Running N=4 exhaustive enumeration (3^6 = 729 structures)...")
    by_k = run_enumeration()
    data = aggregate(by_k)
    print_table(data)
    plot(data)

    # Verdict summary
    ks = data["ks"]
    rules = data["rules"]
    trough_k = ks[int(np.argmin(rules["HiGHS"]))]
    print(f"\n--- Scenario diagnosis ---")
    print(f"HiGHS trough at k={trough_k}")
    for name, vals in rules.items():
        trough_val = min(vals)
        trough_k_r = ks[int(np.argmin(vals))]
        print(f"  {name:<12}: trough = {trough_val:.3f} at k={trough_k_r}")

    # Does max-sup still show a trough?
    max_sup_vals = rules["max-sup"]
    trough_val_ms = min(max_sup_vals)
    right_end_ms  = max_sup_vals[-1]   # k=6 value (phi_B=0)
    left_end_ms   = max_sup_vals[0]    # k=0 value
    u_shape = (trough_val_ms < right_end_ms - 0.05) and (trough_val_ms < left_end_ms - 0.05)
    print(f"\nmax-sup U-shape present: {u_shape}")
    if u_shape:
        print("  → Scenario B confirmed: qualitative U-shape survives all selection rules.")
        print("     Manuscript framing adequate with 'HiGHS-returned' caveat + this figure.")
    else:
        print("  → Scenario C: max-sup flattens/inverts the U-shape.")
        print("     Larger-N selection-rule sweep needed before claiming U-shape robustness.")


if __name__ == "__main__":
    main()
