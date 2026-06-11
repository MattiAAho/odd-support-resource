"""Plot LP parity under four selection rules for phi_B and phi_A sweeps.

Loads results/nonuniqueness_sweep_v1.pkl and plots odd-support fraction vs
phi_B (left panels) and vs phi_A (right panels) for N=7 and N=9.

Each panel shows four curves (HiGHS, min-sup, max-sup, uniform) so that
convergence or divergence of the U-shape across selection rules is visible.
"""

from __future__ import annotations

import argparse
import pickle
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RULES = ["HiGHS", "min-sup", "max-sup", "uniform"]
STYLES = {
    "HiGHS":   dict(color="C0", lw=2,   ls="-",  marker="o", ms=6),
    "min-sup": dict(color="C2", lw=1.5, ls="--", marker="^", ms=6),
    "max-sup": dict(color="C3", lw=1.5, ls=":",  marker="v", ms=6),
    "uniform": dict(color="C1", lw=1.5, ls="-.", marker="s", ms=5),
}


def parity_values(record: dict) -> dict[str, float]:
    highs_sz = record["highs_sz"]
    all_sizes = record["all_sizes"]
    return {
        "HiGHS":   float(highs_sz % 2 == 1),
        "min-sup": float(min(all_sizes) % 2 == 1),
        "max-sup": float(max(all_sizes) % 2 == 1),
        "uniform": sum(sz % 2 == 1 for sz in all_sizes) / len(all_sizes),
    }


def aggregate(records: list[dict]) -> dict:
    """Return {(param_type, n_species, param_val): {rule: mean_odd_frac}}."""
    buckets: dict = defaultdict(lambda: defaultdict(list))
    for r in records:
        key = (r["param_type"], r["n_species"], r["param_val"])
        pv = parity_values(r)
        for rule, val in pv.items():
            buckets[key][rule].append(val)
    out = {}
    for key, rule_vals in buckets.items():
        out[key] = {rule: float(np.mean(vals)) for rule, vals in rule_vals.items()}
    return out


def plot(agg: dict, species: list[int], out_path: str) -> None:
    fig, axes = plt.subplots(
        len(species), 2, figsize=(10, 4 * len(species)), sharey=True
    )
    if len(species) == 1:
        axes = [axes]  # make 2-D indexable

    for row, n in enumerate(species):
        for col, param_type in enumerate(["phi_B", "phi_A"]):
            ax = axes[row][col]

            # Collect x values and mean odd fraction per rule
            xs = sorted(set(
                v for (pt, ns, v), _ in agg.items()
                if pt == param_type and ns == n
            ))
            if not xs:
                ax.set_visible(False)
                continue

            for rule in RULES:
                ys = [
                    agg.get((param_type, n, x), {}).get(rule, np.nan)
                    for x in xs
                ]
                label = rule if row == 0 and col == 0 else None
                ax.plot(xs, ys, label=label, **STYLES[rule])

            ax.axhline(1.0, color="grey", lw=0.6, ls="--", alpha=0.5)
            ax.axhline(0.75, color="grey", lw=0.6, ls=":", alpha=0.5)
            ax.set_xlabel(f"φ_{'B' if param_type == 'phi_B' else 'A'}", fontsize=11)
            ax.set_title(f"N={n}, φ_{'B' if param_type == 'phi_B' else 'A'} sweep", fontsize=11)
            ax.set_ylim(-0.05, 1.08)
            ax.set_xlim(-0.03, max(xs) + 0.03)

        axes[row][0].set_ylabel("LP odd-support fraction", fontsize=10)

    # Single legend at top
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, 1.01))

    # Notes
    fig.text(0.5, -0.02,
             "Dashed grey: 1.0 (A&L baseline)  Dotted grey: 3/4 (random tournament)",
             ha="center", fontsize=8, color="grey")

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


def print_table(agg: dict, param_type: str, species: list[int]) -> None:
    xs = sorted(set(v for (pt, _, v), _ in agg.items() if pt == param_type))
    param_label = "phi_B" if param_type == "phi_B" else "phi_A"
    print(f"\n=== {param_label} sweep ===")
    for n in species:
        print(f"\nN={n}")
        header = f"  {param_label:>6}  " + "  ".join(f"{r:>10}" for r in RULES)
        print(header)
        print("  " + "-" * (len(header) - 2))
        for x in xs:
            vals = agg.get((param_type, n, x), {})
            row = f"  {x:>6.2f}  " + "  ".join(
                f"{vals.get(r, float('nan')):>10.3f}" for r in RULES
            )
            print(row)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pkl", default="results/nonuniqueness_sweep_v1.pkl")
    p.add_argument("--out", default="figures/fig_nonuniqueness_sweep.pdf")
    args = p.parse_args()

    with open(args.pkl, "rb") as f:
        records = pickle.load(f)

    print(f"Loaded {len(records)} records from {args.pkl}")

    agg = aggregate(records)
    species = sorted(set(r["n_species"] for r in records))

    for pt in ["phi_B", "phi_A"]:
        print_table(agg, pt, species)

    plot(agg, species, args.out)

    # Scenario diagnosis
    print("\n=== Scenario diagnosis ===")
    for n in species:
        for pt in ["phi_B", "phi_A"]:
            xs = sorted(set(v for (pty, ns, v), _ in agg.items() if pty == pt and ns == n))
            if not xs:
                continue
            for rule in RULES:
                ys = [agg.get((pt, n, x), {}).get(rule, np.nan) for x in xs]
                ys_clean = [y for y in ys if not np.isnan(y)]
                if not ys_clean:
                    continue
                floor_val = min(ys_clean)
                right_val = ys_clean[-1] if xs[-1] >= 0.6 else np.nan   # high-phi end
                # U-shape: trough exists and endpoints are higher
                trough_idx = int(np.argmin(ys_clean))
                trough_x = xs[trough_idx]
                is_u = (
                    floor_val < ys_clean[0] - 0.05
                    and floor_val < (right_val - 0.05 if not np.isnan(right_val) else np.inf)
                    and trough_idx not in (0, len(ys_clean) - 1)
                )
                print(f"  N={n} {pt} {rule:<12}: floor={floor_val:.3f} at φ={trough_x:.2f}  "
                      f"U-shape={is_u}")


if __name__ == "__main__":
    main()
