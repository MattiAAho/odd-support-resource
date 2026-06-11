"""Plot F sweep: LP support fraction and transitivity vs number of limiting factors.

Three panels:
  (a) Mean LP support fraction (support/N) vs F, by N — both models
  (b) Mean transitivity tau vs F, by N — both models
  (c) Support fraction vs transitivity (scatter over all N, F pairs — mechanism view)

Parity check printed to stdout; if all odd rates == 1.000 it is not shown in the figure.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

N_COLORS = {7: "#1f77b4", 9: "#ff7f0e", 11: "#2ca02c", 13: "#d62728"}
MODEL_STYLES = {"fixed": "--", "bernoulli": "-"}
MODEL_LABELS = {"fixed": "Fixed (sampled)", "bernoulli": "Bernoulli (expected)"}


def load(path: str) -> list[dict]:
    with open(path, "rb") as fh:
        return pickle.load(fh)


def agg(records: list[dict], model: str, N: int, col: str) -> tuple[list, list, list]:
    """Return (factor_vals, means, sems) for given model and N."""
    factor_vals = sorted(set(r["n_factors"] for r in records))
    means, sems = [], []
    for F in factor_vals:
        sub = [r[col] for r in records
               if r["model"] == model and r["n_species"] == N and r["n_factors"] == F]
        if sub:
            arr = np.array(sub)
            means.append(arr.mean())
            sems.append(arr.std() / np.sqrt(len(arr)))
        else:
            means.append(np.nan)
            sems.append(np.nan)
    return factor_vals, means, sems


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="results/f_sweep_v1.pkl")
    p.add_argument("--prefix", default="figures/fig_f_sweep")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    records = load(args.input)

    species_vals = sorted(set(r["n_species"] for r in records))
    factor_vals = sorted(set(r["n_factors"] for r in records))
    models = ["bernoulli", "fixed"]

    # Parity check
    for model in models:
        rates = [r["lp_support_odd"] for r in records if r["model"] == model]
        rate = np.mean(rates)
        if abs(rate - 1.0) > 0.001:
            print(f"WARNING: {model} odd rate = {rate:.4f} (expected 1.000)")
        else:
            print(f"Parity OK: {model} odd rate = {rate:.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Panel (a): mean support fraction vs F, by N, both models
    ax = axes[0]
    for model in models:
        for N in species_vals:
            F_vals, means, sems = agg(records, model, N, "lp_support_frac")
            means = np.array(means)
            sems = np.array(sems)
            label = f"N={N}" if model == "bernoulli" else None
            line, = ax.plot(F_vals, means,
                            linestyle=MODEL_STYLES[model],
                            color=N_COLORS.get(N, "k"),
                            marker="o" if model == "bernoulli" else None,
                            markersize=4,
                            alpha=0.85,
                            label=label)
            if model == "bernoulli":
                ax.fill_between(F_vals, means - sems, means + sems,
                                color=N_COLORS.get(N, "k"), alpha=0.12)

    # Annotate limiting cases
    ax.axhline(1 / min(species_vals), color="grey", linewidth=0.6, linestyle=":")
    ax.set_xlabel("Number of limiting factors $F$")
    ax.set_ylabel("Mean LP support fraction (support/$N$)")
    ax.set_xscale("log")
    ax.set_title("(a) LP support fraction vs $F$\n(solid = Bernoulli, dashed = fixed)", fontsize=9)
    ax.legend(fontsize=8, loc="lower right")

    # Panel (b): mean transitivity vs F, by N, both models
    ax = axes[1]
    for model in models:
        for N in species_vals:
            F_vals, means, sems = agg(records, model, N, "transitivity")
            means = np.array(means)
            sems = np.array(sems)
            ax.plot(F_vals, means,
                    linestyle=MODEL_STYLES[model],
                    color=N_COLORS.get(N, "k"),
                    marker="o" if model == "bernoulli" else None,
                    markersize=4,
                    alpha=0.85,
                    label=f"N={N}" if model == "bernoulli" else None)
            if model == "bernoulli":
                ax.fill_between(F_vals, means - sems, means + sems,
                                color=N_COLORS.get(N, "k"), alpha=0.12)

    ax.axhline(0.75, color="grey", linewidth=0.8, linestyle="--",
               label=r"$\tau \to 3/4$ (random limit)")
    ax.axhline(1.0, color="grey", linewidth=0.6, linestyle=":")
    ax.set_xlabel("Number of limiting factors $F$")
    ax.set_ylabel(r"Mean transitivity fraction $\tau$")
    ax.set_xscale("log")
    ax.set_title(r"(b) Transitivity $\tau$ vs $F$", fontsize=9)
    ax.legend(fontsize=7, loc="lower left")

    # Panel (c): support fraction vs transitivity (scatter, mechanism view)
    # One point per (N, F, model) combination — show the direct relationship
    ax = axes[2]
    for model in models:
        for N in species_vals:
            F_vals, supp_means, _ = agg(records, model, N, "lp_support_frac")
            _, tau_means, _ = agg(records, model, N, "transitivity")
            supp_means = np.array(supp_means)
            tau_means = np.array(tau_means)
            mask = np.isfinite(supp_means) & np.isfinite(tau_means)
            ax.plot(tau_means[mask], supp_means[mask],
                    linestyle=MODEL_STYLES[model],
                    color=N_COLORS.get(N, "k"),
                    marker="o" if model == "bernoulli" else "s",
                    markersize=4,
                    alpha=0.85,
                    label=f"N={N}" if model == "bernoulli" else None)
            # Annotate F=1 and the largest F
            if model == "bernoulli" and len(F_vals) >= 1:
                idx_f1 = [i for i, f in enumerate(F_vals) if f == 1]
                if idx_f1:
                    i = idx_f1[0]
                    ax.annotate("F=1", (tau_means[i], supp_means[i]),
                                fontsize=6, ha="left", va="bottom",
                                xytext=(4, 2), textcoords="offset points",
                                color=N_COLORS.get(N, "k"), alpha=0.6)

    ax.set_xlabel(r"Mean transitivity $\tau$")
    ax.set_ylabel("Mean LP support fraction (support/$N$)")
    ax.set_title("(c) Support fraction vs transitivity\n(mechanism view)", fontsize=9)
    ax.legend(fontsize=7, loc="upper right")
    ax.invert_xaxis()  # high tau (transitive) on left, low tau (cyclic) on right

    fig.suptitle(
        f"F sweep: LP support distribution and transitivity\n"
        f"(φ_B=0, {min(species_vals)}≤N≤{max(species_vals)}, "
        f"F∈{{{','.join(str(f) for f in factor_vals)}}}, "
        f"reps={sum(1 for r in records if r['model']=='bernoulli' and r['n_species']==species_vals[0] and r['n_factors']==factor_vals[0])})",
        fontsize=10,
    )
    fig.tight_layout()

    out = Path(args.prefix)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=150, bbox_inches="tight")
    print(f"Saved {out}.pdf / .png")


if __name__ == "__main__":
    main()
