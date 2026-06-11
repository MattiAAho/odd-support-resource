"""Plot φ_C antisymmetry-break sweep results.

Four panels:
  (a) LP odd fraction vs phi_C — should be 1.0 throughout (sanity check)
  (b) Simulation odd fraction vs phi_C — does breaking complementarity break parity?
  (c) Extinction fraction vs phi_C — how does selection strength change?
  (d) Mean p_ab + p_ba for gamma pairs — shows the distribution of non-complementarity
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


F_COLORS = {3: "#1f77b4", 5: "#ff7f0e", 7: "#2ca02c"}
MODEL_STYLES = {"bernoulli": "--", "gamma": "-"}
MODEL_LABELS = {"bernoulli": "Bernoulli (baseline)", "gamma": "γ (broken complementarity)"}


def load(path: str) -> list[dict]:
    with open(path, "rb") as fh:
        return pickle.load(fh)


def agg_by_phi_c(records: list[dict], model: str, F: int, col: str) -> tuple[list, list]:
    phi_c_vals = sorted(set(r["phi_C"] for r in records))
    vals = []
    for phi_C in phi_c_vals:
        sub = [r for r in records
               if r["phi_C"] == phi_C and r["model"] == model and r["n_factors"] == F]
        vals.append(np.mean([r[col] for r in sub]) if sub else np.nan)
    return phi_c_vals, vals


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="results/phi_c_sweep_v1.pkl")
    p.add_argument("--prefix", default="figures/fig_phi_c_sweep")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    records = load(args.input)

    factor_vals = sorted(set(r["n_factors"] for r in records))
    models = ["bernoulli", "gamma"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # Panel (a): LP odd fraction — sanity check, should always be 1
    ax = axes[0]
    for F in factor_vals:
        for model in models:
            phi_c, vals = agg_by_phi_c(records, model, F, "lp_support_odd")
            ax.plot(phi_c, vals,
                    linestyle=MODEL_STYLES[model],
                    color=F_COLORS.get(F, "k"),
                    marker="o" if model == "gamma" else None,
                    alpha=0.8,
                    label=f"F={F} {model}" if F == factor_vals[0] else None)
    ax.set_xlabel("$\\phi_C$")
    ax.set_ylabel("Fraction LP support odd")
    ax.set_ylim(-0.05, 1.1)
    ax.axhline(1.0, color="grey", linewidth=0.8, linestyle=":")
    ax.set_title("(a) LP parity\n(sanity check: expect = 1.0)", fontsize=9)

    # Panel (b): simulation odd fraction
    ax = axes[1]
    for F in factor_vals:
        for model in models:
            phi_c, vals = agg_by_phi_c(records, model, F, "final_support_odd")
            ax.plot(phi_c, vals,
                    linestyle=MODEL_STYLES[model],
                    color=F_COLORS.get(F, "k"),
                    marker="o" if model == "gamma" else None,
                    alpha=0.8,
                    label=f"F={F} {MODEL_LABELS[model]}" if F == 5 else None)
    ax.set_xlabel("$\\phi_C$")
    ax.set_ylabel("Fraction final support odd")
    ax.set_ylim(-0.05, 1.1)
    ax.axhline(1.0, color="grey", linewidth=0.8, linestyle=":")
    ax.set_title("(b) Simulation parity", fontsize=9)
    ax.legend(fontsize=7, loc="lower left")

    # Panel (c): extinction fraction
    ax = axes[2]
    for F in factor_vals:
        for model in models:
            phi_c_vals = sorted(set(r["phi_C"] for r in records))
            ext_rates = []
            for phi_C in phi_c_vals:
                sub = [r for r in records
                       if r["phi_C"] == phi_C and r["model"] == model and r["n_factors"] == F]
                ext_rates.append(
                    np.mean([r["extinct_time"] is not None for r in sub]) if sub else np.nan
                )
            ax.plot(phi_c_vals, ext_rates,
                    linestyle=MODEL_STYLES[model],
                    color=F_COLORS.get(F, "k"),
                    marker="o" if model == "gamma" else None,
                    alpha=0.8)
    ax.set_xlabel("$\\phi_C$")
    ax.set_ylabel("Extinction fraction")
    ax.set_ylim(-0.05, 1.1)
    ax.set_title("(c) Extinction rate", fontsize=9)

    # Panel (d): per-replicate distribution of mean p_ab+p_ba and deficient fraction
    # The grand mean is always near 1 by CLT, so show spread across reps instead.
    ax = axes[3]
    gamma_recs = [r for r in records if r["model"] == "gamma" and r["n_gamma_pairs"] > 0]
    if gamma_recs:
        phi_c_vals = sorted(set(r["phi_C"] for r in gamma_recs))
        rep_means = []   # mean of per-replicate mean_sums
        rep_stds = []    # std of per-replicate mean_sums
        def_means = []   # mean deficient fraction across reps
        def_stds = []
        for p in phi_c_vals:
            sub = [r for r in gamma_recs if r["phi_C"] == p]
            ms = np.array([r["mean_sum"] for r in sub])
            df = np.array([r["n_deficient_pairs"] / r["n_gamma_pairs"] for r in sub])
            rep_means.append(ms.mean())
            rep_stds.append(ms.std())
            def_means.append(df.mean())
            def_stds.append(df.std())
        rep_means = np.array(rep_means)
        rep_stds = np.array(rep_stds)
        def_means = np.array(def_means)
        def_stds = np.array(def_stds)

        # Show per-replicate mean_sum: mean ± 1 SD band (spread IS substantial)
        ax.plot(phi_c_vals, rep_means, color="#1f77b4", marker="o",
                label="mean $p_{ab}+p_{ba}$ per rep")
        ax.fill_between(phi_c_vals, rep_means - rep_stds, rep_means + rep_stds,
                        color="#1f77b4", alpha=0.25, label="± 1 SD across reps")
        ax.axhline(1.0, color="grey", linewidth=0.8, linestyle=":")

        ax2 = ax.twinx()
        ax2.plot(phi_c_vals, def_means, color="#d62728", marker="s", linestyle="--",
                 label="frac deficient (sum < 1)")
        ax2.fill_between(phi_c_vals, def_means - def_stds, def_means + def_stds,
                         color="#d62728", alpha=0.15)
        ax2.axhline(0.5, color="#d62728", linewidth=0.6, linestyle=":")
        ax2.set_ylabel("Fraction deficient pairs", color="#d62728")
        ax2.tick_params(axis="y", colors="#d62728")
        ax2.set_ylim(-0.05, 1.1)
    ax.set_xlabel("$\\phi_C$")
    ax.set_ylabel("Mean $p_{ab} + p_{ba}$ for $\\gamma$ pairs", color="#1f77b4")
    ax.tick_params(axis="y", colors="#1f77b4")
    ax.set_title("(d) Non-complementarity: per-rep spread\n(shaded = ±1 SD across replicates)", fontsize=9)
    if gamma_recs:
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7)

    fig.suptitle("φ_C antisymmetry-break sweep (Option γ: independent p_ab, p_ba draws)",
                 fontsize=11)
    fig.tight_layout()

    out = Path(args.prefix)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=150, bbox_inches="tight")
    print(f"Saved {out}.pdf / .png")


if __name__ == "__main__":
    main()
