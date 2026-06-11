"""Compare simulation odd-support at 10M vs 100M steps for phi_B=0.3, N=7.

Open Q1: is the sim-vs-LP gap a run-length artefact or genuine long-run divergence?

Loads the original sparsity sweep (baseline, 10M steps) and the horizon check
(100M steps), filters to phi_B=0.3 / N=7, and prints a per-model comparison table
plus a figure with LP reference lines.
"""

from __future__ import annotations

import argparse
import pickle

import matplotlib.pyplot as plt
import numpy as np

MODELS = ["fixed", "bernoulli", "beta_sample"]
MODEL_LABELS = {"fixed": "Fixed", "bernoulli": "Bernoulli", "beta_sample": "Beta-sample"}
MODEL_COLORS = {"fixed": "#1f77b4", "bernoulli": "#ff7f0e", "beta_sample": "#2ca02c"}


def load(path: str) -> list[dict]:
    with open(path, "rb") as fh:
        return pickle.load(fh)


def filter_records(records: list[dict], phi_B: float, n_species: int) -> list[dict]:
    return [r for r in records if abs(r["phi_B"] - phi_B) < 1e-6 and r["n_species"] == n_species]


def stats(records: list[dict], model: str) -> dict:
    rows = [r for r in records if r["model"] == model]
    n = len(rows)
    if n == 0:
        return {}
    lp_odd = np.mean([r["lp_support_odd"] for r in rows])
    final_odd = np.mean([r["final_support_odd"] for r in rows])
    avg_odd = np.mean([r["avg_support_odd"] for r in rows])
    final_se = np.std([r["final_support_odd"] for r in rows]) / np.sqrt(n)
    avg_se = np.std([r["avg_support_odd"] for r in rows]) / np.sqrt(n)
    return {
        "n": n,
        "lp_odd": lp_odd,
        "final_odd": final_odd,
        "avg_odd": avg_odd,
        "final_se": final_se,
        "avg_se": avg_se,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="results/sparsity_sweep_v1.pkl",
                    help="original 10M-step sweep")
    ap.add_argument("--horizon", default="results/horizon_check_v1.pkl",
                    help="100M-step horizon check")
    ap.add_argument("--phi-b", type=float, default=0.3)
    ap.add_argument("--species", type=int, default=7)
    ap.add_argument("--out", default="figures/fig_horizon_check.pdf")
    args = ap.parse_args()

    base = filter_records(load(args.baseline), args.phi_b, args.species)
    long = filter_records(load(args.horizon), args.phi_b, args.species)

    print(f"\nphi_B={args.phi_b}, N={args.species}")
    print(f"Baseline records: {len(base)}  |  Horizon records: {len(long)}")
    print()

    header = f"{'Model':<14}  {'n(base)':>7}  {'LP odd':>7}  "
    header += f"{'final@10M':>9}  {'±SE':>6}  {'final@100M':>10}  {'±SE':>6}  "
    header += f"{'avg@10M':>8}  {'±SE':>6}  {'avg@100M':>9}  {'±SE':>6}"
    print(header)
    print("-" * len(header))

    rows_base = {}
    rows_long = {}
    for model in MODELS:
        b = stats(base, model)
        l = stats(long, model)
        rows_base[model] = b
        rows_long[model] = l
        if not b or not l:
            continue
        print(
            f"{model:<14}  {b['n']:>7}  {b['lp_odd']:>7.3f}  "
            f"{b['final_odd']:>9.3f}  {b['final_se']:>6.3f}  "
            f"{l['final_odd']:>10.3f}  {l['final_se']:>6.3f}  "
            f"{b['avg_odd']:>8.3f}  {b['avg_se']:>6.3f}  "
            f"{l['avg_odd']:>9.3f}  {l['avg_se']:>6.3f}"
        )

    print()
    print("Interpretation guide:")
    print("  If final@100M ≈ LP odd  → convergence artefact (simulations need more time)")
    print("  If final@100M ≈ final@10M  → genuine long-run divergence from LP")
    print()
    for model in MODELS:
        b = rows_base.get(model, {})
        l = rows_long.get(model, {})
        if not b or not l:
            continue
        drift = l["final_odd"] - b["final_odd"]
        gap_to_lp = b["lp_odd"] - l["final_odd"]
        print(
            f"  {model}: drift = {drift:+.3f}  |  remaining gap to LP at 100M = {gap_to_lp:.3f}"
        )

    # --- figure ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax_idx, (metric, label) in enumerate([
        ("final_odd", "Final-sim odd support fraction"),
        ("avg_odd", "Time-avg odd support fraction"),
    ]):
        ax = axes[ax_idx]
        x = np.arange(len(MODELS))
        w = 0.25

        for i, (src_label, rows) in enumerate([("10M steps", rows_base), ("100M steps", rows_long)]):
            vals = [rows.get(m, {}).get(metric, np.nan) for m in MODELS]
            ses = [rows.get(m, {}).get(f"{metric.split('_')[0]}_se", np.nan) for m in MODELS]
            ax.bar(x + (i - 0.5) * w, vals, w, yerr=ses, label=src_label, alpha=0.8, capsize=4)

        lp_vals = [rows_base.get(m, {}).get("lp_odd", np.nan) for m in MODELS]
        for j, (lp, model) in enumerate(zip(lp_vals, MODELS)):
            ax.hlines(lp, j - w, j + w, colors=MODEL_COLORS[model], lw=2, ls="--",
                      label=f"LP ({MODEL_LABELS[model]})" if ax_idx == 0 else "_")

        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS])
        ax.set_ylabel(label)
        ax.set_ylim(0, 1.1)
        ax.axhline(0.5, color="k", lw=0.5, ls=":", alpha=0.4)
        ax.set_title(f"phi_B={args.phi_b}, N={args.species}: {label}")
        if ax_idx == 0:
            ax.legend(fontsize=7, ncol=2)

    fig.suptitle(
        f"Simulation horizon check: phi_B={args.phi_b}, N={args.species}  "
        f"(10M vs 100M steps)"
    )
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    out_png = args.out.replace(".pdf", ".png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"\nSaved {args.out} / {out_png}")
    plt.close(fig)


if __name__ == "__main__":
    main()
