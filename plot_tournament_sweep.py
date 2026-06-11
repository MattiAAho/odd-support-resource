"""Visualise standalone tournament sweep results.

Designed to work with the tiny quick sweep first, then scale to larger sweeps.
"""

from __future__ import annotations

import argparse
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODEL_ORDER = ["fixed", "bernoulli", "factor", "beta_mean", "beta_sample"]
MODEL_LABELS = {
    "fixed": "Fixed\nsampled",
    "bernoulli": "Dynamic\nBernoulli",
    "factor": "Dynamic\nfactor",
    "beta_mean": "Beta\nmean",
    "beta_sample": "Beta\nsample",
}
MODEL_COLORS = {
    "fixed": "#3f6f9f",
    "bernoulli": "#c75d2c",
    "factor": "#d99526",
    "beta_mean": "#4f8f4a",
    "beta_sample": "#7b5ea7",
}


def lp_reference_label(model: str) -> str:
    return "Exact LP" if model == "fixed" else "Expected-game LP"


def lp_vector(row: dict) -> np.ndarray:
    if row.get("lp_exact") is not None:
        return _as_array(row["lp_exact"])
    if row.get("lp_ref") is not None:
        return _as_array(row["lp_ref"])
    return _as_array(row["lp_equilibrium"])


def _as_array(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def load_rows(path: Path) -> list[dict]:
    with path.open("rb") as fh:
        rows = pickle.load(fh)
    if not isinstance(rows, list):
        raise TypeError(f"expected list of records in {path}, got {type(rows)}")
    return rows


def lp_distance(row: dict, field: str = "time_avg") -> float:
    x = _as_array(row[field])
    eq = lp_vector(row)
    return float(np.abs(x - eq).sum() / 2.0)


def richness(vec: np.ndarray, threshold: float = 1e-3) -> int:
    return int(np.sum(vec > threshold))


def prepare_summary(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        final = _as_array(r["final_freq"])
        avg = _as_array(r["time_avg"])
        eq = lp_vector(r)
        out.append(
            {
                **r,
                "avg_lp_l1_half": lp_distance(r, "time_avg"),
                "final_lp_l1_half": lp_distance(r, "final_freq"),
                "lp_support": richness(eq),
                "avg_support": richness(avg),
                "final_support": richness(final),
                "dominant_final": int(np.argmax(final)),
                "dominant_avg": int(np.argmax(avg)),
            }
        )
    return out


def plot_quick_frequency_panels(rows: list[dict], out: Path) -> None:
    """One panel per record: LP, time-average, and final frequencies."""

    rows = sorted(rows, key=lambda r: (MODEL_ORDER.index(r["model"]), r["rep"]))
    n = len(rows)
    ncols = min(5, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.4 * nrows), squeeze=False)

    for ax, r in zip(axes.ravel(), rows):
        s = np.arange(r["n_species"])
        width = 0.25
        lp_label = lp_reference_label(r["model"])
        ax.bar(s - width, lp_vector(r), width=width, color="#222222", alpha=0.85, label=lp_label)
        ax.bar(s, r["time_avg"], width=width, color=MODEL_COLORS[r["model"]], alpha=0.80, label="avg")
        ax.bar(s + width, r["final_freq"], width=width, color="#cccccc", edgecolor="#555555", linewidth=0.4, label="final")
        title = f"{MODEL_LABELS.get(r['model'], r['model'])}  rep {r['rep']}"
        if r["extinct_time"] is not None:
            title += f"\next {r['extinct_time']}"
        else:
            title += "\nno ext"
        ax.set_title(title, fontsize=9)
        ax.set_ylim(0, 1.03)
        ax.set_xticks(s)
        ax.set_xticklabels([str(i) for i in s], fontsize=8)
        ax.grid(axis="y", alpha=0.2)
        if ax is axes[0, 0]:
            ax.legend(fontsize=7, frameon=False)

    for ax in axes.ravel()[len(rows) :]:
        ax.axis("off")

    fig.suptitle(
        "Tournament quick sweep: LP reference vs simulation\n"
        "Exact LP for fixed tournaments; expected-game LP for dynamic models",
        fontsize=12,
    )
    fig.supxlabel("Species")
    fig.supylabel("Frequency")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=160, bbox_inches="tight")


def plot_model_summary(rows: list[dict], out: Path) -> None:
    """Aggregate summary by model; works for quick and larger sweeps."""

    rows = prepare_summary(rows)
    by_model: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)

    models = [m for m in MODEL_ORDER if m in by_model]
    x = np.arange(len(models))

    def mean_sd(field: str) -> tuple[np.ndarray, np.ndarray]:
        means, sds = [], []
        for m in models:
            vals = np.array([r[field] for r in by_model[m]], dtype=np.float64)
            means.append(np.nanmean(vals))
            sds.append(np.nanstd(vals, ddof=1) if len(vals) > 1 else 0.0)
        return np.array(means), np.array(sds)

    dist_mean, dist_sd = mean_sd("avg_lp_l1_half")
    final_support_mean, final_support_sd = mean_sd("final_support")
    lp_support_mean, lp_support_sd = mean_sd("lp_support")

    ext_frac = []
    ext_time_mean = []
    ext_time_sd = []
    for m in models:
        vals = [r["extinct_time"] for r in by_model[m]]
        finite = np.array([v for v in vals if v is not None], dtype=np.float64)
        ext_frac.append(len(finite) / len(vals))
        ext_time_mean.append(np.nan if len(finite) == 0 else np.mean(finite))
        ext_time_sd.append(0.0 if len(finite) <= 1 else np.std(finite, ddof=1))

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    colors = [MODEL_COLORS[m] for m in models]
    labels = [MODEL_LABELS.get(m, m) for m in models]

    ax = axes[0, 0]
    ax.bar(x, dist_mean, yerr=dist_sd, color=colors, capsize=3)
    ax.set_title("Simulation distance from LP reference")
    ax.set_ylabel("0.5 x L1 distance\n(time-average vs LP ref)")
    ax.set_ylim(0, max(0.05, min(1.0, np.nanmax(dist_mean + dist_sd) * 1.25)))
    ax.grid(axis="y", alpha=0.25)

    ax = axes[0, 1]
    ax.bar(x - 0.18, lp_support_mean, width=0.36, yerr=lp_support_sd, color="#222222", capsize=3, label="LP ref")
    ax.bar(x + 0.18, final_support_mean, width=0.36, yerr=final_support_sd, color=colors, capsize=3, label="Final")
    ax.set_title("Support size")
    ax.set_ylabel("Species with frequency > 0.001")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1, 0]
    ax.bar(x, ext_frac, color=colors)
    ax.set_title("Any sampled extinction")
    ax.set_ylabel("Fraction of runs")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1, 1]
    ax.bar(x, ext_time_mean, yerr=ext_time_sd, color=colors, capsize=3)
    ax.set_title("Extinction timing")
    ax.set_ylabel("First sampled extinction step")
    ax.grid(axis="y", alpha=0.25)

    for ax in axes.ravel():
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)

    meta = rows[0]
    fig.suptitle(
        f"Tournament sweep summary: S={meta['n_species']}, F={meta['n_factors']}, "
        f"N={meta['n_individuals']}, steps={meta['n_steps']}\n"
        "LP ref = exact LP for fixed; expected-game LP for dynamic models",
        fontsize=12,
    )
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=160, bbox_inches="tight")


def print_summary(rows: list[dict]) -> None:
    rows = prepare_summary(rows)
    by_model: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)
    print("Model summary")
    for model in [m for m in MODEL_ORDER if m in by_model]:
        group = by_model[model]
        dist = np.array([r["avg_lp_l1_half"] for r in group])
        final_support = np.array([r["final_support"] for r in group])
        extinct = [r["extinct_time"] for r in group]
        print(
            f"  {model:>11}: n={len(group):>3}  "
            f"LPdist={dist.mean():.3f}±{dist.std(ddof=1) if len(dist)>1 else 0:.3f}  "
            f"final_support={final_support.mean():.2f}  "
            f"extinct={sum(v is not None for v in extinct)}/{len(extinct)}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/tournament_sweep_quick.pkl")
    parser.add_argument("--prefix", default="figures/fig_tournament_quick")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_rows(Path(args.input))
    prefix = Path(args.prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    print_summary(rows)
    plot_quick_frequency_panels(rows, prefix.with_name(prefix.name + "_frequencies.pdf"))
    plot_model_summary(rows, prefix.with_name(prefix.name + "_summary.pdf"))
    print(f"Saved {prefix.with_name(prefix.name + '_frequencies.pdf')}")
    print(f"Saved {prefix.with_name(prefix.name + '_summary.pdf')}")


if __name__ == "__main__":
    main()
