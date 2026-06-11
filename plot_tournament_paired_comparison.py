"""Paired replicate comparison plots for tournament sweeps."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_tournament_sweep import (
    MODEL_COLORS,
    MODEL_LABELS,
    MODEL_ORDER,
    load_rows,
    prepare_summary,
)


def group_complete_pairs(rows: list[dict], models: list[str]) -> list[dict[str, dict]]:
    grouped: dict[tuple[int, int, int, int], dict[str, dict]] = defaultdict(dict)
    for row in rows:
        key = (row["n_species"], row["n_factors"], row["rep"], row["rank_seed"])
        grouped[key][row["model"]] = row
    return [bucket for bucket in grouped.values() if all(model in bucket for model in models)]


def split_rows_by_cell(rows: list[dict]) -> dict[tuple[int, int], list[dict]]:
    grouped: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["n_species"], row["n_factors"])].append(row)
    return dict(sorted(grouped.items()))


def extinction_observed(row: dict) -> float:
    return 0.0 if row["extinct_time"] is None else 1.0


def extinction_time_censored(row: dict) -> float:
    if row["extinct_time"] is None:
        return float(row["n_steps"])
    return float(row["extinct_time"])


def model_values(groups: list[dict[str, dict]], models: list[str], metric_fn) -> np.ndarray:
    vals = np.empty((len(groups), len(models)), dtype=np.float64)
    for i, bucket in enumerate(groups):
        for j, model in enumerate(models):
            vals[i, j] = metric_fn(bucket[model])
    return vals


def paired_delta(groups: list[dict[str, dict]], model: str, ref: str, metric_fn) -> np.ndarray:
    deltas = []
    for bucket in groups:
        deltas.append(metric_fn(bucket[model]) - metric_fn(bucket[ref]))
    return np.asarray(deltas, dtype=np.float64)


def print_paired_summary(
    groups: list[dict[str, dict]],
    models: list[str],
    reference: str,
    cell_label: str | None = None,
) -> None:
    print("Paired comparison summary")
    if cell_label is not None:
        print(f"  cell: {cell_label}")
    print(f"  complete paired replicates: {len(groups)}")
    print(f"  reference model: {reference}")

    metrics = [
        ("avg LP distance", lambda r: r["avg_lp_l1_half"], "lower"),
        ("final support", lambda r: float(r["final_support"]), "context"),
        ("any extinction", extinction_observed, "lower"),
        ("extinction step (censored)", extinction_time_censored, "higher"),
    ]

    for model in models:
        if model == reference:
            continue
        print(f"  {model} vs {reference}")
        for label, fn, direction in metrics:
            delta = paired_delta(groups, model, reference, fn)
            mean = float(np.mean(delta))
            median = float(np.median(delta))
            if direction == "lower":
                favored = int(np.sum(delta < 0))
                opposed = int(np.sum(delta > 0))
                direction_text = "lower"
            elif direction == "higher":
                favored = int(np.sum(delta > 0))
                opposed = int(np.sum(delta < 0))
                direction_text = "higher"
            else:
                favored = int(np.sum(delta > 0))
                opposed = int(np.sum(delta < 0))
                direction_text = "higher"
            print(
                f"    {label}: mean delta={mean:+.3f}  median={median:+.3f}  "
                f"{direction_text} in {favored}/{len(delta)}, opposite in {opposed}/{len(delta)}"
            )


def plot_paired_metrics(groups: list[dict[str, dict]], models: list[str], out: Path) -> None:
    x = np.arange(len(models))
    label_x = [MODEL_LABELS.get(model, model) for model in models]
    metric_specs = [
        ("Time-average distance from LP reference", lambda r: r["avg_lp_l1_half"]),
        ("Final support", lambda r: float(r["final_support"])),
        ("Any sampled extinction", extinction_observed),
        ("First extinction step\n(top-coded at n_steps when no extinction)", extinction_time_censored),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    axes = axes.ravel()

    for ax, (title, metric_fn) in zip(axes, metric_specs):
        vals = model_values(groups, models, metric_fn)
        for row in vals:
            ax.plot(
                x,
                row,
                color="#777777",
                alpha=0.35,
                linewidth=1.0,
                marker="o",
                markersize=4,
                markerfacecolor="white",
                markeredgecolor="#666666",
            )

        means = vals.mean(axis=0)
        sds = vals.std(axis=0, ddof=1) if len(groups) > 1 else np.zeros(len(models))
        ax.plot(x, means, color="#111111", linewidth=2.0, zorder=3)
        for j, model in enumerate(models):
            ax.errorbar(
                x[j],
                means[j],
                yerr=sds[j],
                fmt="o",
                markersize=7,
                color="#111111",
                markerfacecolor=MODEL_COLORS.get(model, "#444444"),
                markeredgecolor="#111111",
                capsize=3,
                zorder=4,
            )

        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(label_x, fontsize=8)
        ax.grid(axis="y", alpha=0.25)

    axes[2].set_ylim(-0.05, 1.05)
    axes[2].set_yticks([0.0, 1.0])
    axes[3].set_ylim(bottom=0.0)

    meta = groups[0][models[0]]
    fig.suptitle(
        f"Paired tournament comparison: S={meta['n_species']}, F={meta['n_factors']}, "
        f"N={meta['n_individuals']}, steps={meta['n_steps']}\n"
        "Thin gray lines = same tournament draw (`rank_seed`) across models; "
        "LP ref = exact for fixed, expected-game for dynamic models",
        fontsize=12,
    )
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=160, bbox_inches="tight")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/tournament_sweep_quick_paired.pkl")
    parser.add_argument("--prefix", default="figures/fig_tournament_quick_paired_compare")
    parser.add_argument("--models", nargs="+", default=["fixed", "bernoulli", "beta_sample"])
    parser.add_argument("--reference", default="fixed")
    parser.add_argument("--species", nargs="+", type=int, default=None)
    parser.add_argument("--factors", nargs="+", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = prepare_summary(load_rows(Path(args.input)))
    if args.species is not None:
        keep = set(args.species)
        rows = [row for row in rows if row["n_species"] in keep]
    if args.factors is not None:
        keep = set(args.factors)
        rows = [row for row in rows if row["n_factors"] in keep]
    if not rows:
        raise ValueError("no rows remain after species/factor filtering")

    present_models = {row["model"] for row in rows}
    models = [model for model in MODEL_ORDER if model in args.models and model in present_models]
    if args.reference not in models:
        raise ValueError(f"reference model {args.reference!r} not present in filtered model list {models}")
    if len(models) < 2:
        raise ValueError(f"need at least two models, got {models}")

    prefix = Path(args.prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    cells = split_rows_by_cell(rows)

    if len(cells) == 1:
        (n_species, n_factors), cell_rows = next(iter(cells.items()))
        groups = group_complete_pairs(cell_rows, models)
        if not groups:
            raise ValueError(f"no complete paired replicate sets found for models {models}")
        out = prefix.with_name(prefix.name + "_metrics.pdf")
        print_paired_summary(groups, models, args.reference, f"S={n_species}, F={n_factors}")
        plot_paired_metrics(groups, models, out)
        print(f"Saved {out}")
        return

    print(f"Found {len(cells)} cells in input; writing one paired figure per cell")
    for (n_species, n_factors), cell_rows in cells.items():
        groups = group_complete_pairs(cell_rows, models)
        if not groups:
            print(f"Skipping S={n_species}, F={n_factors}: no complete paired replicate sets")
            continue
        cell_tag = f"_S{n_species}_F{n_factors}"
        out = prefix.with_name(prefix.name + cell_tag + "_metrics.pdf")
        print_paired_summary(groups, models, args.reference, f"S={n_species}, F={n_factors}")
        plot_paired_metrics(groups, models, out)
        print(f"Saved {out}")
        print()


if __name__ == "__main__":
    main()
