"""Plot 2D phase diagram: phi_A (neutrality) × phi_B (sparsity).

Main figure: heatmaps of LP odd-support fraction in the (phi_A, phi_B) plane,
one panel per model (fixed, bernoulli).  Overlaid contour lines test whether
isolines follow (1-phi_A)(1-phi_B) = const (hyperbola prediction) or
phi_A + phi_B = const (diagonal prediction).

Secondary figure: heatmap of effective zero-payoff pair fraction phi_eff
and residuals (observed LP odd − prediction from 1D phi_eff lookup).
"""

from __future__ import annotations

import argparse
import pickle

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

MODELS = ["fixed", "bernoulli"]
MODEL_LABELS = {"fixed": "Fixed", "bernoulli": "Bernoulli", "beta_sample": "Beta-sample"}


def load(path: str) -> list[dict]:
    with open(path, "rb") as fh:
        return pickle.load(fh)


def make_grid(
    records: list[dict],
    model: str,
    metric: str = "lp_support_odd",
    exclude_degenerate: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (phi_A_vals, phi_B_vals, grid[i_phi_A, j_phi_B])."""
    sub = [r for r in records if r["model"] == model]
    if exclude_degenerate:
        sub = [r for r in sub if not r.get("lp_degenerate", False)]
    phi_A_vals = sorted(set(r["phi_A"] for r in sub))
    phi_B_vals = sorted(set(r["phi_B"] for r in sub))
    grid = np.full((len(phi_A_vals), len(phi_B_vals)), np.nan)
    for i, pA in enumerate(phi_A_vals):
        for j, pB in enumerate(phi_B_vals):
            rows = [r for r in sub if abs(r["phi_A"] - pA) < 1e-6 and abs(r["phi_B"] - pB) < 1e-6]
            if rows:
                grid[i, j] = np.mean([r[metric] for r in rows])
    return np.array(phi_A_vals), np.array(phi_B_vals), grid


def phi_eff_grid(phi_A_vals: np.ndarray, phi_B_vals: np.ndarray) -> np.ndarray:
    """Theoretical phi_eff = 1 - (1-phi_A)(1-phi_B)."""
    pA, pB = np.meshgrid(phi_A_vals, phi_B_vals, indexing="ij")
    return 1.0 - (1.0 - pA) * (1.0 - pB)


def build_phi_eff_to_lp_lookup(
    records: list[dict],
    model: str,
    exclude_degenerate: bool = True,
) -> dict[float, float]:
    """Mean LP odd fraction per actual phi_eff (binned to 2 decimal places)."""
    sub = [r for r in records if r["model"] == model]
    if exclude_degenerate:
        sub = [r for r in sub if not r.get("lp_degenerate", False)]
    # Use only the phi_B=0 rows for the 1D phi_eff→LP_odd reference (pure phi_A axis)
    # and the phi_A=0 rows for the pure phi_B axis — they should match if mechanism is shared.
    lookup: dict[float, list[float]] = {}
    for r in sub:
        key = round(r["actual_phi_eff"], 2)
        lookup.setdefault(key, []).append(r["lp_support_odd"])
    return {k: float(np.mean(v)) for k, v in lookup.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="results/phase_2d_v1.pkl")
    ap.add_argument("--prefix", default="figures/fig_phase_2d")
    ap.add_argument("--factors", nargs="+", type=int, default=None,
                    help="restrict to these F values (default: pool all)")
    args = ap.parse_args()

    records = load(args.input)
    if args.factors:
        records = [r for r in records if r["n_factors"] in args.factors]

    # --- Figure 1: LP odd-support heatmaps with contour overlay ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, model in zip(axes, MODELS):
        phi_A_vals, phi_B_vals, grid = make_grid(records, model, "lp_support_odd")

        im = ax.imshow(
            grid,
            origin="lower",
            aspect="auto",
            extent=[phi_B_vals[0] - 0.05, phi_B_vals[-1] + 0.05,
                    phi_A_vals[0] - 0.05, phi_A_vals[-1] + 0.05],
            cmap="RdYlGn",
            vmin=0.4,
            vmax=1.0,
        )
        plt.colorbar(im, ax=ax, label="LP odd-support fraction")

        # Hyperbola contours: (1-phi_A)(1-phi_B) = c
        pA_fine = np.linspace(phi_A_vals[0], phi_A_vals[-1], 200)
        pB_fine = np.linspace(phi_B_vals[0], phi_B_vals[-1], 200)
        pA_m, pB_m = np.meshgrid(pA_fine, pB_fine, indexing="ij")
        eff = 1.0 - (1.0 - pA_m) * (1.0 - pB_m)
        levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        cs = ax.contour(pB_fine, pA_fine, eff, levels=levels,
                        colors="white", linewidths=0.8, linestyles="--", alpha=0.7)
        ax.clabel(cs, fmt="φ_eff=%.1f", fontsize=7, inline=True)

        # Diagonal contours: phi_A + phi_B = c (DeepSeek prediction)
        diag = pA_m + pB_m
        cs2 = ax.contour(pB_fine, pA_fine, diag, levels=levels,
                         colors="cyan", linewidths=0.8, linestyles=":", alpha=0.5)

        ax.set_xlabel("φ_B (sparsity — structural zeros)")
        ax.set_ylabel("φ_A (neutrality — neutral pairs)")
        ax.set_title(f"{MODEL_LABELS[model]}: LP odd-support fraction\n"
                     f"White dashed = φ_eff=(1-φ_A)(1-φ_B) contours  "
                     f"Cyan dotted = φ_A+φ_B contours")

    fig.suptitle("2D phase diagram: LP odd-support fraction\n"
                 "N=7, F pooled", fontsize=12)
    fig.tight_layout()
    out1 = f"{args.prefix}_lp_odd.pdf"
    fig.savefig(out1, bbox_inches="tight")
    fig.savefig(out1.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"Saved {out1}")
    plt.close(fig)

    # --- Figure 2: residual from 1D phi_eff prediction ---
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))

    for ax, model in zip(axes2, MODELS):
        phi_A_vals, phi_B_vals, grid = make_grid(records, model, "lp_support_odd")
        lookup = build_phi_eff_to_lp_lookup(records, model)
        eff_grid = phi_eff_grid(phi_A_vals, phi_B_vals)

        # Interpolate 1D prediction onto grid
        pred_grid = np.full_like(grid, np.nan)
        for i in range(len(phi_A_vals)):
            for j in range(len(phi_B_vals)):
                key = round(eff_grid[i, j], 2)
                if key in lookup:
                    pred_grid[i, j] = lookup[key]

        residual = grid - pred_grid
        vmax = max(abs(np.nanmin(residual)), abs(np.nanmax(residual)), 0.05)

        im = ax.imshow(
            residual,
            origin="lower",
            aspect="auto",
            extent=[phi_B_vals[0] - 0.05, phi_B_vals[-1] + 0.05,
                    phi_A_vals[0] - 0.05, phi_A_vals[-1] + 0.05],
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
        )
        plt.colorbar(im, ax=ax, label="Residual (observed − 1D φ_eff prediction)")
        ax.set_xlabel("φ_B (sparsity)")
        ax.set_ylabel("φ_A (neutrality)")
        ax.set_title(f"{MODEL_LABELS[model]}: residual from 1D φ_eff prediction\n"
                     f"Near-zero = contour parallelism confirmed")

    fig2.suptitle("2D phase diagram: residuals from scalar-φ_eff prediction", fontsize=12)
    fig2.tight_layout()
    out2 = f"{args.prefix}_residuals.pdf"
    fig2.savefig(out2, bbox_inches="tight")
    fig2.savefig(out2.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"Saved {out2}")
    plt.close(fig2)

    # --- Console summary ---
    print("\n=== LP odd-support fraction by (phi_A, phi_B) cell ===")
    phi_A_vals, phi_B_vals, grid = make_grid(records, "bernoulli", "lp_support_odd")
    header = f"{'phi_A':>6}  " + "  ".join(f"pB={pB:.1f}" for pB in phi_B_vals)
    print(f"\nBernoulli model:\n{header}")
    for i, pA in enumerate(phi_A_vals):
        row = f"{pA:>6.1f}  " + "  ".join(f"{'—':>7}" if np.isnan(grid[i, j]) else f"{grid[i,j]:>7.3f}" for j in range(len(phi_B_vals)))
        print(row)

    phi_A_vals, phi_B_vals, grid_f = make_grid(records, "fixed", "lp_support_odd")
    print(f"\nFixed model:\n{header}")
    for i, pA in enumerate(phi_A_vals):
        row = f"{pA:>6.1f}  " + "  ".join(f"{'—':>7}" if np.isnan(grid_f[i, j]) else f"{grid_f[i,j]:>7.3f}" for j in range(len(phi_B_vals)))
        print(row)

    eff_grid = phi_eff_grid(phi_A_vals, phi_B_vals)
    print(f"\nTheoretical phi_eff = 1-(1-phi_A)(1-phi_B):\n{header}")
    for i, pA in enumerate(phi_A_vals):
        row = f"{pA:>6.1f}  " + "  ".join(f"{eff_grid[i,j]:>7.3f}" for j in range(len(phi_B_vals)))
        print(row)


if __name__ == "__main__":
    main()
