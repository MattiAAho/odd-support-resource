"""Plot results of the φ_A (neutral-pairs) sweep.

Three panels:
  (a) Mean extinction step vs φ_A — selection-drift crossover (log y)
  (b) Bernoulli LP odd fraction vs φ_A — U-shape parity result
      with faint horizontal reference band from φ_B at matched zero-pair density
  (c) Cross-sweep comparison: LP_odd vs actual zero pairs in K
      overlaying φ_A (bernoulli) and φ_B (bernoulli) at N=7 and N=9
      (filled = N=7, hollow = N=9) to verify unification holds across N
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

N_COLORS = {7: "#1f77b4", 9: "#ff7f0e"}
SWEEP_COLORS = {"phi_A": "#2ca02c", "phi_B": "#d62728"}


def load(path: str) -> list[dict]:
    with open(path, "rb") as fh:
        return pickle.load(fh)


def bernoulli_lp_odd_by_zeros(records: list[dict], n_species: int,
                                zero_key: str, total_pairs: int,
                                exclude_degen: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Return (mean_zero_pairs, mean_lp_odd) for bernoulli model at given N."""
    phi_key = "phi_A" if zero_key == "n_neutral_pairs" else "phi_B"
    phi_vals = sorted(set(r[phi_key] for r in records))
    zeros, odds = [], []
    for phi in phi_vals:
        rows = [r for r in records
                if r[phi_key] == phi and r["model"] == "bernoulli"
                and r["n_species"] == n_species]
        if exclude_degen:
            rows = [r for r in rows if not r.get("lp_degenerate", False)]
        if not rows:
            continue
        z = np.mean([r[zero_key] for r in rows])
        o = np.mean([r["lp_support_odd"] for r in rows])
        zeros.append(z)
        odds.append(o)
    return np.array(zeros), np.array(odds)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-phi-a", default="results/phi_a_sweep_v1.pkl")
    ap.add_argument("--input-phi-b", default="results/sparsity_sweep_v1.pkl")
    ap.add_argument("--prefix", default="figures/fig_phi_a_sweep")
    ap.add_argument("--factors", nargs="+", type=int, default=None)
    args = ap.parse_args()

    pa = load(args.input_phi_a)
    pb = load(args.input_phi_b)

    if args.factors:
        pa = [r for r in pa if r["n_factors"] in args.factors]
        pb = [r for r in pb if r["n_factors"] in args.factors]

    # Pre-compute φ_B LP_odd trough range for the panel (b) reference band.
    # Use N=7 and N=9, φ_B ∈ [0.2, 0.4] (trough region for both N values).
    pb_trough_odds = []
    for N in [7, 9]:
        for phi in [0.2, 0.3, 0.4]:
            rows = [r for r in pb
                    if r["phi_B"] == phi and r["model"] == "bernoulli"
                    and r["n_species"] == N]
            if rows:
                pb_trough_odds.append(np.mean([r["lp_support_odd"] for r in rows]))
    pb_trough_lo = min(pb_trough_odds)
    pb_trough_hi = max(pb_trough_odds)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # ── Panel (a): Mean extinction step (bernoulli) vs φ_A — log scale ───────
    ax = axes[0]
    for N, c in N_COLORS.items():
        phi_vals = sorted(set(r["phi_A"] for r in pa))
        steps, steps_lo, steps_hi = [], [], []
        for phi in phi_vals:
            rows = [r for r in pa
                    if r["phi_A"] == phi and r["model"] == "bernoulli"
                    and r["n_species"] == N]
            ext = [r["extinct_time"] for r in rows if r["extinct_time"] is not None]
            if not ext:
                steps.append(float("nan"))
                steps_lo.append(float("nan"))
                steps_hi.append(float("nan"))
                continue
            m = np.mean(ext)
            se = np.std(ext) / np.sqrt(len(ext))
            steps.append(m)
            steps_lo.append(m - se)
            steps_hi.append(m + se)
        phi_arr = np.array(phi_vals)
        steps = np.array(steps)
        ax.semilogy(phi_arr, steps, "o-", color=c, label=f"N={N}")
        ax.fill_between(phi_arr,
                        np.maximum(steps_lo, 1.0),
                        np.array(steps_hi),
                        alpha=0.15, color=c)

    ax.set_xlabel("φ_A (neutral-pair fraction)")
    ax.set_ylabel("Mean extinction step (log scale)")
    ax.set_title("(a)  Selection-drift crossover\n(Bernoulli simulation)")
    ax.legend(fontsize=9)

    # ── Panel (b): Bernoulli LP odd fraction vs φ_A — with φ_B reference ─────
    ax = axes[1]

    # Faint horizontal band showing φ_B trough depth at matched density
    ax.axhspan(pb_trough_lo, pb_trough_hi, color="#d62728", alpha=0.08,
               label=f"φ_B trough range [{pb_trough_lo:.2f}–{pb_trough_hi:.2f}]")

    for N, c in N_COLORS.items():
        phi_vals = sorted(set(r["phi_A"] for r in pa))
        lp_odd, lp_se = [], []
        degen_phi, degen_odd = [], []
        for phi in phi_vals:
            rows = [r for r in pa
                    if r["phi_A"] == phi and r["model"] == "bernoulli"
                    and r["n_species"] == N]
            vals = np.array([r["lp_support_odd"] for r in rows], dtype=float)
            m = vals.mean()
            lp_odd.append(m)
            lp_se.append(vals.std() / np.sqrt(len(vals)))
            if any(r.get("lp_degenerate", False) for r in rows):
                degen_phi.append(phi)
                degen_odd.append(m)

        phi_arr = np.array(phi_vals)
        lp_odd = np.array(lp_odd)
        lp_se = np.array(lp_se)
        ax.plot(phi_arr, lp_odd, "o-", color=c, label=f"N={N}")
        ax.fill_between(phi_arr, lp_odd - lp_se, lp_odd + lp_se,
                        alpha=0.15, color=c)
        if degen_phi:
            ax.scatter(degen_phi, degen_odd, marker="x", color=c,
                       s=80, zorder=5)

    ax.axhline(1.0, color="k", lw=0.6, ls="--", alpha=0.35)
    ax.axhline(0.5, color="k", lw=0.5, ls=":", alpha=0.25)
    ax.set_ylim(0.45, 1.05)
    ax.set_xlabel("φ_A (neutral-pair fraction)")
    ax.set_ylabel("Bernoulli LP odd-support fraction")
    ax.set_title("(b)  LP parity U-shape\n(× = degenerate; shading = φ_B trough)")
    ax.legend(fontsize=8, loc="upper right")
    ax.text(1.0, 1.004, "degen.", fontsize=7, ha="center", color="gray")

    # ── Panel (c): Cross-sweep: LP_odd vs zero pairs (N=7 filled, N=9 hollow) ─
    ax = axes[2]

    markers_filled = {"phi_A": "o", "phi_B": "s"}
    markers_hollow = {"phi_A": "o", "phi_B": "s"}

    for N, total_pairs in [(7, 21), (9, 36)]:
        filled = (N == 7)
        mfc_A = SWEEP_COLORS["phi_A"] if filled else "none"
        mfc_B = SWEEP_COLORS["phi_B"] if filled else "none"
        mew = 1.0 if filled else 1.5

        # φ_A sweep
        zeros_a, odds_a = bernoulli_lp_odd_by_zeros(
            pa, N, "n_neutral_pairs", total_pairs, exclude_degen=True)
        ax.scatter(zeros_a, odds_a,
                   color=SWEEP_COLORS["phi_A"], marker=markers_filled["phi_A"],
                   s=60, zorder=4, facecolors=mfc_A,
                   edgecolors=SWEEP_COLORS["phi_A"], linewidths=mew)

        # φ_B sweep
        def n_zeros_phi_b(r, tp=total_pairs):
            return tp - r["n_active_pairs"]

        phi_b_vals = sorted(set(r["phi_B"] for r in pb))
        for phi in phi_b_vals:
            rows = [r for r in pb
                    if r["phi_B"] == phi and r["model"] == "bernoulli"
                    and r["n_species"] == N]
            if not rows:
                continue
            z = np.mean([n_zeros_phi_b(r) for r in rows])
            o = np.mean([r["lp_support_odd"] for r in rows])
            ax.scatter(z, o,
                       color=SWEEP_COLORS["phi_B"], marker=markers_filled["phi_B"],
                       s=60, zorder=4, facecolors=mfc_B,
                       edgecolors=SWEEP_COLORS["phi_B"], linewidths=mew)

    ax.axhline(1.0, color="k", lw=0.6, ls="--", alpha=0.35)
    ax.axhline(0.5, color="k", lw=0.5, ls=":", alpha=0.25)
    ax.set_ylim(0.45, 1.05)
    ax.set_xlabel("Mean zero pairs in K (N=7: /21, N=9: /36)")
    ax.set_ylabel("Bernoulli LP odd-support fraction")
    ax.set_title("(c)  Shared trough mechanism\n(filled = N=7, hollow = N=9)")

    leg_handles = [
        mlines.Line2D([], [], color=SWEEP_COLORS["phi_A"], marker="o",
                      ls="None", markersize=8, label="φ_A sweep (neutral pairs)"),
        mlines.Line2D([], [], color=SWEEP_COLORS["phi_B"], marker="s",
                      ls="None", markersize=8, label="φ_B sweep (struct. zeros)"),
        mlines.Line2D([], [], color="k", marker="o", ls="None", markersize=7,
                      label="N=7 (filled)"),
        mlines.Line2D([], [], color="k", marker="o", ls="None", markersize=7,
                      mfc="none", label="N=9 (hollow)"),
    ]
    ax.legend(handles=leg_handles, fontsize=8)

    fig.suptitle("φ_A sweep: selection-drift crossover × LP parity", fontsize=13)
    fig.tight_layout()

    prefix = Path(args.prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".pdf", ".png"):
        out = f"{prefix}{ext}"
        try:
            fig.savefig(out, bbox_inches="tight", dpi=150 if ext == ".png" else None)
            print(f"Saved {out}")
        except PermissionError:
            alt = str(prefix) + "_v2" + ext
            fig.savefig(alt, bbox_inches="tight", dpi=150 if ext == ".png" else None)
            print(f"Saved {alt} (permission denied on original)")
    plt.close(fig)


if __name__ == "__main__":
    main()
