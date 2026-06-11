"""Combined Figure 4: neutrality (phi_A) vs sparsity (phi_B) and the phi_eff phase diagram.

Three panels, all Bernoulli LP odd-support fraction:
  (a) vs phi_A, with the phi_B trough band overlaid (neutrality reproduces the trough).
  (b) Collapse: vs mean zero pairs in K, phi_A and phi_B overlaid
      (N=7 filled, N=9 hollow) -- agreement at matched zero density.
  (c) 2D phase diagram over (phi_B, phi_A) with phi_eff=(1-phi_A)(1-phi_B)
      contours (N=7, F pooled).

Reuses make_grid (plot_phase_2d) and bernoulli_lp_odd_by_zeros (plot_phi_a_sweep)
so the panels match the standalone figures exactly.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

from plot_phase_2d import make_grid
from plot_phi_a_sweep import bernoulli_lp_odd_by_zeros, N_COLORS, SWEEP_COLORS


def load(path: str) -> list[dict]:
    with open(path, "rb") as fh:
        return pickle.load(fh)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phi-a", default="results/phi_a_sweep_v1.pkl")
    ap.add_argument("--phi-b", default="results/sparsity_sweep_v1.pkl")
    ap.add_argument("--phase-2d", default="results/phase_2d_v1.pkl")
    ap.add_argument("--prefix", default="figures/fig_phi_combined")
    args = ap.parse_args()

    pa = load(args.phi_a)
    pb = load(args.phi_b)
    p2 = load(args.phase_2d)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # ── Panel (a): Bernoulli LP odd vs phi_A, with phi_B trough band ──────────
    trough = []
    for N in (7, 9):
        for phi in (0.2, 0.3, 0.4):
            rows = [r for r in pb if r["phi_B"] == phi and r["model"] == "bernoulli"
                    and r["n_species"] == N]
            if rows:
                trough.append(np.mean([r["lp_support_odd"] for r in rows]))
    lo, hi = min(trough), max(trough)

    ax = axes[0]
    ax.axhspan(lo, hi, color=SWEEP_COLORS["phi_B"], alpha=0.08,
               label=f"φ_B trough [{lo:.2f}–{hi:.2f}]")
    for N, c in N_COLORS.items():
        phis = sorted(set(r["phi_A"] for r in pa))
        odd, se = [], []
        for phi in phis:
            rows = [r for r in pa if r["phi_A"] == phi and r["model"] == "bernoulli"
                    and r["n_species"] == N]
            v = np.array([r["lp_support_odd"] for r in rows], dtype=float)
            odd.append(v.mean())
            se.append(v.std() / np.sqrt(len(v)))
        phis, odd, se = np.array(phis), np.array(odd), np.array(se)
        ax.plot(phis, odd, "o-", color=c, label=f"N={N}")
        ax.fill_between(phis, odd - se, odd + se, alpha=0.15, color=c)
    ax.axhline(1.0, color="k", lw=0.6, ls="--", alpha=0.35)
    ax.set_ylim(0.45, 1.05)
    ax.set_xlabel("φ_A (neutral-pair fraction)")
    ax.set_ylabel("Bernoulli LP odd-support fraction")
    ax.set_title("(a)  Neutrality reproduces the trough")
    ax.legend(fontsize=8, loc="lower left")

    # ── Panel (b): collapse vs zero pairs (N=7 filled, N=9 hollow) ────────────
    ax = axes[1]
    for N, total in ((7, 21), (9, 36)):
        filled = (N == 7)
        za, oa = bernoulli_lp_odd_by_zeros(pa, N, "n_neutral_pairs", total,
                                           exclude_degen=True)
        ax.scatter(za, oa, marker="o", s=60, zorder=4,
                   facecolors=SWEEP_COLORS["phi_A"] if filled else "none",
                   edgecolors=SWEEP_COLORS["phi_A"], linewidths=1.0 if filled else 1.5)
        for phi in sorted(set(r["phi_B"] for r in pb)):
            rows = [r for r in pb if r["phi_B"] == phi and r["model"] == "bernoulli"
                    and r["n_species"] == N]
            if not rows:
                continue
            z = np.mean([total - r["n_active_pairs"] for r in rows])
            o = np.mean([r["lp_support_odd"] for r in rows])
            ax.scatter(z, o, marker="s", s=60, zorder=4,
                       facecolors=SWEEP_COLORS["phi_B"] if filled else "none",
                       edgecolors=SWEEP_COLORS["phi_B"], linewidths=1.0 if filled else 1.5)
    ax.axhline(1.0, color="k", lw=0.6, ls="--", alpha=0.35)
    ax.set_ylim(0.45, 1.05)
    ax.set_xlabel("Mean zero pairs in K  (N=7: /21, N=9: /36)")
    ax.set_ylabel("Bernoulli LP odd-support fraction")
    ax.set_title("(b)  Collapse at matched zero density")
    handles = [
        mlines.Line2D([], [], color=SWEEP_COLORS["phi_A"], marker="o", ls="None",
                      markersize=8, label="φ_A (neutral pairs)"),
        mlines.Line2D([], [], color=SWEEP_COLORS["phi_B"], marker="s", ls="None",
                      markersize=8, label="φ_B (struct. zeros)"),
        mlines.Line2D([], [], color="k", marker="o", ls="None", markersize=7,
                      label="N=7 (filled)"),
        mlines.Line2D([], [], color="k", marker="o", ls="None", markersize=7,
                      mfc="none", label="N=9 (hollow)"),
    ]
    ax.legend(handles=handles, fontsize=8, loc="lower left")

    # ── Panel (c): phi_eff 2D phase diagram (Bernoulli) ──────────────────────
    ax = axes[2]
    phiA, phiB, grid = make_grid(p2, "bernoulli", "lp_support_odd")
    im = ax.imshow(grid, origin="lower", aspect="auto",
                   extent=[phiB[0] - 0.05, phiB[-1] + 0.05,
                           phiA[0] - 0.05, phiA[-1] + 0.05],
                   cmap="RdYlGn", vmin=0.4, vmax=1.0)
    plt.colorbar(im, ax=ax, label="LP odd-support fraction")
    pA_fine = np.linspace(phiA[0], phiA[-1], 200)
    pB_fine = np.linspace(phiB[0], phiB[-1], 200)
    pAm, pBm = np.meshgrid(pA_fine, pB_fine, indexing="ij")
    eff = 1.0 - (1.0 - pAm) * (1.0 - pBm)
    cs = ax.contour(pB_fine, pA_fine, eff,
                    levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                    colors="blue", linewidths=1.4, linestyles="--", alpha=0.9)
    ax.clabel(cs, fmt="φ_eff=%.1f", fontsize=7, inline=True)
    ax.set_xlabel("φ_B (sparsity)")
    ax.set_ylabel("φ_A (neutrality)")
    ax.set_title("(c)  φ_eff phase diagram (Bernoulli, N=7)")

    fig.suptitle("Neutrality and sparsity unify through the zero-in-K density", fontsize=13)
    fig.tight_layout()

    prefix = Path(args.prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".pdf", ".png"):
        out = f"{prefix}{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=150 if ext == ".png" else None)
        print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
