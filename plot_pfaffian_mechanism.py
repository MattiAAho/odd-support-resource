"""plot_pfaffian_mechanism.py — Track B figure for the Pfaffian/matching result.

Loads results/pfaffian_matching_v1.pkl and draws a 1x3 panel:

  A  Mechanism: of all even-support equilibria, the fraction whose support has
     NO perfect matching (structural Pf=0) vs accidental, by phi_B.  ~0.98.
  B  Family-level even-support admissibility is MONOTONE in phi_B (no U-shape);
     connectivity drop marks where the selection-driven recovery begins.
  C  Unification: even-eq fraction vs realized zero-pair count collapses the
     phi_B and phi_A axes onto one curve (same zero-in-K mechanism).

Outputs fig_pfaffian_mechanism.{pdf,png} in figures/.
"""

from __future__ import annotations

import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PKL = Path("results/pfaffian_matching_v1.pkl")
OUT = Path("figures/fig_pfaffian_mechanism")
NCOL = {7: "#377eb8", 9: "#e41a1c"}     # ColorBrewer blue / red


def load():
    with open(PKL, "rb") as f:
        return pickle.load(f)


def agg_by(records, key, pred):
    """mean of pred(r) grouped by key(r), returned sorted by key."""
    g = defaultdict(list)
    for r in records:
        g[key(r)].append(pred(r))
    return sorted(g), [np.mean(g[k]) for k in sorted(g)]


def main():
    recs = load()
    phi_b = [r for r in recs if r["param_type"] == "phi_B"]
    Ns = sorted({r["N"] for r in recs})

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(14, 4.4),
                                        constrained_layout=True)

    # ── Panel A: structural fraction among even-support equilibria ──────────
    for N in Ns:
        sub = [r for r in phi_b if r["N"] == N]
        bvals = sorted({r["param_val"] for r in sub})
        bvals = [b for b in bvals if b > 0]            # phi_B=0 has no even eqs
        struct = []
        for b in bvals:
            rows = [r for r in sub if r["param_val"] == b]
            num = sum(r["n_eveneq_structural"] for r in rows)
            den = sum(r["n_eveneq"] for r in rows)
            struct.append(num / den if den else np.nan)
        axA.plot(bvals, struct, "o-", color=NCOL[N], label=f"N={N}")
    pooled = (sum(r["n_eveneq_structural"] for r in phi_b)
              / sum(r["n_eveneq"] for r in phi_b))
    axA.axhline(pooled, ls=":", color="0.4", lw=1)
    axA.text(0.62, pooled - 0.04, f"pooled {pooled:.3f}", color="0.4", fontsize=9)
    axA.set_ylim(0.85, 1.005)
    axA.set_xlabel(r"$\varphi_B$")
    axA.set_ylabel("fraction structural\n(support has no perfect matching)")
    axA.set_title("A  Even-support eqs are structural\n(100% singular, Pf$=0$)",
                  fontsize=11)
    axA.legend(frameon=False, loc="lower right")

    # ── Panel B: family even-eq fraction (monotone) + connectivity ─────────
    for N in Ns:
        sub = [r for r in phi_b if r["N"] == N]
        bv, even = agg_by(sub, lambda r: r["param_val"], lambda r: r["has_even_eq"])
        _, conn = agg_by(sub, lambda r: r["param_val"], lambda r: r["connected"])
        axB.plot(bv, even, "o-", color=NCOL[N], label=f"N={N}  even-eq")
        axB.plot(bv, conn, "s--", color=NCOL[N], alpha=0.45, ms=4,
                 label=f"N={N}  connected")
    axB.set_xlabel(r"$\varphi_B$")
    axB.set_ylabel("fraction of tournaments")
    axB.set_title("B  Family-level even admissibility\nis monotone (no U-shape)",
                  fontsize=11)
    axB.set_ylim(-0.02, 1.02)
    axB.legend(frameon=False, fontsize=8, loc="upper left")
    axB.text(0.33, 0.06, "U-shape recovery is a selection\neffect, not Pfaffian",
             transform=axB.transAxes, fontsize=8.5, color="0.3")

    # ── Panel C: phi_A / phi_B unification by realized zero-pair count ──────
    for N in Ns:
        sN = [r for r in recs if r["N"] == N]
        for ptype, marker, fill in (("phi_B", "o", NCOL[N]),
                                    ("phi_A", "^", "none")):
            pts = [r for r in sN if r["param_type"] == ptype]
            z, ev = agg_by(pts, lambda r: r["n_zero_pairs"],
                           lambda r: r["has_even_eq"])
            # keep buckets with enough support for a stable mean
            cnt = defaultdict(int)
            for r in pts:
                cnt[r["n_zero_pairs"]] += 1
            zz = [zi for zi in z if cnt[zi] >= 15]
            ee = [ev[z.index(zi)] for zi in zz]
            lbl = (f"N={N}  " + (r"$\varphi_B$" if ptype == "phi_B"
                                 else r"$\varphi_A$"))
            axC.plot(zz, ee, marker=marker, ls="-" if ptype == "phi_B" else "--",
                     mfc=fill, mec=NCOL[N], color=NCOL[N],
                     alpha=0.9 if ptype == "phi_B" else 0.7, label=lbl)
    axC.set_xlabel("realized zero pairs in $K$")
    axC.set_ylabel("even-eq fraction")
    axC.set_title(r"C  $\varphi_A$ and $\varphi_B$ collapse" "\n"
                  "(one zero-in-$K$ mechanism)", fontsize=11)
    axC.legend(frameon=False, fontsize=8, loc="lower right")

    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT.with_suffix(".pdf"), dpi=200)
    fig.savefig(OUT.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"Saved {OUT}.pdf / .png")


if __name__ == "__main__":
    main()
