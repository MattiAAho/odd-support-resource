"""N=4 equilibrium non-uniqueness analysis.

For each of the 729 tournament structures (N=4, 6 pairs each ∈ {-1, 0, +1}),
enumerate ALL valid Nash equilibrium support sets using direct linear algebra
(null-space method, no LP-solver internals).  Compare the parity of every valid
equilibrium against the HiGHS-selected one to determine whether the U-shape
result in the phi_B sweep is selection-dependent (Scenario B/C) or robust to
which equilibrium the solver returns (Scenario A).

Three outcome scenarios (from Opus 4.7 briefing):
  A: all valid equilibria for each structure have the same parity → result robust
  B: absolute parity fractions differ across selection rules, but U-shape shape holds
  C: qualitatively different parity stories under different selection rules
"""

from __future__ import annotations

import itertools
from collections import Counter, defaultdict

import numpy as np

from tournament.lp import all_equilibrium_support_sets, solve_zero_sum_equilibrium

PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
N = 4
SUPPORT_TOL = 1e-6


def build_payoff(assignment: tuple[int, ...]) -> np.ndarray:
    K = np.zeros((N, N))
    for val, (a, b) in zip(assignment, PAIRS):
        K[a, b] = val
        K[b, a] = -val
    return K


def main() -> None:
    by_k: dict[int, list[dict]] = defaultdict(list)

    for assignment in itertools.product([-1, 0, 1], repeat=6):
        k = sum(1 for v in assignment if v != 0)
        K = build_payoff(assignment)

        x_highs, _ = solve_zero_sum_equilibrium(K)
        highs_sz = int(np.sum(x_highs > SUPPORT_TOL))

        all_eq = all_equilibrium_support_sets(K)
        all_sizes = sorted(len(s) for s, _ in all_eq)
        all_parities = set(sz % 2 for sz in all_sizes)  # 0=even, 1=odd

        parity_determinate = len(all_parities) <= 1
        highs_in_family = any(
            len(s) == highs_sz for s, _ in all_eq
        )

        by_k[k].append({
            "assignment": assignment,
            "highs_sz": highs_sz,
            "highs_odd": highs_sz % 2 == 1,
            "n_valid": len(all_eq),
            "all_sizes": all_sizes,
            "parity_determinate": parity_determinate,
            "highs_in_family": highs_in_family,
        })

    # ── Summary table ──────────────────────────────────────────────────────
    print("=" * 72)
    print("N=4 equilibrium non-uniqueness analysis  (3^6 = 729 structures)")
    print("=" * 72)
    print(
        f"\n{'k':>3}  {'n':>5}  {'HiGHS odd':>10}  {'multi-eq':>9}  "
        f"{'mixed parity':>13}  {'HiGHS∈family':>13}"
    )
    print("-" * 60)

    total = dict(n=0, mixed=0, multi=0, not_in_fam=0)

    for k in range(7):
        rows = by_k.get(k, [])
        if not rows:
            continue
        n = len(rows)
        highs_odd = sum(r["highs_odd"] for r in rows) / n
        multi_eq = sum(r["n_valid"] > 1 for r in rows) / n
        mixed_par = sum(not r["parity_determinate"] for r in rows) / n
        in_fam = sum(r["highs_in_family"] for r in rows) / n
        total["n"] += n
        total["mixed"] += sum(not r["parity_determinate"] for r in rows)
        total["multi"] += sum(r["n_valid"] > 1 for r in rows)
        total["not_in_fam"] += sum(not r["highs_in_family"] for r in rows)
        print(
            f"{k:>3}  {n:>5}  {highs_odd:>10.3f}  {multi_eq:>9.3f}  "
            f"{mixed_par:>13.3f}  {in_fam:>13.3f}"
        )

    N_total = total["n"]
    print(
        f"{'ALL':>3}  {N_total:>5}  {'':>10}  "
        f"{total['multi']/N_total:>9.3f}  "
        f"{total['mixed']/N_total:>13.3f}  "
        f"{1-total['not_in_fam']/N_total:>13.3f}"
    )

    # ── Detailed look at mixed-parity structures ────────────────────────────
    mixed = [r for rows in by_k.values() for r in rows if not r["parity_determinate"]]
    if mixed:
        print(f"\n--- Mixed-parity structures: {len(mixed)} / {N_total} ---")
        print("  (first 10 shown; 'sizes' = set of support sizes across all valid equilibria)")
        for r in mixed[:10]:
            k = sum(1 for v in r["assignment"] if v != 0)
            print(
                f"  k={k}  HiGHS_sz={r['highs_sz']}  "
                f"sizes={sorted(set(r['all_sizes']))}  "
                f"assign={r['assignment']}"
            )
        # Distribution: what support-size pairs appear in mixed structures?
        size_pairs: Counter = Counter()
        for r in mixed:
            size_pairs[tuple(sorted(set(r["all_sizes"])))] += 1
        print(f"\n  Support-size combinations in mixed structures:")
        for pair, cnt in size_pairs.most_common():
            print(f"    {pair}: {cnt} structures")
    else:
        print("\n✓ No mixed-parity structures — Scenario A confirmed.")
        print("  Parity is uniquely determined by the tournament structure.")
        print("  The U-shape result in the phi_B sweep is not selection-dependent.")

    # ── Distribution of n_valid_supports across all structures ─────────────
    all_rows = [r for rows in by_k.values() for r in rows]
    dist = Counter(r["n_valid"] for r in all_rows)
    print(f"\n--- Distribution of #valid equilibrium support sets across 729 structures ---")
    for n_eq in sorted(dist):
        frac = dist[n_eq] / N_total
        bar = "█" * int(frac * 30)
        print(f"  n_eq={n_eq}: {dist[n_eq]:>4} ({frac:.3f})  {bar}")


if __name__ == "__main__":
    main()
