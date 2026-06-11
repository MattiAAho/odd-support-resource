"""Exhaustive N=4 tournament enumeration.

For N=4, there are 6 unordered pairs.  Each pair can be in one of 3 states:
  0  — no interaction (structural zero)
 +1  — species a beats species b
 -1  — species b beats species a

Total distinct directed tournaments: 3^6 = 729.

For each structure this script:
  1. Builds the antisymmetric payoff matrix (Bernoulli LP variant).
  2. Solves the LP and records support size and parity.
  3. For the fixed-LP variant (complete k=6 tournaments only):
     enumerates all 2^6 = 64 directed complete tournaments and checks that
     A&L parity holds exactly.

The enumeration is exact — no sampling.  Runtime: < 1 second.
"""

from __future__ import annotations

import itertools
from collections import defaultdict

import numpy as np

from tournament.lp import solve_zero_sum_equilibrium

PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
N = 4
SUPPORT_TOL = 1e-3


def support_size(x: np.ndarray) -> int:
    return int(np.sum(x > SUPPORT_TOL))


def build_payoff(assignment: tuple[int, ...]) -> np.ndarray:
    """Build N×N antisymmetric payoff from a 6-tuple of {-1,0,+1} values."""
    payoff = np.zeros((N, N))
    for val, (a, b) in zip(assignment, PAIRS):
        payoff[a, b] = val
        payoff[b, a] = -val
    return payoff


def classify_structure(assignment: tuple[int, ...]) -> str:
    """Return a human-readable label for notable 4-species structures."""
    edges = [(PAIRS[i], assignment[i]) for i in range(6) if assignment[i] != 0]
    k = len(edges)

    # 4-Hamiltonian-cycle: exactly 4 edges forming 0→1→2→3→0
    ham = {((0,1),1),((1,2),1),((2,3),1),((3,0),-1)}  # payoff from pair (0,3): b beats a = -1 → a=0,b=3, val=-1
    # Alternative: use undirected + directed check
    edge_set = frozenset((p, v) for p, v in edges)

    # Pure 4-cycle: 0→1→2→3→0 encoded in our pairs
    # (0,1)=+1, (1,2)=+1, (2,3)=+1, (0,3)=-1 [i.e. 3→0], (0,2)=0, (1,3)=0
    if assignment == (1, 0, -1, 1, 0, 1):
        return "4-cycle (0→1→2→3→0)"

    # 3-cycle + isolated species (one of {0,1,2,3} has no interactions)
    nonzero_species = set()
    for (a, b), v in edges:
        nonzero_species.add(a)
        nonzero_species.add(b)
    if k == 3 and len(nonzero_species) == 3:
        isolated = (set(range(N)) - nonzero_species).pop()
        return f"3-cycle + isolated (species {isolated})"

    # Transitive complete tournament: 0>1>2>3 etc.
    if k == 6:
        # Check transitivity: for all i<j<k, if i>j and j>k then i>k
        winner = {(a, b): v > 0 for (a, b), v in edges}
        winner.update({(b, a): v < 0 for (a, b), v in edges})
        is_transitive = True
        for i in range(N):
            for j in range(N):
                for k_ in range(N):
                    if i != j and j != k_ and i != k_:
                        if winner.get((i, j), False) and winner.get((j, k_), False):
                            if not winner.get((i, k_), False):
                                is_transitive = False
        if is_transitive:
            return "complete transitive"

    return f"k={k}"


def bernoulli_enumeration() -> dict:
    """Enumerate all 729 structures, solve Bernoulli LP, return aggregated stats."""
    stats: dict[int, list[int]] = defaultdict(list)  # k → [support_sizes]
    per_structure: list[dict] = []

    for assignment in itertools.product([-1, 0, 1], repeat=6):
        k = sum(1 for v in assignment if v != 0)
        payoff = build_payoff(assignment)
        lp_x, _ = solve_zero_sum_equilibrium(payoff)
        sz = support_size(lp_x)
        stats[k].append(sz)
        label = classify_structure(assignment)
        per_structure.append({
            "assignment": assignment,
            "k": k,
            "support_size": sz,
            "support_odd": sz % 2 == 1,
            "label": label,
        })

    return {"by_k": dict(stats), "per_structure": per_structure}


def fixed_complete_enumeration() -> dict:
    """Enumerate all 2^6 = 64 complete tournaments, solve fixed LP.

    A&L theorem guarantees all should have odd support.
    """
    results = []
    for bits in range(64):
        assignment = tuple(1 if (bits >> i) & 1 else -1 for i in range(6))
        payoff = build_payoff(assignment)
        # For fixed LP: payoff is already {-1,+1} (winner matrix)
        winner = (payoff > 0).astype(np.float64)
        lp_x, _ = solve_zero_sum_equilibrium(winner - winner.T)
        sz = support_size(lp_x)
        results.append(sz)

    return {"support_sizes": results}


def print_summary(bern: dict, fixed: dict) -> None:
    print("=" * 60)
    print(f"N=4 exhaustive enumeration  (3^6 = 729 Bernoulli structures)")
    print("=" * 60)

    print("\n--- Bernoulli LP odd-support fraction by # interacting pairs ---")
    print(f"{'k':>4}  {'n_structs':>10}  {'lp_odd':>8}  {'mean_supp':>10}  {'supp distribution'}")
    print("-" * 70)
    for k in range(7):
        sizes = bern["by_k"].get(k, [])
        if not sizes:
            continue
        n = len(sizes)
        odd_frac = sum(1 for s in sizes if s % 2 == 1) / n
        mean_s = sum(sizes) / n
        from collections import Counter
        dist = Counter(sizes)
        dist_str = "  ".join(f"s{s}:{dist[s]}" for s in sorted(dist))
        print(f"{k:>4}  {n:>10}  {odd_frac:>8.3f}  {mean_s:>10.3f}  {dist_str}")

    print("\n--- Notable structures (Bernoulli LP) ---")
    notable = [r for r in bern["per_structure"]
               if "cycle" in r["label"] or "transitive" in r["label"]]
    for r in notable[:15]:
        print(f"  {r['label']:<40}  support={r['support_size']}  odd={r['support_odd']}")

    print("\n--- Fixed LP: complete tournaments (k=6, 2^6=64) ---")
    sizes = fixed["support_sizes"]
    odd_frac = sum(1 for s in sizes if s % 2 == 1) / len(sizes)
    from collections import Counter
    dist = Counter(sizes)
    print(f"  n={len(sizes)}  lp_odd={odd_frac:.3f}  distribution: {dict(sorted(dist.items()))}")
    print("  (A&L theorem: should be 100% odd for complete antisymmetric tournament)")

    print("\n--- Bernoulli LP: k=6 complete structure (single payoff from win probs) ---")
    # k=6 Bernoulli: assignment has no zeros, same as a directed tournament but using
    # payoff matrix directly.  For Bernoulli with payoff[a,b] ∈ {-1,+1}: same as fixed.
    k6 = [r for r in bern["per_structure"] if r["k"] == 6]
    odd_frac_k6 = sum(1 for r in k6 if r["support_odd"]) / len(k6)
    print(f"  n={len(k6)}  lp_odd={odd_frac_k6:.3f}")

    print("\n--- Key: where does A&L break? ---")
    for k in range(7):
        sizes = bern["by_k"].get(k, [])
        if not sizes:
            continue
        odd_frac = sum(1 for s in sizes if s % 2 == 1) / len(sizes)
        n_structs_with_k_pairs = len(set(
            r["assignment"] for r in bern["per_structure"]
            if r["k"] == k and all(v in (-1, 0, 1) for v in r["assignment"])
        ))
        bar = "█" * int(odd_frac * 20)
        print(f"  k={k}: {odd_frac:.3f}  {bar}")


def main() -> None:
    bern = bernoulli_enumeration()
    fixed = fixed_complete_enumeration()
    print_summary(bern, fixed)


if __name__ == "__main__":
    main()
