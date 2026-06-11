"""Track B: Pfaffian / perfect-matching test of the sparsity-trough mechanism.

Conjecture (manuscript-2 open thread): the odd-support trough under incompleteness
arises because structural zeros in the skew-symmetric payoff K destroy perfect
matchings in the present-pair graph, which forces the Pfaffian of even-order
support blocks to vanish, making even-support equilibria admissible.

Mechanism, stated as a per-support equivalence chain (even support S):
    even-support equilibrium admissible
        => K_S singular                       (necessary for an interior eq on S)
        <=> Pf(K_S) = 0                        (even-order skew block: det = Pf^2)
    and, structurally (Tutte / Pfaffian-as-signed-matching-sum):
        present-graph on S has no perfect matching  =>  Pf(K_S) == 0  identically.

This script regenerates the exact `nonuniqueness_sweep_v1` Bernoulli tournaments
(same seed scheme) and tests:

  Test 1 (necessity + structural split): for every even support that HOSTS an
          equilibrium, is K_S singular (Pf=0)?  Is the singularity structural
          (no perfect matching) or accidental (matching exists, Pf=0 anyway)?
  Test 2 (sufficiency gap): among ALL singular even subsets (Pf=0), what fraction
          actually host a valid equilibrium?  (necessary != sufficient)
  Test 3 (trough): does a parameter-free matching statistic track the family-level
          even-support fraction across phi_B, in the connected regime?
  Unification: phi_A vs phi_B agreement at matched zero-pair density.

Cross-validation: regenerated family support sizes are checked against the stored
`results/nonuniqueness_sweep_v1.pkl` records (identical seeds => identical families).
"""

from __future__ import annotations

import argparse
import pickle
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path
from time import perf_counter

import numpy as np
import networkx as nx

from tournament.generation import (
    apply_neutrality,
    apply_sparsity,
    independent_factor_ranks,
    payoff_from_probabilities,
    win_probability_matrix,
)
from tournament.lp import all_equilibrium_support_sets

SING_TOL = 1e-8          # smallest singular value below this => K_S singular
SUPPORT_TOL = 1e-3       # support-size threshold (matches the sweep convention)


# ── Pfaffian (explicit, for the det = Pf^2 cross-check) ─────────────────────

def pfaffian(A: np.ndarray) -> float:
    """Recursive Pfaffian of an even-order skew-symmetric matrix (small n)."""
    n = A.shape[0]
    if n == 0:
        return 1.0
    if n % 2 == 1:
        return 0.0
    # expand along the first row
    total = 0.0
    rest = list(range(1, n))
    for k, j in enumerate(rest):
        if A[0, j] == 0.0:
            continue
        minor_idx = [r for r in rest if r != j]
        sub = A[np.ix_(minor_idx, minor_idx)]
        total += ((-1) ** k) * A[0, j] * pfaffian(sub)
    return total


def is_singular(K_S: np.ndarray) -> bool:
    if K_S.shape[0] == 0:
        return False
    sv = np.linalg.svd(K_S, compute_uv=False)
    return sv[-1] < SING_TOL


def has_perfect_matching(nodes: list[int], interact: np.ndarray) -> bool:
    """True iff the present-pair graph induced on `nodes` has a perfect matching."""
    if len(nodes) % 2 == 1:
        return False
    G = nx.Graph()
    G.add_nodes_from(nodes)
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if interact[nodes[i], nodes[j]]:
                G.add_edge(nodes[i], nodes[j])
    m = nx.max_weight_matching(G, maxcardinality=True)
    return 2 * len(m) == len(nodes)


# ── per-tournament analysis ─────────────────────────────────────────────────

def analyse_tournament(K: np.ndarray, interact: np.ndarray) -> dict:
    n = K.shape[0]

    # present-pair graph + connectivity (on all n species)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if interact[i, j]:
                G.add_edge(i, j)
    connected = nx.is_connected(G) if n > 0 else True
    n_components = nx.number_connected_components(G)
    match_number = len(nx.max_weight_matching(G, maxcardinality=True))
    deficiency = n - 2 * match_number          # unmatched vertices under a max matching

    # equilibrium family
    family = all_equilibrium_support_sets(K)
    fam_sizes = sorted(len(s) for s, _ in family)
    fam_supports = [frozenset(s) for s, _ in family]
    eq_supports = set(fam_supports)
    has_even_eq = any(sz % 2 == 0 for sz in fam_sizes)
    fam_odd_frac = (np.mean([sz % 2 == 1 for sz in fam_sizes]) if fam_sizes else 1.0)

    # ── per even-support classification over ALL even subsets ──
    n_even = 0
    n_sing = 0                 # singular (Pf=0)
    n_sing_structural = 0      # singular AND no perfect matching
    n_sing_accidental = 0      # singular BUT matching exists (Pf vanishes on values)
    n_sing_hosts_eq = 0        # singular AND hosts a family equilibrium (sufficiency)
    pf_det_max_err = 0.0       # max |det - Pf^2| sanity over even subsets

    # eq-hosting even supports, classified
    n_eveneq = 0
    n_eveneq_singular = 0
    n_eveneq_structural = 0
    n_eveneq_accidental = 0

    for size in range(2, n + 1, 2):
        for S in combinations(range(n), size):
            S = list(S)
            K_S = K[np.ix_(S, S)]
            n_even += 1
            sing = is_singular(K_S)
            pm = has_perfect_matching(S, interact)
            hosts = frozenset(S) in eq_supports

            # det = Pf^2 sanity (cheap; sizes <= 8)
            pf = pfaffian(K_S)
            det = np.linalg.det(K_S)
            pf_det_max_err = max(pf_det_max_err, abs(det - pf * pf))

            if sing:
                n_sing += 1
                if pm:
                    n_sing_accidental += 1
                else:
                    n_sing_structural += 1
                if hosts:
                    n_sing_hosts_eq += 1

            if hosts:
                n_eveneq += 1
                if sing:
                    n_eveneq_singular += 1
                if not pm:
                    n_eveneq_structural += 1
                else:
                    n_eveneq_accidental += 1

    return {
        "fam_sizes": fam_sizes,
        "has_even_eq": has_even_eq,
        "fam_odd_frac": float(fam_odd_frac),
        "highs_in_family": fam_sizes,           # placeholder (family-level only)
        "connected": bool(connected),
        "n_components": int(n_components),
        "match_number": int(match_number),
        "deficiency": int(deficiency),
        # all-even-subset tallies
        "n_even": n_even,
        "n_sing": n_sing,
        "n_sing_structural": n_sing_structural,
        "n_sing_accidental": n_sing_accidental,
        "n_sing_hosts_eq": n_sing_hosts_eq,
        # eq-hosting even-support tallies
        "n_eveneq": n_eveneq,
        "n_eveneq_singular": n_eveneq_singular,
        "n_eveneq_structural": n_eveneq_structural,
        "n_eveneq_accidental": n_eveneq_accidental,
        "pf_det_max_err": float(pf_det_max_err),
    }


# ── sweep driver (mirrors run_nonuniqueness_sweep.make_jobs seeds) ───────────

def seed_for(seed_base, rep, n_species, param_type, param_val) -> int:
    return (
        seed_base
        + 1_000_000 * rep
        + 10_000 * n_species
        + int(round(param_val * 10_000))
        + (0 if param_type == "phi_B" else 500_000_000)
    )


def _worker(job: tuple) -> dict:
    param_type, param_val, N, rep, seed, factors = job
    rng = np.random.default_rng(seed)
    ranks = independent_factor_ranks(N, factors, rng)
    p = win_probability_matrix(ranks)
    if param_type == "phi_B":
        p_mod, mask = apply_sparsity(p, param_val, rng)
        interact = mask
    else:
        p_mod, neutral = apply_neutrality(p, param_val, rng)
        # neutral pairs are zeros in K too -> not "present" for matching
        interact = np.ones((N, N), dtype=bool)
        np.fill_diagonal(interact, False)
        interact[neutral] = False
    K = payoff_from_probabilities(p_mod)
    n_total = N * (N - 1) // 2
    n_zero = int((~interact[np.triu_indices(N, 1)]).sum())
    rec = {
        "param_type": param_type,
        "param_val": param_val,
        "N": N,
        "rep": rep,
        "seed": seed,
        "n_zero_pairs": n_zero,
        "actual_phi": n_zero / n_total,
    }
    rec.update(analyse_tournament(K, interact))
    return rec


def run(args) -> list[dict]:
    grids = [("phi_B", args.phi_b), ("phi_A", args.phi_a)]
    jobs = []
    for param_type, vals in grids:
        for param_val in vals:
            for N in args.species:
                for rep in range(args.reps):
                    seed = seed_for(args.seed_base, rep, N, param_type, param_val)
                    jobs.append((param_type, param_val, N, rep, seed, args.factors))
    total = len(jobs)
    print(f"Jobs: {total}  workers: {args.workers}")
    records: list[dict] = []
    t0 = perf_counter()
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_worker, j) for j in jobs]
        for fut in as_completed(futures):
            records.append(fut.result())
            done += 1
            if done % max(1, total // 20) == 0:
                el = perf_counter() - t0
                print(f"  {done}/{total}  {el:.0f}s  ETA {el/done*(total-done):.0f}s",
                      flush=True)
    print(f"Done {len(records)} tournaments in {perf_counter()-t0:.1f}s")
    return records


# ── reporting ───────────────────────────────────────────────────────────────

def report(records: list[dict]) -> None:
    phi_b = [r for r in records if r["param_type"] == "phi_B"]

    print("\n" + "=" * 74)
    print("TEST 1  —  even-support equilibria: is the singularity structural?")
    print("=" * 74)
    n_eq = sum(r["n_eveneq"] for r in phi_b)
    n_eq_sing = sum(r["n_eveneq_singular"] for r in phi_b)
    n_eq_struct = sum(r["n_eveneq_structural"] for r in phi_b)
    n_eq_acc = sum(r["n_eveneq_accidental"] for r in phi_b)
    if n_eq:
        print(f"  even supports hosting an equilibrium : {n_eq}")
        print(f"  ... with K_S singular  (Pf=0)        : {n_eq_sing}  ({n_eq_sing/n_eq:.4f})")
        print(f"  ... structural (no perfect matching) : {n_eq_struct}  ({n_eq_struct/n_eq:.4f})")
        print(f"  ... accidental (matching exists)     : {n_eq_acc}  ({n_eq_acc/n_eq:.4f})")
    pf_err = max(r["pf_det_max_err"] for r in records)
    print(f"  max |det(K_S) - Pf(K_S)^2| over all even subsets : {pf_err:.2e}")

    print("\n" + "=" * 74)
    print("TEST 2  —  sufficiency gap: singular even subsets that host an eq")
    print("=" * 74)
    n_sing = sum(r["n_sing"] for r in phi_b)
    n_sing_struct = sum(r["n_sing_structural"] for r in phi_b)
    n_sing_acc = sum(r["n_sing_accidental"] for r in phi_b)
    n_sing_hosts = sum(r["n_sing_hosts_eq"] for r in phi_b)
    if n_sing:
        print(f"  singular even subsets (Pf=0)         : {n_sing}")
        print(f"  ... structural (no perfect matching) : {n_sing_struct}  ({n_sing_struct/n_sing:.4f})")
        print(f"  ... accidental (matching exists)     : {n_sing_acc}  ({n_sing_acc/n_sing:.4f})")
        print(f"  ... that host a family equilibrium   : {n_sing_hosts}  ({n_sing_hosts/n_sing:.4f})")
        print("  (gap = singular but no eq: needs nonneg kernel + invasion)")

    print("\n" + "=" * 74)
    print("TEST 3  —  family-level even-admissibility: structural supply vs realized")
    print("=" * 74)
    print("  evenEq = frac of tournaments whose family contains an even support")
    print("  supply = mean # structural (no-PM) singular even subsets per tournament")
    print("  realiz = mean # even supports actually hosting an equilibrium")
    print(f"  {'N':>2} {'phi_B':>6} {'reps':>5} {'evenEq':>7} {'conn':>5} "
          f"{'supply':>8} {'realiz':>8} {'realiz/supply':>13}")
    by = defaultdict(list)
    for r in phi_b:
        by[(r["N"], r["param_val"])].append(r)
    for (N, pv) in sorted(by):
        rows = by[(N, pv)]
        conn = [r for r in rows if r["connected"]]
        even_all = np.mean([r["has_even_eq"] for r in rows])
        supply = np.mean([r["n_sing_structural"] for r in rows])
        realiz = np.mean([r["n_eveneq"] for r in rows])
        rr = realiz / supply if supply > 0 else float("nan")
        print(f"  {N:>2} {pv:>6.2f} {len(rows):>5} {even_all:>7.3f} "
              f"{len(conn)/len(rows):>5.2f} {supply:>8.3f} {realiz:>8.3f} {rr:>13.3f}")

    print("\n" + "=" * 74)
    print("UNIFICATION  —  phi_A vs phi_B even-eq fraction at matched zero-pair density")
    print("=" * 74)
    for N in sorted({r["N"] for r in records}):
        print(f"  N={N}:")
        print(f"    {'E[zero]':>8} {'phi_B evenEq':>13} {'phi_A evenEq':>13}")
        # bin by realized zero-pair count
        for ptype in ("phi_B", "phi_A"):
            pass
        buckets = defaultdict(lambda: {"phi_B": [], "phi_A": []})
        for r in records:
            if r["N"] != N:
                continue
            buckets[r["n_zero_pairs"]][r["param_type"]].append(r["has_even_eq"])
        for z in sorted(buckets):
            b = buckets[z]
            if b["phi_B"] and b["phi_A"]:
                fb = np.mean(b["phi_B"])
                fa = np.mean(b["phi_A"])
                print(f"    {z:>8} {fb:>13.3f} {fa:>13.3f}")


def cross_validate(records: list[dict], stored_path: str) -> None:
    p = Path(stored_path)
    if not p.exists():
        print(f"\n[cross-validate] {stored_path} not found; skipping.")
        return
    with open(p, "rb") as f:
        stored = pickle.load(f)
    # index stored by (param_type, param_val, N, rep)
    idx = {}
    for s in stored:
        key = (s["param_type"], round(s["param_val"], 6), s["n_species"], s["rep"])
        idx[key] = sorted(s["all_sizes"])
    matched = mism = missing = 0
    for r in records:
        key = (r["param_type"], round(r["param_val"], 6), r["N"], r["rep"])
        if key not in idx:
            missing += 1
            continue
        if idx[key] == r["fam_sizes"]:
            matched += 1
        else:
            mism += 1
    print("\n" + "=" * 74)
    print("CROSS-VALIDATION vs results/nonuniqueness_sweep_v1.pkl")
    print("=" * 74)
    print(f"  matched family sizes : {matched}")
    print(f"  mismatched           : {mism}")
    print(f"  not in stored set    : {missing}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phi-b", nargs="+", type=float,
                    default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    ap.add_argument("--phi-a", nargs="+", type=float, default=[0.0, 0.1, 0.25, 0.5])
    ap.add_argument("--species", nargs="+", type=int, default=[7, 9])
    ap.add_argument("--factors", type=int, default=5)
    ap.add_argument("--reps", type=int, default=500)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed-base", type=int, default=99)
    ap.add_argument("--out", default="results/pfaffian_matching_v1.pkl")
    ap.add_argument("--quick", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.species = [7]
        args.phi_b = [0.0, 0.2, 0.3, 0.5]
        args.phi_a = [0.0, 0.25]
        args.reps = 50
        args.out = "results/pfaffian_matching_quick.pkl"
    print(f"N={args.species} F={args.factors} reps={args.reps} "
          f"phi_B={args.phi_b} phi_A={args.phi_a}")
    records = run(args)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(records, f)
    print(f"Saved {args.out}")
    report(records)
    cross_validate(records, "results/nonuniqueness_sweep_v1.pkl")


if __name__ == "__main__":
    main()
