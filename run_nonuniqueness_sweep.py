"""Selection-rule robustness sweep for LP parity in incomplete tournaments.

For each replicate, builds the Bernoulli LP (K = p - p^T) from a factor-rank
tournament, applies phi_B (structural zeros) or phi_A (neutral pairs), then
enumerates the full Nash equilibrium family using all_equilibrium_support_sets().

Four selection rules are derived from the same family — no extra LP solves:
  HiGHS   — HiGHS default (vertex of the LP polytope)
  min-sup  — smallest-support member of the family
  max-sup  — largest-support member of the family
  uniform  — mean odd-parity fraction over family members

Design:
  N ∈ {7, 9},  F=5,  seed_base=99 (distinct from sparsity sweep seed_base=42)
  phi_B ∈ {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7}
  phi_A ∈ {0.0, 0.1, 0.25, 0.5}  (comparison axis)
  reps = 500 per condition
"""

from __future__ import annotations

import argparse
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from time import perf_counter

import numpy as np

from tournament.generation import (
    apply_neutrality,
    apply_sparsity,
    factor_win_counts,
    independent_factor_ranks,
    payoff_from_probabilities,
    win_probability_matrix,
)
from tournament.lp import all_equilibrium_support_sets, solve_zero_sum_equilibrium

SUPPORT_TOL = 1e-3


def support_size(x: np.ndarray) -> int:
    return int(np.sum(x > SUPPORT_TOL))


@dataclass(frozen=True)
class Job:
    param_type: str   # "phi_B" or "phi_A"
    param_val: float
    n_species: int
    n_factors: int
    rep: int
    rank_seed: int


def _run_job(job: Job) -> dict:
    rng = np.random.default_rng(job.rank_seed)

    ranks = independent_factor_ranks(job.n_species, job.n_factors, rng)
    p_win = win_probability_matrix(ranks)

    if job.param_type == "phi_B":
        p_mod, interact = apply_sparsity(p_win, job.param_val, rng)
        n_total = job.n_species * (job.n_species - 1) // 2
        n_active = int(interact[np.triu_indices(job.n_species, k=1)].sum())
        actual_phi = 1.0 - n_active / n_total if n_total > 0 else 0.0
    else:
        p_mod, neutral = apply_neutrality(p_win, job.param_val, rng)
        n_total = job.n_species * (job.n_species - 1) // 2
        n_neutral = int(neutral[np.triu_indices(job.n_species, k=1)].sum())
        actual_phi = n_neutral / n_total if n_total > 0 else 0.0

    K = payoff_from_probabilities(p_mod)

    # HiGHS default
    x_highs, _ = solve_zero_sum_equilibrium(K)
    highs_sz = support_size(x_highs)

    # Full equilibrium family (linear algebra, no extra LP objectives)
    t0 = perf_counter()
    family = all_equilibrium_support_sets(K)
    enum_time = perf_counter() - t0

    all_sizes = [len(s) for s, _ in family] if family else [highs_sz]

    return {
        **asdict(job),
        "actual_phi": actual_phi,
        "highs_sz": highs_sz,
        "n_family": len(all_sizes),
        "all_sizes": all_sizes,
        "enum_time_s": enum_time,
    }


def make_jobs(args: argparse.Namespace) -> list[Job]:
    jobs: list[Job] = []
    for param_type, param_vals in [("phi_B", args.phi_b), ("phi_A", args.phi_a)]:
        for param_val in param_vals:
            for n_species in args.species:
                for rep in range(args.reps):
                    seed = (
                        args.seed_base
                        + 1_000_000 * rep
                        + 10_000 * n_species
                        + int(round(param_val * 10_000))
                        + (0 if param_type == "phi_B" else 500_000_000)
                    )
                    jobs.append(Job(
                        param_type=param_type,
                        param_val=param_val,
                        n_species=n_species,
                        n_factors=args.factors,
                        rep=rep,
                        rank_seed=seed,
                    ))
    return jobs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--phi-b", nargs="+", type=float,
                   default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    p.add_argument("--phi-a", nargs="+", type=float,
                   default=[0.0, 0.1, 0.25, 0.5])
    p.add_argument("--species", nargs="+", type=int, default=[7, 9])
    p.add_argument("--factors", type=int, default=5)
    p.add_argument("--reps", type=int, default=500)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed-base", type=int, default=99)
    p.add_argument("--out", default="results/nonuniqueness_sweep_v1.pkl")
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


def apply_quick(args: argparse.Namespace) -> argparse.Namespace:
    if args.quick:
        args.phi_b = [0.0, 0.2, 0.5]
        args.phi_a = [0.0, 0.25]
        args.species = [7]
        args.reps = 10
        args.workers = 2
        if args.out == "results/nonuniqueness_sweep_v1.pkl":
            args.out = "results/nonuniqueness_sweep_quick.pkl"
    return args


def main() -> None:
    args = apply_quick(parse_args())
    jobs = make_jobs(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Jobs: {len(jobs)}  workers: {args.workers}  out: {out}")
    print(f"N={args.species}  F={args.factors}  reps={args.reps}")

    t_start = perf_counter()
    records: list[dict] = []
    done = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run_job, j): j for j in jobs}
        for fut in as_completed(futures):
            records.append(fut.result())
            done += 1
            if done % max(1, len(jobs) // 20) == 0:
                elapsed = perf_counter() - t_start
                eta = elapsed / done * (len(jobs) - done)
                print(f"  {done}/{len(jobs)}  elapsed {elapsed:.0f}s  ETA {eta:.0f}s")

    elapsed = perf_counter() - t_start
    print(f"Done: {len(records)} records in {elapsed:.1f}s")

    with open(out, "wb") as f:
        pickle.dump(records, f)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
