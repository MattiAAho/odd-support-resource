"""Neutral-pairs sweep (φ_A): selection-drift crossover in A&L tournaments.

φ_A is the probability that any unordered species pair is made neutral
(p_ab = p_ba = 0.5).  At φ_A=0 the tournament is the standard factor-rank
model (selection dominant); at φ_A=1 every encounter is a coin flip (drift).

Primary questions:
  1. How does extinction timing shift as φ_A increases?
  2. How does final support size (surviving species) change with φ_A?
  3. At what φ_A does drift start dominating selection?

LP notes:
  - Fixed model LP: computed from the sampled winner matrix (always a
    complete antisymmetric tournament, always odd support by A&L theorem).
  - Bernoulli LP: computed from p_neutral (payoff=0 for neutral pairs).
    At φ_A=1 the payoff is all-zero → LP is degenerate (corner solution,
    support=1 artefact); flagged by lp_degenerate=True.
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
    factor_win_counts,
    independent_factor_ranks,
    payoff_from_probabilities,
    payoff_from_winner,
    sample_fixed_tournament,
    win_probability_matrix,
)
from tournament.kernels import (
    extinction_step,
    simulate_bernoulli_tournament_counts,
    simulate_fixed_tournament_counts,
)
from tournament.lp import solve_zero_sum_equilibrium


def support_size(x: np.ndarray, tol: float = 1e-3) -> int:
    return int(np.sum(x > tol))


@dataclass(frozen=True)
class PhiAJob:
    phi_A: float
    n_species: int
    n_factors: int
    n_individuals: int
    n_steps: int
    sample_every: int
    burn_in: int
    rep: int
    rank_seed: int


def _run_job(job: PhiAJob) -> list[dict]:
    rng = np.random.default_rng(job.rank_seed)

    ranks = independent_factor_ranks(job.n_species, job.n_factors, rng)
    p_win = win_probability_matrix(ranks)

    # Both models share the same neutrality mask (paired replicate design).
    p_neutral, neutral_mask = apply_neutrality(p_win, job.phi_A, rng)
    n_total_pairs = job.n_species * (job.n_species - 1) // 2
    n_neutral_pairs = int(neutral_mask[np.triu_indices(job.n_species, k=1)].sum())
    actual_phi_A = n_neutral_pairs / n_total_pairs if n_total_pairs > 0 else 0.0

    base = {
        **asdict(job),
        "n_neutral_pairs": n_neutral_pairs,
        "actual_phi_A": actual_phi_A,
    }

    results = []

    # ── Fixed model ──────────────────────────────────────────────────────────
    sim_seed_fixed = job.rank_seed + 10_000_000
    t0 = perf_counter()
    winner = sample_fixed_tournament(p_neutral, rng)
    payoff_fixed = payoff_from_winner(winner)
    lp_x_fixed, _ = solve_zero_sum_equilibrium(payoff_fixed)
    _, freqs_f, avg_f = simulate_fixed_tournament_counts(
        winner, job.n_individuals, job.n_steps,
        job.sample_every, sim_seed_fixed, job.burn_in,
    )
    extinct_idx_f = extinction_step(freqs_f)
    lp_sz_f = support_size(lp_x_fixed)
    results.append({
        **base,
        "model": "fixed",
        "sim_seed": sim_seed_fixed,
        "lp_x": lp_x_fixed,
        "lp_support_size": lp_sz_f,
        "lp_support_odd": lp_sz_f % 2 == 1,
        "lp_degenerate": False,
        "final_support_size": support_size(freqs_f[-1]),
        "final_support_odd": support_size(freqs_f[-1]) % 2 == 1,
        "avg_support_size": support_size(avg_f),
        "final_freq": freqs_f[-1],
        "extinct_time": (
            None if extinct_idx_f < 0 else extinct_idx_f * job.sample_every
        ),
        "runtime_s": perf_counter() - t0,
    })

    # ── Bernoulli model ──────────────────────────────────────────────────────
    sim_seed_bern = job.rank_seed + 10_000_000 + 100_000
    t0 = perf_counter()
    payoff_bern = payoff_from_probabilities(p_neutral)
    lp_degenerate = bool(np.allclose(payoff_bern, 0.0, atol=1e-10))
    lp_x_bern, _ = solve_zero_sum_equilibrium(payoff_bern)
    _, freqs_b, avg_b = simulate_bernoulli_tournament_counts(
        p_neutral, job.n_individuals, job.n_steps,
        job.sample_every, sim_seed_bern, job.burn_in,
    )
    extinct_idx_b = extinction_step(freqs_b)
    lp_sz_b = support_size(lp_x_bern)
    results.append({
        **base,
        "model": "bernoulli",
        "sim_seed": sim_seed_bern,
        "lp_x": lp_x_bern,
        "lp_support_size": lp_sz_b,
        "lp_support_odd": lp_sz_b % 2 == 1,
        "lp_degenerate": lp_degenerate,
        "final_support_size": support_size(freqs_b[-1]),
        "final_support_odd": support_size(freqs_b[-1]) % 2 == 1,
        "avg_support_size": support_size(avg_b),
        "final_freq": freqs_b[-1],
        "extinct_time": (
            None if extinct_idx_b < 0 else extinct_idx_b * job.sample_every
        ),
        "runtime_s": perf_counter() - t0,
    })

    return results


def make_jobs(args: argparse.Namespace) -> list[PhiAJob]:
    jobs: list[PhiAJob] = []
    for phi_A in args.phi_a:
        for n_species in args.species:
            for n_factors in args.factors:
                for rep in range(args.reps):
                    seed = (
                        args.seed_base
                        + 1_000_000 * rep
                        + 10_000 * n_species
                        + 100 * n_factors
                        + int(round(phi_A * 1000))
                    )
                    jobs.append(PhiAJob(
                        phi_A=phi_A,
                        n_species=n_species,
                        n_factors=n_factors,
                        n_individuals=args.n_individuals,
                        n_steps=args.n_steps,
                        sample_every=args.sample_every,
                        burn_in=args.burn_in,
                        rep=rep,
                        rank_seed=seed,
                    ))
    return jobs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--phi-a", nargs="+", type=float,
                   default=[0.0, 0.1, 0.25, 0.5, 0.75, 1.0])
    p.add_argument("--species", nargs="+", type=int, default=[7, 9])
    p.add_argument("--factors", nargs="+", type=int, default=[3, 5, 7])
    p.add_argument("--reps", type=int, default=200)
    p.add_argument("--n-individuals", type=int, default=1000)
    p.add_argument("--n-steps", type=int, default=5_000_000)
    p.add_argument("--sample-every", type=int, default=5_000)
    p.add_argument("--burn-in", type=int, default=0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed-base", type=int, default=42)
    p.add_argument("--out", default="results/phi_a_sweep_v1.pkl")
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


def apply_quick_defaults(args: argparse.Namespace) -> argparse.Namespace:
    if args.quick:
        args.phi_a = [0.0, 0.5, 1.0]
        args.species = [7]
        args.factors = [5]
        args.reps = 4
        args.n_individuals = 200
        args.n_steps = 200_000
        args.sample_every = 2_000
        args.burn_in = 0
        args.workers = 2
        if args.out == "results/phi_a_sweep_v1.pkl":
            args.out = "results/phi_a_sweep_quick.pkl"
    return args


def main() -> None:
    args = apply_quick_defaults(parse_args())
    jobs = make_jobs(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_rows = len(jobs) * 2  # fixed + bernoulli
    print(
        f"Phi-A sweep: {len(jobs)} paired jobs ({n_rows} model rows), "
        f"phi_A={args.phi_a}, S={args.species}, F={args.factors}, "
        f"reps={args.reps}, N_ind={args.n_individuals}, "
        f"n_steps={args.n_steps:,}, workers={args.workers}"
    )

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(_run_job, job) for job in jobs]
        for i, fut in enumerate(as_completed(futures), start=1):
            batch = fut.result()
            results.extend(batch)
            m = batch[0]
            rt = sum(r["runtime_s"] for r in batch)
            extinct_frac = sum(
                1 for r in batch if r["extinct_time"] is not None
            ) / len(batch)
            print(
                f"[{i:>5}/{len(jobs)}] phi_A={m['phi_A']:.3f} "
                f"S={m['n_species']} F={m['n_factors']} rep={m['rep']} "
                f"neutral_pairs={m['n_neutral_pairs']} "
                f"extinct={extinct_frac:.0%} runtime={rt:.2f}s"
            )

    with out.open("wb") as fh:
        pickle.dump(results, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved {out} ({len(results)} records)")


if __name__ == "__main__":
    main()
