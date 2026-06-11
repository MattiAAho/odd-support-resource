"""Sparsity sweep: how does odd-support fraction change as phi_B rises?

phi_B is the probability that any given unordered species pair is declared
non-interacting (structural zero in the payoff matrix).  At phi_B=0 the
tournament is complete antisymmetric; at phi_B=1 no species interact.

The sweep uses the same paired-replicate design as run_tournament_sweep.py:
a shared rank_seed produces one factor-rank draw, and all three models
(fixed, bernoulli, beta_sample) share the same underlying tournament structure
and the same sparsity mask.
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
    apply_sparsity,
    factor_win_counts,
    independent_factor_ranks,
    payoff_from_probabilities,
    payoff_from_winner,
    sample_beta_probability_matrix,
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
class SparsityJob:
    phi_B: float
    n_species: int
    n_factors: int
    n_individuals: int
    n_steps: int
    sample_every: int
    burn_in: int
    rep: int
    rank_seed: int


def _run_job(job: SparsityJob) -> list[dict]:
    rng = np.random.default_rng(job.rank_seed)

    ranks = independent_factor_ranks(job.n_species, job.n_factors, rng)
    counts = factor_win_counts(ranks)
    p_win = win_probability_matrix(ranks)

    # Apply the same sparsity mask to all models in this paired replicate.
    p_sparse, interact = apply_sparsity(p_win, job.phi_B, rng)
    n_total_pairs = job.n_species * (job.n_species - 1) // 2
    n_active_pairs = int(interact[np.triu_indices(job.n_species, k=1)].sum())
    actual_phi = 1.0 - n_active_pairs / n_total_pairs if n_total_pairs > 0 else 0.0

    base = {
        **asdict(job),
        "n_active_pairs": n_active_pairs,
        "actual_phi": actual_phi,
    }

    results = []
    for model_idx, model in enumerate(["fixed", "bernoulli", "beta_sample"]):
        sim_seed = job.rank_seed + 10_000_000 + 100_000 * model_idx
        t0 = perf_counter()

        lp_x = None

        if model == "fixed":
            winner = sample_fixed_tournament(p_sparse, rng)
            payoff = payoff_from_winner(winner)
            lp_x, _ = solve_zero_sum_equilibrium(payoff)
            _, freqs, avg = simulate_fixed_tournament_counts(
                winner, job.n_individuals, job.n_steps,
                job.sample_every, sim_seed, job.burn_in,
            )

        elif model == "bernoulli":
            payoff = payoff_from_probabilities(p_sparse)
            lp_x, _ = solve_zero_sum_equilibrium(payoff)
            _, freqs, avg = simulate_bernoulli_tournament_counts(
                p_sparse, job.n_individuals, job.n_steps,
                job.sample_every, sim_seed, job.burn_in,
            )

        elif model == "beta_sample":
            p_beta = sample_beta_probability_matrix(counts, job.n_factors, rng)
            # Apply the same structural zeros as the other models.
            off_diag_non_interact = ~interact.copy()
            np.fill_diagonal(off_diag_non_interact, False)
            p_beta[off_diag_non_interact] = 0.0
            payoff = payoff_from_probabilities(p_beta)
            lp_x, _ = solve_zero_sum_equilibrium(payoff)
            _, freqs, avg = simulate_bernoulli_tournament_counts(
                p_beta, job.n_individuals, job.n_steps,
                job.sample_every, sim_seed, job.burn_in,
            )

        lp_size = support_size(lp_x)
        final_size = support_size(freqs[-1])
        avg_size = support_size(avg)
        extinct_idx = extinction_step(freqs)

        results.append({
            **base,
            "model": model,
            "sim_seed": sim_seed,
            "lp_x": lp_x,
            "lp_support_size": lp_size,
            "lp_support_odd": lp_size % 2 == 1,
            "final_support_size": final_size,
            "final_support_odd": final_size % 2 == 1,
            "avg_support_size": avg_size,
            "avg_support_odd": avg_size % 2 == 1,
            "final_freq": freqs[-1],
            "time_avg": avg,
            "extinct_time": None if extinct_idx < 0 else extinct_idx,
            "runtime_s": perf_counter() - t0,
        })

    return results


def make_jobs(args: argparse.Namespace) -> list[SparsityJob]:
    jobs: list[SparsityJob] = []
    for phi_B in args.phi_b:
        for n_species in args.species:
            for n_factors in args.factors:
                for rep in range(args.reps):
                    seed = (
                        args.seed_base
                        + 1_000_000 * rep
                        + 10_000 * n_species
                        + 100 * n_factors
                        + int(round(phi_B * 1000))
                    )
                    jobs.append(SparsityJob(
                        phi_B=phi_B,
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
    p.add_argument("--phi-b", nargs="+", type=float,
                   default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    p.add_argument("--species", nargs="+", type=int, default=[7, 9])
    p.add_argument("--factors", nargs="+", type=int, default=[3, 5, 7])
    p.add_argument("--reps", type=int, default=200)
    p.add_argument("--n-individuals", type=int, default=25_000)
    p.add_argument("--n-steps", type=int, default=10_000_000)
    p.add_argument("--sample-every", type=int, default=10_000)
    p.add_argument("--burn-in", type=int, default=1_000_000)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed-base", type=int, default=42)
    p.add_argument("--out", default="results/sparsity_sweep_v1.pkl")
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


def apply_quick_defaults(args: argparse.Namespace) -> argparse.Namespace:
    if args.quick:
        args.phi_b = [0.0, 0.3, 0.7]
        args.species = [7]
        args.factors = [5]
        args.reps = 4
        args.n_individuals = 2_000
        args.n_steps = 50_000
        args.sample_every = 1_000
        args.burn_in = 5_000
        args.workers = 2
        if args.out == "results/sparsity_sweep_v1.pkl":
            args.out = "results/sparsity_sweep_quick.pkl"
    return args


def main() -> None:
    args = apply_quick_defaults(parse_args())
    jobs = make_jobs(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_rows = len(jobs) * 3  # fixed + bernoulli + beta_sample
    print(
        f"Sparsity sweep: {len(jobs)} paired jobs ({n_rows} model rows), "
        f"phi_B={args.phi_b}, S={args.species}, F={args.factors}, "
        f"reps={args.reps}, workers={args.workers}"
    )

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(_run_job, job) for job in jobs]
        for i, fut in enumerate(as_completed(futures), start=1):
            batch = fut.result()
            results.extend(batch)
            m = batch[0]
            rt = sum(r["runtime_s"] for r in batch)
            print(
                f"[{i:>5}/{len(jobs)}] phi_B={m['phi_B']:.2f} "
                f"S={m['n_species']} F={m['n_factors']} rep={m['rep']} "
                f"active_pairs={m['n_active_pairs']} runtime={rt:.2f}s"
            )

    with out.open("wb") as fh:
        pickle.dump(results, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved {out} ({len(results)} records)")


if __name__ == "__main__":
    main()
