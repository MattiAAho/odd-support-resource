"""Parallel sweeps for standalone tournament experiments."""

from __future__ import annotations

import argparse
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from time import perf_counter

import numpy as np

from tournament.generation import (
    factor_win_counts,
    independent_factor_ranks,
    payoff_from_probabilities,
    payoff_from_winner,
    posterior_mean_probability_matrix,
    sample_beta_probability_matrix,
    sample_fixed_tournament,
    win_probability_matrix,
)
from tournament.kernels import (
    extinction_step,
    simulate_bernoulli_tournament,
    simulate_bernoulli_tournament_counts,
    simulate_factor_resampling_tournament,
    simulate_factor_resampling_tournament_counts,
    simulate_fixed_tournament,
    simulate_fixed_tournament_counts,
)
from tournament.lp import solve_zero_sum_equilibrium


@dataclass(frozen=True)
class TournamentJob:
    models: tuple[str, ...]
    engine: str
    n_species: int
    n_factors: int
    n_individuals: int
    n_steps: int
    sample_every: int
    burn_in: int
    rep: int
    rank_seed: int


def _run_job(job: TournamentJob) -> dict:
    rng = np.random.default_rng(job.rank_seed)
    ranks = independent_factor_ranks(job.n_species, job.n_factors, rng)
    counts = factor_win_counts(ranks)
    p_win = win_probability_matrix(ranks)
    base_record = {
        **asdict(job),
        "factor_counts": counts,
        "p_win": p_win,
    }

    if job.engine == "counts":
        fixed_kernel = simulate_fixed_tournament_counts
        bernoulli_kernel = simulate_bernoulli_tournament_counts
        factor_kernel = simulate_factor_resampling_tournament_counts
    elif job.engine == "population":
        fixed_kernel = simulate_fixed_tournament
        bernoulli_kernel = simulate_bernoulli_tournament
        factor_kernel = simulate_factor_resampling_tournament
    else:
        raise ValueError(f"unknown engine: {job.engine}")

    results = []
    for model_idx, model in enumerate(job.models):
        sim_seed = job.rank_seed + 10_000_000 + 100_000 * model_idx
        t0 = perf_counter()

        lp_exact = None
        lp_ref = None
        lp_value = None
        winner = None

        if model == "fixed":
            winner = sample_fixed_tournament(p_win, rng)
            lp_exact, lp_value = solve_zero_sum_equilibrium(payoff_from_winner(winner))
            times, freqs, avg = fixed_kernel(
                winner,
                job.n_individuals,
                job.n_steps,
                job.sample_every,
                sim_seed,
                job.burn_in,
            )
        elif model == "bernoulli":
            lp_ref, lp_value = solve_zero_sum_equilibrium(payoff_from_probabilities(p_win))
            times, freqs, avg = bernoulli_kernel(
                p_win,
                job.n_individuals,
                job.n_steps,
                job.sample_every,
                sim_seed,
                job.burn_in,
            )
        elif model == "factor":
            lp_ref, lp_value = solve_zero_sum_equilibrium(payoff_from_probabilities(p_win))
            times, freqs, avg = factor_kernel(
                ranks,
                job.n_individuals,
                job.n_steps,
                job.sample_every,
                sim_seed,
                job.burn_in,
            )
        elif model == "beta_mean":
            p_beta = posterior_mean_probability_matrix(counts, job.n_factors)
            lp_ref, lp_value = solve_zero_sum_equilibrium(payoff_from_probabilities(p_beta))
            times, freqs, avg = bernoulli_kernel(
                p_beta,
                job.n_individuals,
                job.n_steps,
                job.sample_every,
                sim_seed,
                job.burn_in,
            )
        elif model == "beta_sample":
            p_beta = sample_beta_probability_matrix(counts, job.n_factors, rng)
            lp_ref, lp_value = solve_zero_sum_equilibrium(payoff_from_probabilities(p_beta))
            times, freqs, avg = bernoulli_kernel(
                p_beta,
                job.n_individuals,
                job.n_steps,
                job.sample_every,
                sim_seed,
                job.burn_in,
            )
        else:
            raise ValueError(f"unknown model: {model}")

        extinct_idx = extinction_step(freqs)
        extinct_time = None if extinct_idx < 0 else int(times[extinct_idx])
        elapsed = perf_counter() - t0

        results.append(
            {
                **base_record,
                "model": model,
                "sim_seed": sim_seed,
                "lp_exact": lp_exact,
                "lp_ref": lp_ref,
                "lp_value": lp_value,
                "time_avg": avg,
                "final_freq": freqs[-1],
                "extinct_time": extinct_time,
                "runtime_s": elapsed,
            }
        )

    return results


def make_jobs(args: argparse.Namespace) -> list[TournamentJob]:
    jobs: list[TournamentJob] = []
    for n_species in args.species:
        for n_factors in args.factors:
            for rep in range(args.reps):
                rank_seed = args.seed_base + 1_000_000 * rep + 10_000 * n_species + 100 * n_factors
                jobs.append(
                    TournamentJob(
                        engine=args.engine,
                        n_species=n_species,
                        n_factors=n_factors,
                        n_individuals=args.n_individuals,
                        n_steps=args.n_steps,
                        sample_every=args.sample_every,
                        burn_in=args.burn_in,
                        rep=rep,
                        rank_seed=rank_seed,
                        models=tuple(args.models),
                    )
                )
    return jobs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["fixed", "bernoulli", "beta_sample"])
    parser.add_argument("--species", nargs="+", type=int, default=[7, 9, 11])
    parser.add_argument("--factors", nargs="+", type=int, default=[3, 5, 7])
    parser.add_argument("--reps", type=int, default=26)
    parser.add_argument("--n-individuals", type=int, default=25_000)
    parser.add_argument("--n-steps", type=int, default=10_000_000)
    parser.add_argument("--sample-every", type=int, default=10_000)
    parser.add_argument("--burn-in", type=int, default=1_000_000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument("--engine", choices=["counts", "population"], default="counts")
    parser.add_argument("--out", default="results/tournament_sweep_v1.pkl")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def apply_quick_defaults(args: argparse.Namespace) -> argparse.Namespace:
    if args.quick:
        user_engine = args.engine
        user_out = args.out
        args.models = ["fixed", "bernoulli", "beta_mean", "beta_sample"]
        args.species = [5]
        args.factors = [3]
        args.reps = 2
        args.n_individuals = 2_000
        args.n_steps = 20_000
        args.sample_every = 1_000
        args.burn_in = 5_000
        args.workers = 2
        args.engine = user_engine
        args.out = user_out if user_out != "results/tournament_sweep_v1.pkl" else "results/tournament_sweep_quick.pkl"
    return args


def main() -> None:
    args = apply_quick_defaults(parse_args())
    jobs = make_jobs(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_rows = len(jobs) * len(args.models)
    print(
        f"Running {len(jobs)} paired tournament replicates "
        f"({n_rows} model rows) with {args.workers} workers using {args.engine} engine"
    )

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(_run_job, job) for job in jobs]
        for i, fut in enumerate(as_completed(futures), start=1):
            batch = fut.result()
            results.extend(batch)
            meta = batch[0]
            runtime_total = sum(r["runtime_s"] for r in batch)
            print(
                f"[{i:>4}/{len(jobs)}] paired rep S={meta['n_species']} F={meta['n_factors']} "
                f"rep={meta['rep']} rows={len(batch)} runtime={runtime_total:.2f}s"
            )

    with out.open("wb") as fh:
        pickle.dump(results, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved {out} ({len(results)} records)")


if __name__ == "__main__":
    main()
