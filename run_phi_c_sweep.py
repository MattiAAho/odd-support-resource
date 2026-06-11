"""φ_C sweep: break complementarity for a fraction of pairs (Option γ).

For each affected pair {a, b}, p_ab and p_ba are drawn independently from
Uniform(0, 1) rather than constrained to sum to 1.  The LP K matrix
(K[a,b] = p_ab − p_ba) remains skew-symmetric regardless, so LP parity should
stay odd at all φ_C — this is a sanity check and not the interesting result.

The interesting result is in the simulation: encounters may fizzle (sum < 1)
or be renormalised (sum > 1), changing selection strength and coexistence.

Paired replicate design: each replicate uses one shared rank_seed to draw
the factor-rank structure.  The same underlying p_win is used for:
  bernoulli — standard complementarity baseline (phi_C=0 throughout)
  gamma     — phi_C fraction of pairs have independent p_ab, p_ba
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
    apply_complementarity_break,
    independent_factor_ranks,
    payoff_from_probabilities,
    win_probability_matrix,
)
from tournament.kernels import (
    extinction_step,
    simulate_bernoulli_tournament_counts,
    simulate_gamma_tournament_counts,
)
from tournament.lp import solve_zero_sum_equilibrium


def support_size(x: np.ndarray, tol: float = 1e-3) -> int:
    return int(np.sum(x > tol))


@dataclass(frozen=True)
class PhiCJob:
    phi_C: float
    n_species: int
    n_factors: int
    n_individuals: int
    n_steps: int
    sample_every: int
    burn_in: int
    rep: int
    rank_seed: int


def _run_job(job: PhiCJob) -> list[dict]:
    rng = np.random.default_rng(job.rank_seed)

    ranks = independent_factor_ranks(job.n_species, job.n_factors, rng)
    p_win = win_probability_matrix(ranks)

    p_gamma, gamma_mask = apply_complementarity_break(p_win, job.phi_C, rng)

    n_total_pairs = job.n_species * (job.n_species - 1) // 2
    idx = np.triu_indices(job.n_species, k=1)
    n_gamma_pairs = int(gamma_mask[idx].sum())

    pair_sums = []
    for a in range(job.n_species):
        for b in range(a + 1, job.n_species):
            if gamma_mask[a, b]:
                pair_sums.append(p_gamma[a, b] + p_gamma[b, a])
    mean_sum = float(np.mean(pair_sums)) if pair_sums else 1.0
    n_deficient = int(sum(1 for s in pair_sums if s < 1.0))
    n_excess = int(sum(1 for s in pair_sums if s > 1.0))

    base = {
        **asdict(job),
        "n_gamma_pairs": n_gamma_pairs,
        "n_total_pairs": n_total_pairs,
        "mean_sum": mean_sum,
        "n_deficient_pairs": n_deficient,
        "n_excess_pairs": n_excess,
    }

    results = []
    for model_idx, model in enumerate(["bernoulli", "gamma"]):
        sim_seed = job.rank_seed + 10_000_000 + 100_000 * model_idx
        t0 = perf_counter()

        if model == "bernoulli":
            p_use = p_win
        else:
            p_use = p_gamma

        payoff = payoff_from_probabilities(p_use)
        lp_x, _ = solve_zero_sum_equilibrium(payoff)

        if model == "bernoulli":
            _, freqs, avg = simulate_bernoulli_tournament_counts(
                p_use, job.n_individuals, job.n_steps,
                job.sample_every, sim_seed, job.burn_in,
            )
        else:
            _, freqs, avg = simulate_gamma_tournament_counts(
                p_use, job.n_individuals, job.n_steps,
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
            "lp_support_size": lp_size,
            "lp_support_odd": lp_size % 2 == 1,
            "final_support_size": final_size,
            "final_support_odd": final_size % 2 == 1,
            "avg_support_size": avg_size,
            "avg_support_odd": avg_size % 2 == 1,
            "extinct_time": None if extinct_idx < 0 else extinct_idx,
            "runtime_s": perf_counter() - t0,
        })

    return results


def make_jobs(args: argparse.Namespace) -> list[PhiCJob]:
    jobs = []
    for phi_C in args.phi_c:
        for n_species in args.species:
            for n_factors in args.factors:
                for rep in range(args.reps):
                    seed = (
                        args.seed_base
                        + 1_000_000 * rep
                        + 10_000 * n_species
                        + 100 * n_factors
                        + int(round(phi_C * 1000))
                    )
                    jobs.append(PhiCJob(
                        phi_C=phi_C,
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
    p.add_argument("--phi-c", nargs="+", type=float,
                   default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    p.add_argument("--species", nargs="+", type=int, default=[7])
    p.add_argument("--factors", nargs="+", type=int, default=[3, 5, 7])
    p.add_argument("--reps", type=int, default=200)
    p.add_argument("--n-individuals", type=int, default=25_000)
    p.add_argument("--n-steps", type=int, default=10_000_000)
    p.add_argument("--sample-every", type=int, default=10_000)
    p.add_argument("--burn-in", type=int, default=1_000_000)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed-base", type=int, default=42)
    p.add_argument("--out", default="results/phi_c_sweep_v1.pkl")
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


def apply_quick_defaults(args: argparse.Namespace) -> argparse.Namespace:
    if args.quick:
        args.phi_c = [0.0, 0.5, 1.0]
        args.species = [7]
        args.factors = [5]
        args.reps = 4
        args.n_individuals = 2_000
        args.n_steps = 50_000
        args.sample_every = 1_000
        args.burn_in = 5_000
        args.workers = 2
        if args.out == "results/phi_c_sweep_v1.pkl":
            args.out = "results/phi_c_sweep_quick.pkl"
    return args


def main() -> None:
    args = apply_quick_defaults(parse_args())
    jobs = make_jobs(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_rows = len(jobs) * 2
    print(
        f"phi_C sweep: {len(jobs)} paired jobs ({n_rows} model rows), "
        f"phi_C={args.phi_c}, S={args.species}, F={args.factors}, "
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
                f"[{i:>5}/{len(jobs)}] phi_C={m['phi_C']:.2f} "
                f"S={m['n_species']} F={m['n_factors']} rep={m['rep']} "
                f"n_gamma={m['n_gamma_pairs']} mean_sum={m['mean_sum']:.3f} "
                f"runtime={rt:.2f}s"
            )

    with out.open("wb") as fh:
        pickle.dump(results, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved {out} ({len(results)} records)")


if __name__ == "__main__":
    main()
