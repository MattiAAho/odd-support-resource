"""2D phase diagram sweep: phi_A (neutrality) × phi_B (sparsity).

Combines both axes to test whether LP odd-support fraction is a scalar function
of total zero-payoff pair count.  Structural zeros and neutral pairs are applied
to disjoint sets: sparsity is applied first, neutrality only to the surviving
(interact=True) pairs.

Effective zero-payoff fraction: phi_eff = 1 - (1-phi_A)(1-phi_B).
Contour prediction: curves of constant LP odd-support follow (1-phi_A)(1-phi_B) = const.
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
class Phase2DJob:
    phi_A: float
    phi_B: float
    n_species: int
    n_factors: int
    n_individuals: int
    n_steps: int
    sample_every: int
    burn_in: int
    rep: int
    rank_seed: int


def _apply_neutrality_masked(
    p: np.ndarray,
    interact: np.ndarray,
    phi_A: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply neutrality only to interact=True pairs, leaving structural zeros untouched."""
    n = p.shape[0]
    p_out = p.copy()
    neutral_mask = np.zeros((n, n), dtype=np.bool_)
    for a in range(n):
        for b in range(a + 1, n):
            if interact[a, b] and rng.random() < phi_A:
                p_out[a, b] = 0.5
                p_out[b, a] = 0.5
                neutral_mask[a, b] = True
                neutral_mask[b, a] = True
    return p_out, neutral_mask


def _run_job(job: Phase2DJob) -> list[dict]:
    rng = np.random.default_rng(job.rank_seed)

    ranks = independent_factor_ranks(job.n_species, job.n_factors, rng)
    counts = factor_win_counts(ranks)
    p_win = win_probability_matrix(ranks)

    # Step 1: structural zeros (phi_B).
    p_sparse, interact = apply_sparsity(p_win, job.phi_B, rng)
    # Step 2: neutrality only to surviving pairs (phi_A).
    p_combined, neutral_mask = _apply_neutrality_masked(p_sparse, interact, job.phi_A, rng)

    n_total_pairs = job.n_species * (job.n_species - 1) // 2
    n_sparse = int(interact[np.triu_indices(job.n_species, k=1)].sum())
    n_neutral = int(neutral_mask[np.triu_indices(job.n_species, k=1)].sum())
    n_zero_payoff = (n_total_pairs - n_sparse) + n_neutral  # structural zeros + neutral pairs
    actual_phi_eff = n_zero_payoff / n_total_pairs if n_total_pairs > 0 else 0.0

    base = {
        **asdict(job),
        "n_active_pairs": n_sparse,
        "n_neutral_pairs": n_neutral,
        "n_zero_payoff_pairs": n_zero_payoff,
        "actual_phi_eff": actual_phi_eff,
    }

    results = []
    for model_idx, model in enumerate(["fixed", "bernoulli", "beta_sample"]):
        sim_seed = job.rank_seed + 10_000_000 + 100_000 * model_idx
        t0 = perf_counter()

        lp_x = None
        lp_degenerate = False

        if model == "fixed":
            winner = sample_fixed_tournament(p_combined, rng)
            payoff = payoff_from_winner(winner)
            lp_x, lp_status = solve_zero_sum_equilibrium(payoff)
            _, freqs, avg = simulate_fixed_tournament_counts(
                winner, job.n_individuals, job.n_steps,
                job.sample_every, sim_seed, job.burn_in,
            )

        elif model == "bernoulli":
            payoff = payoff_from_probabilities(p_combined)
            lp_degenerate = bool(np.allclose(payoff, 0.0))
            lp_x, lp_status = solve_zero_sum_equilibrium(payoff)
            _, freqs, avg = simulate_bernoulli_tournament_counts(
                p_combined, job.n_individuals, job.n_steps,
                job.sample_every, sim_seed, job.burn_in,
            )

        elif model == "beta_sample":
            p_beta = sample_beta_probability_matrix(counts, job.n_factors, rng)
            # Apply the same structural zeros and neutrality mask.
            off_diag_non_interact = ~interact.copy()
            np.fill_diagonal(off_diag_non_interact, False)
            p_beta[off_diag_non_interact] = 0.0
            p_beta[neutral_mask] = 0.5
            payoff = payoff_from_probabilities(p_beta)
            lp_x, lp_status = solve_zero_sum_equilibrium(payoff)
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
            "lp_support_size": lp_size,
            "lp_support_odd": lp_size % 2 == 1,
            "lp_degenerate": lp_degenerate,
            "final_support_size": final_size,
            "final_support_odd": final_size % 2 == 1,
            "avg_support_size": avg_size,
            "avg_support_odd": avg_size % 2 == 1,
            "extinct_time": None if extinct_idx < 0 else extinct_idx,
            "runtime_s": perf_counter() - t0,
        })

    return results


def make_jobs(args: argparse.Namespace) -> list[Phase2DJob]:
    jobs: list[Phase2DJob] = []
    for phi_A in args.phi_a:
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
                            + int(round(phi_A * 100_000))
                        )
                        jobs.append(Phase2DJob(
                            phi_A=phi_A,
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
    p.add_argument("--phi-a", nargs="+", type=float,
                   default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    p.add_argument("--phi-b", nargs="+", type=float,
                   default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    p.add_argument("--species", nargs="+", type=int, default=[7])
    p.add_argument("--factors", nargs="+", type=int, default=[3, 5, 7])
    p.add_argument("--reps", type=int, default=200)
    p.add_argument("--n-individuals", type=int, default=25_000)
    p.add_argument("--n-steps", type=int, default=10_000_000)
    p.add_argument("--sample-every", type=int, default=10_000)
    p.add_argument("--burn-in", type=int, default=1_000_000)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed-base", type=int, default=42)
    p.add_argument("--out", default="results/phase_2d_v1.pkl")
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


def apply_quick_defaults(args: argparse.Namespace) -> argparse.Namespace:
    if args.quick:
        args.phi_a = [0.0, 0.3, 0.6]
        args.phi_b = [0.0, 0.3, 0.6]
        args.species = [7]
        args.factors = [5]
        args.reps = 4
        args.n_individuals = 2_000
        args.n_steps = 50_000
        args.sample_every = 1_000
        args.burn_in = 5_000
        args.workers = 2
        if args.out == "results/phase_2d_v1.pkl":
            args.out = "results/phase_2d_quick.pkl"
    return args


def main() -> None:
    args = apply_quick_defaults(parse_args())
    jobs = make_jobs(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_rows = len(jobs) * 3
    n_cells = len(args.phi_a) * len(args.phi_b)
    print(
        f"2D phase sweep: {len(jobs)} paired jobs ({n_rows} model rows), "
        f"{n_cells} (phi_A, phi_B) cells, S={args.species}, F={args.factors}, "
        f"reps={args.reps}, workers={args.workers}"
    )

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(_run_job, job) for job in jobs]
        for i, fut in enumerate(as_completed(futures), start=1):
            batch = fut.result()
            results.extend(batch)
            if i % 500 == 0 or i == len(jobs):
                m = batch[0]
                rt = sum(r["runtime_s"] for r in batch)
                print(
                    f"[{i:>6}/{len(jobs)}] phi_A={m['phi_A']:.2f} phi_B={m['phi_B']:.2f} "
                    f"S={m['n_species']} F={m['n_factors']} rep={m['rep']} "
                    f"phi_eff={m['actual_phi_eff']:.3f} runtime={rt:.2f}s"
                )

    with out.open("wb") as fh:
        pickle.dump(results, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved {out} ({len(results)} records)")


if __name__ == "__main__":
    main()
