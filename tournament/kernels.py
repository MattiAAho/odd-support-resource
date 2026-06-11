"""Numba kernels for fixed and dynamic tournament simulations."""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def _init_counts_uniform(n_species: int, n_individuals: int) -> np.ndarray:
    counts = np.zeros(n_species, dtype=np.int64)
    for _ in range(n_individuals):
        counts[np.random.randint(0, n_species)] += 1
    return counts


@njit(cache=True)
def _draw_species_from_counts(counts: np.ndarray, n_individuals: int) -> int:
    r = np.random.random() * n_individuals
    acc = 0.0
    for s in range(counts.shape[0]):
        acc += counts[s]
        if r < acc:
            return s
    return counts.shape[0] - 1


@njit(cache=True)
def _record_counts(
    counts: np.ndarray,
    n_individuals: int,
    step: int,
    sample_idx: int,
    times: np.ndarray,
    freqs: np.ndarray,
    avg: np.ndarray,
    burn_in: int,
    avg_n: int,
) -> tuple[int, int]:
    times[sample_idx] = step
    for s in range(counts.shape[0]):
        f = counts[s] / n_individuals
        freqs[sample_idx, s] = f
        if step >= burn_in:
            avg[s] += f
    if step >= burn_in:
        avg_n += 1
    sample_idx += 1
    return sample_idx, avg_n


@njit(cache=True)
def simulate_fixed_tournament(
    winner: np.ndarray,
    n_individuals: int,
    n_steps: int,
    sample_every: int,
    seed: int,
    burn_in: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Individual-based simulation with a frozen pairwise tournament."""

    np.random.seed(seed)
    n_species = winner.shape[0]
    population = np.empty(n_individuals, dtype=np.int16)
    counts = np.zeros(n_species, dtype=np.int64)

    for i in range(n_individuals):
        s = np.random.randint(0, n_species)
        population[i] = s
        counts[s] += 1

    n_samples = n_steps // sample_every + 1
    times = np.empty(n_samples, dtype=np.int64)
    freqs = np.empty((n_samples, n_species), dtype=np.float64)
    sample_idx = 0
    avg = np.zeros(n_species, dtype=np.float64)
    avg_n = 0

    for step in range(n_steps + 1):
        if step % sample_every == 0:
            times[sample_idx] = step
            for s in range(n_species):
                f = counts[s] / n_individuals
                freqs[sample_idx, s] = f
                if step >= burn_in:
                    avg[s] += f
            if step >= burn_in:
                avg_n += 1
            sample_idx += 1

        if step == n_steps:
            break

        i = np.random.randint(0, n_individuals)
        j = np.random.randint(0, n_individuals)
        if i == j:
            continue
        a = population[i]
        b = population[j]
        if a == b:
            continue

        if winner[a, b]:
            population[j] = a
            counts[a] += 1
            counts[b] -= 1
        elif winner[b, a]:
            population[i] = b
            counts[b] += 1
            counts[a] -= 1

    if avg_n > 0:
        avg /= avg_n
    return times, freqs, avg


@njit(cache=True)
def simulate_fixed_tournament_counts(
    winner: np.ndarray,
    n_individuals: int,
    n_steps: int,
    sample_every: int,
    seed: int,
    burn_in: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Counts-only simulation with a frozen pairwise tournament."""

    np.random.seed(seed)
    n_species = winner.shape[0]
    counts = _init_counts_uniform(n_species, n_individuals)

    n_samples = n_steps // sample_every + 1
    times = np.empty(n_samples, dtype=np.int64)
    freqs = np.empty((n_samples, n_species), dtype=np.float64)
    sample_idx = 0
    avg = np.zeros(n_species, dtype=np.float64)
    avg_n = 0

    for step in range(n_steps + 1):
        if step % sample_every == 0:
            sample_idx, avg_n = _record_counts(
                counts, n_individuals, step, sample_idx, times, freqs, avg, burn_in, avg_n
            )

        if step == n_steps:
            break

        a = _draw_species_from_counts(counts, n_individuals)
        b = _draw_species_from_counts(counts, n_individuals)
        if a == b:
            continue

        if winner[a, b]:
            counts[a] += 1
            counts[b] -= 1
        elif winner[b, a]:
            counts[b] += 1
            counts[a] -= 1

    if avg_n > 0:
        avg /= avg_n
    return times, freqs, avg


@njit(cache=True)
def simulate_bernoulli_tournament(
    p_win: np.ndarray,
    n_individuals: int,
    n_steps: int,
    sample_every: int,
    seed: int,
    burn_in: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dynamic tournament where each encounter samples from p_win[a,b]."""

    np.random.seed(seed)
    n_species = p_win.shape[0]
    population = np.empty(n_individuals, dtype=np.int16)
    counts = np.zeros(n_species, dtype=np.int64)

    for i in range(n_individuals):
        s = np.random.randint(0, n_species)
        population[i] = s
        counts[s] += 1

    n_samples = n_steps // sample_every + 1
    times = np.empty(n_samples, dtype=np.int64)
    freqs = np.empty((n_samples, n_species), dtype=np.float64)
    sample_idx = 0
    avg = np.zeros(n_species, dtype=np.float64)
    avg_n = 0

    for step in range(n_steps + 1):
        if step % sample_every == 0:
            times[sample_idx] = step
            for s in range(n_species):
                f = counts[s] / n_individuals
                freqs[sample_idx, s] = f
                if step >= burn_in:
                    avg[s] += f
            if step >= burn_in:
                avg_n += 1
            sample_idx += 1

        if step == n_steps:
            break

        i = np.random.randint(0, n_individuals)
        j = np.random.randint(0, n_individuals)
        if i == j:
            continue
        a = population[i]
        b = population[j]
        if a == b:
            continue
        if p_win[a, b] == 0.0 and p_win[b, a] == 0.0:
            continue

        if np.random.random() < p_win[a, b]:
            population[j] = a
            counts[a] += 1
            counts[b] -= 1
        else:
            population[i] = b
            counts[b] += 1
            counts[a] -= 1

    if avg_n > 0:
        avg /= avg_n
    return times, freqs, avg


@njit(cache=True)
def simulate_bernoulli_tournament_counts(
    p_win: np.ndarray,
    n_individuals: int,
    n_steps: int,
    sample_every: int,
    seed: int,
    burn_in: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Counts-only dynamic tournament with Bernoulli pairwise wins."""

    np.random.seed(seed)
    n_species = p_win.shape[0]
    counts = _init_counts_uniform(n_species, n_individuals)

    n_samples = n_steps // sample_every + 1
    times = np.empty(n_samples, dtype=np.int64)
    freqs = np.empty((n_samples, n_species), dtype=np.float64)
    sample_idx = 0
    avg = np.zeros(n_species, dtype=np.float64)
    avg_n = 0

    for step in range(n_steps + 1):
        if step % sample_every == 0:
            sample_idx, avg_n = _record_counts(
                counts, n_individuals, step, sample_idx, times, freqs, avg, burn_in, avg_n
            )

        if step == n_steps:
            break

        a = _draw_species_from_counts(counts, n_individuals)
        b = _draw_species_from_counts(counts, n_individuals)
        if a == b:
            continue
        if p_win[a, b] == 0.0 and p_win[b, a] == 0.0:
            continue

        if np.random.random() < p_win[a, b]:
            counts[a] += 1
            counts[b] -= 1
        else:
            counts[b] += 1
            counts[a] -= 1

    if avg_n > 0:
        avg /= avg_n
    return times, freqs, avg


@njit(cache=True)
def simulate_factor_resampling_tournament(
    ranks: np.ndarray,
    n_individuals: int,
    n_steps: int,
    sample_every: int,
    seed: int,
    burn_in: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dynamic tournament that resamples one limiting factor per encounter."""

    np.random.seed(seed)
    n_factors, n_species = ranks.shape
    population = np.empty(n_individuals, dtype=np.int16)
    counts = np.zeros(n_species, dtype=np.int64)

    for i in range(n_individuals):
        s = np.random.randint(0, n_species)
        population[i] = s
        counts[s] += 1

    n_samples = n_steps // sample_every + 1
    times = np.empty(n_samples, dtype=np.int64)
    freqs = np.empty((n_samples, n_species), dtype=np.float64)
    sample_idx = 0
    avg = np.zeros(n_species, dtype=np.float64)
    avg_n = 0

    for step in range(n_steps + 1):
        if step % sample_every == 0:
            times[sample_idx] = step
            for s in range(n_species):
                f = counts[s] / n_individuals
                freqs[sample_idx, s] = f
                if step >= burn_in:
                    avg[s] += f
            if step >= burn_in:
                avg_n += 1
            sample_idx += 1

        if step == n_steps:
            break

        i = np.random.randint(0, n_individuals)
        j = np.random.randint(0, n_individuals)
        if i == j:
            continue
        a = population[i]
        b = population[j]
        if a == b:
            continue

        factor = np.random.randint(0, n_factors)
        if ranks[factor, a] < ranks[factor, b]:
            population[j] = a
            counts[a] += 1
            counts[b] -= 1
        else:
            population[i] = b
            counts[b] += 1
            counts[a] -= 1

    if avg_n > 0:
        avg /= avg_n
    return times, freqs, avg


@njit(cache=True)
def simulate_factor_resampling_tournament_counts(
    ranks: np.ndarray,
    n_individuals: int,
    n_steps: int,
    sample_every: int,
    seed: int,
    burn_in: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Counts-only dynamic tournament with factor resampling per encounter."""

    np.random.seed(seed)
    n_factors, n_species = ranks.shape
    counts = _init_counts_uniform(n_species, n_individuals)

    n_samples = n_steps // sample_every + 1
    times = np.empty(n_samples, dtype=np.int64)
    freqs = np.empty((n_samples, n_species), dtype=np.float64)
    sample_idx = 0
    avg = np.zeros(n_species, dtype=np.float64)
    avg_n = 0

    for step in range(n_steps + 1):
        if step % sample_every == 0:
            sample_idx, avg_n = _record_counts(
                counts, n_individuals, step, sample_idx, times, freqs, avg, burn_in, avg_n
            )

        if step == n_steps:
            break

        a = _draw_species_from_counts(counts, n_individuals)
        b = _draw_species_from_counts(counts, n_individuals)
        if a == b:
            continue

        factor = np.random.randint(0, n_factors)
        if ranks[factor, a] < ranks[factor, b]:
            counts[a] += 1
            counts[b] -= 1
        else:
            counts[b] += 1
            counts[a] -= 1

    if avg_n > 0:
        avg /= avg_n
    return times, freqs, avg


@njit(cache=True)
def simulate_gamma_tournament_counts(
    p_win: np.ndarray,
    n_individuals: int,
    n_steps: int,
    sample_every: int,
    seed: int,
    burn_in: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Counts-only tournament where p_ab + p_ba may not equal 1 (Option γ).

    a wins with prob p_win[a, b]; b wins with prob p_win[b, a].
    When sum > 1: both probabilities are renormalised so they sum to 1 (every
    encounter has an outcome, just with adjusted odds).
    When sum <= 1: a wins with prob p_win[a,b], b wins with prob p_win[b,a],
    no change otherwise — the gap (1 - sum) is a no-outcome probability.
    Structural zeros (both entries 0.0) are skipped as usual.
    """

    np.random.seed(seed)
    n_species = p_win.shape[0]
    counts = _init_counts_uniform(n_species, n_individuals)

    n_samples = n_steps // sample_every + 1
    times = np.empty(n_samples, dtype=np.int64)
    freqs = np.empty((n_samples, n_species), dtype=np.float64)
    sample_idx = 0
    avg = np.zeros(n_species, dtype=np.float64)
    avg_n = 0

    for step in range(n_steps + 1):
        if step % sample_every == 0:
            sample_idx, avg_n = _record_counts(
                counts, n_individuals, step, sample_idx, times, freqs, avg, burn_in, avg_n
            )

        if step == n_steps:
            break

        a = _draw_species_from_counts(counts, n_individuals)
        b = _draw_species_from_counts(counts, n_individuals)
        if a == b:
            continue

        p_ab = p_win[a, b]
        p_ba = p_win[b, a]
        if p_ab == 0.0 and p_ba == 0.0:
            continue

        total = p_ab + p_ba
        r = np.random.random()
        if total > 1.0:
            if r < p_ab / total:
                counts[a] += 1
                counts[b] -= 1
            else:
                counts[b] += 1
                counts[a] -= 1
        else:
            if r < p_ab:
                counts[a] += 1
                counts[b] -= 1
            elif r < total:
                counts[b] += 1
                counts[a] -= 1

    if avg_n > 0:
        avg /= avg_n
    return times, freqs, avg


@njit(cache=True)
def extinction_step(freqs: np.ndarray) -> int:
    """Return first sampled index with an extinct species, or -1 if none."""

    for t in range(freqs.shape[0]):
        for s in range(freqs.shape[1]):
            if freqs[t, s] <= 0.0:
                return t
    return -1
