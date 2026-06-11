"""Tournament construction for fixed and dynamic competition models.

The functions here are deliberately independent of the spatial Go model.  They
construct pairwise dominance objects that can be consumed by fast Numba kernels.
"""

from __future__ import annotations

import numpy as np


def fixed_rps_winner() -> np.ndarray:
    """Return the 3-species rock-paper-scissors tournament.

    winner[a, b] is True when species a beats species b.  The diagonal is False.
    """

    winner = np.zeros((3, 3), dtype=np.bool_)
    winner[0, 1] = True
    winner[1, 2] = True
    winner[2, 0] = True
    return winner


def independent_factor_ranks(
    n_species: int,
    n_factors: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate independent random rankings for each limiting factor.

    Lower rank value means the species is superior on that limiting factor.
    Shape: (n_factors, n_species).
    """

    rng = np.random.default_rng() if rng is None else rng
    ranks = np.empty((n_factors, n_species), dtype=np.int16)
    for f in range(n_factors):
        order = rng.permutation(n_species)
        ranks[f, order] = np.arange(n_species, dtype=np.int16)
    return ranks


def factor_win_counts(ranks: np.ndarray) -> np.ndarray:
    """Count factors on which species a beats species b.

    ranks[f, a] < ranks[f, b] means a is superior to b under factor f.
    Returns an int matrix N where N[a, b] is the number of factors won by a.
    """

    ranks = np.asarray(ranks)
    if ranks.ndim != 2:
        raise ValueError("ranks must have shape (n_factors, n_species)")
    n_factors, n_species = ranks.shape
    counts = np.zeros((n_species, n_species), dtype=np.int16)
    for a in range(n_species):
        for b in range(n_species):
            if a != b:
                counts[a, b] = int(np.sum(ranks[:, a] < ranks[:, b]))
    return counts


def win_probability_matrix(ranks: np.ndarray) -> np.ndarray:
    """Return pairwise win probabilities implied by limiting-factor ranks."""

    counts = factor_win_counts(ranks).astype(np.float64)
    n_factors = ranks.shape[0]
    p = counts / float(n_factors)
    np.fill_diagonal(p, 0.5)
    return p


def posterior_mean_probability_matrix(
    counts: np.ndarray,
    n_factors: int,
    prior_alpha: float = 0.5,
    prior_beta: float = 0.5,
) -> np.ndarray:
    """Posterior mean pairwise probabilities from factor win counts.

    Jeffreys prior is prior_alpha=prior_beta=0.5.  A uniform prior is 1.0, 1.0.
    """

    counts = np.asarray(counts, dtype=np.float64)
    p = (counts + prior_alpha) / (n_factors + prior_alpha + prior_beta)
    np.fill_diagonal(p, 0.5)
    return p


def sample_beta_probability_matrix(
    counts: np.ndarray,
    n_factors: int,
    rng: np.random.Generator | None = None,
    prior_alpha: float = 0.5,
    prior_beta: float = 0.5,
) -> np.ndarray:
    """Sample pairwise probabilities from Beta posteriors.

    For each unordered pair {a,b}, p_ab is sampled from
    Beta(N_ab + prior_alpha, N_ba + prior_beta), and p_ba = 1 - p_ab.
    """

    rng = np.random.default_rng() if rng is None else rng
    counts = np.asarray(counts)
    if counts.ndim != 2 or counts.shape[0] != counts.shape[1]:
        raise ValueError("counts must be a square matrix")

    n_species = counts.shape[0]
    p = np.full((n_species, n_species), 0.5, dtype=np.float64)
    for a in range(n_species):
        for b in range(a + 1, n_species):
            alpha = float(counts[a, b]) + prior_alpha
            beta = float(n_factors - counts[a, b]) + prior_beta
            p_ab = rng.beta(alpha, beta)
            p[a, b] = p_ab
            p[b, a] = 1.0 - p_ab
    return p


def apply_sparsity(
    p_win: np.ndarray,
    phi_B: float,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Zero out a fraction phi_B of unordered pairs (structural non-interactions).

    For each unordered pair {a, b}, with probability phi_B both p[a,b] and
    p[b,a] are set to 0.0 — the pair is removed from the tournament entirely.
    This breaks the completeness assumption of the A&L parity argument.

    Returns (p_sparse, interact_mask).  interact_mask[a, b] is True iff species
    a and b still compete (False on the diagonal and for zeroed-out pairs).
    """

    rng = np.random.default_rng() if rng is None else rng
    n = p_win.shape[0]
    p_sparse = p_win.copy()
    interact = np.ones((n, n), dtype=np.bool_)
    np.fill_diagonal(interact, False)
    for a in range(n):
        for b in range(a + 1, n):
            if rng.random() < phi_B:
                p_sparse[a, b] = 0.0
                p_sparse[b, a] = 0.0
                interact[a, b] = False
                interact[b, a] = False
    return p_sparse, interact


def apply_neutrality(
    p_win: np.ndarray,
    phi_A: float,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Set a fraction phi_A of unordered pairs to neutral (p_ab = p_ba = 0.5).

    Neutral pairs preserve completeness (both directions are 0.5, not 0),
    so the A&L parity theorem still applies to the fixed-model LP.  The
    Bernoulli LP becomes degenerate when all pairs are neutral (phi_A=1,
    payoff matrix = all zeros).

    Returns (p_neutral, neutral_mask).  neutral_mask[a, b] is True iff pair
    {a, b} was made neutral.
    """
    rng = np.random.default_rng() if rng is None else rng
    n = p_win.shape[0]
    p_neutral = p_win.copy()
    neutral_mask = np.zeros((n, n), dtype=np.bool_)
    for a in range(n):
        for b in range(a + 1, n):
            if rng.random() < phi_A:
                p_neutral[a, b] = 0.5
                p_neutral[b, a] = 0.5
                neutral_mask[a, b] = True
                neutral_mask[b, a] = True
    return p_neutral, neutral_mask


def apply_complementarity_break(
    p_win: np.ndarray,
    phi_C: float,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Break complementarity for a fraction phi_C of pairs (Option γ).

    For each affected pair {a, b}, p_ab and p_ba are redrawn independently
    from Uniform(0, 1), so p_ab + p_ba is no longer constrained to equal 1.

    The K matrix (K[a,b] = p_ab − p_ba) remains skew-symmetric because
    K[b,a] = p_ba − p_ab = −K[a,b], so LP parity is theoretically unaffected.
    The simulation dynamics change: encounters may fizzle (sum < 1) or be
    renormalised (sum > 1) — see simulate_gamma_tournament_counts in kernels.py.

    Returns (p_gamma, gamma_mask).  gamma_mask[a, b] is True for affected pairs.
    """
    rng = np.random.default_rng() if rng is None else rng
    n = p_win.shape[0]
    p_gamma = p_win.copy()
    gamma_mask = np.zeros((n, n), dtype=np.bool_)
    for a in range(n):
        for b in range(a + 1, n):
            if rng.random() < phi_C:
                p_gamma[a, b] = rng.random()
                p_gamma[b, a] = rng.random()
                gamma_mask[a, b] = True
                gamma_mask[b, a] = True
    return p_gamma, gamma_mask


def sample_fixed_tournament(
    p_win: np.ndarray,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample a fixed tournament from pairwise win probabilities.

    For every unordered pair {a,b}, draw one Bernoulli trial with probability
    p_win[a,b].  The resulting edge is frozen for all future encounters.
    Structural zeros (p_win[a,b] == p_win[b,a] == 0) are skipped: neither
    species wins, leaving both winner entries False.
    """

    rng = np.random.default_rng() if rng is None else rng
    p_win = np.asarray(p_win, dtype=np.float64)
    if p_win.ndim != 2 or p_win.shape[0] != p_win.shape[1]:
        raise ValueError("p_win must be a square matrix")
    n_species = p_win.shape[0]
    winner = np.zeros((n_species, n_species), dtype=np.bool_)
    for a in range(n_species):
        for b in range(a + 1, n_species):
            if p_win[a, b] == 0.0 and p_win[b, a] == 0.0:
                continue  # structural zero; neither species wins
            if rng.random() < p_win[a, b]:
                winner[a, b] = True
            else:
                winner[b, a] = True
    return winner


def payoff_from_winner(winner: np.ndarray) -> np.ndarray:
    """Convert a boolean tournament to a skew-symmetric zero-sum payoff matrix."""

    winner = np.asarray(winner, dtype=np.bool_)
    if winner.ndim != 2 or winner.shape[0] != winner.shape[1]:
        raise ValueError("winner must be a square matrix")
    payoff = winner.astype(np.float64) - winner.T.astype(np.float64)
    np.fill_diagonal(payoff, 0.0)
    return payoff


def payoff_from_probabilities(p_win: np.ndarray) -> np.ndarray:
    """Expected skew-symmetric payoff matrix from pairwise win probabilities."""

    p_win = np.asarray(p_win, dtype=np.float64)
    if p_win.ndim != 2 or p_win.shape[0] != p_win.shape[1]:
        raise ValueError("p_win must be a square matrix")
    payoff = p_win - p_win.T
    np.fill_diagonal(payoff, 0.0)
    return payoff


def win_probabilities_from_values(x: np.ndarray) -> np.ndarray:
    """Pairwise win probabilities from continuous species-factor values.

    x[i, k] = competitive value of species i on factor k (higher = better).
    p[i, j] = fraction of factors where x[i, k] > x[j, k].
    Diagonal set to 0.5.  Works for all A&L factor-correlation models.
    """
    x = np.asarray(x, dtype=np.float64)
    wins = (x[:, np.newaxis, :] > x[np.newaxis, :, :]).mean(axis=2)
    np.fill_diagonal(wins, 0.5)
    return wins


def autocorrelated_factor_values(
    n_species: int,
    n_factors: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Factor values with positive autocorrelation across factors (A&L Fig. 3B).

    x[i, 0] ~ U(0, 1); x[i, k] = x[i, k-1] + U(0, 1) for k > 0.
    Species good on factor 1 tend to be good on all factors — reduces effective
    dimensionality, slowing convergence toward the random-tournament baseline.
    Returns shape (n_species, n_factors).
    """
    rng = np.random.default_rng() if rng is None else rng
    x = np.empty((n_species, n_factors), dtype=np.float64)
    x[:, 0] = rng.random(n_species)
    for k in range(1, n_factors):
        x[:, k] = x[:, k - 1] + rng.random(n_species)
    return x


def tradeoff_factor_values(
    n_species: int,
    n_factors: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Factor values under resource trade-off (A&L Fig. 3C).

    Each species allocates a unit budget across F factors: x[i, :] ~ Dirichlet(1,...,1),
    sum_k x[i, k] = 1.  For even F, p_ij = 0.5 for all pairs when a fixed tournament
    is sampled — identical to an iid random tournament, reaching the N/2 baseline
    immediately.  For odd F, p_ij != 0.5 (producing a 'zigzag' vs F).
    Returns shape (n_species, n_factors).
    """
    rng = np.random.default_rng() if rng is None else rng
    return rng.dirichlet(np.ones(n_factors), size=n_species)


def random_iid_payoff(
    n_species: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Iid random tournament payoff matrix (F→∞ proxy for fixed-model f-factor).

    p[a, b] ~ U(0, 1), p[b, a] = 1 - p[a, b]; K = p - p^T.
    When p_ij → 0.5 for all pairs (large F), fixed-tournament samples become
    iid random tournaments and E[LP support] → N/2.  This function draws the
    limiting distribution directly.
    """
    rng = np.random.default_rng() if rng is None else rng
    p = np.full((n_species, n_species), 0.5, dtype=np.float64)
    for a in range(n_species):
        for b in range(a + 1, n_species):
            p_ab = rng.random()
            p[a, b] = p_ab
            p[b, a] = 1.0 - p_ab
    K = p - p.T
    np.fill_diagonal(K, 0.0)
    return K
