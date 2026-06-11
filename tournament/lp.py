"""Linear-program equilibria for zero-sum tournament games."""

from __future__ import annotations

import itertools

import numpy as np
from scipy.optimize import linprog


def solve_zero_sum_equilibrium(payoff: np.ndarray, tol: float = 1e-10) -> tuple[np.ndarray, float]:
    """Solve the uninvadable population equilibrium for a tournament game.

    The payoff matrix should be skew-symmetric for tournament games: payoff[a,b]
    is positive if a beats b and negative if a loses to b.  A population state x
    is uninvadable when no rare pure species has positive payoff against it:
    payoff @ x <= 0.  For skew-symmetric games the value is theoretically zero,
    but the LP returns the numerical upper bound.
    """

    payoff = np.asarray(payoff, dtype=np.float64)
    if payoff.ndim != 2 or payoff.shape[0] != payoff.shape[1]:
        raise ValueError("payoff must be a square matrix")

    n_species = payoff.shape[0]

    # Variables are x_0..x_{n-1}, v.  Minimise v subject to payoff @ x <= v.
    c = np.zeros(n_species + 1, dtype=np.float64)
    c[-1] = 1.0

    # payoff @ x <= v
    a_ub = np.zeros((n_species, n_species + 1), dtype=np.float64)
    a_ub[:, :n_species] = payoff
    a_ub[:, -1] = -1.0
    b_ub = np.zeros(n_species, dtype=np.float64)

    a_eq = np.zeros((1, n_species + 1), dtype=np.float64)
    a_eq[0, :n_species] = 1.0
    b_eq = np.array([1.0], dtype=np.float64)

    bounds = [(0.0, 1.0)] * n_species + [(None, None)]
    res = linprog(c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"LP equilibrium failed: {res.message}")

    x = np.asarray(res.x[:n_species], dtype=np.float64)
    x[np.abs(x) < tol] = 0.0
    total = x.sum()
    if total <= 0:
        raise RuntimeError("LP returned zero-mass strategy")
    x /= total
    return x, float(res.x[-1])


def _interior_equilibrium_on_support(
    payoff: np.ndarray, support: list[int], tol: float
) -> np.ndarray | None:
    """Return an interior Nash equilibrium on exactly this support, or None.

    Uses the null space of K_SS (the payoff submatrix restricted to `support`).
    For 1-D null spaces the candidate is the unique null vector (up to sign).
    For higher-dimensional null spaces an interior point is found via a small LP.
    Domination by species outside the support is checked at the end.
    """
    K = payoff
    n = K.shape[0]
    s = len(support)

    if s == 1:
        i = support[0]
        x = np.zeros(n)
        x[i] = 1.0
        return x if np.all(K @ x <= tol) else None

    K_SS = K[np.ix_(support, support)]
    _, sv, Vt = np.linalg.svd(K_SS)
    null_basis = Vt[sv < tol]   # rows are null vectors

    if null_basis.shape[0] == 0:
        return None

    if null_basis.shape[0] == 1:
        v = null_basis[0]
        for sign in (1.0, -1.0):
            x_S = sign * v
            if np.all(x_S > tol):
                x_S = x_S / x_S.sum()
                x = np.zeros(n)
                for i, idx in enumerate(support):
                    x[idx] = x_S[i]
                if np.all(K @ x <= tol):
                    return x
        return None

    # Higher-dimensional null space: parameterise x_S = V @ alpha and maximise
    # the minimum component t subject to V@alpha >= t and sum(x_S) = 1.
    V = null_basis.T          # shape (s, dim_null)
    dim_null = V.shape[1]

    c = np.zeros(dim_null + 1)
    c[-1] = -1.0              # minimise -t

    A_ub = np.zeros((s, dim_null + 1))
    A_ub[:, :dim_null] = -V
    A_ub[:, dim_null] = 1.0
    b_ub = np.zeros(s)

    A_eq = np.zeros((1, dim_null + 1))
    A_eq[0, :dim_null] = V.sum(axis=0)
    b_eq = np.array([1.0])

    bounds = [(None, None)] * dim_null + [(0.0, 1.0)]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds,
                  method="highs")
    if not res.success or res.x[-1] < tol:
        return None

    x_S = V @ res.x[:dim_null]
    x_S /= x_S.sum()
    x = np.zeros(n)
    for i, idx in enumerate(support):
        x[idx] = x_S[i]
    return x if np.all(K @ x <= tol) else None


def all_equilibrium_support_sets(
    payoff: np.ndarray, tol: float = 1e-8
) -> list[tuple[frozenset[int], np.ndarray]]:
    """Enumerate all Nash equilibrium support sets for a zero-sum game.

    For each non-empty subset S of species, checks whether an interior Nash
    equilibrium exists with support *exactly* S (all species in S have strictly
    positive density; no species outside S can invade).  Uses direct linear
    algebra — SVD null-space computation — rather than repeated LP solves with
    different objectives.

    Returns a list of (frozenset_of_species, x_full_vector) pairs.  The list
    may contain multiple entries if the game has non-unique equilibria across
    different support sets (parity-heterogeneous equilibrium family).
    """
    K = np.asarray(payoff, dtype=np.float64)
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError("payoff must be square")
    n = K.shape[0]

    valid: list[tuple[frozenset[int], np.ndarray]] = []
    for size in range(1, n + 1):
        for support in itertools.combinations(range(n), size):
            x = _interior_equilibrium_on_support(K, list(support), tol)
            if x is not None:
                valid.append((frozenset(support), x))
    return valid
