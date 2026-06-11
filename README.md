# Odd-Support Resource — Tournament Parity under Incomplete Competition

Replication code for the paper
**"Reproducing the Odd-Support Invariant and its Incompleteness Limit: A Bridge from
Tournament Parity Theory to the Allesina–Levine Ecological Model."**

---

## Overview

This repository contains the tournament-generation engine and the analysis code needed
to reproduce the sweeps, enumerations, and figures in the paper. No pre-computed data is
distributed; everything is regenerated from scratch by the scripts described below.

The study concerns the **odd-support parity invariant** of competitive tournaments — that
generic complete antisymmetric zero-sum tournaments support an *odd* number of coexisting
species at equilibrium (Kaplansky 1945; Fisher & Ryan 1992; Brandl 2017), brought into
ecology by Allesina & Levine (2011). Real interaction networks are rarely complete, so we
ask what happens to the invariant once completeness is lost. Using large-scale numerical
sweeps inside the Allesina–Levine factor-rank model, the code measures how

- **structural sparsity** (`φ_B`, absent interactions),
- **competitive neutrality** (`φ_A`, tied contests),
- **complementarity breaking** (`φ_C`), and
- the number of **limiting factors** (`F`)

perturb the returned linear-programming (LP) support parity, and it ties the resulting
U-shaped support-versus-connectivity curve to the established Pfaffian / perfect-matching
mechanism. A finite-population simulation engine contrasts the infinite-population LP
predictions with stochastic dynamics under absorbing extinction.

---

## Requirements

Python 3.11 or 3.12 recommended. Install dependencies with:

```bash
pip install -r requirements.txt
```

| Package    | Minimum |
|------------|---------|
| numpy      | 2.0     |
| scipy      | 1.12    |
| networkx   | 3.4     |
| numba      | 0.60    |
| matplotlib | 3.9     |

The LP equilibria are solved with SciPy's HiGHS backend. The finite-population kernels in
`tournament/kernels.py` are JIT-compiled with Numba; the first run is slower while Numba
compiles, after which the on-disk cache is reused.

---

## Repository structure

```
tournament/                         Core package (the only module the scripts import)
  generation.py                     Factor-rank tournament construction; φ_A / φ_B / φ_C perturbations
  kernels.py                        Numba finite-population simulation engines (counts + reference)
  lp.py                             Zero-sum LP equilibrium solver and equilibrium-family enumeration
  README.md                         Package-level notes on the tournament models

Sweeps  (write results/*.pkl)
  run_sparsity_sweep.py             φ_B structural-sparsity sweep
  run_phi_a_sweep.py                φ_A competitive-neutrality sweep
  run_phi_c_sweep.py                φ_C complementarity-breaking sweep
  run_f_sweep.py                    Limiting-factors F sweep
  run_phase_2d_sweep.py             2D φ_A × φ_B phase grid
  run_nonuniqueness_sweep.py        Equilibrium-family / vertex-selection-rule sweep
  run_al_fig3.py                    Allesina & Levine (2011) Fig. 3A factor-coexistence reproduction
  run_tournament_sweep.py           Paired-replicate baseline sweep (fixed / bernoulli / beta_sample)
  run_tournament_demo.py            Small direct demo

Analyses & figure scripts  (read results/*.pkl, write figures/*)
  analysis_sparsity_mechanism.py    U-shape + monodominance recovery        (Fig. 1)
  analysis_ultra_small_phiB.py      Immediate parity break near φ_B = 0      (Fig. 2)
  analysis_pfaffian_matching.py     Structural/accidental Pfaffian split
  plot_pfaffian_mechanism.py        Pfaffian / matching mechanism            (Fig. 3)
  plot_phi_combined.py              Neutrality + φ_eff unification panels     (Fig. 4)
  plot_phi_a_sweep.py               φ_A neutrality curves
  plot_phase_2d.py                  φ_eff phase diagram + residuals          (Fig. 4c, Fig. S1)
  analysis_horizon_check.py         LP vs finite-population divergence       (Fig. 5)
  plot_nonuniqueness_sweep.py       Returned support under four rules        (Fig. 6)
  analysis_n4_enumeration.py        Exhaustive N = 4 integer-payoff census
  plot_n4_selection_rules.py        N = 4 selection-rule census              (Fig. S5)
  analysis_n4_nonuniqueness.py      N = 4 equilibrium-family non-uniqueness
  plot_phi_c_sweep.py               Complementarity-breaking control         (Fig. S2)
  plot_f_sweep.py                   Limiting-factors control                 (Fig. S3)
  analysis_unification_check.py     φ_A/φ_B selection-rule robustness         (Fig. S6)
  analysis_bernoulli_fixed_gap.py   Fixed-vs-Bernoulli representation gap
  plot_tournament_oscillation.py    Cycling dynamics illustration
  plot_tournament_sweep.py          Baseline-sweep summary plot
  plot_tournament_paired_comparison.py  Paired fixed/bernoulli/beta comparison

Tests
  test_tournament_smoke.py          Fast self-check of the package API
```

Figure scripts read the `results/*.pkl` written by the corresponding sweep and write to
`figures/`. Both directories are regenerated and are git-ignored.

---

## Reproduction pipeline

```bash
# 1 — Simulate (writes results/*.pkl; the heavier sweeps benefit from multiple cores)
python run_sparsity_sweep.py
python run_phi_a_sweep.py
python run_phi_c_sweep.py
python run_f_sweep.py
python run_phase_2d_sweep.py
python run_nonuniqueness_sweep.py

# 2 — Render main-text and supplementary figures (reads results/, writes figures/)
python analysis_sparsity_mechanism.py       # Fig. 1
python analysis_ultra_small_phiB.py         # Fig. 2
python plot_pfaffian_mechanism.py           # Fig. 3
python plot_phi_combined.py                 # Fig. 4
python analysis_horizon_check.py            # Fig. 5
python plot_nonuniqueness_sweep.py          # Fig. 6
python plot_phase_2d.py                     # Fig. S1
python plot_phi_c_sweep.py                  # Fig. S2
python plot_f_sweep.py                      # Fig. S3
python plot_n4_selection_rules.py           # Fig. S5
python analysis_unification_check.py        # Fig. S6
```

Most scripts accept `--help` for their input/output paths and parameters.

---

## Minimal example

```python
import numpy as np
from tournament.generation import (
    independent_factor_ranks, win_probability_matrix,
    payoff_from_probabilities, apply_sparsity,
)
from tournament.lp import solve_zero_sum_equilibrium

rng = np.random.default_rng(42)

# Factor-rank tournament: N = 7 species, F = 5 limiting factors (Allesina & Levine 2011)
ranks = independent_factor_ranks(n_species=7, n_factors=5, rng=rng)
p = win_probability_matrix(ranks)            # pairwise win probabilities
K = payoff_from_probabilities(p)             # skew-symmetric payoff  K = p - p^T

# Complete tournament -> odd support (the A&L invariant)
x, value = solve_zero_sum_equilibrium(K)
print("complete:   support =", int((x > 1e-8).sum()), " value =", round(value, 9))

# Structural sparsity (phi_B): completeness breaks, even support becomes admissible
p_sparse, _ = apply_sparsity(p, phi_B=0.3, rng=rng)
x2, _ = solve_zero_sum_equilibrium(payoff_from_probabilities(p_sparse))
print("phi_B = 0.3: support =", int((x2 > 1e-8).sum()))
```

Swap `apply_sparsity` for `apply_neutrality` (φ_A) or `apply_complementarity_break` (φ_C)
to reproduce the other perturbation axes, and use `all_equilibrium_support_sets` from
`tournament.lp` to enumerate the full equilibrium family under the alternative
vertex-selection rules.

---

## Citation

*(to be added on publication)*

---

## License

Released under the [GNU General Public License v3.0](LICENSE).
