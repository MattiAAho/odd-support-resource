# Tournament Module

Standalone code for Allesina-Levine style tournament experiments.

## Models

- `fixed`: sample pairwise dominance once from limiting-factor win probabilities, then freeze the tournament.
- `bernoulli`: resample the pairwise winner at each encounter from the marginal probability matrix.
- `factor`: resample one limiting factor at each encounter; the better-ranked species on that factor wins.
- `beta_mean`: use the posterior mean pairwise probabilities from factor win counts.
- `beta_sample`: sample pairwise probabilities from Beta posteriors, then run Bernoulli encounters.

Jeffreys prior is `Beta(0.5, 0.5)`.  A uniform prior is `Beta(1, 1)`.

The sweep runner supports two simulation engines:

- `counts` (default): production engine; state is tracked only as species counts.
- `population`: educational/reference engine; state is tracked as an explicit population array.

For the current iid-factor model, `factor` and `bernoulli` have the same marginal pairwise
win probabilities. They become meaningfully different only after adding temporal
structure, correlation, or trade-off constraints to the limiting factors.

The production sweep uses a paired replicate design:

- one shared `rank_seed` per replicate
- one shared rank matrix / probability matrix per replicate
- all baseline models are run from that same underlying tournament draw

This makes `fixed` vs `bernoulli` vs `beta_sample` comparisons cleaner because
between-replicate tournament variation is controlled.

## Commands

Small direct demo:

```bash
python run_tournament_demo.py
```

Fast multiprocessing smoke sweep:

```bash
python run_tournament_sweep.py --quick
```

Paired baseline comparison plot:

```bash
python plot_tournament_paired_comparison.py \
  --input results/tournament_sweep_quick_paired.pkl
```

Explicit reference engine:

```bash
python run_tournament_sweep.py --quick --engine population
```

Default heavier sweep:

```bash
python run_tournament_sweep.py
```

The default baseline models are `fixed`, `bernoulli`, and `beta_sample`.
The current iid `factor` model is not included by default because it is marginally
equivalent to `bernoulli`. Use `--models fixed bernoulli factor beta_sample` only
after adding structure that makes `factor` genuinely distinct.

The default sweep is intentionally conservative but still large enough to benefit
from Numba and parallel workers. Use `--n-steps 10000000` for A&L-style long-run
dynamic comparisons.
