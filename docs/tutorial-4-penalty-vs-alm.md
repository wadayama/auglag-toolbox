# Tutorial 4 — Penalty vs ALM, and KKT diagnostics

## The pure penalty baseline

`penalty_method` solves a single subproblem with a fixed weight `mu`:

```
maximise f(x) - mu * ( sum_c [g_c]_+^2 + sum_k h_k^2 )
```

```python
from auglag_toolbox import penalty_method

res = penalty_method(objective, [F], inequality=[g], easy_projector=projector, mu=1.0)
```

Sweeping `mu` shows the difficulty:

| `mu` | feasibility | objective |
| --- | --- | --- |
| small | violated | high (but infeasible) |
| large | ~feasible | **degraded** (stiff landscape) |

The penalty only becomes feasible as `mu -> infinity`, and the large-`mu`
problem is ill-conditioned, so the attainable objective drops.

## Why ALM wins

The augmented Lagrangian carries multipliers, so it reaches a feasible KKT point
at a **finite** `rho` while keeping a high objective:

```python
alm = augmented_lagrangian(objective, [F], inequality=[g], easy_projector=projector)
assert alm.feasibility <= 1e-6
assert alm.objective > res.objective       # beats the large-mu penalty
print("finite rho:", alm.rho)
```

This contrast is regression-tested in
[`tests/test_penalty_vs_alm.py`](../tests/test_penalty_vs_alm.py).

## KKT diagnostics, standalone

`kkt_residuals` evaluates the three optimality residuals at any point and
multiplier estimate — useful to audit a solution from elsewhere:

```python
from auglag_toolbox import kkt_residuals

feas, compl, stat = kkt_residuals(
    objective, [F],
    inequality=[g], equality=None,
    multipliers_ineq=alm.multipliers_ineq, multipliers_eq=alm.multipliers_eq,
    easy_projector=projector,
)
```

- **feasibility** — `max(max_c [g_c]_+, max_k |h_k|)`; how far outside the
  feasible set.
- **complementarity** — `max_c |lambda_c g_c|`; should be ~0 (active ⇒ `g≈0`,
  inactive ⇒ `lambda≈0`).
- **stationarity** — projected-gradient-map residual of the Lagrangian; ~0 means
  no feasible ascent direction remains.

Default pass tolerances: `feas <= 1e-6`, `compl <= 1e-6`, `stat <= 1e-5`. All
three are tunable via `feas_tol`, `compl_tol`, `stat_tol`.

## Did it actually converge?

The solver does not raise on non-convergence — it returns the last iterate. Use
the self-reported flags instead of guessing:

```python
res = augmented_lagrangian(objective, [F], inequality=[g], easy_projector=projector)
if res.converged:                       # all three KKT tolerances met
    assert res.stop_reason == "kkt_tol"
    # res is an approximate KKT point
else:
    assert res.stop_reason == "max_outer_iters"
    # budget exhausted; NOT a KKT point — raise outer_iters or relax tolerances
```

`converged` gates on `feas_tol`, `compl_tol`, and `stat_tol` together, so a True
flag certifies an approximate KKT point (subject to a constraint qualification;
KKT is necessary, not sufficient, under non-convexity).

## Non-convexity: multistart

For non-convex problems, run several seeded starts and keep the best feasible
one:

```python
res = augmented_lagrangian(objective, [F], inequality=[g],
                           easy_projector=projector, multistart=8, seed=0)
```

This runs the starts **sequentially**. To run them **in parallel** (one batched
solve, GPU-friendly), use `augmented_lagrangian_batched`: stack a leading batch
dimension, have the closures return `(B,)`, pass a batch-aware projector (e.g.
`project_frobenius_ball_batched`), and read the global incumbent from
`res.winner` / `res.winner_result()`. The closures must be NaN-safe (a diverged
element returns NaN, never raises). See the README "Batched (parallel
multi-start)" section.

Next: [Reproducing the node-power figure](tutorial-5-reproduce.md).
