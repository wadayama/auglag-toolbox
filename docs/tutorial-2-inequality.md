# Tutorial 2 — Inequality constraints and multipliers

## Sign convention

Every inequality is written `g_c(x) <= 0`. A budget `tr(F W F^T) <= P1` becomes

```python
def g():
    return torch.trace(F @ W @ F.T) - P1     # <= 0 means feasible
```

## What the multiplier tells you

After the solve, `res.multipliers_ineq[c]` is the converged $\lambda_c\ge 0$:

- **Active** constraint (binding at the optimum): $\lambda_c > 0$, and
  $g_c(x)\approx 0$. The constraint is exerting force on the solution.
- **Inactive** constraint (slack at the optimum): $\lambda_c \to 0$, and
  $g_c(x) < 0$. The easy-only optimum already satisfied it.

This is the complementary-slackness condition $\lambda_c g_c = 0$, reported as
`res.complementarity = max_c |lambda_c g_c|`.

```python
res = augmented_lagrangian(objective, [F], inequality=[g], easy_projector=projector)
if res.multipliers_ineq[0] > 1e-6:
    print("constraint is ACTIVE, lambda =", res.multipliers_ineq[0])
else:
    print("constraint is inactive (slack)")
```

## Multiple constraints

Pass a list; multipliers come back in the same order:

```python
res = augmented_lagrangian(objective, params,
                           inequality=[g1, g2, g3], easy_projector=projector)
res.multipliers_ineq        # [lambda_1, lambda_2, lambda_3]
```

## Scaling tip

The dual update is $\lambda_c \leftarrow [\lambda_c+\rho g_c]_+$, so constraints
of very different magnitudes get incomparable multipliers and the single shared
`rho` serves them unequally. Normalise each constraint by its budget:

```python
def g():
    return torch.trace(F @ W @ F.T) / P1 - 1.0   # dimensionless, O(1)
```

## Reading convergence

`res.history` is one record per outer iteration with `feas`, `compl`, `stat`,
`lam`, `mu`, `rho`, `inner_evals`. Pass `verbose=True` to print it live:

```python
res = augmented_lagrangian(objective, [F], inequality=[g],
                           easy_projector=projector, verbose=True)
```

Next: [Covariance-dependent, coupled constraints](tutorial-3-coupled.md).
