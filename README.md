# auglag-toolbox

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-%E2%89%A53.12-blue.svg)](https://www.python.org/)

Augmented-Lagrangian / penalty constraint handling on top of a first-order
projected solver ([`pga-toolbox`](https://github.com/wadayama/pga-toolbox)), for
complex-valued (Wirtinger) and real parameters. Built on PyTorch.

This library owns exactly one thing: the **outer loop that enforces nonlinear,
coupled, covariance-dependent constraints** which have no closed-form
projection. The inner maximisation over the easy (projectable) set is delegated
to a pluggable first-order solver — by default `pga_toolbox.pga_ascent_spg`.

```
auglag-toolbox  — outer constraint meta-solver (PHR augmented Lagrangian / penalty)
      │  depends on  (one-directional; pga-toolbox knows nothing of this library)
      ▼
  pga-toolbox   — inner first-order solver (projected gradient, Wirtinger)
```

See [`MATH.md`](MATH.md) for the implementation-side mathematics (PHR augmented
Lagrangian, dual updates, the `rho` rule, KKT residuals, complex Wirtinger).

## Why

`pga-toolbox` only handles **easy** constraints — those with a closed-form
Euclidean projection (a Frobenius ball, a total-power ball). Many problems also
carry **hard** constraints that are nonlinear, couple several variables, and
depend on covariances, e.g. a per-node transmit-power budget

```
tr( R K_jj(x) R^H ) <= P_j .
```

There is no projection for these. The augmented Lagrangian moves them into the
objective via multipliers and a penalty parameter, solves the resulting
*easy-only* subproblem with the inner solver, and updates the multipliers in an
outer loop until the KKT conditions hold. Design choices:

- **Single concept.** `pga-toolbox` = inner projected first-order solver;
  `auglag-toolbox` = outer constraint meta-solver.
- **One-directional dependency.** `auglag-toolbox` depends on `pga-toolbox`;
  never the reverse. No vendoring.
- **Pluggable inner solver.** `inner=` accepts any callable that maximises a
  scalar closure by updating `params` in place (the pga-toolbox contract).
  `pga_ascent_spg` (default) and `pga_ascent_armijo` both qualify.
- **Model-agnostic.** Objective and constraints are just closures returning real
  scalars. The library knows nothing about DAGs, mutual information, or channels.

## Install

```bash
git clone https://github.com/wadayama/auglag-toolbox.git
cd auglag-toolbox
uv sync --extra test       # pulls pga-toolbox (git+https) + torch
uv run pytest
```

Before `pga-toolbox` is published, inject the sibling checkout for development:

```bash
uv run --with-editable ../pga-toolbox --extra test pytest
```

## Quick start

Inequality constraint (covariance-weighted power), maximisation:

```python
import torch
from pga_toolbox import project_frobenius_ball
from auglag_toolbox import augmented_lagrangian

F = torch.randn(3, 3, dtype=torch.float64, requires_grad=True)
H = torch.randn(3, 3, dtype=torch.float64)
W = torch.diag(torch.tensor([2.5, 1.0, 0.4], dtype=torch.float64))

def objective():                                  # maximise (real scalar)
    HF = H @ F
    return torch.linalg.slogdet(torch.eye(3) + HF @ HF.T / 0.25).logabsdet

def power():                                      # g(x) <= 0
    return torch.trace(F @ W @ F.T) - 4.0

def projector(params):                            # S_easy = Frobenius ball
    return [project_frobenius_ball(params[0], 4.0)]

res = augmented_lagrangian(objective, [F], inequality=[power], easy_projector=projector)
print(res.objective, res.feasibility, res.multipliers_ineq)
```

Coupled, covariance-dependent constraint on two variables `(F, R)` — see
[`examples/node_power_constraint.py`](examples/node_power_constraint.py):

```python
def g_relay():                                    # tr(R K11 R^H) - P_relay <= 0
    K11 = F @ F.T + sigma**2 * I
    return torch.trace(R @ K11 @ R.T) - P_relay

res = augmented_lagrangian(lambda: mi(F, R), [F, R],
                           inequality=[g_relay], easy_projector=easy_proj)
```

### Batched (parallel multi-start)

For non-convex problems, run `B` seeded starts **in parallel** (one batched
solve instead of `B` sequential ones; GPU-friendly). Stack a leading batch
dimension, make the closures return `(B,)`, and use the batch-aware projector:

```python
from pga_toolbox import project_frobenius_ball_batched
from auglag_toolbox import augmented_lagrangian_batched

B = 64
Fb = (0.3 * torch.randn(B, 3, 3, dtype=torch.float64)).requires_grad_(True)

def objective():                                   # returns (B,)
    HF = torch.matmul(H, Fb)
    return torch.linalg.slogdet(I + HF @ HF.transpose(-1, -2) / 0.25).logabsdet

def power():                                        # returns (B,)
    return torch.diagonal(Fb @ W @ Fb.transpose(-1, -2), dim1=-2, dim2=-1).sum(-1) - 4.0

def projector(params):
    return [project_frobenius_ball_batched(params[0], 4.0)]

res = augmented_lagrangian_batched(objective, [Fb], inequality=[power], easy_projector=projector)
best = res.winner_result()        # global incumbent as a scalar AugLagResult
print(res.objective.shape, res.winner, best.objective)
```

Closures must be **NaN-safe** (a diverged element returns NaN, never raises) and
have independent batch elements; the projector must be batch-aware.

## Public API

| Symbol | Purpose |
| --- | --- |
| `augmented_lagrangian(objective, params, *, inequality, equality, easy_projector, inner, ...)` | PHR augmented-Lagrangian maximisation; returns `AugLagResult`. |
| `augmented_lagrangian_descent(cost, params, ...)` | Minimise `cost` via `max(-cost)`. |
| `augmented_lagrangian_batched(objective, params, ...)` | Parallel multi-start: `B` solves at once (GPU-friendly); returns `BatchedAugLagResult`. |
| `augmented_lagrangian_descent_batched(cost, params, ...)` | Batched minimisation. |
| `penalty_method(objective, params, *, mu, ...)` | Pure quadratic-penalty baseline. |
| `kkt_residuals(...)` / `kkt_residuals_batched(...)` | `(feasibility, complementarity, stationarity)` diagnostics (scalar / per-element). |
| `AugLagResult` / `BatchedAugLagResult` | Result dataclasses (residuals, multipliers, `rho`, history; batched adds `winner` + `winner_result()`). |
| `InnerSolver`, `Projector` | Type aliases for the pluggable inner solver / projector. |

## Conventions

- **Maximisation form.** Inequalities `g_c(x) <= 0`; equalities `h_k(x) = 0`.
- **Real-valued constraints.** Each `g_c` / `h_k` returns a real scalar. For
  complex parameters take the real part, e.g. `tr(F W F^H).real`.
- **Inner-solver contract.** `inner(closure, params, *, projector=None, **kw)`
  must maximise `closure()` by updating `params` in place. The objective for the
  inner solve is the PHR augmented objective built by this library.
- **In-place params.** `params` is updated to the best point found and is also
  returned in `AugLagResult.params`.
- **KKT residuals.** feasibility `max(max_c [g_c]_+, max_k |h_k|)`;
  complementarity `max_c |lambda_c g_c|`; stationarity is the projected-gradient
  map residual of the Lagrangian (complex via the Wirtinger gradient).
- **Convergence reporting.** The solve stops early once all three residuals meet
  the user-tunable tolerances `feas_tol`, `compl_tol`, `stat_tol`; then
  `res.converged is True` and `res.stop_reason == "kkt_tol"`. If the
  `outer_iters` budget is hit first, `res.converged is False` /
  `stop_reason == "max_outer_iters"` and the point is **not** a KKT point —
  always check `res.converged` (or the residuals) before trusting a solution.

## Examples

- [`examples/node_power_constraint.py`](examples/node_power_constraint.py) —
  flagship: relay node transmit-power budget under MI maximisation, with a
  reproduced figure and `.npz` results. All numbers are from the actual run.

## Known limitations

- **KKT / stationary points only.** For non-convex problems the method finds a
  KKT point; use `multistart > 1` (a non-convexity safeguard) and constraint
  scaling (normalise `g_c` by `P_j`).
- **Convexity lives in the inner projection.** Correctness of the inner SPG line
  search assumes `S_easy` is convex (a Frobenius / total-power ball is).
- **Deterministic objective.** Stochastic (e.g. fading) objectives are out of
  scope.
- **GPU-parallel multistart** is available via `augmented_lagrangian_batched`
  (batched ALM on `pga_ascent_spg_batched`); it requires NaN-safe, batch-aware
  closures and a batch-aware projector.

## Sister libraries

`auglag-toolbox` adds constraint handling to the shared optimisation core
`pga-toolbox`, used by a family of open-source linear-Gaussian-DAG information
libraries by the same author.

- [**pga-toolbox**](https://github.com/wadayama/pga-toolbox) — the inner
  projected first-order solver (fixed-step / Armijo / SPG / batched SPG).
  Constrained extensions live here in `auglag-toolbox`.
- [**gaussian-dag**](https://github.com/wadayama/gaussian-dag) — mutual
  information of linear Gaussian DAGs via the K-recursion (exact log-det MI,
  Wirtinger PGA). Paper: [arXiv:2606.06982](https://arxiv.org/abs/2606.06982).
- [**cmi-dag**](https://github.com/wadayama/cmi-dag) — conditional mutual
  information and rate regions for multi-terminal linear Gaussian DAGs.
- [**fading-dag**](https://github.com/wadayama/fading-dag) — fading-channel MI
  with SGD optimisation (ergodic / outage objectives).
- [**bussgang-dag**](https://github.com/wadayama/bussgang-dag) — Bussgang
  surrogate MI for nonlinear linear-Gaussian DAGs.

## License

MIT — see [`LICENSE`](LICENSE).

## Citation

If this toolbox underpins a publication, please cite the originating
methodology paper:

> T. Wadayama and Na Siqi, *Mutual Information Optimization via K-Recursion and
> Automatic Differentiation for Linear Gaussian Wireless Networks*,
> arXiv:2606.06982 \[cs.IT\], 2026.
