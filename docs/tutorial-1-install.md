# Tutorial 1 — Installation and your first constrained optimisation

## Install

```bash
git clone https://github.com/wadayama/auglag-toolbox.git
cd auglag-toolbox
uv sync --extra test       # pulls pga-toolbox (git+https) and torch
uv run pytest              # should be green (SLSQP test skips without scipy)
```

During development, before `pga-toolbox` is published, inject the sibling
checkout instead of fetching over the network:

```bash
uv run --with-editable ../pga-toolbox --extra test pytest
```

## The shape of every problem

`auglag-toolbox` solves

```
maximise f(x)   s.t.   g_c(x) <= 0,   h_k(x) = 0,   x in S_easy
```

You provide:

- an **objective** closure `() -> real scalar tensor` (we *maximise* it),
- a list of **leaf params** (`requires_grad=True`),
- optional **inequality** / **equality** closures (each `() -> real scalar`),
- an optional **easy projector** onto `S_easy` (a Frobenius / power ball).

## A minimal example

Maximise a log-det objective subject to a single weighted-power inequality, with
a Frobenius ball as the easy set:

```python
import torch
from pga_toolbox import project_frobenius_ball
from auglag_toolbox import augmented_lagrangian

torch.set_default_dtype(torch.float64)

H = torch.randn(3, 3)
W = torch.diag(torch.tensor([2.5, 1.0, 0.4]))
F = torch.randn(3, 3, requires_grad=True)

def objective():
    HF = H @ F
    return torch.linalg.slogdet(torch.eye(3) + HF @ HF.T / 0.25).logabsdet

def power():
    return torch.trace(F @ W @ F.T) - 4.0      # g(x) <= 0

def projector(params):
    return [project_frobenius_ball(params[0], 4.0)]

res = augmented_lagrangian(objective, [F], inequality=[power], easy_projector=projector)
print("objective     :", res.objective)
print("feasibility    :", res.feasibility)
print("multiplier     :", res.multipliers_ineq)
print("KKT stationarity:", res.stationarity)
```

`res` is an `AugLagResult` — objective value, multipliers, KKT residuals, the
final `rho`, and a per-outer-iteration `history`. `F` is updated in place to the
solution.

Next: [Inequality constraints and multipliers](tutorial-2-inequality.md).
