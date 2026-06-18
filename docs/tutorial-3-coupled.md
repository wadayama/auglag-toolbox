# Tutorial 3 — Covariance-dependent, coupled constraints (node power)

The point of this library is constraints that have **no closed-form
projection** — they are nonlinear, couple several variables, and depend on
covariances. The canonical case is a per-node transmit-power budget in a relay
network.

## The problem

Two-hop amplify-and-forward relay, source precoder `F` and relay matrix `R`.
The relay's transmit covariance is `K11 = F F^H + sigma^2 I`, so its node power

```
tr( R K11 R^H ) <= P_relay
```

depends on `F` (through `K11`) **and** `R` — coupled and covariance-dependent.
There is no projection for this set; it goes into the multiplier loop. The
source budget `||F||_F^2 <= Pf` *does* have a projection, so it stays easy.

## Setting it up

```python
import torch
from pga_toolbox import project_frobenius_ball
from auglag_toolbox import augmented_lagrangian

torch.set_default_dtype(torch.float64)
d, sigma, Pf, P_relay = 2, 0.5, 4.0, 9.0
H2 = torch.randn(d, d)
I = torch.eye(d)

F = torch.randn(d, d, requires_grad=True)
R = torch.randn(d, d, requires_grad=True)

def mi(F, R):                                   # end-to-end mutual information
    G = H2 @ R @ F
    C = sigma**2 * (H2 @ R @ R.T @ H2.T) + sigma**2 * I
    sld = lambda A: torch.linalg.slogdet(0.5 * (A + A.T)).logabsdet
    return sld(G @ G.T + C) - sld(C)

def g_relay():                                  # hard: coupled, covariance-dependent
    K11 = F @ F.T + sigma**2 * I
    return torch.trace(R @ K11 @ R.T) - P_relay

def easy_proj(params):                          # easy: project F only, R free
    F, R = params
    return [project_frobenius_ball(F, Pf), R]

res = augmented_lagrangian(lambda: mi(F, R), [F, R],
                           inequality=[g_relay], easy_projector=easy_proj)
```

## Key points

- **Multiple params.** `params = [F, R]` are optimised jointly; gradients of both
  the objective and the constraint flow by autograd. No gradients are derived by
  hand.
- **Easy projector touches a subset.** Here it projects `F` and returns `R`
  unchanged — the easy set constrains only `F`.
- **Activity by construction.** MI saturates in the relay gain, so the
  unconstrained optimum drives the relay power away to infinity; a finite
  `P_relay` therefore binds. Expect `res.multipliers_ineq[0] > 0` and relay
  power equal to the cap.

This is exactly the flagship [`examples/node_power_constraint.py`](../examples/node_power_constraint.py).

Next: [Penalty vs ALM, and KKT diagnostics](tutorial-4-penalty-vs-alm.md).
