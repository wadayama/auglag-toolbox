# examples

Runnable, self-contained demonstrations. All printed/plotted/saved numbers come
from the actual optimiser run -- nothing is hard-coded.

```sh
uv run --extra examples python examples/node_power_constraint.py
```

(During development, before `pga-toolbox` is published:
`uv run --with-editable ../pga-toolbox --extra examples python examples/node_power_constraint.py`.)

## `node_power_constraint.py` (flagship)

Two-hop amplify-and-forward relay `X --H1--> [relay R] --H2--> Y`. Maximise the
end-to-end mutual information by jointly shaping the source precoder `F` and the
relay matrix `R`, subject to a **relay node transmit-power** budget

```
tr( R K11 R^H ) <= P_relay,   K11 = F F^H + sigma^2 I
```

— nonlinear, coupled (through `K11`), covariance-dependent, no closed-form
projection — handled by the augmented Lagrangian. The source Frobenius budget
`||F||_F^2 <= Pf` is the easy constraint handled by projection inside the inner
SPG solve. This is the "future work" constraint from the gaussian-dag paper,
demonstrated end-to-end.

Outputs (written next to the script):
- `node_power_constraint_results.npz` — config, final KKT residuals, multiplier,
  per-outer-iteration history arrays.
- `node_power_constraint.pdf` — KKT-residual convergence (left) and relay power
  vs budget: projection-only runs away while the ALM solution sits at the cap
  (right, log scale).

Expected outcome (the constraint is **active**): feasibility ~3e-7,
complementarity ~2e-8, stationarity ~1e-7, multiplier `lambda ~ 0.088`, relay
power equal to the cap, at finite `rho = 1.0`.

## `visual_abstract.py`

Generates the README visual abstract (`docs/figures/visual_abstract.png`) from
actual runs: panel A is the coupled covariance constraint converging to a KKT
point; panel B is batched multi-start escaping local optima on a multimodal
problem. Re-run to regenerate the committed figure.
