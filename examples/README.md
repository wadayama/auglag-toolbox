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

## `check_cuda.py`

Runs the batched solver on CPU and (if available) GPU and checks they agree;
also serves as a clean-environment install check. Run with
`uv run python examples/check_cuda.py`.

## `visual_abstract.py`

Draws the README visual abstract (`docs/figures/visual_abstract.png`) — a
conceptual schematic (not a data plot): panel A is the two-layer architecture
(outer augmented Lagrangian over inner pga-toolbox); panel B is the geometry of
projection onto `S_easy` plus a multiplier on the projection-free hard
constraint, converging to a KKT point. Re-run to regenerate the committed figure.
