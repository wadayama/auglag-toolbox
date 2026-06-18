"""CUDA verification: the batched solver must agree on CPU and GPU.

Runs ``augmented_lagrangian_batched`` on the same problem on CPU and (if a CUDA
device is present) on GPU, and checks that the global best objective agrees. Also
doubles as a clean-environment install check: if ``uv sync`` resolved the
``pga-toolbox`` git dependency, this script imports and runs.

Run on a machine with a GPU:
    uv sync
    uv run python examples/check_cuda.py
"""

from __future__ import annotations

import torch
from pga_toolbox import project_frobenius_ball_batched

from auglag_toolbox import augmented_lagrangian_batched

torch.set_default_dtype(torch.float64)


def run(device):
    d, sigma, P, B = 3, 0.5, 4.0, 8
    H = torch.randn(d, d, generator=torch.Generator().manual_seed(1),
                    dtype=torch.float64).to(device)
    W = torch.diag(torch.tensor([2.5, 1.0, 0.4], dtype=torch.float64)).to(device)
    I = torch.eye(d, dtype=torch.float64, device=device)
    Fb = torch.stack([
        0.3 * torch.randn(d, d, generator=torch.Generator().manual_seed(100 + b),
                          dtype=torch.float64)
        for b in range(B)
    ]).to(device).requires_grad_(True)

    def obj():
        HF = torch.matmul(H, Fb)
        return torch.linalg.slogdet(I + HF @ HF.transpose(-1, -2) / sigma**2).logabsdet

    def g1():
        FWF = torch.matmul(torch.matmul(Fb, W), Fb.transpose(-1, -2))
        return torch.diagonal(FWF, dim1=-2, dim2=-1).sum(-1) - 2.0

    return augmented_lagrangian_batched(
        obj, [Fb], inequality=[g1],
        easy_projector=lambda ps: [project_frobenius_ball_batched(ps[0], P)])


def main():
    cpu = run(torch.device("cpu"))
    print("CPU   best=%.10f  feas=%.2e  converged=%s"
          % (cpu.objective.max(), cpu.feasibility.amax(), bool(cpu.converged.all())))

    if torch.cuda.is_available():
        gpu = run(torch.device("cuda"))
        print("CUDA  best=%.10f  feas=%.2e  converged=%s"
              % (gpu.objective.max(), gpu.feasibility.amax(), bool(gpu.converged.all())))
        agree = abs(float(cpu.objective.max()) - float(gpu.objective.max())) < 1e-6
        print("CPU/GPU agree (best obj within 1e-6):", agree)
    else:
        print("no CUDA device available; CPU result above. "
              "The batched path is device-agnostic (tensors inherit device).")


if __name__ == "__main__":
    main()
