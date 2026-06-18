"""Shared test helpers (mirrors the smoke E1-E5 problem builders)."""

from __future__ import annotations

import torch


def randmat(d, seed, scale=1.0):
    """Deterministic real d x d Gaussian matrix."""
    g = torch.Generator().manual_seed(seed)
    return scale * torch.randn(d, d, generator=g, dtype=torch.float64)


def randc(d, seed, scale=1.0):
    """Deterministic complex d x d Gaussian matrix."""
    g = torch.Generator().manual_seed(seed)
    return scale * torch.complex(
        torch.randn(d, d, generator=g, dtype=torch.float64),
        torch.randn(d, d, generator=g, dtype=torch.float64),
    )


def logdet_pd(A):
    """log det of a Hermitian PD matrix (symmetrised against FP drift)."""
    A = 0.5 * (A + A.transpose(-1, -2).conj())
    return torch.linalg.slogdet(A).logabsdet
