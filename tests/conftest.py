"""Pytest configuration: use float64 throughout (matches the reference smoke)."""

import torch

torch.set_default_dtype(torch.float64)
