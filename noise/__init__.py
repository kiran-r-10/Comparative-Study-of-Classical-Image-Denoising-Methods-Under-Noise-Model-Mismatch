"""Noise generation and variance-stabilizing transforms (Part A)."""

from .awgn import add_awgn
from .poisson_gaussian import add_poisson_gaussian, alpha_from_peak
from .gat import (
    gat_forward,
    gat_inverse_algebraic,
    gat_inverse_closed_form,
    gat_inverse_exact_unbiased,
)
from .sigma import sigma_blind, sigma_equal_variance, sigma_read_only

__all__ = [
    "add_awgn",
    "add_poisson_gaussian",
    "alpha_from_peak",
    "gat_forward",
    "gat_inverse_algebraic",
    "gat_inverse_closed_form",
    "gat_inverse_exact_unbiased",
    "sigma_blind",
    "sigma_equal_variance",
    "sigma_read_only",
]
