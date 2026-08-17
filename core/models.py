"""Generic numerical model-building blocks and multi-component evaluation.

Physical, species-specific models (which lines, which library) live in
the science modules (stellar/emission/absorption). This module holds
only the numerical primitives shared by all of them (Gaussian, Voigt)
and a generic driver that sums an arbitrary number of components defined
by a core.parameters.ModelParameters object using an injected per-
component model function -- so the driver itself never needs to know
whether it's summing emission-line Gaussians or absorption Voigt
profiles.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.special import wofz

from core.parameters import ModelParameters


__all__ = ["gaussian", "voigt_profile", "sum_components"]


def gaussian(wave, amplitude, mean, sigma):
    """Gaussian line profile, peak-normalized to `amplitude` (not area)."""
    wave = np.asarray(wave, dtype=float)
    if sigma <= 0:
        raise ValueError("sigma must be positive.")
    return amplitude * np.exp(-0.5 * ((wave - mean) / sigma) ** 2)


def voigt_profile(wave, amplitude, mean, sigma, gamma):
    """Voigt line profile via the Faddeeva function, peak-normalized to `amplitude`.

    For physical (optical-depth/column-density) absorption-line Voigt
    profiles, build the parameterization directly rather than through
    this peak-normalized convenience wrapper -- this is meant as a
    generic, reusable numerical primitive.

    Parameters
    ----------
    sigma : float
        Gaussian component standard deviation (same units as `wave`).
    gamma : float
        Lorentzian half-width at half-maximum (same units as `wave`).
    """
    wave = np.asarray(wave, dtype=float)
    if sigma <= 0:
        raise ValueError("sigma must be positive.")
    if gamma < 0:
        raise ValueError("gamma must be non-negative.")

    scale = sigma * np.sqrt(2.0)
    z = ((wave - mean) + 1j * gamma) / scale
    profile = np.real(wofz(z))
    peak = np.real(wofz(1j * gamma / scale))
    if peak <= 0:
        raise ValueError("Degenerate Voigt profile (non-positive peak); check sigma/gamma.")
    return amplitude * profile / peak


def sum_components(wave, model_parameters: ModelParameters, component_func: Callable, **shared_kwargs) -> np.ndarray:
    """Sum every component's contribution using an injected per-component model function.

    Parameters
    ----------
    wave : array-like
        Wavelength grid to evaluate on.
    model_parameters : ModelParameters
        Explicit component/parameter container (see core.parameters).
    component_func : callable
        ``component_func(wave, **parameter_values, **shared_kwargs) ->
        np.ndarray``, called once per component with that component's
        current parameter values passed as keyword arguments by name
        (including any fixed parameters, at their current fixed value).
    **shared_kwargs
        Extra keyword arguments passed to every call of `component_func`
        (e.g. redshift, a ResolutionModel).

    Returns
    -------
    np.ndarray
        Sum of every component's contribution, same length as `wave`.
    """
    wave = np.asarray(wave, dtype=float)
    total = np.zeros_like(wave)
    for component in model_parameters.components:
        parameter_values = {parameter.name: parameter.value for parameter in component.parameters}
        total = total + component_func(wave, **parameter_values, **shared_kwargs)
    return total
