"""Generic deterministic fitting: wave[mask], flux[mask], flux_unc[mask] -> FitResult.

One optimizer wrapper shared by every science module, implementing the
FitSpec fundamentals request for "a generic script that takes in
wave[mask], flux[mask], flux_unc[mask], *pars, *models". The optimizer
itself never knows whether it's fitting a stellar continuum, an
emission-line complex, or absorption Voigt components -- that is
entirely encoded in `model_func` and the ModelParameters object it is
given. Posterior sampling (Dynesty/emcee) is a separate concern living
in `inference`, not here -- this module only does bounded nonlinear
least squares.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit

from core.parameters import ModelParameters
from core.results import FitResult
from core.statistics import compute_fit_statistics


__all__ = ["fit_deterministic"]


def fit_deterministic(
    wave, flux, flux_unc, model_parameters: ModelParameters, model_func, *,
    mask=None, redshift: float = 0.0, resolution_source: "str | None" = None,
    max_function_evaluations: int = 20000, **statistics_kwargs,
) -> FitResult:
    """Fit `model_func` to a spectrum via bounded nonlinear least squares.

    Parameters
    ----------
    wave, flux, flux_unc : array-like
        Full (unmasked) spectrum arrays; `mask` selects which points are
        actually fit (None means every point).
    model_parameters : ModelParameters
        Explicit component/parameter container providing the initial
        guess (`to_vector`), bounds (`bounds`), and how fitted values are
        written back (`from_vector`).
    model_func : callable
        ``model_func(wave, model_parameters) -> np.ndarray``, evaluated
        on the *full* `wave` grid at each trial point, reading current
        parameter values from `model_parameters` (e.g. via
        `core.models.sum_components`) rather than from separate
        positional arguments -- this keeps `model_func` responsible for
        translating components into a physical model (profile shapes,
        resolution convolution, redshift, etc.), while this function only
        deals with the flat free-parameter vector the optimizer sees.
    mask : array-like of bool, optional
        Which points to fit. If None, every point is used.
    redshift, resolution_source :
        Recorded in the returned FitResult for provenance; not used by
        the optimizer itself.
    max_function_evaluations : int, default 20000
        Passed to the underlying optimizer.
    **statistics_kwargs
        Passed through to core.statistics.compute_fit_statistics (e.g.
        pix_per_resel, fit_jitter, jitter_min).

    Returns
    -------
    FitResult
    """
    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)
    flux_unc = np.asarray(flux_unc, dtype=float)
    if not (wave.shape == flux.shape == flux_unc.shape):
        raise ValueError("wave, flux, and flux_unc must have equal shapes.")

    mask = np.ones(wave.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if mask.shape != wave.shape:
        raise ValueError("mask must have the same shape as wave.")
    if not np.any(mask):
        raise ValueError("mask selects no points; nothing to fit.")

    lower_bounds, upper_bounds = model_parameters.bounds()
    initial_guess = model_parameters.to_vector()
    if initial_guess.size == 0:
        raise ValueError("model_parameters has no free parameters to fit.")

    def _wrapped(_wave_subset, *free_values):
        model_parameters.from_vector(np.array(free_values))
        full_model = np.asarray(model_func(wave, model_parameters), dtype=float)
        return full_model[mask]

    best_fit_values, covariance = curve_fit(
        _wrapped, wave[mask], flux[mask], p0=initial_guess, sigma=flux_unc[mask],
        absolute_sigma=True, bounds=(lower_bounds, upper_bounds), maxfev=max_function_evaluations,
    )

    model_parameters.from_vector(best_fit_values)
    uncertainty_values = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    parameter_uncertainties = dict(zip(model_parameters.parameter_names(), uncertainty_values))

    full_model = np.asarray(model_func(wave, model_parameters), dtype=float)
    residuals = flux[mask] - full_model[mask]
    statistics = compute_fit_statistics(
        residuals, flux_unc[mask], k_params=len(best_fit_values), **statistics_kwargs)

    return FitResult(
        parameters=model_parameters, parameter_uncertainties=parameter_uncertainties,
        wave=wave, flux=flux, flux_unc=flux_unc, mask=mask, model=full_model,
        statistics=statistics, redshift=redshift, resolution_source=resolution_source,
        method="curve_fit",
    )
