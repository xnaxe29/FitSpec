"""Reusable Bayesian problem definitions for FitSpec.

The inference package deliberately samples a flat numerical vector.  A
:class:`PosteriorProblem` supplies the names, prior, and likelihood needed by
samplers without requiring them to know anything about stellar populations,
emission lines, or absorption systems.

For the common component-based models used by emission and absorption,
:func:`model_parameters_problem` adapts a ``ModelParameters`` + ``model_func``
pair directly.  Specialized science modules (notably the stellar
variable-projection fitter) can instead construct a PosteriorProblem from a
custom log-likelihood callable while retaining the same sampler API.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from core.parameters import ModelParameters
from inference.priors import PriorSet

__all__ = ["PosteriorProblem", "model_parameters_problem", "problem_from_fit_result", "gaussian_log_likelihood"]


def gaussian_log_likelihood(data, model, uncertainty, *, covariance=None) -> float:
    """Normalized Gaussian log likelihood for diagonal errors or covariance.

    Exactly one of ``uncertainty`` and ``covariance`` must carry the error
    model.  ``uncertainty`` may be None only when ``covariance`` is supplied.
    The normalization is retained because nested sampling uses the absolute
    likelihood to estimate the evidence.
    """
    data = np.asarray(data, dtype=float)
    model = np.asarray(model, dtype=float)
    if data.shape != model.shape:
        raise ValueError("data and model must have equal shapes.")
    residual = data - model

    if covariance is not None:
        cov = np.asarray(covariance, dtype=float)
        if cov.shape != (data.size, data.size):
            raise ValueError("covariance must have shape (n_data, n_data).")
        if not np.all(np.isfinite(cov)):
            raise ValueError("covariance must be finite.")
        sign, logdet = np.linalg.slogdet(cov)
        if sign <= 0:
            raise ValueError("covariance must be positive definite.")
        try:
            solved = np.linalg.solve(cov, residual)
        except np.linalg.LinAlgError as exc:
            raise ValueError("covariance must be nonsingular.") from exc
        return float(-0.5 * (residual @ solved + logdet + data.size * np.log(2.0 * np.pi)))

    if uncertainty is None:
        raise ValueError("uncertainty is required when covariance is not supplied.")
    sigma = np.asarray(uncertainty, dtype=float)
    if sigma.shape != data.shape:
        raise ValueError("uncertainty must have the same shape as data.")
    if np.any(sigma <= 0) or not np.all(np.isfinite(sigma)):
        raise ValueError("uncertainty must be finite and strictly positive.")
    return float(-0.5 * np.sum((residual / sigma) ** 2 + np.log(2.0 * np.pi * sigma**2)))


@dataclass
class PosteriorProblem:
    """Sampler-independent posterior definition.

    Parameters
    ----------
    parameter_names
        Ordered names corresponding to every dimension in the sample vector.
    log_likelihood
        Callable ``log_likelihood(theta) -> float``.
    priors
        A :class:`~inference.priors.PriorSet` with the same dimensionality.
    initial_position
        Optional deterministic best-fit/reference vector.  emcee uses it to
        initialize walkers; dynesty does not require it.
    metadata
        Reproducibility/provenance information supplied by the science module.
    """

    parameter_names: list[str]
    log_likelihood: Callable[[np.ndarray], float]
    priors: PriorSet
    initial_position: "np.ndarray | None" = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.parameter_names = list(self.parameter_names)
        if not self.parameter_names:
            raise ValueError("PosteriorProblem requires at least one parameter.")
        if len(set(self.parameter_names)) != len(self.parameter_names):
            raise ValueError("parameter_names must be unique.")
        if len(self.priors) != len(self.parameter_names):
            raise ValueError("priors dimensionality must match parameter_names.")
        if self.initial_position is not None:
            self.initial_position = np.asarray(self.initial_position, dtype=float)
            if self.initial_position.shape != (self.ndim,):
                raise ValueError("initial_position must have shape (ndim,).")

    @property
    def ndim(self) -> int:
        return len(self.parameter_names)

    def log_prior(self, theta) -> float:
        return float(self.priors.log_probability(theta))

    def log_probability(self, theta) -> float:
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (self.ndim,):
            raise ValueError(f"theta must have shape ({self.ndim},).")
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        ll = float(self.log_likelihood(theta))
        if not np.isfinite(ll):
            return -np.inf
        return lp + ll

    def prior_transform(self, unit_cube) -> np.ndarray:
        return self.priors.transform(unit_cube)


def model_parameters_problem(
    wave, flux, flux_unc, model_parameters: ModelParameters, model_func, *,
    mask=None, covariance=None, priors: "PriorSet | None" = None, metadata=None,
) -> PosteriorProblem:
    """Adapt a FitSpec component model into a :class:`PosteriorProblem`.

    The supplied ``model_parameters`` object is copied, so sampling never
    mutates the deterministic result or live GUI state.  Bounds become proper
    uniform priors by default; therefore every free parameter must have finite
    lower and upper bounds unless an explicit ``priors`` object is supplied.
    """
    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)
    if wave.shape != flux.shape:
        raise ValueError("wave and flux must have equal shapes.")
    mask_arr = np.ones(wave.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if mask_arr.shape != wave.shape or not np.any(mask_arr):
        raise ValueError("mask must match wave and select at least one point.")

    unc = None if flux_unc is None else np.asarray(flux_unc, dtype=float)
    if unc is not None and unc.shape != wave.shape:
        raise ValueError("flux_unc must have the same shape as wave.")

    cov_used = None
    if covariance is not None:
        cov = np.asarray(covariance, dtype=float)
        if cov.shape == (wave.size, wave.size):
            cov_used = cov[np.ix_(mask_arr, mask_arr)]
        elif cov.shape == (int(np.count_nonzero(mask_arr)),) * 2:
            cov_used = cov
        else:
            raise ValueError("covariance must describe either the full or masked spectrum.")

    params = model_parameters.copy()
    names = params.parameter_names()
    if priors is None:
        priors = PriorSet.from_model_parameters(params)

    def log_likelihood(theta):
        try:
            params.from_vector(theta)
            full_model = np.asarray(model_func(wave, params), dtype=float)
        except (ValueError, FloatingPointError, OverflowError):
            return -np.inf
        if full_model.shape != wave.shape or not np.all(np.isfinite(full_model[mask_arr])):
            return -np.inf
        try:
            return gaussian_log_likelihood(
                flux[mask_arr], full_model[mask_arr],
                None if unc is None else unc[mask_arr], covariance=cov_used,
            )
        except ValueError:
            return -np.inf

    return PosteriorProblem(
        parameter_names=names,
        log_likelihood=log_likelihood,
        priors=priors,
        initial_position=params.to_vector(),
        metadata={} if metadata is None else dict(metadata),
    )


def problem_from_fit_result(fit_result, model_func, *, priors=None, covariance=None, metadata=None) -> PosteriorProblem:
    """Reconstruct a component-model posterior problem from a saved FitResult.

    ``model_func`` is rebuilt by the owning science module from its saved/configured
    physical context (line list, redshift, resolution, tying mode, etc.).  All data,
    mask, fitted parameter state, and deterministic initialization come directly
    from the result product, so no GUI state is required.
    """
    if fit_result.flux_unc is None and covariance is None:
        raise ValueError("Posterior sampling requires flux_unc or a covariance matrix.")
    merged = dict(getattr(fit_result, "metadata", {}) or {})
    if metadata:
        merged.update(metadata)
    merged.setdefault("deterministic_method", getattr(fit_result, "method", "unknown"))
    return model_parameters_problem(
        fit_result.wave, fit_result.flux, fit_result.flux_unc, fit_result.parameters, model_func,
        mask=fit_result.mask, covariance=covariance, priors=priors, metadata=merged,
    )
