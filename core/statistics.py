"""Generic goodness-of-fit and model-selection statistics.

This module is FitSpec's single implementation of chi-square, reduced
chi-square, information criteria (BIC/AIC/AICc), and the supporting
quantities they need (effective sample size, error-jitter scale, -2 ln L).
It is deliberately independent of any particular science module: stellar,
emission, and absorption fits all call the same functions here so their
statistics are directly comparable, instead of each maintaining its own
copy of the chi-square/BIC math.

Explicitly NOT here: searching over a number of model components and
comparing their statistics (that is a model-selection *driver*, living in
``inference``), and any per-component significance filtering that depends
on a specific model's parameter layout (that lives with the relevant
science module). This module only turns residuals into numbers; it does
not decide how many components a model should have.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


__all__ = [
    "FitStatistics",
    "chi_square",
    "reduced_chi_square",
    "effective_sample_size",
    "estimate_error_jitter_scale",
    "neg2_log_likelihood",
    "bayesian_information_criterion",
    "akaike_information_criterion",
    "compute_fit_statistics",
]


@dataclass
class FitStatistics:
    """Bundled goodness-of-fit and model-selection statistics.

    Attributes
    ----------
    n_data : int
        Number of data points actually used (after masking).
    n_eff : float
        Effective sample size after correcting for oversampling
        (``n_data / pix_per_resel``); equals ``n_data`` if
        ``pix_per_resel == 1``.
    k_params : int
        Number of free model parameters.
    dof : int
        Degrees of freedom, ``n_data - k_params``.
    chi_square : float
        Unweighted-by-jitter chi-square, using the input uncertainties
        as given.
    reduced_chi_square : float
        ``chi_square / dof``.
    jitter_scale : float
        Multiplicative scaling ``s`` applied to the input uncertainties
        (``sigma_eff = s * sigma``) before computing the likelihood-based
        quantities below. 1.0 if jitter fitting was disabled.
    neg2_log_likelihood : float
        ``-2 ln L`` under a Gaussian likelihood with ``sigma_eff``.
    bic : float
        Bayesian Information Criterion.
    aic : float
        Akaike Information Criterion.
    aicc : float
        Small-sample-corrected AIC. NaN if the sample size is too small
        relative to ``k_params`` for the correction to be defined
        (requires ``n_data - k_params - 1 > 0``).
    """

    n_data: int
    n_eff: float
    k_params: int
    dof: int
    chi_square: float
    reduced_chi_square: float
    jitter_scale: float
    neg2_log_likelihood: float
    bic: float
    aic: float
    aicc: float


def chi_square(residuals, uncertainty, mask=None) -> float:
    """Chi-square statistic, ``sum(((residuals)/uncertainty)**2)``.

    Parameters
    ----------
    residuals : array-like
        Data minus model.
    uncertainty : array-like
        1-sigma uncertainty on each residual. Must be the same length as
        ``residuals``.
    mask : array-like of bool, optional
        If given, only points where ``mask`` is True are included.
    """
    residuals = np.asarray(residuals, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    if residuals.shape != uncertainty.shape:
        raise ValueError("residuals and uncertainty must have the same shape.")
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        residuals = residuals[mask]
        uncertainty = uncertainty[mask]
    if residuals.size == 0:
        raise ValueError("No data points remain after masking.")
    if np.any(uncertainty <= 0) or not np.all(np.isfinite(uncertainty)):
        raise ValueError("uncertainty must be finite and strictly positive everywhere used.")
    return float(np.sum((residuals / uncertainty) ** 2))


def reduced_chi_square(chi2: float, n_data: int, k_params: int) -> float:
    """Reduced chi-square, ``chi2 / (n_data - k_params)``.

    Raises
    ------
    ValueError
        If the degrees of freedom ``n_data - k_params`` is not positive.
    """
    dof = n_data - k_params
    if dof <= 0:
        raise ValueError(
            f"Non-positive degrees of freedom (n_data={n_data}, k_params={k_params}); "
            "reduced chi-square is undefined."
        )
    return chi2 / dof


def effective_sample_size(n_data: int, pix_per_resel: float = 1.0) -> float:
    """Effective, decorrelated sample size, ``n_data / pix_per_resel``.

    Oversampled or correlated pixels (e.g. several detector pixels per
    resolution element) otherwise inflate BIC's ``ln(N)`` penalty beyond
    what the actual independent information content supports.

    Parameters
    ----------
    n_data : int
        Raw number of data points.
    pix_per_resel : float, default 1.0
        Number of (correlated) pixels per resolution element. Must be
        >= 1.
    """
    if pix_per_resel < 1:
        raise ValueError("pix_per_resel must be >= 1.")
    return n_data / pix_per_resel


def estimate_error_jitter_scale(chi2: float, n_data: int, jitter_min: "float | None" = 1.0) -> float:
    """Maximum-likelihood multiplicative error-jitter scale ``s``.

    Under a Gaussian likelihood with ``sigma_eff = s * sigma``, the MLE
    for ``s`` given the (unscaled) chi-square is ``s = sqrt(chi2 / n_data)``.
    Fitting this lets quoted uncertainties self-calibrate against the
    actual scatter of the residuals, rather than BIC/AIC penalizing
    additional model components simply because quoted uncertainties were
    too small.

    Parameters
    ----------
    chi2 : float
        Chi-square computed with the *unscaled* input uncertainties.
    n_data : int
        Number of data points chi2 was computed from.
    jitter_min : float or None, default 1.0
        Floor on the returned scale. The default of 1.0 means
        uncertainties are only ever inflated, never shrunk below their
        quoted values; pass None to allow shrinking, or 0.0 for no floor.
    """
    if n_data <= 0:
        raise ValueError("n_data must be positive.")
    scale = np.sqrt(chi2 / n_data)
    if jitter_min is not None:
        scale = max(scale, jitter_min)
    return float(scale)


def neg2_log_likelihood(residuals, uncertainty, jitter_scale: float = 1.0, mask=None, full: bool = True) -> float:
    """``-2 ln L`` under an independent Gaussian likelihood.

    Parameters
    ----------
    residuals, uncertainty : array-like
        As in :func:`chi_square`; ``uncertainty`` is the unscaled, quoted
        1-sigma uncertainty.
    jitter_scale : float, default 1.0
        Multiplicative scale ``s`` applied to ``uncertainty`` before
        evaluating the likelihood (see :func:`estimate_error_jitter_scale`).
    mask : array-like of bool, optional
        As in :func:`chi_square`.
    full : bool, default True
        If True, include the normalization term
        ``sum(log(2 * pi * sigma_eff**2))`` so the result is the true
        ``-2 ln L`` (required for BIC/AIC to be meaningful, since the
        normalization depends on ``jitter_scale``). If False, returns
        the scaled chi-square term alone.
    """
    residuals = np.asarray(residuals, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    if residuals.shape != uncertainty.shape:
        raise ValueError("residuals and uncertainty must have the same shape.")
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        residuals = residuals[mask]
        uncertainty = uncertainty[mask]
    if jitter_scale <= 0:
        raise ValueError("jitter_scale must be positive.")

    sigma_eff = jitter_scale * uncertainty
    chi2_term = np.sum((residuals / sigma_eff) ** 2)
    if not full:
        return float(chi2_term)
    normalization_term = np.sum(np.log(2.0 * np.pi * sigma_eff ** 2))
    return float(chi2_term + normalization_term)


def bayesian_information_criterion(neg2lnL: float, k_params: int, n_data: float) -> float:
    """BIC = -2 ln L + k_params * ln(n_data).

    ``n_data`` may be an effective sample size (see
    :func:`effective_sample_size`) rather than the raw pixel count.
    """
    if n_data <= 0:
        raise ValueError("n_data must be positive.")
    return neg2lnL + k_params * np.log(n_data)


def akaike_information_criterion(
    neg2lnL: float, k_params: int, n_data: "float | None" = None, corrected: bool = False,
) -> float:
    """AIC = -2 ln L + 2 * k_params, or the small-sample-corrected AICc.

    Parameters
    ----------
    neg2lnL : float
        -2 ln L, e.g. from :func:`neg2_log_likelihood`.
    k_params : int
        Number of free model parameters.
    n_data : float, optional
        Sample size (raw or effective); required if ``corrected=True``.
    corrected : bool, default False
        If True, apply the AICc small-sample correction
        ``+ 2*k*(k+1) / (n_data - k - 1)``, recommended whenever
        ``n_data`` is not much larger than ``k_params`` (as is common
        when searching a small number of model components). Returns NaN
        if ``n_data - k_params - 1 <= 0``, where the correction is
        undefined.
    """
    aic = neg2lnL + 2 * k_params
    if not corrected:
        return aic
    if n_data is None:
        raise ValueError("n_data is required when corrected=True.")
    denominator = n_data - k_params - 1
    if denominator <= 0:
        return float("nan")
    return aic + (2 * k_params * (k_params + 1)) / denominator


def compute_fit_statistics(
    residuals, uncertainty, k_params: int, *, mask=None, pix_per_resel: float = 1.0,
    fit_jitter: bool = True, jitter_min: "float | None" = 1.0, use_neff_for_bic: bool = True,
) -> FitStatistics:
    """Compute the full standard set of fit statistics in one call.

    This is the entry point every science module should use after a fit,
    so stellar/emission/absorption results are always directly comparable.

    Parameters
    ----------
    residuals, uncertainty : array-like
        Data minus model, and its 1-sigma uncertainty (unscaled/quoted).
    k_params : int
        Number of free model parameters.
    mask : array-like of bool, optional
        Restrict to a subset of points.
    pix_per_resel : float, default 1.0
        Passed to :func:`effective_sample_size`.
    fit_jitter : bool, default True
        Whether to fit a multiplicative error-jitter scale (see
        :func:`estimate_error_jitter_scale`) before computing the
        likelihood-based quantities. If False, ``jitter_scale`` is fixed
        at 1.0.
    jitter_min : float or None, default 1.0
        Passed to :func:`estimate_error_jitter_scale`.
    use_neff_for_bic : bool, default True
        If True, BIC and AICc use the effective sample size ``n_eff``
        rather than the raw ``n_data``.

    Returns
    -------
    FitStatistics
    """
    residuals = np.asarray(residuals, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    if mask is not None:
        mask_arr = np.asarray(mask, dtype=bool)
        residuals_used = residuals[mask_arr]
        uncertainty_used = uncertainty[mask_arr]
    else:
        residuals_used = residuals
        uncertainty_used = uncertainty

    n_data = residuals_used.size
    chi2 = chi_square(residuals_used, uncertainty_used)
    reduced_chi2 = reduced_chi_square(chi2, n_data, k_params)
    n_eff = effective_sample_size(n_data, pix_per_resel)

    jitter_scale = estimate_error_jitter_scale(chi2, n_data, jitter_min) if fit_jitter else 1.0
    neg2lnL = neg2_log_likelihood(residuals_used, uncertainty_used, jitter_scale=jitter_scale, full=True)

    bic_n = n_eff if use_neff_for_bic else n_data
    bic = bayesian_information_criterion(neg2lnL, k_params, bic_n)
    aic = akaike_information_criterion(neg2lnL, k_params)
    aicc = akaike_information_criterion(neg2lnL, k_params, n_data=bic_n, corrected=True)

    return FitStatistics(
        n_data=int(n_data), n_eff=float(n_eff), k_params=int(k_params), dof=int(n_data - k_params),
        chi_square=float(chi2), reduced_chi_square=float(reduced_chi2), jitter_scale=float(jitter_scale),
        neg2_log_likelihood=float(neg2lnL), bic=float(bic), aic=float(aic), aicc=float(aicc),
    )
