"""Instrumental spectral resolution: lookup, and application to models.

Resolution must never be silently ignored (see design principles). This
module distinguishes two genuinely different ways resolution enters a
fit, rather than forcing both through one numerical convolution:

1. **Single-line models** (a Gaussian or Voigt line profile): convolving
   with an instrumental Gaussian needs no numerical convolution at all.
   Gaussian (*) Gaussian is a Gaussian, and a Voigt profile is already
   Gaussian (*) Lorentzian, so convolving a Voigt line with an
   instrumental Gaussian is *exactly* equivalent to using a larger
   effective Doppler parameter and leaving the Lorentzian damping
   untouched (see :func:`effective_doppler_b_kms`; this is the standard
   parameterization, e.g. PyAstronomy's VoigtAstroP). Use
   :func:`combine_gaussian_sigma` / :func:`effective_doppler_b_kms`
   for these, not :func:`convolve_variable_gaussian`.

2. **Full-spectrum / template models** (e.g. stellar population
   templates), where resolution genuinely varies across the fitted
   bandpass: these need real numerical convolution with a
   wavelength-dependent kernel width. :func:`convolve_variable_gaussian`
   implements the standard direct-summation approach (as in Cappellari's
   ppxf_util.gaussian_filter1d / Westfall et al. 2019), evaluated
   directly on the target (data) wavelength grid -- which also handles
   "the convolved model must be resampled onto the input spectrum's
   grid" in the same pass, rather than as a separate rebinning step.

A :class:`ResolutionModel` wraps *where the resolution came from*
(a user-supplied table, a user-supplied constant FWHM in km/s, or an
explicit, file-backed fallback) so that whenever a fallback is actually
used, a ``DefaultResolutionWarning`` is raised immediately -- not
deferred, not silent, and not something a caller can accidentally not
see. There is deliberately no built-in fabricated default table here:
:meth:`ResolutionModel.default_fallback` requires an explicit file path
(e.g. pointed to by config), since inventing plausible-looking
instrument resolution numbers would be a silent correctness bug of
exactly the kind this module exists to prevent.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from astropy import constants as const

from core.rebinning import rebin_spectrum

_C_KMS = const.c.to("km/s").value
_FWHM_TO_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))


__all__ = [
    "DefaultResolutionWarning",
    "sigma_angstrom_from_R",
    "sigma_angstrom_from_fwhm_kms",
    "combine_gaussian_sigma",
    "effective_doppler_b_kms",
    "ResolutionModel",
    "convolve_variable_gaussian",
]


class DefaultResolutionWarning(UserWarning):
    """Raised whenever a fallback/default resolution is actually used."""


def sigma_angstrom_from_R(wave, R):
    """Instrumental Gaussian sigma [Angstrom], assuming FWHM = wave / R."""
    wave = np.asarray(wave, dtype=float)
    R = np.asarray(R, dtype=float)
    if np.any(R <= 0) or not np.all(np.isfinite(R)):
        raise ValueError("R must be finite and strictly positive.")
    return (wave / R) / _FWHM_TO_SIGMA


def sigma_angstrom_from_fwhm_kms(wave, fwhm_kms):
    """Instrumental Gaussian sigma [Angstrom] from a constant FWHM [km/s]."""
    wave = np.asarray(wave, dtype=float)
    if fwhm_kms <= 0 or not np.isfinite(fwhm_kms):
        raise ValueError("fwhm_kms must be finite and strictly positive.")
    sigma_kms = fwhm_kms / _FWHM_TO_SIGMA
    return wave * sigma_kms / _C_KMS


def combine_gaussian_sigma(*sigmas):
    """Quadrature-combine independent Gaussian sigmas (any consistent unit).

    Convolving independent Gaussian broadening mechanisms (e.g. intrinsic
    line width and instrumental resolution) combines as
    ``sigma_total = sqrt(sum(sigma_i**2))``.
    """
    total_variance = sum(np.asarray(s, dtype=float) ** 2 for s in sigmas)
    return np.sqrt(total_variance)


def effective_doppler_b_kms(intrinsic_b_kms, instrumental_sigma_kms):
    """Effective Doppler parameter for a Voigt line broadened by resolution.

    A Voigt profile is Gaussian(b) (*) Lorentzian(gamma). Convolving with
    an additional independent instrumental Gaussian of dispersion
    ``instrumental_sigma_kms`` is exactly equivalent to replacing the
    intrinsic Doppler parameter ``b`` (with ``sigma = b / sqrt(2)``) by
    an effective ``b_eff``, and leaving ``gamma`` unchanged -- no
    numerical convolution needed. This is the standard parameterization
    (e.g. PyAstronomy's VoigtAstroP "R" parameter).

    Parameters
    ----------
    intrinsic_b_kms : float or array-like
        Intrinsic (thermal + turbulent) Doppler parameter, km/s.
    instrumental_sigma_kms : float or array-like
        Instrumental Gaussian dispersion, km/s (i.e. ``sigma``, not
        ``b`` -- convert with ``sigma = fwhm / (2*sqrt(2*ln2))`` first
        if starting from an instrumental FWHM).

    Returns
    -------
    b_eff_kms : float or np.ndarray
    """
    intrinsic_b_kms = np.asarray(intrinsic_b_kms, dtype=float)
    instrumental_sigma_kms = np.asarray(instrumental_sigma_kms, dtype=float)
    instrumental_b_kms = instrumental_sigma_kms * np.sqrt(2.0)
    return np.sqrt(intrinsic_b_kms ** 2 + instrumental_b_kms ** 2)


@dataclass
class ResolutionModel:
    """Where instrumental resolution comes from, and how to evaluate it.

    Construct via one of the classmethods rather than directly.

    Attributes
    ----------
    sigma_func : callable
        ``sigma_func(wave) -> sigma_angstrom``, vectorized.
    source : str
        Human-readable description of where this came from, e.g.
        ``"table:/path/to/resolution.dat"``, ``"constant_fwhm_kms=120.0"``,
        or ``"default_fallback:/path/to/resolution_average.dat"``.
    is_default : bool
        True if this resolution model is a fallback rather than something
        the user actually supplied for this spectrum.
    """

    sigma_func: "callable"
    source: str
    is_default: bool = False

    def sigma_angstrom(self, wave):
        return np.asarray(self.sigma_func(np.asarray(wave, dtype=float)), dtype=float)

    def fwhm_angstrom(self, wave):
        return self.sigma_angstrom(wave) * _FWHM_TO_SIGMA

    def R(self, wave):
        wave = np.asarray(wave, dtype=float)
        return wave / self.fwhm_angstrom(wave)

    @classmethod
    def from_table(cls, wave_grid, *, R=None, fwhm_angstrom=None, sigma_angstrom=None, source: str = "table"):
        """Build from a wavelength-dependent resolution table.

        Exactly one of ``R``, ``fwhm_angstrom``, or ``sigma_angstrom`` must
        be given, evaluated at ``wave_grid``. Interpolation is linear in
        wavelength (constant extrapolation beyond the table's range).
        """
        wave_grid = np.asarray(wave_grid, dtype=float)
        order = np.argsort(wave_grid)
        wave_grid = wave_grid[order]

        given = [name for name, value in (("R", R), ("fwhm_angstrom", fwhm_angstrom), ("sigma_angstrom", sigma_angstrom)) if value is not None]
        if len(given) != 1:
            raise ValueError("Exactly one of R, fwhm_angstrom, or sigma_angstrom must be given.")

        if R is not None:
            sigma_grid = sigma_angstrom_from_R(wave_grid, np.asarray(R, dtype=float)[order])
        elif fwhm_angstrom is not None:
            sigma_grid = np.asarray(fwhm_angstrom, dtype=float)[order] / _FWHM_TO_SIGMA
        else:
            sigma_grid = np.asarray(sigma_angstrom, dtype=float)[order]

        def sigma_func(wave):
            return np.interp(wave, wave_grid, sigma_grid)

        return cls(sigma_func=sigma_func, source=source)

    @classmethod
    def from_constant_fwhm_kms(cls, fwhm_kms: float):
        """Build from a single Gaussian FWHM [km/s] applied across the whole spectrum."""
        def sigma_func(wave):
            return sigma_angstrom_from_fwhm_kms(wave, fwhm_kms)
        return cls(sigma_func=sigma_func, source=f"constant_fwhm_kms={fwhm_kms}")

    @classmethod
    def from_file(cls, path, *, wave_col: str = "lambda_A", R_col: str = "R", delimiter: str = ",", source: "str | None" = None):
        """Build from a delimited resolution-table file (wavelength, R)."""
        table = np.genfromtxt(path, delimiter=delimiter, names=True, encoding=None, dtype=float)
        if wave_col not in table.dtype.names or R_col not in table.dtype.names:
            raise ValueError(f"{path}: expected columns {wave_col!r} and {R_col!r}; found {table.dtype.names!r}.")
        return cls.from_table(
            np.asarray(table[wave_col], dtype=float), R=np.asarray(table[R_col], dtype=float),
            source=source if source is not None else f"table:{path}",
        )

    @classmethod
    def default_fallback(cls, path, **kwargs):
        """Build the fallback resolution model, warning loudly that it's a fallback.

        ``path`` must point to an actual default resolution table file --
        there is no built-in fabricated data. Callers that end up using
        this (i.e. because the user provided no resolution information
        for their spectrum) must not swallow the warning this raises.
        """
        model = cls.from_file(path, source=f"default_fallback:{path}", **kwargs)
        model.is_default = True
        warnings.warn(
            "No spectral resolution was provided for this spectrum. Falling back to the "
            f"default resolution table at {path}. Fitted line widths and any resolution-"
            "dependent quantities will be biased if this default does not match the "
            "actual instrument used.",
            DefaultResolutionWarning,
            stacklevel=2,
        )
        return model


def convolve_variable_gaussian(model_wave, model_flux, target_wave, sigma_angstrom_at_target, *, truncate_sigma: float = 4.0):
    """Convolve a model with a wavelength-varying Gaussian kernel, resampled onto target_wave.

    Standard direct-summation approach for matching template resolution
    to data resolution when the kernel width varies across the bandpass
    (e.g. Cappellari's ppxf_util.gaussian_filter1d / Westfall et al. 2019).
    Evaluates the convolution integral directly on ``target_wave``, so
    this also performs "resample the convolved model onto the observed
    wavelength grid" in the same pass.

    Parameters
    ----------
    model_wave, model_flux : array-like
        The model, on its own (possibly finer, possibly irregular)
        wavelength grid. Must be sorted ascending.
    target_wave : array-like
        Wavelength grid to evaluate the convolved model on (typically the
        observed spectrum's grid).
    sigma_angstrom_at_target : float or array-like
        Gaussian kernel standard deviation [Angstrom] at each point of
        ``target_wave`` (e.g. from ``ResolutionModel.sigma_angstrom``).
        If the intrinsic model also has nonzero width, combine it in
        first via :func:`combine_gaussian_sigma`.
    truncate_sigma : float, default 4.0
        Kernel is evaluated out to +/- this many sigma; contributions
        beyond that are neglected.

    Returns
    -------
    np.ndarray
        Convolved model flux, same length as ``target_wave``.

    Notes
    -----
    This is the standard direct-summation algorithm, adequate for typical
    spectral fitting use. For very large spectra where performance
    matters, an FFT-based variable-sigma method exists (Cappellari 2023,
    ppxf_util.varsmooth) and could replace this implementation without
    changing its interface.
    """
    model_wave = np.asarray(model_wave, dtype=float)
    model_flux = np.asarray(model_flux, dtype=float)
    target_wave = np.asarray(target_wave, dtype=float)
    sigma = np.broadcast_to(np.asarray(sigma_angstrom_at_target, dtype=float), target_wave.shape)

    if model_wave.ndim != 1 or model_flux.shape != model_wave.shape:
        raise ValueError("model_wave and model_flux must be one-dimensional and equal length.")
    if np.any(np.diff(model_wave) <= 0):
        raise ValueError("model_wave must be strictly increasing.")
    if np.any(sigma < 0) or not np.all(np.isfinite(sigma)):
        raise ValueError("sigma_angstrom_at_target must be finite and non-negative.")

    # Trapezoidal weights for the model's native grid, for accurate
    # quadrature of the convolution integral even on an irregular grid.
    edges = np.empty(model_wave.size + 1)
    edges[1:-1] = 0.5 * (model_wave[:-1] + model_wave[1:])
    edges[0] = model_wave[0] - 0.5 * (model_wave[1] - model_wave[0])
    edges[-1] = model_wave[-1] + 0.5 * (model_wave[-1] - model_wave[-2])
    pixel_width = np.diff(edges)

    # IMPORTANT: do not initialize uncovered/under-resolved samples to zero.
    # A very narrow requested kernel (e.g. sigma_star ~ 1 km/s in the UV)
    # can be much narrower than the native template sampling.  In that case
    # +/- truncate_sigma*sigma may contain zero or one native template point.
    # Zero is not a physical convolution result; the correct limiting behavior
    # is simply interpolation of the native model because the additional
    # broadening is unresolved on that grid.
    result = np.empty(target_wave.size, dtype=float)
    for j, (center, sig) in enumerate(zip(target_wave, sigma)):
        if sig == 0:
            result[j] = np.interp(center, model_wave, model_flux)
            continue
        lower = np.searchsorted(model_wave, center - truncate_sigma * sig, side="left")
        upper = np.searchsorted(model_wave, center + truncate_sigma * sig, side="right")

        # Fewer than two native samples cannot support a numerical convolution.
        # Fall back to linear interpolation rather than leaving a zero-valued
        # hole in the transformed stellar template.
        if upper - lower < 2:
            result[j] = np.interp(center, model_wave, model_flux)
            continue

        local_wave = model_wave[lower:upper]
        local_flux = model_flux[lower:upper]
        local_width = pixel_width[lower:upper]
        kernel = np.exp(-0.5 * ((local_wave - center) / sig) ** 2)
        weights = kernel * local_width
        normalization = np.sum(weights)
        if not np.isfinite(normalization) or normalization <= 0:
            result[j] = np.interp(center, model_wave, model_flux)
            continue
        value = np.sum(local_flux * weights) / normalization
        # A pathological numerical convolution must never silently create a
        # zero trough.  Interpolation is the well-defined narrow-kernel limit.
        result[j] = value if np.isfinite(value) else np.interp(center, model_wave, model_flux)

    return result
