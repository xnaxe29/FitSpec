"""Numerical primitives specific to emission-line profiles.

The generic peak-normalized Gaussian lives in ``core.models``; emission
lines are conventionally parameterized by *integrated flux* rather than
peak amplitude, and their width comes from a stellar/gas velocity
dispersion plus instrumental resolution rather than a raw sigma in
Angstrom -- this module holds those line-specific conversions.
"""
from __future__ import annotations

import numpy as np
from astropy.constants import c as _c

from core.resolution import ResolutionModel

__all__ = [
    "C_KMS", "doppler_shifted_wavelength", "velocity_dispersion_to_angstrom",
    "gaussian_line_flux",
]

C_KMS = _c.to("km/s").value


def doppler_shifted_wavelength(rest_wavelength_angstrom, velocity_kms, redshift=0.0):
    """Observed-frame line center for a systemic redshift plus a residual velocity."""
    rest = np.asarray(rest_wavelength_angstrom, dtype=float)
    beta = float(velocity_kms) / C_KMS
    if abs(beta) >= 1:
        raise ValueError("velocity magnitude must be below c.")
    return rest * (1.0 + float(redshift)) * np.sqrt((1 + beta) / (1 - beta))


def velocity_dispersion_to_angstrom(center_angstrom, sigma_kms, resolution: "ResolutionModel | None" = None):
    """Combine an intrinsic velocity-dispersion width with instrumental resolution.

    Both broadenings are treated as independent Gaussians and combined in
    quadrature at the line center, per the "resolution must never be
    silently ignored" design principle -- if ``resolution`` is None the
    returned width is intrinsic-only (the caller is responsible for
    surfacing the DefaultResolutionWarning upstream, e.g. via
    ``core.resolution``).
    """
    center = np.asarray(center_angstrom, dtype=float)
    if float(sigma_kms) < 0:
        raise ValueError("sigma_kms must be non-negative.")
    sigma_intrinsic = center * float(sigma_kms) / C_KMS
    if resolution is None:
        return sigma_intrinsic
    sigma_instrumental = resolution.sigma_angstrom(center)
    return np.hypot(sigma_intrinsic, sigma_instrumental)


def gaussian_line_flux(wave, integrated_flux, center_angstrom, sigma_angstrom):
    """Area-normalized Gaussian: integral over wave equals ``integrated_flux``.

    Distinct from ``core.models.gaussian`` (peak-normalized), since
    emission-line amplitudes are fit and reported as integrated line flux.
    """
    wave = np.asarray(wave, dtype=float)
    sigma = float(sigma_angstrom)
    if sigma <= 0:
        raise ValueError("sigma_angstrom must be positive.")
    normalization = float(integrated_flux) / (sigma * np.sqrt(2.0 * np.pi))
    return normalization * np.exp(-0.5 * ((wave - float(center_angstrom)) / sigma) ** 2)
