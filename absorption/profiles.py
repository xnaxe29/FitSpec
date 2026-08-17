"""Physical (optical-depth) Voigt profiles and partial-covering mixing.

Distinct from the generic, peak-normalized Voigt primitive in
``core.models`` (built from the Faddeeva function, dimensionless
amplitude/sigma/gamma) -- absorption-line optical depth is
parameterized directly by physical column density, Doppler parameter,
oscillator strength, and radiative damping constant, and evaluated via
the Voigt-Hjerting function using the Tepper-Garcia (2006) rational
approximation, which is fast, accurate to better than 1% over the full
range of astrophysically relevant optical depths, and avoids the
special-function dependency of an exact Faddeeva evaluation.
"""
from __future__ import annotations

import numpy as np
from astropy import constants as const

__all__ = [
    "C_CMS", "C_KMS", "voigt_hjerting", "optical_depth_voigt",
    "transmission_voigt", "apply_partial_covering",
]

C_CMS = const.c.to("cm/s").value
C_KMS = const.c.to("km/s").value
_M_E_CGS = const.m_e.cgs.value
_E_ESU = 4.80320425e-10  # electron charge, esu (cgs)


def voigt_hjerting(a, x):
    """Tepper-Garcia (2006) rational approximation to the Voigt-Hjerting function H(a, x).

    Accurate to <1% for the damping parameters relevant to resonance
    absorption lines. ``a`` is the ratio of the Lorentzian (radiative
    damping) to Gaussian (Doppler) half-widths; ``x`` is wavelength
    offset from line center in Doppler-width units.
    """
    a = np.asarray(a, dtype=float)
    x = np.asarray(x, dtype=float)
    p = x ** 2
    h0 = np.exp(-p)
    q = 1.5 / p
    # The exact expression is singular at x=0 (p=0); Tepper-Garcia's own
    # reference implementation offsets x by a small epsilon for this
    # reason -- callers should do the same (see optical_depth_voigt).
    return h0 - (a / np.sqrt(np.pi) / p) * (h0 * h0 * (4.0 * p * p + 7.0 * p + 4.0 + q) - q - 1.0)


def optical_depth_voigt(wave, rest_wavelength_angstrom, oscillator_strength, damping_constant_s,
                         column_density_cm2, doppler_b_kms, velocity_kms=0.0, redshift: float = 0.0):
    """Absorption optical depth tau(wave) for one transition, one velocity component.

    Parameters
    ----------
    wave : array-like
        Observed-frame wavelength grid [Angstrom].
    rest_wavelength_angstrom, oscillator_strength, damping_constant_s : float
        Transition rest wavelength, oscillator strength ``f``, and
        radiative damping constant Gamma [s^-1] (see
        ``absorption.atomic.AtomicTransition``).
    column_density_cm2 : float
        Ionic column density N [cm^-2] (linear, not log).
    doppler_b_kms : float
        Doppler parameter b [km/s] (b = sqrt(2) * 1D velocity dispersion).
    velocity_kms : float, default 0.0
        Residual velocity of this component relative to the spectrum's
        systemic redshift.
    redshift : float, default 0.0
        Systemic redshift.

    Returns
    -------
    np.ndarray
        Optical depth at each wavelength in ``wave``.
    """
    wave = np.asarray(wave, dtype=float)
    l0 = float(rest_wavelength_angstrom)
    f = float(oscillator_strength)
    gamma = float(damping_constant_s)
    b_cms = float(doppler_b_kms) * 1.0e5
    if b_cms <= 0:
        raise ValueError("doppler_b_kms must be positive.")

    beta = float(velocity_kms) / C_KMS
    if abs(beta) >= 1:
        raise ValueError("velocity magnitude must be below c.")
    z_total = (1.0 + float(redshift)) * np.sqrt((1 + beta) / (1 - beta)) - 1.0

    l0_cm = l0 * 1.0e-8
    wave_rest_cm = (wave * 1.0e-8) / (1.0 + z_total)

    c_a = np.sqrt(np.pi) * _E_ESU ** 2 * f * l0_cm / (_M_E_CGS * C_CMS * b_cms)
    a = l0_cm * gamma / (4.0 * np.pi * b_cms)
    doppler_width_cm = (b_cms / C_CMS) * l0_cm
    x = (wave_rest_cm - l0_cm) / doppler_width_cm + 1.0e-8  # tiny offset avoids the x=0 singularity

    tau = c_a * column_density_cm2 * voigt_hjerting(a, x)
    return np.clip(tau, 0.0, None)


def transmission_voigt(wave, rest_wavelength_angstrom, oscillator_strength, damping_constant_s,
                        column_density_cm2, doppler_b_kms, velocity_kms=0.0, redshift: float = 0.0):
    """Normalized transmission exp(-tau) for one transition, one component (full coverage)."""
    tau = optical_depth_voigt(
        wave, rest_wavelength_angstrom, oscillator_strength, damping_constant_s,
        column_density_cm2, doppler_b_kms, velocity_kms, redshift,
    )
    return np.exp(-tau)


def apply_partial_covering(full_coverage_transmission, covering_fraction):
    """Mix a fully-covered transmission profile with the unabsorbed continuum.

    ``T_obs = (1 - C_f) + C_f * T_full``, the standard partial-covering
    formalism for a background source only partially covered by the
    absorbing gas (e.g. Barlow & Sargent 1997; Ganguly et al. 2003; Arav
    et al. 2005). ``covering_fraction = 1`` (full coverage) reduces this
    identically to ``T_full``. This mixing is intrinsic to the source
    geometry and must be applied to the *unconvolved* transmission,
    before any instrumental resolution convolution.
    """
    covering_fraction = float(covering_fraction)
    if not (0.0 <= covering_fraction <= 1.0):
        raise ValueError("covering_fraction must be in [0, 1].")
    t_full = np.asarray(full_coverage_transmission, dtype=float)
    return (1.0 - covering_fraction) + covering_fraction * t_full
