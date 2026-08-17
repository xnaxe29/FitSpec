"""Column-density upper limits for non-detected ions.

Implements the method described in RDGEN's documentation (its ``uc``
command), attributed there to Schaye et al. (2007, MNRAS, 379, 1169,
Appendix): given a redshift and Doppler parameter determined from some
reference ion (e.g. from a clean, detected line), estimate the range of
physically plausible Doppler parameters for a different, non-detected
test ion via a turbulent/thermal decomposition, then, for each trial b
in that range, find the largest column density whose predicted
absorption profile remains statistically consistent with the observed
(non-detection) data. The overall upper limit is the maximum of these
over the b range -- the most conservative value, valid even in the
presence of blends, since only pixels where the trial profile predicts
more absorption than the data allow ever contribute to the statistic.

Note: this is FitSpec's own implementation of the method as documented
in RDGEN's manual, built on FitSpec's Voigt-profile machinery -- not a
byte-for-byte port of VPFIT/RDGEN's Fortran implementation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from core.resolution import ResolutionModel, convolve_variable_gaussian

from absorption.atomic import AtomicTransition
from absorption.profiles import C_KMS, optical_depth_voigt

__all__ = ["ColumnDensityUpperLimit", "estimate_column_density_upper_limit"]


@dataclass
class ColumnDensityUpperLimit:
    """Result of :func:`estimate_column_density_upper_limit`."""

    logN_limit: float
    b_kms_at_limit: float
    probability_threshold: float
    b_grid_kms: np.ndarray
    logN_limit_per_b: np.ndarray
    dof_at_limit: int


def _trial_transmission(wave, transitions, logN, b_kms, redshift, resolution):
    column_density_cm2 = 10.0 ** logN
    total_tau = np.zeros_like(wave)
    for transition in transitions:
        total_tau = total_tau + optical_depth_voigt(
            wave, transition.rest_wavelength_angstrom, transition.oscillator_strength,
            transition.damping_constant_s, column_density_cm2, b_kms, velocity_kms=0.0, redshift=redshift,
        )
    transmission = np.exp(-total_tau)
    if resolution is not None:
        transmission = convolve_variable_gaussian(wave, transmission, wave, resolution.sigma_angstrom(wave))
    return transmission


def _logN_limit_for_b(
    wave, flux, flux_unc, transitions, b_kms, redshift, resolution, *,
    probability_threshold, xn, xb, min_pixels,
    logN_start, logN_step_small, logN_step_large, logN_step_boundary, logN_max,
):
    line_centers = np.array([t.rest_wavelength_angstrom * (1.0 + redshift) for t in transitions])
    half_window_angstrom = np.array([xb * b_kms / C_KMS * center for center in line_centers])
    in_window = np.zeros(wave.shape, dtype=bool)
    for center, half_window in zip(line_centers, half_window_angstrom):
        in_window |= np.abs(wave - center) <= half_window

    logN = logN_start
    dof = 0
    while logN <= logN_max:
        model = _trial_transmission(wave, transitions, logN, b_kms, redshift, resolution)
        qualifying = in_window & (model < flux + xn * flux_unc)
        dof = int(np.count_nonzero(qualifying))
        if dof < min_pixels:
            step = logN_step_small if logN < logN_step_boundary else logN_step_large
            logN += step
            continue
        chi_square = float(np.sum(((flux[qualifying] - model[qualifying]) / flux_unc[qualifying]) ** 2))
        probability = float(stats.chi2.sf(chi_square, dof))
        if probability < probability_threshold:
            return logN, dof
        step = logN_step_small if logN < logN_step_boundary else logN_step_large
        logN += step
    return logN_max, dof


def estimate_column_density_upper_limit(
    wave, flux, flux_unc, transitions: "list[AtomicTransition]", *,
    reference_b_kms: float, reference_mass_amu: float, test_mass_amu: float,
    redshift: float, reference_b_uncertainty_kms: float = 0.0,
    resolution: "ResolutionModel | None" = None,
    probability_threshold: float = 0.16, xn: float = 1.0, xb: float = 2.0, min_pixels: int = 5,
    b_grid_size: int = 7, logN_start: float = 11.0,
    logN_step_small: float = 0.05, logN_step_large: float = 0.1, logN_step_boundary: float = 13.5,
    logN_max: float = 23.0,
) -> ColumnDensityUpperLimit:
    """Estimate a column-density upper limit for a non-detected ion.

    Parameters
    ----------
    wave, flux, flux_unc : array-like
        Continuum-normalized spectrum (flux ~ 1 in unabsorbed regions).
    transitions : list[AtomicTransition]
        The test ion's transitions available in the spectral range
        (typically ``absorption.atomic.select_group(line_list, ion)``).
    reference_b_kms, reference_b_uncertainty_kms : float
        Doppler parameter (and its 1-sigma uncertainty) determined from
        the reference ion at this redshift.
    reference_mass_amu, test_mass_amu : float
        Atomic/molecular masses of the reference and test species (see
        ``AtomicTransition.atomic_mass_amu``).
    redshift : float
        Redshift of the (non-detected) system, from the reference ion.
    probability_threshold : float, default 0.16
        Chi-square right-tail probability below which a trial profile
        is judged inconsistent with the data (0.16 corresponds roughly
        to a 1-sigma one-sided limit).
    xn : float, default 1.0
        Only pixels where the trial model is more than ``xn`` sigma
        below the data contribute to the statistic.
    xb : float, default 2.0
        Comparison window half-width, in units of the trial b, around
        each transition center.
    min_pixels : int, default 5
        Minimum qualifying-pixel count before the chi-square statistic
        is trusted; the column density is increased until this many
        pixels qualify.
    b_grid_size : int, default 7
        Number of trial Doppler parameters spanning the turbulent/
        thermal extreme range (using ``reference_b_kms +/-
        reference_b_uncertainty_kms`` at both extremes).

    Returns
    -------
    ColumnDensityUpperLimit
    """
    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)
    flux_unc = np.asarray(flux_unc, dtype=float)
    if not (wave.shape == flux.shape == flux_unc.shape):
        raise ValueError("wave, flux, and flux_unc must have equal shapes.")
    if not transitions:
        raise ValueError("transitions must be non-empty.")

    mass_ratio = np.sqrt(reference_mass_amu / test_mass_amu)
    b_candidates = []
    for b_ref in (reference_b_kms - reference_b_uncertainty_kms, reference_b_kms + reference_b_uncertainty_kms):
        b_candidates.append(max(b_ref, 1e-3))          # fully turbulent: mass-independent
        b_candidates.append(max(b_ref * mass_ratio, 1e-3))  # fully thermal: mass-scaled
    b_min, b_max = min(b_candidates), max(b_candidates)
    b_grid = np.linspace(b_min, b_max, max(2, b_grid_size))

    logN_limits = np.empty(b_grid.shape)
    dof_at = np.empty(b_grid.shape, dtype=int)
    for i, b_kms in enumerate(b_grid):
        logN_limits[i], dof_at[i] = _logN_limit_for_b(
            wave, flux, flux_unc, transitions, float(b_kms), redshift, resolution,
            probability_threshold=probability_threshold, xn=xn, xb=xb, min_pixels=min_pixels,
            logN_start=logN_start, logN_step_small=logN_step_small, logN_step_large=logN_step_large,
            logN_step_boundary=logN_step_boundary, logN_max=logN_max,
        )

    best_index = int(np.argmax(logN_limits))
    return ColumnDensityUpperLimit(
        logN_limit=float(logN_limits[best_index]), b_kms_at_limit=float(b_grid[best_index]),
        probability_threshold=probability_threshold, b_grid_kms=b_grid, logN_limit_per_b=logN_limits,
        dof_at_limit=int(dof_at[best_index]),
    )
