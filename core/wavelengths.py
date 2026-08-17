"""Air <-> vacuum wavelength conversion.

FitSpec's internal wavelength standard is vacuum: the stellar libraries
(SB99, BPASS, eMILES) are all natively vacuum, and vacuum wavelength is
the physically native quantity for redshift/Doppler calculations. This
module converts anything expressed in air wavelengths -- an emission/
absorption line list, or an observed spectrum whose reduction pipeline
delivered air-calibrated wavelengths -- to vacuum before it enters any
fitting code.

Conversion uses the standard dispersion-of-air formula (Morton 1991,
ApJS 77, 119; as adopted by the IDL Astronomy User's Library
airtovac/vactoair routines and NIST ASD), valid for the standard-air
regime 2000-100000 Angstrom.

Convention (also used throughout FitSpec's line lists, and standard in
atomic spectroscopy -- see e.g. NIST ASD, CLOUDY): wavelengths below
2000 Angstrom are, by convention, already vacuum wavelengths (no
meaningful "standard air" exists that far into the UV); wavelengths at
or above 2000 Angstrom are, by convention, air wavelengths unless
otherwise stated. ``standard_medium`` / the ``medium="auto"`` option
implement exactly this rule -- use it for line lists that follow the
convention, and pass an explicit ``medium="air"`` or ``medium="vacuum"``
for an observed spectrum's wavelength solution, which is uniformly one
medium or the other regardless of wavelength.
"""
from __future__ import annotations

import numpy as np


__all__ = [
    "AIR_VACUUM_BOUNDARY_ANGSTROM",
    "air_to_vacuum",
    "vacuum_to_air",
    "standard_medium",
    "to_vacuum",
    "to_air",
]

#: Standard atomic-spectroscopy convention boundary (Angstrom): below
#: this, wavelengths are conventionally vacuum; at/above, air.
AIR_VACUUM_BOUNDARY_ANGSTROM = 2000.0


def _refractive_index_factor(wave_angstrom: np.ndarray) -> np.ndarray:
    """n(sigma) - 1 dispersion-of-air factor, Morton (1991) formula.

    ``wave_angstrom`` is used to compute the vacuum wavenumber sigma in
    both directions; per Morton, using either the air or vacuum
    wavelength to compute sigma introduces a fractional error far below
    the ~1e-8 level the formula is quoted to, so no iteration is needed
    for astronomical purposes.
    """
    sigma2 = (1.0e4 / wave_angstrom) ** 2
    return 1.0 + 5.792105e-2 / (238.0185 - sigma2) + 1.67917e-3 / (57.362 - sigma2)


def air_to_vacuum(wave_air):
    """Convert standard-air wavelength(s) [Angstrom] to vacuum.

    Valid for the standard-air regime (>= 2000 Angstrom); FitSpec never
    calls this below that boundary (see module docstring).
    """
    wave_air = np.asarray(wave_air, dtype=float)
    return wave_air * _refractive_index_factor(wave_air)


def vacuum_to_air(wave_vacuum):
    """Convert vacuum wavelength(s) [Angstrom] to standard air.

    Valid for the standard-air regime (>= 2000 Angstrom).
    """
    wave_vacuum = np.asarray(wave_vacuum, dtype=float)
    return wave_vacuum / _refractive_index_factor(wave_vacuum)


def standard_medium(wave, boundary: float = AIR_VACUUM_BOUNDARY_ANGSTROM):
    """Which medium a wavelength is conventionally expressed in.

    Returns an array of ``"vacuum"``/``"air"`` strings following the
    standard atomic-spectroscopy convention: vacuum below ``boundary``,
    air at or above it. Only meaningful for values that already follow
    the convention (e.g. a compiled line list) -- not for an observed
    spectrum's wavelength solution, which is uniformly one medium.
    """
    wave = np.asarray(wave, dtype=float)
    return np.where(wave < boundary, "vacuum", "air")


def to_vacuum(wave, medium: str = "auto", boundary: float = AIR_VACUUM_BOUNDARY_ANGSTROM):
    """Convert wavelength(s) to vacuum, given their current medium.

    Parameters
    ----------
    wave : array-like
        Wavelength(s) in Angstrom.
    medium : {"air", "vacuum", "auto"}, default "auto"
        - "air": every value is currently air; convert all of it.
        - "vacuum": already vacuum; returned unchanged.
        - "auto": each value's current medium is inferred from its own
          wavelength via :func:`standard_medium` (i.e. follows the
          standard convention). Appropriate for a line list built to
          that convention; NOT appropriate for an observed spectrum's
          wavelength axis (use an explicit "air"/"vacuum" for that).
    """
    wave = np.asarray(wave, dtype=float)
    if medium == "vacuum":
        return wave.copy()
    if medium == "air":
        return air_to_vacuum(wave)
    if medium == "auto":
        is_air = wave >= boundary
        result = wave.copy()
        if np.any(is_air):
            result[is_air] = air_to_vacuum(wave[is_air])
        return result
    raise ValueError('medium must be "air", "vacuum", or "auto".')


def to_air(wave, medium: str = "auto", boundary: float = AIR_VACUUM_BOUNDARY_ANGSTROM):
    """Convert wavelength(s) to standard air, given their current medium.

    See :func:`to_vacuum` for the meaning of ``medium``. Note that
    values below ``boundary`` have no meaningful standard-air
    representation by convention; with ``medium="auto"`` these are left
    unchanged (treated as already vacuum, per the standard rule) rather
    than passed through the air formula outside its valid range.
    """
    wave = np.asarray(wave, dtype=float)
    if medium == "air":
        return wave.copy()
    if medium == "vacuum":
        return vacuum_to_air(wave)
    if medium == "auto":
        is_vacuum = wave < boundary
        result = wave.copy()
        if np.any(~is_vacuum):
            result[~is_vacuum] = vacuum_to_air(wave[~is_vacuum])
        return result
    raise ValueError('medium must be "air", "vacuum", or "auto".')
