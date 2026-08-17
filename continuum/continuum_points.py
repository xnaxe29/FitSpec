"""Interactive, point-based continuum editing.

This is the actual continuum used to fit emission (or absorption) lines
against -- distinct from a "ghost" reference continuum (e.g. a separately
fit stellar model) shown only for visual comparison. The workflow this
ports (originally ``bic_emission_fitting.py``'s key-press continuum
editor) is:

1. Estimate an initial continuum automatically (``continuum.continuum``).
2. Reduce it to a sparse set of "anchor" points at ~regular intervals.
3. Let the user add/remove/move anchor points interactively; the
   continuum everywhere else is rebuilt from the current anchor points
   by spline interpolation on every edit.
4. Save the anchor points (and, implicitly, the continuum they define)
   once the user is satisfied, ready to be subtracted from the data
   before the emission-line fit.

:class:`ContinuumPointsState` mirrors the working/saved lifecycle of
``core.masking.FitMaskState``: interactive edits change ``working_*``
immediately; ``save()``/``load()`` explicitly commit or revert against
``saved_*``, and nothing is silently persisted to disk on every edit.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import splrep, splev

__all__ = [
    "continuum_from_points", "default_anchor_points",
    "ContinuumPointsState", "save_continuum_points_file", "load_continuum_points_file",
]


def continuum_from_points(wave, wave_points, flux_points):
    """Evaluate a continuum at `wave` from sparse (wave, flux) anchor points.

    Cubic-spline through the anchor points (sorted and deduplicated by
    wavelength first, matching the original ``get_continuum_from_points``)
    when at least 4 are given. Fewer points fall back to a well-defined
    lower-order interpolation -- linear for 2-3 points, flat for exactly
    1 -- rather than the original's silent all-zero array whenever a
    cubic spline wasn't possible.

    Parameters
    ----------
    wave : array-like
        Wavelengths to evaluate the continuum at.
    wave_points, flux_points : array-like
        Anchor point coordinates, any order (sorted internally).

    Returns
    -------
    np.ndarray
        Continuum flux, same shape as `wave`.
    """
    wave = np.asarray(wave, dtype=float)
    wave_points = np.asarray(wave_points, dtype=float)
    flux_points = np.asarray(flux_points, dtype=float)
    if wave_points.size == 0:
        raise ValueError("continuum_from_points needs at least one anchor point.")
    if wave_points.shape != flux_points.shape:
        raise ValueError("wave_points and flux_points must have the same shape.")

    order = np.argsort(wave_points)
    wave_points = wave_points[order]
    flux_points = flux_points[order]
    _, unique_index = np.unique(wave_points, return_index=True)
    unique_index = np.sort(unique_index)
    wave_points = wave_points[unique_index]
    flux_points = flux_points[unique_index]

    if wave_points.size == 1:
        return np.full_like(wave, float(flux_points[0]))
    if wave_points.size < 4:
        return np.interp(wave, wave_points, flux_points)

    tck = splrep(wave_points, flux_points, k=3)
    return np.asarray(splev(wave, tck, der=0), dtype=float)


def default_anchor_points(wave, continuum, n_points=50):
    """Pick ~evenly (pixel-index-)spaced anchor points along an estimated continuum.

    Matches the original workflow's default of 50 points spaced evenly by
    pixel index across the spectrum (not by wavelength -- the two only
    coincide for a uniform wavelength grid, but that's what the legacy
    tool did and it's a reasonable, simple default either way).
    """
    wave = np.asarray(wave, dtype=float)
    continuum = np.asarray(continuum, dtype=float)
    if wave.size == 0:
        raise ValueError("default_anchor_points needs a non-empty wave array.")
    n_points = max(2, min(int(n_points), wave.size))
    indices = np.unique(np.linspace(0, wave.size - 1, n_points).astype(int))
    return wave[indices].copy(), continuum[indices].copy()


@dataclass
class ContinuumPointsState:
    """Working/saved anchor-point continuum state.

    Mirrors ``core.masking.FitMaskState``'s working-vs-saved lifecycle:
    interactive add/remove/move edits change ``working_wave``/
    ``working_flux`` immediately; nothing touches ``saved_wave``/
    ``saved_flux`` until :meth:`save` is called explicitly, and
    :meth:`load` discards unsaved working edits back to the last save.
    """

    working_wave: np.ndarray
    working_flux: np.ndarray
    saved_wave: np.ndarray
    saved_flux: np.ndarray

    @classmethod
    def from_points(cls, wave_points, flux_points) -> "ContinuumPointsState":
        wave_points = np.asarray(wave_points, dtype=float).copy()
        flux_points = np.asarray(flux_points, dtype=float).copy()
        return cls(
            working_wave=wave_points, working_flux=flux_points,
            saved_wave=wave_points.copy(), saved_flux=flux_points.copy(),
        )

    @property
    def is_modified(self) -> bool:
        return not (
            np.array_equal(self.working_wave, self.saved_wave)
            and np.array_equal(self.working_flux, self.saved_flux)
        )

    @property
    def n_points(self) -> int:
        return int(self.working_wave.size)

    def continuum_on(self, wave) -> np.ndarray:
        return continuum_from_points(wave, self.working_wave, self.working_flux)

    def add_point(self, wave_value: float, flux_value: float) -> None:
        self.working_wave = np.append(self.working_wave, float(wave_value))
        self.working_flux = np.append(self.working_flux, float(flux_value))

    def remove_nearest(self, wave_value: float) -> None:
        if self.working_wave.size == 0:
            return
        index = int(np.argmin(np.abs(self.working_wave - float(wave_value))))
        self.working_wave = np.delete(self.working_wave, index)
        self.working_flux = np.delete(self.working_flux, index)

    def move_nearest(self, wave_value: float, new_wave: float, new_flux: float) -> None:
        if self.working_wave.size == 0:
            return
        index = int(np.argmin(np.abs(self.working_wave - float(wave_value))))
        self.working_wave[index] = float(new_wave)
        self.working_flux[index] = float(new_flux)

    def reset_to(self, wave_points, flux_points) -> None:
        self.working_wave = np.asarray(wave_points, dtype=float).copy()
        self.working_flux = np.asarray(flux_points, dtype=float).copy()

    def save(self) -> None:
        self.saved_wave = self.working_wave.copy()
        self.saved_flux = self.working_flux.copy()

    def load(self) -> None:
        self.working_wave = self.saved_wave.copy()
        self.working_flux = self.saved_flux.copy()


def save_continuum_points_file(path, state: ContinuumPointsState, *, metadata=None) -> Path:
    """Persist the *saved* anchor points to a compressed NPZ file.

    Call ``state.save()`` first (the GUI Save action does this) -- this
    only writes ``saved_wave``/``saved_flux``, never unsaved working edits.
    """
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        wave_points=state.saved_wave,
        flux_points=state.saved_flux,
        metadata=np.asarray([repr({} if metadata is None else dict(metadata))]),
    )
    return path


def load_continuum_points_file(path) -> ContinuumPointsState:
    """Load anchor points from a file, as both saved and working state."""
    path = Path(path).expanduser()
    with np.load(path, allow_pickle=False) as data:
        wave_points = np.asarray(data["wave_points"], dtype=float)
        flux_points = np.asarray(data["flux_points"], dtype=float)
    return ContinuumPointsState.from_points(wave_points, flux_points)
