"""Structural spectrum cleaning and the universal Spectrum object.

This module defines FitSpec's single spectrum-cleaning routine (see
:func:`clean_spectrum`), used as a preprocessing step immediately after
loading a spectrum and before it is handed to rebinning, masking, or
fitting, and the :class:`Spectrum` object that every science module uses
as its common data container.

Cleaning here means only what genuinely cannot be represented downstream:

- non-finite wavelengths (a pixel with no defined wavelength cannot be
  placed on the grid at all), and
- duplicate wavelengths (required for a strictly increasing wavelength
  axis, e.g. by ``core.rebinning``).

Both are structural problems, so affected rows are removed and every
array shrinks together.

Non-finite flux or non-finite/non-positive flux uncertainty, by contrast,
are *not* removed here. Per the FitSpec masking design principle, "all
masking mechanisms should resolve to one boolean spectrum mask" -- so
these are reported back as an ``invalid`` boolean array for
``core.masking`` to fold into the combined mask, keeping every array
pixel-aligned with the input instead of silently changing its length.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from core.masking import combine_masks


__all__ = ["CleanedSpectrum", "clean_spectrum", "Spectrum"]


@dataclass
class CleanedSpectrum:
    """Result of :func:`clean_spectrum`.

    Attributes
    ----------
    wave, flux : np.ndarray
        Structurally cleaned wavelength and flux arrays (sorted,
        non-finite-wavelength and duplicate-wavelength rows removed).
        Same length as each other and as every array in ``extra``.
    flux_unc : np.ndarray or None
        Structurally cleaned uncertainty array, or None if no
        uncertainty was supplied at all (a valid two-column
        wavelength/flux spectrum).
    invalid : np.ndarray of bool
        True where flux is non-finite, or (if ``flux_unc`` was supplied)
        flux_unc is non-finite or non-positive. Not yet removed or
        combined with any other mask -- pass to ``core.masking`` alongside
        other mask sources.
    extra : dict[str, np.ndarray]
        Any companion arrays passed in (e.g. continuum, previously fitted
        model, SNR), carried through the same sort/dedup as wave/flux.
    n_dropped_nonfinite_wave : int
        Number of rows removed for a non-finite wavelength.
    n_dropped_duplicate_wave : int
        Number of rows removed as duplicate wavelengths.
    """

    wave: np.ndarray
    flux: np.ndarray
    flux_unc: "np.ndarray | None"
    invalid: np.ndarray
    extra: dict = field(default_factory=dict)
    n_dropped_nonfinite_wave: int = 0
    n_dropped_duplicate_wave: int = 0


def clean_spectrum(
    wave, flux, flux_unc=None, *, extra: "dict[str, np.ndarray] | None" = None,
    fill_invalid_uncertainty: bool = False,
) -> CleanedSpectrum:
    """Structurally clean a spectrum: finite, sorted, unique wavelengths.

    Parameters
    ----------
    wave, flux : array-like
        Input wavelength and flux density. Need not be sorted and may
        contain NaNs.
    flux_unc : array-like, optional
        1-sigma uncertainty. May be omitted entirely (a valid two-column
        wavelength/flux spectrum, per the "missing optional information
        stays explicitly absent" principle); in that case ``invalid``
        only flags non-finite flux, and the returned ``flux_unc`` is None.
    extra : dict[str, array-like], optional
        Any additional companion arrays that must be carried through the
        same sort/deduplication as ``wave``/``flux``/``flux_unc`` (e.g.
        ``{"continuum": ..., "model": ..., "snr": ...}``). Each must have
        the same length as ``wave``.
    fill_invalid_uncertainty : bool, default False
        If True (and ``flux_unc`` was supplied), replace non-finite or
        non-positive uncertainties with the median of the finite,
        positive uncertainties, in-place in the returned ``flux_unc``.
        This is an explicit opt-in convenience (it does not affect
        ``invalid``, which still flags those pixels) -- left off by
        default since silently substituting uncertainties can mask real
        data-quality problems.

    Returns
    -------
    CleanedSpectrum
    """
    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)
    has_uncertainty = flux_unc is not None
    if has_uncertainty:
        flux_unc = np.asarray(flux_unc, dtype=float)

    if not (wave.ndim == flux.ndim == 1) or (has_uncertainty and flux_unc.ndim != 1):
        raise ValueError("wave, flux, and flux_unc must be one-dimensional.")
    if wave.size != flux.size or (has_uncertainty and flux_unc.size != wave.size):
        raise ValueError("wave, flux, and flux_unc must have equal lengths.")

    extra = {} if extra is None else {k: np.asarray(v) for k, v in extra.items()}
    for key, arr in extra.items():
        if arr.shape[0] != wave.size:
            raise ValueError(f"extra['{key}'] must have the same length as wave.")

    n_input = wave.size

    # --- Step 1: drop non-finite wavelengths (cannot be placed on a grid) ---
    finite_wave = np.isfinite(wave)
    n_dropped_nonfinite_wave = int(n_input - np.count_nonzero(finite_wave))

    wave = wave[finite_wave]
    flux = flux[finite_wave]
    if has_uncertainty:
        flux_unc = flux_unc[finite_wave]
    extra = {k: v[finite_wave] for k, v in extra.items()}

    if wave.size == 0:
        raise ValueError("No finite wavelength values remain after cleaning.")

    # --- Step 2: sort by wavelength (carrying every companion array) ---
    order = np.argsort(wave, kind="stable")
    wave = wave[order]
    flux = flux[order]
    if has_uncertainty:
        flux_unc = flux_unc[order]
    extra = {k: v[order] for k, v in extra.items()}

    # --- Step 3: drop duplicate wavelengths, keep first occurrence ---
    unique_wave, unique_indices = np.unique(wave, return_index=True)
    keep = np.sort(unique_indices)
    n_dropped_duplicate_wave = int(wave.size - keep.size)

    wave = wave[keep]
    flux = flux[keep]
    if has_uncertainty:
        flux_unc = flux_unc[keep]
    extra = {k: v[keep] for k, v in extra.items()}

    # --- Flag (do not drop) non-finite flux or bad uncertainty ---
    invalid = ~np.isfinite(flux)
    if has_uncertainty:
        invalid = invalid | ~np.isfinite(flux_unc) | (flux_unc <= 0)

        if fill_invalid_uncertainty:
            bad_unc = ~np.isfinite(flux_unc) | (flux_unc <= 0)
            good_unc = np.isfinite(flux_unc) & (flux_unc > 0)
            if np.any(bad_unc):
                if not np.any(good_unc):
                    raise ValueError(
                        "fill_invalid_uncertainty=True but no finite, positive "
                        "uncertainties are available to compute a replacement median."
                    )
                flux_unc = flux_unc.copy()
                flux_unc[bad_unc] = np.nanmedian(flux_unc[good_unc])

    return CleanedSpectrum(
        wave=wave, flux=flux, flux_unc=(flux_unc if has_uncertainty else None),
        invalid=invalid, extra=extra,
        n_dropped_nonfinite_wave=n_dropped_nonfinite_wave,
        n_dropped_duplicate_wave=n_dropped_duplicate_wave,
    )


@dataclass
class Spectrum:
    """FitSpec's universal spectrum container.

    Every science module (stellar, emission, absorption) reads and
    writes this same object rather than passing wave/flux/uncertainty/
    continuum/mask arrays separately. Optional fields that were never
    supplied stay ``None`` rather than being fabricated -- e.g. a plain
    two-column wavelength/flux input is a fully valid ``Spectrum`` with
    ``flux_unc=None``.

    Attributes
    ----------
    wave, flux : np.ndarray
        Wavelength [Angstrom, vacuum -- see ``core.wavelengths``] and
        flux density.
    flux_unc : np.ndarray or None
        1-sigma flux uncertainty, if available.
    continuum, model : np.ndarray or None
        A previously determined continuum and/or fitted model, if
        available.
    mask : np.ndarray of bool or None
        The current combined fit mask (see ``core.masking``); True means
        "use this pixel". None until a mask has actually been built.
    redshift : float
        Systemic redshift.
    resolution : object or None
        Instrumental resolution information; typed loosely for now since
        ``core.resolution`` (the module that will define a concrete
        resolution model/class) has not been built yet. Replace this
        annotation with that type once it exists.
    metadata : dict
        Anything else worth carrying along (instrument name, original
        file path, header keywords, wavelength medium at load time, etc.).
    """

    wave: np.ndarray
    flux: np.ndarray
    flux_unc: "np.ndarray | None" = None
    continuum: "np.ndarray | None" = None
    model: "np.ndarray | None" = None
    mask: "np.ndarray | None" = None
    redshift: float = 0.0
    resolution: object = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.wave = np.asarray(self.wave, dtype=float)
        self.flux = np.asarray(self.flux, dtype=float)
        n = self.wave.size
        if self.flux.shape != (n,):
            raise ValueError("flux must have the same length as wave.")
        for name in ("flux_unc", "continuum", "model"):
            value = getattr(self, name)
            if value is not None:
                value = np.asarray(value, dtype=float)
                if value.shape != (n,):
                    raise ValueError(f"{name} must have the same length as wave.")
                setattr(self, name, value)
        if self.mask is not None:
            mask = np.asarray(self.mask, dtype=bool)
            if mask.shape != (n,):
                raise ValueError("mask must have the same length as wave.")
            self.mask = mask

    @classmethod
    def from_arrays(
        cls, wave, flux, flux_unc=None, *, continuum=None, model=None,
        redshift: float = 0.0, resolution=None, metadata: "dict | None" = None,
        fill_invalid_uncertainty: bool = False,
    ) -> "Spectrum":
        """Build a cleaned, mask-initialized Spectrum from raw arrays.

        Runs :func:`clean_spectrum` (structural cleaning: finite,
        sorted, unique wavelengths; non-finite flux/uncertainty flagged
        rather than dropped) and initializes ``mask`` from the resulting
        ``invalid`` flags via ``core.masking.combine_masks`` -- i.e. a
        fresh Spectrum's mask already excludes bad pixels, before any
        region/velocity/interactive selection is layered on later.
        """
        extra = {}
        if continuum is not None:
            extra["continuum"] = continuum
        if model is not None:
            extra["model"] = model

        cleaned = clean_spectrum(
            wave, flux, flux_unc, extra=extra, fill_invalid_uncertainty=fill_invalid_uncertainty)

        mask_components = combine_masks(cleaned.wave.size, invalid=cleaned.invalid)

        return cls(
            wave=cleaned.wave, flux=cleaned.flux, flux_unc=cleaned.flux_unc,
            continuum=cleaned.extra.get("continuum"), model=cleaned.extra.get("model"),
            mask=mask_components.combined, redshift=redshift, resolution=resolution,
            metadata=dict(metadata) if metadata is not None else {},
        )
