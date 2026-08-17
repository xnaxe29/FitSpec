"""Flux-conservative, gap-aware spectral rebinning.

This module defines FitSpec's single numerical rebinning implementation.
Per the FitSpec design principles, there is exactly one flux-conservative,
gap-aware rebinning routine; "permanent rebinning" and "display smoothing"
are two different *applications* of this same routine, not two different
numerical methods:

- ``apply_permanent_rebinning`` overwrites the stored :class:`Spectrum`
  (wavelength, flux, uncertainty, mask). It is only invoked when requested
  in configuration, immediately after loading/validation and before any
  fitting. The result becomes the spectrum used by all subsequent fits and
  saved products.

- ``compute_display_smoothing`` never touches the stored spectrum. It
  returns display-only arrays for GUI visualization. It runs the same
  flux-conservative binning at a (typically coarser) bin size, but skips
  full covariance propagation by default since only a display uncertainty
  envelope is needed.

The numerical core (``rebin_spectrum``) is a sparse-matrix, exact
wavelength-overlap implementation in the spirit of standard flux-conserving
resampling used for e.g. HST spectra: each output bin's flux is a coverage-
weighted integral of the input pixels it overlaps,

    flux_bin = A @ flux

and uncertainty is propagated exactly through the same linear operator,

    covariance_bin = A @ covariance_input @ A.T

Gaps (missing/invalid pixels, or instrumental gaps) are handled explicitly
via two independent controls:

- ``min_coverage``: an output bin is dropped (set to NaN) if the fraction
  of its wavelength range actually covered by valid input pixels falls
  below this threshold.
- ``max_gap_pixels``: when gap-filling interpolation is enabled, caps how
  many consecutive missing native pixels may be bridged before a point is
  left unsupported (never extrapolated).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix, diags, eye, issparse


__all__ = [
    "RebinResult",
    "rebin_spectrum",
    "apply_permanent_rebinning",
    "compute_display_smoothing",
]


@dataclass
class RebinResult:
    """Result of a flux-conservative rebinning operation.

    Attributes
    ----------
    wave, flux, flux_unc, snr : np.ndarray
        Output-bin wavelength centers, flux density, 1-sigma uncertainty,
        and signal-to-noise ratio.
    bin_width : np.ndarray
        Wavelength width of each output bin.
    coverage_fraction : np.ndarray
        Fraction of each output bin's wavelength range actually covered by
        valid input pixels. Bins below ``min_coverage`` are NaN in
        ``flux``/``flux_unc`` but still reported here for diagnostics.
    covariance : scipy.sparse.csr_matrix or None
        Full covariance matrix of the output flux vector, or None if not
        requested / not applicable.
    rebin_matrix : scipy.sparse.csr_matrix
        The complete linear operator A such that ``flux = A @ flux_input``
        (including any gap-filling/resampling preprocessing).
    """

    wave: np.ndarray
    flux: np.ndarray
    flux_unc: np.ndarray
    snr: np.ndarray
    bin_width: np.ndarray
    coverage_fraction: np.ndarray
    covariance: "csr_matrix | None"
    rebin_matrix: csr_matrix


# ---------------------------------------------------------------------------
# Core: exact wavelength-overlap flux-conservative binning
# ---------------------------------------------------------------------------

def _flux_conserving_bin(
    wave, flux, err, binsize, *,
    covariance=None, min_coverage=1.0, keep_partial_bin=False,
    return_covariance=True,
):
    """Bin ``binsize`` consecutive *already-gap-processed* pixels using an
    exact wavelength-overlap operator. Assumes ``wave`` is strictly
    increasing. This is the innermost numerical step; use
    :func:`rebin_spectrum` for the public, gap-aware entry point.
    """
    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)
    err = np.asarray(err, dtype=float)

    if not (wave.ndim == flux.ndim == err.ndim == 1):
        raise ValueError("wave, flux, and err must be one-dimensional.")
    if not (wave.size == flux.size == err.size):
        raise ValueError("wave, flux, and err must have equal lengths.")
    if wave.size < 2:
        raise ValueError("At least two input pixels are required.")

    binsize = int(binsize)
    if binsize < 1:
        raise ValueError("binsize must be an integer >= 1.")
    if not 0 <= min_coverage <= 1:
        raise ValueError("min_coverage must lie between 0 and 1.")
    if not np.all(np.isfinite(wave)) or not np.all(np.diff(wave) > 0):
        raise ValueError("wave must be finite and strictly increasing.")

    n_input = wave.size

    # Native pixel edges (midpoints between neighbors; edge pixels mirror
    # their nearest neighbor spacing).
    input_edges = np.empty(n_input + 1, dtype=float)
    input_edges[1:-1] = 0.5 * (wave[:-1] + wave[1:])
    input_edges[0] = wave[0] - 0.5 * (wave[1] - wave[0])
    input_edges[-1] = wave[-1] + 0.5 * (wave[-1] - wave[-2])
    input_width = np.diff(input_edges)
    if np.any(input_width <= 0):
        raise ValueError("Input pixels must have positive wavelength widths.")

    # Output bin edges: group `binsize` consecutive native pixels.
    if keep_partial_bin:
        edge_indices = np.arange(0, n_input + 1, binsize, dtype=int)
        if edge_indices[-1] != n_input:
            edge_indices = np.append(edge_indices, n_input)
    else:
        n_complete = n_input // binsize
        if n_complete == 0:
            raise ValueError("binsize exceeds the number of input pixels.")
        edge_indices = np.arange(0, n_complete * binsize + 1, binsize, dtype=int)

    output_edges = input_edges[edge_indices]
    output_lower, output_upper = output_edges[:-1], output_edges[1:]
    bin_width = output_upper - output_lower
    wave_bin = 0.5 * (output_lower + output_upper)
    n_output = wave_bin.size

    valid_pixel = np.isfinite(flux) & np.isfinite(err) & (err > 0)

    # Build the sparse overlap matrix A: flux_bin = A @ flux (coverage-weighted).
    rows, cols, vals = [], [], []
    coverage_fraction = np.zeros(n_output, dtype=float)
    for i in range(n_output):
        lower, upper = output_lower[i], output_upper[i]
        overlap = np.clip(
            np.minimum(input_edges[1:], upper) - np.maximum(input_edges[:-1], lower),
            0.0, None,
        )
        overlap[~valid_pixel] = 0.0
        coverage_fraction[i] = np.sum(overlap) / bin_width[i]
        if coverage_fraction[i] < min_coverage:
            continue
        contributing = np.flatnonzero(overlap > 0)
        weights = overlap[contributing] / bin_width[i]
        rows.extend(np.full(contributing.size, i, dtype=int))
        cols.extend(contributing)
        vals.extend(weights)

    rebin_matrix = csr_matrix((vals, (rows, cols)), shape=(n_output, n_input))

    safe_flux = np.zeros_like(flux)
    safe_flux[valid_pixel] = flux[valid_pixel]
    flux_bin = np.asarray(rebin_matrix @ safe_flux).ravel()
    invalid_output = coverage_fraction < min_coverage
    flux_bin[invalid_output] = np.nan

    # Input covariance (full matrix if supplied, else diagonal from err).
    if covariance is not None:
        if covariance.shape != (n_input, n_input):
            raise ValueError(f"covariance must have shape ({n_input}, {n_input}).")
        covariance_input = covariance.tocsr() if issparse(covariance) else csr_matrix(
            np.asarray(covariance, dtype=float))
    else:
        variance = np.zeros(n_input, dtype=float)
        variance[valid_pixel] = err[valid_pixel] ** 2
        covariance_input = diags(variance, offsets=0, format="csr")

    validity_operator = diags(valid_pixel.astype(float), offsets=0, format="csr")
    covariance_input = (validity_operator @ covariance_input @ validity_operator).tocsr()

    covariance_bin = (rebin_matrix @ covariance_input @ rebin_matrix.T).tocsr()
    variance_bin = covariance_bin.diagonal()
    tolerance = 100.0 * np.finfo(float).eps * max(1.0, np.nanmax(np.abs(variance_bin)))
    if np.any(variance_bin < -tolerance):
        raise ValueError(
            "The propagated covariance contains negative variances; the "
            "supplied covariance may not be positive semidefinite."
        )
    variance_bin = np.clip(variance_bin, 0.0, None)
    err_bin = np.sqrt(variance_bin)
    err_bin[invalid_output] = np.nan

    with np.errstate(divide="ignore", invalid="ignore"):
        snr_bin = flux_bin / err_bin
    snr_bin[~np.isfinite(snr_bin)] = np.nan

    return RebinResult(
        wave=wave_bin, flux=flux_bin, flux_unc=err_bin, snr=snr_bin,
        bin_width=bin_width, coverage_fraction=coverage_fraction,
        covariance=covariance_bin if return_covariance else None,
        rebin_matrix=rebin_matrix,
    )


# ---------------------------------------------------------------------------
# Public entry point: gap-aware preprocessing + flux-conservative binning
# ---------------------------------------------------------------------------

def rebin_spectrum(
    wave, flux, err, binsize, *,
    covariance=None, min_coverage=1.0, keep_partial_bin=False,
    return_covariance=True, fill_gaps=False, uniform_grid=False,
    uniform_step=None, max_gap_pixels=None, weighted=False,
) -> RebinResult:
    """Flux-conservatively rebin a spectrum, with explicit gap handling.

    Parameters
    ----------
    wave, flux, err : array-like
        Input pixel-center wavelength, flux density, and 1-sigma
        uncertainty. ``wave`` must be strictly increasing (sort first if
        needed).
    binsize : int
        Number of processed pixels per output bin. Use ``binsize=1`` to
        pass the spectrum through unchanged (e.g. for uniform-grid
        resampling only).
    covariance : array-like or scipy sparse matrix, optional
        Full input-pixel covariance matrix. If omitted, an independent-pixel
        diagonal covariance is built from ``err``.
    min_coverage : float, default 1.0
        Minimum fraction of an output bin's wavelength range that must be
        covered by valid input pixels; otherwise the bin is NaN. Use this
        (rather than filling) to make instrumental gaps propagate honestly
        into the output rather than being silently interpolated over.
    keep_partial_bin : bool, default False
        Keep a final bin with fewer than ``binsize`` pixels.
    fill_gaps : bool, default False
        If True, interpolate across missing/invalid native pixels before
        binning (bounded by ``max_gap_pixels``). If False, gaps are only
        handled via ``min_coverage`` at the binning step.
    uniform_grid : bool, default False
        If True, resample onto a uniform wavelength grid (step
        ``uniform_step``, or the median native spacing) before binning.
        Implies gap-filling interpolation.
    uniform_step : float, optional
        Spacing for the uniform grid. Defaults to the median native
        spacing.
    max_gap_pixels : int, optional
        Maximum number of consecutive missing native pixels that may be
        bridged by interpolation. None means no limit (never extrapolates
        past the valid-data edges regardless).
    weighted : bool, default False
        If True, gap-filling interpolation is additionally inverse-variance
        weighted between the two bracketing valid pixels.
    return_covariance : bool, default True
        Whether to return the full output covariance matrix.

    Returns
    -------
    RebinResult
    """
    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)
    err = np.asarray(err, dtype=float)

    if not (wave.ndim == flux.ndim == err.ndim == 1):
        raise ValueError("wave, flux, and err must be one-dimensional.")
    if not (wave.size == flux.size == err.size):
        raise ValueError("wave, flux, and err must have equal lengths.")
    if wave.size < 2:
        raise ValueError("At least two wavelength pixels are required.")
    if not np.all(np.isfinite(wave)) or not np.all(np.diff(wave) > 0):
        raise ValueError("wave must be finite and strictly increasing.")
    if max_gap_pixels is not None and int(max_gap_pixels) < 1:
        raise ValueError("max_gap_pixels must be >= 1 or None.")

    n_input = wave.size
    valid_input = np.isfinite(flux) & np.isfinite(err) & (err > 0)
    if np.count_nonzero(valid_input) < 2:
        raise ValueError("At least two valid pixels are required.")

    # Build input covariance once, up front.
    if covariance is not None:
        if covariance.shape != (n_input, n_input):
            raise ValueError(f"covariance must have shape ({n_input}, {n_input}).")
        covariance_input = covariance.tocsr() if issparse(covariance) else csr_matrix(
            np.asarray(covariance, dtype=float))
    else:
        variance = np.zeros(n_input, dtype=float)
        variance[valid_input] = err[valid_input] ** 2
        covariance_input = diags(variance, offsets=0, shape=(n_input, n_input), format="csr")

    validity_matrix = diags(valid_input.astype(float), offsets=0, shape=(n_input, n_input), format="csr")
    covariance_input = (validity_matrix @ covariance_input @ validity_matrix).tocsr()

    safe_flux = np.zeros(n_input, dtype=float)
    safe_flux[valid_input] = flux[valid_input]

    needs_preprocessing = fill_gaps or uniform_grid

    if not needs_preprocessing:
        # No gap-filling requested: gaps are handled purely through
        # min_coverage at the binning step.
        return _flux_conserving_bin(
            wave, flux, err, binsize, covariance=covariance_input,
            min_coverage=min_coverage, keep_partial_bin=keep_partial_bin,
            return_covariance=return_covariance,
        )

    # --- Preprocessing grid: native (gap-filled) or uniform -------------
    if uniform_grid:
        step = float(uniform_step) if uniform_step is not None else float(np.nanmedian(np.diff(wave)))
        if not np.isfinite(step) or step <= 0:
            raise ValueError("uniform_step must be finite and positive.")
        n_uniform = int(np.floor((wave[-1] - wave[0]) / step)) + 1
        wave_work = wave[0] + step * np.arange(n_uniform, dtype=float)
    else:
        wave_work = wave.copy()
    n_work = wave_work.size

    valid_indices = np.flatnonzero(valid_input)
    valid_wave = wave[valid_indices]
    wavelength_tolerance = 100.0 * np.finfo(float).eps * max(1.0, np.nanmax(np.abs(wave)))

    rows, cols, vals = [], [], []
    for work_index, target_wave in enumerate(wave_work):
        insertion = np.searchsorted(wave, target_wave)
        exact_native_index = None
        for candidate in (i for i in (insertion, insertion - 1) if 0 <= i < n_input):
            if abs(wave[candidate] - target_wave) <= wavelength_tolerance and valid_input[candidate]:
                exact_native_index = candidate
                break
        if exact_native_index is not None:
            rows.append(work_index)
            cols.append(exact_native_index)
            vals.append(1.0)
            continue

        valid_position = np.searchsorted(valid_wave, target_wave)
        if valid_position == 0 or valid_position == valid_wave.size:
            continue  # never extrapolate past valid-data coverage
        left_index = valid_indices[valid_position - 1]
        right_index = valid_indices[valid_position]
        if not (wave[left_index] < target_wave < wave[right_index]):
            continue

        missing_between = right_index - left_index - 1
        if max_gap_pixels is not None and missing_between > max_gap_pixels:
            continue

        span = wave[right_index] - wave[left_index]
        if span <= 0:
            continue
        right_frac = (target_wave - wave[left_index]) / span
        left_frac = 1.0 - right_frac

        if weighted:
            lw = left_frac / err[left_index] ** 2
            rw = right_frac / err[right_index] ** 2
            wsum = lw + rw
            if not np.isfinite(wsum) or wsum <= 0:
                continue
            lw, rw = lw / wsum, rw / wsum
        else:
            lw, rw = left_frac, right_frac

        rows.extend([work_index, work_index])
        cols.extend([left_index, right_index])
        vals.extend([lw, rw])

    interpolation_matrix = csr_matrix((vals, (rows, cols)), shape=(n_work, n_input))
    flux_work = np.asarray(interpolation_matrix @ safe_flux).ravel()
    covariance_work = (interpolation_matrix @ covariance_input @ interpolation_matrix.T).tocsr()
    work_variance = covariance_work.diagonal()
    support_count = np.asarray(interpolation_matrix.getnnz(axis=1)).ravel()
    valid_work = (support_count > 0) & np.isfinite(work_variance) & (work_variance > 0)
    flux_work[~valid_work] = np.nan
    err_work = np.full(n_work, np.nan, dtype=float)
    err_work[valid_work] = np.sqrt(np.clip(work_variance[valid_work], 0.0, None))

    if binsize <= 1 and not uniform_grid:
        # Gap-filled but not rebinned/resampled further.
        rebin_matrix = interpolation_matrix
        bin_width = np.diff(np.concatenate((
            [wave_work[0] - 0.5 * (wave_work[1] - wave_work[0])],
            0.5 * (wave_work[:-1] + wave_work[1:]),
            [wave_work[-1] + 0.5 * (wave_work[-1] - wave_work[-2])],
        )))
        coverage_fraction = valid_work.astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            snr_work = flux_work / err_work
        snr_work[~np.isfinite(snr_work)] = np.nan
        return RebinResult(
            wave=wave_work, flux=flux_work, flux_unc=err_work, snr=snr_work,
            bin_width=bin_width, coverage_fraction=coverage_fraction,
            covariance=covariance_work if return_covariance else None,
            rebin_matrix=rebin_matrix,
        )

    result = _flux_conserving_bin(
        wave_work, flux_work, err_work, binsize, covariance=covariance_work,
        min_coverage=min_coverage, keep_partial_bin=keep_partial_bin,
        return_covariance=return_covariance,
    )
    result.rebin_matrix = (result.rebin_matrix @ interpolation_matrix).tocsr()
    return result


# ---------------------------------------------------------------------------
# Semantic wrappers: same method, different application to the Spectrum
# ---------------------------------------------------------------------------

def apply_permanent_rebinning(spectrum, binsize, **kwargs) -> "Spectrum":
    """Permanently rebin ``spectrum`` and return a new, rebinned Spectrum.

    Only call this when rebinning is requested in configuration, once
    immediately after loading/validation and before any fitting. The
    returned spectrum (and its mask) is what all subsequent fits and saved
    products should use.

    ``spectrum`` is expected to expose ``wave``, ``flux``, ``flux_unc``,
    and (optionally) ``mask`` attributes, matching ``core.spectrum.Spectrum``.
    """
    result = rebin_spectrum(spectrum.wave, spectrum.flux, spectrum.flux_unc, binsize, **kwargs)
    new_spectrum = spectrum.__class__(
        wave=result.wave, flux=result.flux, flux_unc=result.flux_unc,
        **{k: v for k, v in vars(spectrum).items() if k not in ("wave", "flux", "flux_unc", "mask")},
    )
    if getattr(spectrum, "mask", None) is not None:
        # Resample the boolean mask through the same operator: a bin is
        # considered masked if any contributing input pixel was masked.
        mask_float = np.asarray(spectrum.mask, dtype=float)
        new_spectrum.mask = np.asarray(result.rebin_matrix @ mask_float).ravel() > 0
    return new_spectrum


def compute_display_smoothing(spectrum, binsize, **kwargs):
    """Compute display-only smoothed arrays; never modifies ``spectrum``.

    Uses the same flux-conservative binning as permanent rebinning, at
    (typically) a coarser ``binsize``, purely for GUI visualization.
    Covariance propagation is skipped by default for speed since only a
    display uncertainty envelope is needed.

    Returns
    -------
    wave_display, flux_display, flux_unc_display : np.ndarray
    """
    kwargs.setdefault("return_covariance", False)
    result = rebin_spectrum(spectrum.wave, spectrum.flux, spectrum.flux_unc, binsize, **kwargs)
    return result.wave, result.flux, result.flux_unc
