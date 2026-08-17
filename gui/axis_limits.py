"""Shared "sensible initial view" axis-limit logic for FitSpec science GUIs.

Matplotlib's default ``Axes`` view is ``(0, 1)`` on both axes, and that is
exactly what gets shown whenever ``ax.relim()``/``ax.autoscale_view()`` fail
to find any finite data to scale to (e.g. before a plotted line has data, or
when non-finite values are mixed into an otherwise valid spectrum). This is
the root cause of a wavelength axis that opens at 0 and a flux axis that
opens at 0--1 instead of the spectrum's own range.

This module factors out the fix originally written for ``gui.stellar``
(explicit, mask-aware, outlier-robust axis limits computed up front) so the
other science GUIs can share it rather than re-deriving it.
"""
from __future__ import annotations

import numpy as np

__all__ = ["compute_sensible_limits"]


def compute_sensible_limits(wave, flux, mask=None, *, x_pad_frac=0.015, y_pad_frac=0.08, y_widen_frac=None,
                             extra_y=None, y_percentiles=None):
    """Compute robust, non-degenerate ``(xlim, ylim)`` for a spectrum plot.

    Parameters
    ----------
    wave, flux : array-like
        The plotted arrays (any frame -- rest or observed -- is fine; this
        function just describes what is finite and in range).
    mask : array-like of bool, optional
        Good-pixel mask (``True`` = keep). Only used to pick sensible y
        limits; the x range always spans the full finite wavelength array
        so the whole spectrum stays reachable.
    x_pad_frac : float
        Fractional padding added to each side of the x range.
    y_pad_frac : float
        Fractional padding added to each side of the y range.
    y_widen_frac : float, optional
        If given, widen the y range by this fraction beyond the padded
        range (e.g. ``0.2`` for "20% wider, for convenience"). Applied in
        addition to, not instead of, ``y_pad_frac``.
    extra_y : list[array-like], optional
        Additional flux-like arrays (e.g. a reference/"ghost" model) that
        should stay inside the y range even though they aren't the primary
        spectrum -- their finite values are folded into the same y-range
        calculation as the main spectrum's.
    y_percentiles : (float, float), optional
        If given (e.g. ``(1.0, 99.0)``), clip the y range to these
        percentiles instead of the true min/max -- trades off showing the
        full extent of the spectrum against isolated cosmic rays/bad
        pixels being able to blow out the view. Default (``None``) is the
        true min/max, so the full spectrum is always visible; pass this
        explicitly only if the data is known to have such outliers that
        aren't already excluded via `mask`.

    Returns
    -------
    (xlim, ylim) : tuple[tuple[float, float], tuple[float, float]] | None
        ``None`` if no finite data is available at all (caller should leave
        the axes untouched in that case).
    """
    wave = np.asarray(wave, float)
    flux = np.asarray(flux, float)
    good_x = np.isfinite(wave) & np.isfinite(flux)
    if not np.any(good_x):
        return None

    x = wave[good_x]
    xmin = float(np.nanmin(x))
    xmax = float(np.nanmax(x))
    dx = xmax - xmin
    xpad = x_pad_frac * dx if dx > 0 else max(abs(xmin) * 0.01, 1.0)
    xlim = (xmin - xpad, xmax + xpad)

    # Prefer mask-selected pixels for the y range (so masked-out
    # cosmic rays/bad pixels can't blow out the view), but fall back to all
    # finite pixels if masking would leave nothing.
    good_y = good_x if mask is None else good_x & np.asarray(mask, bool)
    if not np.any(good_y):
        good_y = good_x
    y = flux[good_y]

    if extra_y:
        parts = [y]
        for array in extra_y:
            array = np.asarray(array, float)
            finite = array[np.isfinite(array)]
            if finite.size:
                parts.append(finite)
        y = np.concatenate(parts)

    # True min/max by default, so the full extent of the spectrum is
    # always visible -- only clip to percentiles if the caller explicitly
    # asks for that outlier-robustness trade-off.
    if y_percentiles is not None and y.size >= 20:
        ylo, yhi = np.nanpercentile(y, list(y_percentiles))
    else:
        ylo, yhi = float(np.nanmin(y)), float(np.nanmax(y))
    if not np.isfinite(ylo) or not np.isfinite(yhi):
        return xlim, None
    if yhi <= ylo:
        scale = max(abs(ylo), abs(yhi), 1e-30)
        ylo -= 0.1 * scale
        yhi += 0.1 * scale

    if y_widen_frac:
        span = yhi - ylo
        extra = 0.5 * y_widen_frac * span
        ylo -= extra
        yhi += extra

    ypad = y_pad_frac * (yhi - ylo)
    ylim = (ylo - ypad, yhi + ypad)
    return xlim, ylim
