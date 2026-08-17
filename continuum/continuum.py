"""Continuum estimation.

Ported from the original continuum_function_module.py, converted from a
stateful class (spectrum held as `self.flux`) to explicit, stateless
functions taking `wave, flux` directly, consistent with the rest of
FitSpec.

Two confirmed bugs were fixed relative to the original, both index-
alignment issues that only manifest when `flux` contains NaNs (i.e.
whenever there is a masked/bad-pixel gap):

1. In the default ("custom") method, the internal NaN-stripping step
   (``pick = isfinite(flux); flux = flux[pick]``) shrinks `flux` but the
   original code then indexed the *original, unshrunk* `wave` with
   indices computed from the *shrunk* `flux` (e.g. `wave[mask_less]`,
   and again when building `absolute_array`). Every point after a NaN
   gap was silently assigned the wrong wavelength, offset by exactly the
   gap size. Fixed by consistently indexing `wave[pick]` wherever `flux`
   had already been filtered.
2. The fallback branch (used when fewer than 4 continuum points survive
   the primary selection) built a boolean mask sized to the *original*
   `wave` but applied it to the *filtered* `flux`, which raises a
   shape-mismatch error whenever NaNs are actually present. Fixed by
   sizing the mask to the filtered array and re-deriving wavelengths
   through `wave[pick]`.

``pca_continuum`` is kept as in the original but is a mathematical
no-op: PCA with 1 component on a single-feature matrix cannot discard
any information, so it returns the input flux unchanged. This is not
fixed here (that would mean redesigning the method, not porting it) --
calling it raises ``ContinuumNoOpWarning`` rather than silently doing
nothing.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema, find_peaks, savgol_filter
from scipy import interpolate
from scipy.interpolate import interp1d
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from statsmodels.nonparametric.smoothers_lowess import lowess


__all__ = [
    "ContinuumNoOpWarning",
    "continuum_finder",
    "poly_fit_continuum",
    "spline_fit_continuum",
    "pca_continuum",
    "lowess_continuum",
    "gaussian_fit_continuum",
    "peak_find_continuum",
    "median_filter_continuum",
    "continuum_using_fof",
    "estimate_continuum",
]


class ContinuumNoOpWarning(UserWarning):
    """Raised when a continuum method is known not to do what its name implies."""


def _boxcar_smooth(y, box_pts):
    """Internal boxcar smoothing used only for extrema-finding within
    continuum_finder -- unrelated to core.rebinning's display smoothing.
    """
    box = np.ones(box_pts) / box_pts
    return np.convolve(y, box, mode="same")


# This function's structure follows Martin+2021 (2021MNRAS.500.4937M).
def continuum_finder(
    wave, flux, *pars,
    n_smooth=5, allowed_percentile=75, filter_points_len=10, data_height_upscale=2,
    poly_order=3, window_size_default=999, fwhm_galaxy_min=10, noise_level_sigma=10,
    fwhm_ratio_upscale=10, **kwargs,
):
    """Custom savgol/interpolation-based continuum finder.

    Parameters as in the original continuum_fitClass, now explicit
    keyword arguments instead of instance attributes. `*pars`, if given,
    overrides all of n_smooth..fwhm_ratio_upscale positionally (matching
    the original calling convention used by callers that pass a
    parameter vector, e.g. for optimizing the continuum-finder's own
    tuning parameters).

    Returns
    -------
    fit : callable
        Interpolating function giving the continuum at any wavelength.
    std_cont : float
        Standard deviation of (flux - continuum) at the selected
        continuum points.
    flux_valid : np.ndarray
        `flux` with non-finite pixels removed (aligned with `pick`).
    pick : np.ndarray of bool
        Which input pixels were finite (i.e. which `wave` entries
        `flux_valid` corresponds to).
    pick_2 : np.ndarray of int
        Indices into `flux_valid`/`wave[pick]` selected as continuum points.
    peaks : tuple
        Output of scipy.signal.find_peaks on the feature-detection array.
    """
    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)

    if pars:
        (n_smooth, allowed_percentile, filter_points_len, data_height_upscale, poly_order,
         window_size_default, fwhm_galaxy_min, noise_level_sigma, fwhm_ratio_upscale) = pars

    n_smooth = int(n_smooth)
    allowed_percentile = int(allowed_percentile)
    filter_points_len = int(filter_points_len)
    poly_order = int(poly_order)
    window_size_default = int(window_size_default)
    fwhm_galaxy_min = int(fwhm_galaxy_min)
    noise_level_sigma = int(noise_level_sigma)
    fwhm_ratio_upscale = int(fwhm_ratio_upscale)

    pick = np.isfinite(flux)
    flux_valid = flux[pick]
    wave_valid = wave[pick]  # kept aligned with flux_valid throughout (fixes the alignment bug)

    smoothed_data = _boxcar_smooth(flux_valid, n_smooth)
    local_std = np.median([np.std(s) for s in np.array_split(flux_valid, n_smooth)])
    mask_less = argrelextrema(smoothed_data, np.less)[0]
    mask_greater = argrelextrema(smoothed_data, np.greater)[0]
    mask_less_interpolate_func = interpolate.interp1d(
        wave_valid[mask_less], flux_valid[mask_less], kind="cubic", fill_value="extrapolate")
    mask_greater_interpolate_func = interpolate.interp1d(
        wave_valid[mask_greater], flux_valid[mask_greater], kind="cubic", fill_value="extrapolate")
    absolute_array = mask_greater_interpolate_func(wave_valid) - mask_less_interpolate_func(wave_valid)

    filter_points = np.array([int(i * len(absolute_array) / filter_points_len) for i in range(1, filter_points_len)])
    noise_height_max_default = noise_level_sigma * np.nanmin(np.array(
        [np.nanstd(absolute_array[filter_points[i] - 10:filter_points[i] + 10]) for i in range(len(filter_points))]))
    data_height_max_default = data_height_upscale * np.nanmax(
        np.array([np.abs(np.nanmax(absolute_array)), np.abs(np.nanmin(absolute_array))]))
    noise_height_max = kwargs.get("noise_height_max", noise_height_max_default)
    data_height_max = kwargs.get("data_height_max", data_height_max_default)
    peaks = find_peaks(
        absolute_array, height=[noise_height_max, data_height_max], prominence=(local_std * 3.0),
        width=[fwhm_galaxy_min, int(fwhm_ratio_upscale * fwhm_galaxy_min)])
    edges = np.int32([np.round(peaks[1]["left_ips"]), np.round(peaks[1]["right_ips"])])

    d = np.diff(flux_valid, n=1)
    w = 1.0 / np.concatenate((np.asarray([np.median(d)] * 1), d))
    w[0] = np.max(w)
    w[-1] = np.max(w)
    for edge in edges.T:
        diff_tmp = int((edge[1] - edge[0]) / 2)
        w[edge[0] - diff_tmp:edge[1] + diff_tmp] = 1.0 / 10000.0
    w = np.abs(w)

    pick_2 = np.where(w > np.percentile(w, allowed_percentile * (float(len(flux_valid)) / float(len(wave)))))[0]

    if len(wave_valid[pick_2]) > 3:
        xx = np.linspace(np.min(wave_valid[pick_2]), np.max(wave_valid[pick_2]), 1000)
        itp = interpolate.interp1d(wave_valid[pick_2], flux_valid[pick_2], kind="linear")
    else:
        keep = np.ones_like(flux_valid, dtype=np.bool_)  # fixed: sized to flux_valid, not wave
        ynew = np.abs(np.diff(flux_valid[keep], prepend=1e-10))
        ynew2 = np.percentile(ynew, allowed_percentile)
        xx = wave_valid[keep][ynew < ynew2]  # fixed: derived from wave_valid, not raw wave
        y_rev = flux_valid[keep][ynew < ynew2]
        itp = interpolate.interp1d(xx, y_rev, axis=0, fill_value="extrapolate", kind="linear")

    window_size = int(fwhm_ratio_upscale * fwhm_galaxy_min)
    if window_size >= len(xx):
        window_size = len(xx) - 1
    if window_size % 2 == 0:
        window_size = window_size + 1
    window_size = max(window_size, poly_order + 1 + (poly_order % 2 == 0))  # guard: savgol needs window > poly_order
    fit_savgol = savgol_filter(itp(xx), window_size, poly_order)
    fit = interpolate.interp1d(xx, fit_savgol, kind="cubic", fill_value="extrapolate")
    std_cont = np.std(flux_valid[pick_2] - fit(wave_valid[pick_2]))

    return fit, std_cont, flux_valid, pick, pick_2, peaks, std_cont


def poly_fit_continuum(wave, flux, *, poly_order=3):
    """Polynomial continuum fit."""
    coefficients = np.polyfit(wave, flux, poly_order)
    return np.polyval(coefficients, wave)


def spline_fit_continuum(wave, flux):
    """Cubic-spline continuum fit, evaluated at the input wavelengths."""
    continuum = interp1d(wave, flux, kind="cubic")
    return continuum(wave)


def pca_continuum(wave, flux, *, n_components=1):
    """PCA-based continuum estimate.

    NOTE: as implemented (single-feature matrix, n_components=1), this
    is a mathematical no-op and returns `flux` unchanged -- kept as in
    the original rather than redesigned; see module docstring.
    """
    warnings.warn(
        "pca_continuum has no effect as currently implemented (PCA on a "
        "single-feature matrix with 1 component cannot discard any "
        "information); it returns the input flux unchanged.",
        ContinuumNoOpWarning, stacklevel=2,
    )
    pca = PCA(n_components=n_components)
    X = flux.reshape(-1, 1)
    pca.fit(X)
    return pca.inverse_transform(pca.transform(X)).flatten()


def lowess_continuum(wave, flux, *, frac=0.05):
    """LOWESS-smoothed continuum estimate."""
    return lowess(flux, wave, frac=frac)[:, 1]


def _local_gaussian_with_offset(x, a, b, c, d):
    return a * np.exp(-((x - b) / c) ** 2) + d


def gaussian_fit_continuum(wave, flux, *, window=200):
    """Local-Gaussian-fit continuum estimate (one fit per pixel, in a sliding window)."""
    continuum = np.zeros_like(flux)
    for i in range(len(flux)):
        low = max(0, i - window // 2)
        high = min(len(flux), i + window // 2)
        x = wave[low:high]
        y = flux[low:high]
        try:
            popt, _ = curve_fit(_local_gaussian_with_offset, x, y, p0=[1, wave[i], 5, 0])
            continuum[i] = _local_gaussian_with_offset(wave[i], *popt)
        except (RuntimeError, ValueError):  # broadened from RuntimeError-only: curve_fit also
            continuum[i] = np.nan          # raises ValueError on NaN input in the local window
    mask = np.isnan(continuum)
    if np.all(mask):
        raise ValueError("gaussian_fit_continuum: every local fit failed; no valid continuum points.")
    continuum[mask] = np.interp(wave[mask], wave[~mask], continuum[~mask])
    return continuum


def peak_find_continuum(wave, flux, *, prominence=0.05, width=10):
    """Continuum via linear interpolation between detected peaks/troughs."""
    peaks, _ = find_peaks(flux, prominence=prominence, width=width)
    troughs, _ = find_peaks(-flux, prominence=prominence, width=width)
    indices = np.concatenate([peaks, troughs, [0, len(flux) - 1]])
    return np.interp(wave, wave[indices], flux[indices])


def median_filter_continuum(wave, flux, *, window=101):
    """Sliding-median continuum estimate (NaN-safe)."""
    continuum = np.zeros_like(flux)
    for i in range(len(flux)):
        low = max(0, i - window // 2)
        high = min(len(flux), i + window // 2)
        continuum[i] = np.nanmedian(flux[low:high])
    mask = np.isnan(continuum)
    if np.any(mask):
        continuum[mask] = np.interp(wave[mask], wave[~mask], continuum[~mask])
    return continuum


def continuum_using_fof(wave, flux, *, eps=3, min_samples=3):
    """Friends-of-friends (DBSCAN) continuum level.

    NOTE: this returns a single flat (wavelength-independent) value --
    the median flux of the largest wavelength-proximity cluster -- not a
    wavelength-dependent continuum shape. `eps` is an absolute
    wavelength distance, so behavior depends strongly on pixel spacing;
    kept as in the original.

    Fixed relative to the original: the original built the pairwise
    distance matrix with a custom callable metric
    (``lambda a, b: np.abs(a - b)``) that always raised ValueError,
    since scipy's cdist requires a custom metric to return a scalar but
    was passing length-1 arrays here. For 1-D wavelength data, Euclidean
    distance is exactly absolute difference, so this uses cdist's
    built-in 'euclidean' metric directly instead.
    """
    wave = np.asarray(wave, dtype=float)
    X = wave.reshape(-1, 1)
    D = cdist(X, X, metric="euclidean")
    db = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed").fit(D)
    mask = db.labels_ == 0
    return np.median(flux[mask])


_METHODS = {
    "poly": poly_fit_continuum,
    "spline": spline_fit_continuum,
    "pca": pca_continuum,
    "lowess": lowess_continuum,
    "gauss": gaussian_fit_continuum,
    "peak_find": peak_find_continuum,
    "median_filtering": median_filter_continuum,
}


def estimate_continuum(wave, flux, method="custom", **kwargs):
    """Dispatch to one of the continuum-estimation methods by name.

    Parameters
    ----------
    method : str, default "custom"
        One of "custom" (continuum_finder), "poly", "spline", "pca",
        "lowess", "gauss", "peak_find", "median_filtering", "fof".
    **kwargs
        Passed through to the selected method.

    Returns
    -------
    np.ndarray
        Continuum flux, same length as `wave` for every method except
        "custom" (which is evaluated at the finite-flux subset -- call
        the returned callable, or use `continuum_finder` directly, to
        control where it's evaluated).
    """
    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)

    if method == "custom":
        fit, std_cont, flux_valid, pick, pick_2, peaks, _ = continuum_finder(wave, flux, **kwargs)
        return fit(wave[pick])
    if method == "fof":
        return continuum_using_fof(wave, flux, **kwargs) * np.ones_like(wave)
    if method in _METHODS:
        return _METHODS[method](wave, flux, **kwargs)

    raise ValueError(f"Unknown continuum method {method!r}; expected one of "
                      f"{['custom', 'fof'] + list(_METHODS)}.")
