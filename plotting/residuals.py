"""Reusable residual-panel plotting, shared by every science module."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from core.results import FitResult

__all__ = ["add_residual_panel", "plot_residuals"]


def add_residual_panel(
    fit_result: FitResult, *, ax=None, wave_range=None, xlabel="Wavelength (\u00c5)",
    normalized: bool = True, sigma_lines=(-1, 1), fontsize=11,
):
    """Draw (data - model) -- normalized by flux_unc if available -- onto ``ax``.

    Masked-out points are shown dimmed rather than dropped, so a
    residual panel makes clear both how well the fit describes the
    pixels it used and how it would have looked on the excluded ones.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 2.5))

    wave = np.asarray(fit_result.wave, dtype=float)
    finite = np.isfinite(wave) & np.isfinite(fit_result.flux) & np.isfinite(fit_result.model)
    mask = np.asarray(fit_result.mask, dtype=bool)

    residual = fit_result.flux - fit_result.model
    ylabel = "Residual"
    if normalized and fit_result.flux_unc is not None:
        flux_unc = np.asarray(fit_result.flux_unc, dtype=float)
        good = finite & np.isfinite(flux_unc) & (flux_unc > 0)
        residual = np.where(good, residual / np.where(flux_unc > 0, flux_unc, np.nan), np.nan)
        ylabel = "Residual / \u03c3"
        finite = finite & good

    used = finite & mask
    excluded = finite & ~mask
    if np.any(excluded):
        ax.scatter(wave[excluded], residual[excluded], s=4, alpha=0.25, color="0.5")
    ax.scatter(wave[used], residual[used], s=4, color="C0")
    ax.axhline(0, lw=0.8, color="k")
    if normalized and fit_result.flux_unc is not None:
        for level in sigma_lines:
            ax.axhline(level, lw=0.6, ls="--", color="0.6")
        ax.set_ylim(min(-4, min(sigma_lines) - 1), max(4, max(sigma_lines) + 1))

    if wave_range is not None:
        ax.set_xlim(*wave_range)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.tick_params(labelsize=fontsize - 1)
    return ax


def plot_residuals(fit_result: FitResult, **kwargs):
    """Standalone residual-panel figure. See :func:`add_residual_panel` for kwargs."""
    fig, ax = plt.subplots(figsize=kwargs.pop("figsize", (10, 3)))
    add_residual_panel(fit_result, ax=ax, **kwargs)
    fig.tight_layout()
    return fig
