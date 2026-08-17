"""Generic data+model spectrum plotting, shared across every science module.

Operates on ``core.results.FitResult`` directly, so the same function
works for stellar, emission, and absorption fits alike -- plotting is
kept separate from fitting and receives result objects rather than
running any science calculation itself, per the project's plotting
design principle.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from core.results import FitResult

__all__ = ["plot_spectrum_fit", "plot_spectrum_only"]


def plot_spectrum_only(
    wave, flux, *, flux_unc=None, model=None, mask=None, ax=None,
    wave_range=None, title=None, xlabel="Wavelength (\u00c5)", ylabel="Flux",
    data_label="data", model_label="model", figsize=(10, 4), fontsize=11, show_legend=True,
):
    """Plot flux (optionally with error shading, a model curve, and masked
    points dimmed) on a single axes. The building block every other
    plotting function in this module composes.

    Returns
    -------
    matplotlib.axes.Axes
    """
    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    finite = np.isfinite(wave) & np.isfinite(flux)
    used = finite if mask is None else finite & np.asarray(mask, dtype=bool)
    excluded = finite & ~used if mask is not None else np.zeros_like(finite)

    if flux_unc is not None:
        flux_unc = np.asarray(flux_unc, dtype=float)
        good = used & np.isfinite(flux_unc)
        ax.fill_between(wave[good], (flux - flux_unc)[good], (flux + flux_unc)[good],
                         color="0.8", step="mid", zorder=1, label="_nolegend_")

    if np.any(excluded):
        ax.plot(wave[excluded], flux[excluded], ".", ms=2, alpha=0.25, color="0.5", label="masked")
    ax.plot(wave[used], flux[used], ".", ms=2, alpha=0.6, color="C0", label=data_label)

    if model is not None:
        model = np.asarray(model, dtype=float)
        ax.plot(wave[finite], model[finite], lw=1.5, color="C1", label=model_label)

    if wave_range is not None:
        ax.set_xlim(*wave_range)
    if title:
        ax.set_title(title, fontsize=fontsize + 1)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.tick_params(labelsize=fontsize - 1)
    if show_legend:
        ax.legend(fontsize=fontsize - 2, loc="best")
    return ax


def plot_spectrum_fit(
    fit_result: FitResult, *, show_residuals: bool = True, wave_range=None,
    title=None, xlabel="Wavelength (\u00c5)", ylabel="Flux",
    figsize=(10, 6), fontsize=11,
):
    """Standard data+model(+residual panel) figure for any ``FitResult``.

    Works identically for stellar, emission, and absorption results,
    since all three share the same ``FitResult`` shape (wave, flux,
    flux_unc, mask, model). Per-module wrappers (``plotting.emission``,
    ``plotting.absorption``, ``plotting.stellar_plotting``) call this
    directly rather than re-implementing the layout.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if show_residuals:
        fig, (ax_main, ax_resid) = plt.subplots(
            2, 1, figsize=figsize, sharex=True,
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.06},
            constrained_layout=True,
        )
    else:
        fig, ax_main = plt.subplots(figsize=figsize, constrained_layout=True)
        ax_resid = None

    plot_spectrum_fit_axes(fit_result, ax_main, ax_resid, wave_range=wave_range,
                            title=title, xlabel=xlabel, ylabel=ylabel, fontsize=fontsize)
    return fig


def plot_spectrum_fit_axes(
    fit_result: FitResult, ax_main, ax_resid=None, *, wave_range=None,
    title=None, xlabel="Wavelength (\u00c5)", ylabel="Flux", fontsize=11,
):
    """Draw a FitResult's data/model (and, if ``ax_resid`` is given, its
    residual panel) onto already-created axes -- the composable form of
    :func:`plot_spectrum_fit`, for callers building their own multi-panel
    layouts (e.g. alongside a velocity stack or a diagnostics panel).
    """
    plot_spectrum_only(
        fit_result.wave, fit_result.flux, flux_unc=fit_result.flux_unc, model=fit_result.model,
        mask=fit_result.mask, ax=ax_main, wave_range=wave_range, title=title,
        xlabel=("" if ax_resid is not None else xlabel), ylabel=ylabel, fontsize=fontsize,
    )
    if ax_resid is not None:
        _plot_residual_axes(fit_result, ax_resid, wave_range=wave_range, xlabel=xlabel, fontsize=fontsize)
    return ax_main, ax_resid


def _plot_residual_axes(fit_result: FitResult, ax, *, wave_range=None, xlabel="Wavelength (\u00c5)", fontsize=11):
    from plotting.residuals import add_residual_panel
    return add_residual_panel(fit_result, ax=ax, wave_range=wave_range, xlabel=xlabel, fontsize=fontsize)
