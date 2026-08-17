"""Fit-quality and posterior diagnostic plots.

Residual diagnostics operate on a deterministic FitResult. Posterior marginal
and corner-style diagnostics operate on core.results.PosteriorResult and use
only NumPy/Matplotlib, keeping external plotting dependencies optional.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from core.results import FitResult

__all__ = ["plot_residual_diagnostics", "plot_posterior_corner", "plot_posterior_marginals"]


def plot_residual_diagnostics(fit_result: FitResult, *, figsize=(10, 7), fontsize=10, bins=40):
    """Residual-vs-wavelength, a residual histogram against the expected
    unit Gaussian (if flux_unc is available), and a text panel
    summarizing the fit statistics -- a quick "did this fit go well?"
    check without needing a posterior.

    Returns
    -------
    matplotlib.figure.Figure
    """
    from plotting.residuals import add_residual_panel

    fig, axes = plt.subplot_mosaic(
        [["resid", "resid"], ["hist", "summary"]],
        figsize=figsize, height_ratios=[1.4, 1], constrained_layout=True,
    )

    add_residual_panel(fit_result, ax=axes["resid"], fontsize=fontsize)
    axes["resid"].set_title("Residuals vs. wavelength", fontsize=fontsize + 1)

    wave = np.asarray(fit_result.wave, dtype=float)
    mask = np.asarray(fit_result.mask, dtype=bool)
    finite = np.isfinite(wave) & np.isfinite(fit_result.flux) & np.isfinite(fit_result.model) & mask
    residual = fit_result.flux - fit_result.model
    ax_hist = axes["hist"]
    if fit_result.flux_unc is not None:
        flux_unc = np.asarray(fit_result.flux_unc, dtype=float)
        good = finite & np.isfinite(flux_unc) & (flux_unc > 0)
        normalized = residual[good] / flux_unc[good]
        ax_hist.hist(normalized, bins=bins, density=True, color="C0", alpha=0.75, label="residual/\u03c3")
        grid = np.linspace(-4, 4, 200)
        ax_hist.plot(grid, np.exp(-0.5 * grid ** 2) / np.sqrt(2 * np.pi), color="k", lw=1.2, label="unit Gaussian")
        ax_hist.set_xlabel("Normalized residual", fontsize=fontsize)
        ax_hist.legend(fontsize=fontsize - 2)
    else:
        ax_hist.hist(residual[finite], bins=bins, color="C0", alpha=0.75)
        ax_hist.set_xlabel("Residual", fontsize=fontsize)
    ax_hist.set_ylabel("Density", fontsize=fontsize)
    ax_hist.set_title("Residual distribution", fontsize=fontsize + 1)
    ax_hist.tick_params(labelsize=fontsize - 1)

    ax_summary = axes["summary"]
    ax_summary.axis("off")
    stats = fit_result.statistics
    rows = [
        ("Method", fit_result.method),
        ("N data (fit)", str(stats.n_data)),
        ("k params", str(stats.k_params)),
        ("dof", str(stats.dof)),
        ("chi2", f"{stats.chi_square:.4g}"),
        ("reduced chi2", f"{stats.reduced_chi_square:.4g}"),
        ("BIC", f"{stats.bic:.4g}"),
        ("AIC", f"{stats.aic:.4g}"),
        ("AICc", f"{stats.aicc:.4g}"),
        ("jitter scale", f"{stats.jitter_scale:.4g}"),
    ]
    ax_summary.table(cellText=rows, colLabels=["Statistic", "Value"], loc="center", cellLoc="left")
    ax_summary.set_title("Fit statistics", fontsize=fontsize + 1)

    return fig


def plot_posterior_corner(posterior_result, *, parameter_names=None, max_parameters=8,
                          figsize_per_parameter=2.2, bins=35, point_alpha=0.12,
                          point_size=4.0):
    """Lightweight corner-style posterior diagnostic without extra dependencies.

    Diagonal panels show one-dimensional marginalized histograms; lower-triangle
    panels show pairwise posterior samples.  To keep figures usable for very
    high-dimensional line fits, at most ``max_parameters`` are plotted unless
    the caller explicitly supplies a shorter ``parameter_names`` selection.

    Returns
    -------
    matplotlib.figure.Figure
    """
    samples = np.asarray(posterior_result.samples, dtype=float)
    all_names = list(posterior_result.parameter_names)
    if parameter_names is None:
        names = all_names[:int(max_parameters)]
    else:
        names = list(parameter_names)
        missing = [name for name in names if name not in all_names]
        if missing:
            raise ValueError(f"Unknown posterior parameter(s): {missing}")
        if len(names) > int(max_parameters):
            raise ValueError("Selected parameter count exceeds max_parameters.")
    if not names:
        raise ValueError("No posterior parameters selected for plotting.")

    indices = [all_names.index(name) for name in names]
    data = samples[:, indices]
    npar = len(names)
    fig, axes = plt.subplots(
        npar, npar, figsize=(figsize_per_parameter*npar, figsize_per_parameter*npar),
        squeeze=False, constrained_layout=True,
    )
    for row in range(npar):
        for col in range(npar):
            ax = axes[row, col]
            if row < col:
                ax.axis("off")
                continue
            if row == col:
                ax.hist(data[:, col], bins=bins, density=True, histtype="step")
                q16, q50, q84 = np.percentile(data[:, col], [16, 50, 84])
                ax.axvline(q50, lw=1.0)
                ax.axvline(q16, lw=0.8, ls="--")
                ax.axvline(q84, lw=0.8, ls="--")
            else:
                ax.scatter(data[:, col], data[:, row], s=point_size, alpha=point_alpha, rasterized=True)
            if row == npar - 1:
                ax.set_xlabel(names[col])
            else:
                ax.set_xticklabels([])
            if col == 0 and row > 0:
                ax.set_ylabel(names[row])
            elif col > 0:
                ax.set_yticklabels([])
    return fig


def plot_posterior_marginals(posterior_result, *, parameter_names=None, columns=3,
                             bins=40, figsize_per_panel=(4.0, 3.0)):
    """Plot marginalized 1-D posteriors with 16/50/84 percentile markers."""
    samples = np.asarray(posterior_result.samples, dtype=float)
    all_names = list(posterior_result.parameter_names)
    names = all_names if parameter_names is None else list(parameter_names)
    missing = [name for name in names if name not in all_names]
    if missing:
        raise ValueError(f"Unknown posterior parameter(s): {missing}")
    if not names:
        raise ValueError("No posterior parameters selected for plotting.")
    columns = max(1, int(columns))
    rows = int(np.ceil(len(names) / columns))
    fig, axes = plt.subplots(
        rows, columns,
        figsize=(figsize_per_panel[0]*columns, figsize_per_panel[1]*rows),
        squeeze=False, constrained_layout=True,
    )
    for ax, name in zip(axes.flat, names):
        values = samples[:, all_names.index(name)]
        q16, q50, q84 = np.percentile(values, [16, 50, 84])
        ax.hist(values, bins=bins, density=True, histtype="step")
        ax.axvline(q50, lw=1.2)
        ax.axvline(q16, lw=0.8, ls="--")
        ax.axvline(q84, lw=0.8, ls="--")
        ax.set_title(f"{name}\n{q50:.4g} -{q50-q16:.3g}/+{q84-q50:.3g}")
        ax.set_xlabel(name)
        ax.set_ylabel("Posterior density")
    for ax in list(axes.flat)[len(names):]:
        ax.axis("off")
    return fig
