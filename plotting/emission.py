"""Plotting wrappers for ``emission.emission_results.EmissionFitResult``.

Composes the generic primitives in ``plotting.spectrum``/``residuals``/
``stamps``/``diagnostics`` with emission-specific context (the fitted
line list, per-component kinematics) rather than duplicating any
plotting logic -- per the "define universal functionality once" design
principle. Config keys consumed where given (``emission_fig_*``, see
``config/default_config_emission.dat``) are optional; every function
also works with sensible defaults if no config is passed.
"""
from __future__ import annotations

import numpy as np

from emission.emission_results import EmissionFitResult
from plotting.spectrum import plot_spectrum_fit
from plotting.stamps import plot_line_stamps, plot_velocity_stack
from plotting.diagnostics import plot_residual_diagnostics

__all__ = ["plot_emission_fit", "plot_emission_line_stamps", "plot_emission_velocity_stack", "plot_emission_line_fluxes"]


def _get(config, key, default=None):
    return config.get(key, default) if (config is not None and hasattr(config, "get")) else default


def plot_emission_fit(result: EmissionFitResult, *, show_residuals: bool = True, title=None, **kwargs):
    """Full-spectrum data+model(+residual) plot for an emission fit."""
    return plot_spectrum_fit(
        result.fit_result, show_residuals=show_residuals,
        title=(title or f"Emission fit ({result.fit_result.parameters.n_components} component(s))"),
        **kwargs,
    )


def plot_emission_line_stamps(result: EmissionFitResult, *, config=None, velocity_half_width_kms=None, **kwargs):
    """Grid of per-fitted-line zoomed panels. See ``plotting.stamps.plot_line_stamps``."""
    half_width = velocity_half_width_kms
    if half_width is None:
        half_width = float(_get(config, "emission_fig_velocity_limit_kms", 500.0))
    return plot_line_stamps(
        result.fit_result.wave, result.fit_result.flux, result.line_list,
        flux_unc=result.fit_result.flux_unc, model=result.fit_result.model,
        redshift=result.fit_result.redshift, velocity_half_width_kms=half_width, **kwargs,
    )


def plot_emission_velocity_stack(result: EmissionFitResult, *, config=None, lines=None, velocity_range_kms=None, **kwargs):
    """Stacked common-velocity-scale plot (RDGEN-style) across the fitted lines.

    Parameters
    ----------
    lines : list[EmissionLine], optional
        Subset/order of lines to stack; defaults to
        ``emission_fig_plot_lines`` from ``config`` if given, else every
        fitted line, in fit order.
    """
    if lines is None:
        requested = _get(config, "emission_fig_plot_lines", None)
        if requested:
            names = [name.strip() for name in str(requested).split(",") if name.strip()]
            by_name = {line.name: line for line in result.line_list}
            lines = [by_name[name] for name in names if name in by_name]
        if not lines:
            lines = result.line_list

    half_range = velocity_range_kms
    if half_range is None:
        limit = float(_get(config, "emission_fig_velocity_limit_kms", 500.0))
        half_range = (-limit, limit)

    y_boost = float(_get(config, "emission_fig_y_boost", 1.0))
    return plot_velocity_stack(
        result.fit_result.wave, result.fit_result.flux, lines,
        flux_unc=result.fit_result.flux_unc, model=result.fit_result.model,
        reference_redshift=result.fit_result.redshift, velocity_range_kms=half_range,
        component_velocities_kms=list(result.component_velocities_kms), y_boost=y_boost,
        title="Emission-line velocity stack", **kwargs,
    )


def plot_emission_line_fluxes(result: EmissionFitResult, *, figsize=(8, 4), fontsize=10):
    """Bar chart of every measured line's integrated flux (+ uncertainty)."""
    import matplotlib.pyplot as plt

    names = [measurement.name for measurement in result.measurements]
    fluxes = np.array([measurement.integrated_flux for measurement in result.measurements])
    errors = np.array([measurement.integrated_flux_uncertainty for measurement in result.measurements])

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    positions = np.arange(len(names))
    ax.bar(positions, fluxes, yerr=errors, color="C0", alpha=0.8, capsize=3)
    ax.set_xticks(positions)
    ax.set_xticklabels(names, rotation=60, ha="right", fontsize=fontsize - 1)
    ax.set_ylabel("Integrated flux", fontsize=fontsize)
    ax.set_title("Fitted emission-line fluxes", fontsize=fontsize + 1)
    ax.tick_params(labelsize=fontsize - 1)
    return fig
