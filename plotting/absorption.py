"""Plotting wrappers for ``absorption.absorption_results.AbsorptionFitResult``.

The flagship visualization here is :func:`plot_absorption_velocity_stack`,
FitSpec's non-interactive counterpart to RDGEN's cursor-mode stacked
velocity plot (its ``v``/``u`` commands, Section 6.1 of the RDGEN
manual): every fitted transition stacked on one shared velocity axis
relative to the systemic redshift, with each kinematic component's
fitted velocity marked -- letting the reader see at a glance whether
the same kinematic structure is present across every transition in the
group. As elsewhere, plotting composes the generic
``plotting.spectrum``/``residuals``/``stamps``/``diagnostics``
primitives rather than duplicating logic.
"""
from __future__ import annotations

import numpy as np

from absorption.absorption_results import AbsorptionFitResult
from plotting.spectrum import plot_spectrum_fit
from plotting.stamps import plot_line_stamps, plot_velocity_stack

__all__ = [
    "plot_absorption_fit", "plot_absorption_line_stamps", "plot_absorption_velocity_stack",
    "plot_absorption_component_summary",
]


def _get(config, key, default=None):
    return config.get(key, default) if (config is not None and hasattr(config, "get")) else default


def plot_absorption_fit(result: AbsorptionFitResult, *, show_residuals: bool = True, title=None, **kwargs):
    """Full-spectrum transmission data+model(+residual) plot for an absorption fit."""
    n = result.fit_result.parameters.n_components
    coverage_note = "" if not result.partial_coverage else f", C_f={result.covering_fraction:.2f}"
    return plot_spectrum_fit(
        result.fit_result, show_residuals=show_residuals,
        title=(title or f"Absorption fit ({n} component(s){coverage_note})"),
        ylabel="Transmission", **kwargs,
    )


def plot_absorption_line_stamps(result: AbsorptionFitResult, *, velocity_half_width_kms: float = 300.0, **kwargs):
    """Grid of per-transition zoomed panels. See ``plotting.stamps.plot_line_stamps``."""
    return plot_line_stamps(
        result.fit_result.wave, result.fit_result.flux, result.transitions,
        flux_unc=result.fit_result.flux_unc, model=result.fit_result.model,
        redshift=result.fit_result.redshift, velocity_half_width_kms=velocity_half_width_kms, **kwargs,
    )


def plot_absorption_velocity_stack(
    result: AbsorptionFitResult, *, config=None, transitions=None, velocity_range_kms=None, **kwargs,
):
    """Stacked common-velocity-scale plot (RDGEN-style) across the fitted transitions.

    Parameters
    ----------
    transitions : list[AtomicTransition], optional
        Subset/order to stack; defaults to
        ``absorption_fig_plot_transitions`` from ``config`` if given,
        else every transition in the fit, in fit order.
    velocity_range_kms : (float, float), optional
        Defaults to +/- ``absorption_fig_velocity_limit_kms`` from
        ``config`` if given, else +/-500 km/s.
    """
    if transitions is None:
        requested = _get(config, "absorption_fig_plot_transitions", None)
        if requested:
            names = [name.strip() for name in str(requested).split(",") if name.strip()]
            by_name = {transition.name: transition for transition in result.transitions}
            transitions = [by_name[name] for name in names if name in by_name]
        if not transitions:
            transitions = result.transitions

    if velocity_range_kms is None:
        limit = float(_get(config, "absorption_fig_velocity_limit_kms", 500.0))
        velocity_range_kms = (-limit, limit)

    component_velocities = [measurement.velocity_kms for measurement in result.measurements]
    return plot_velocity_stack(
        result.fit_result.wave, result.fit_result.flux, transitions,
        flux_unc=result.fit_result.flux_unc, model=result.fit_result.model,
        reference_redshift=result.fit_result.redshift, velocity_range_kms=velocity_range_kms,
        component_velocities_kms=component_velocities, title="Absorption velocity stack", **kwargs,
    )


def plot_absorption_component_summary(result: AbsorptionFitResult, *, figsize=(9, 4), fontsize=10):
    """Per-component logN/b/velocity summary: error bars vs. velocity,
    with insignificant (frozen/upper-limit) components marked distinctly
    -- see ``AbsorptionComponentMeasurement.is_upper_limit`` (set by
    ``absorption.rejection.fit_joint_absorption_spectrum_with_rejection``).
    """
    import matplotlib.pyplot as plt

    fig, (ax_n, ax_b) = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    for measurement in result.measurements:
        color = "0.6" if measurement.is_upper_limit else "C0"
        marker = "v" if measurement.is_upper_limit else "o"
        n_uncertainty = 0.0 if not np.isfinite(measurement.logN_uncertainty) else measurement.logN_uncertainty
        ax_n.errorbar(measurement.velocity_kms, measurement.logN,
                       yerr=(None if measurement.is_upper_limit else n_uncertainty),
                       xerr=measurement.velocity_kms_uncertainty if np.isfinite(measurement.velocity_kms_uncertainty) else None,
                       fmt=marker, color=color, capsize=3)
        b_uncertainty = 0.0 if not np.isfinite(measurement.b_kms_uncertainty) else measurement.b_kms_uncertainty
        ax_b.errorbar(measurement.velocity_kms, measurement.b_kms,
                       yerr=(None if measurement.is_upper_limit else b_uncertainty),
                       xerr=measurement.velocity_kms_uncertainty if np.isfinite(measurement.velocity_kms_uncertainty) else None,
                       fmt=marker, color=color, capsize=3)

    ax_n.set_xlabel("Velocity (km/s)", fontsize=fontsize)
    ax_n.set_ylabel("log N (cm$^{-2}$)", fontsize=fontsize)
    ax_n.set_title("Column density per component", fontsize=fontsize + 1)
    ax_b.set_xlabel("Velocity (km/s)", fontsize=fontsize)
    ax_b.set_ylabel("b (km/s)", fontsize=fontsize)
    ax_b.set_title("Doppler parameter per component", fontsize=fontsize + 1)
    for ax in (ax_n, ax_b):
        ax.tick_params(labelsize=fontsize - 1)
    return fig
