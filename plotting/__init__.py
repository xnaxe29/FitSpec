"""FitSpec plotting: generic, reusable visualization for any science module's results.

Plotting is kept strictly separate from fitting: every function here
receives a result object (``core.results.FitResult`` or a science
module's wrapper around it) and never runs a science calculation
itself. Three layers:

* Generic primitives (``plotting.spectrum``, ``plotting.residuals``,
  ``plotting.stamps``, ``plotting.diagnostics``) work on any
  ``FitResult`` or bare wave/flux arrays, and are shared by every
  science module.
* Per-module wrappers (``plotting.emission``, ``plotting.absorption``,
  ``plotting.stellar_plotting``) compose those primitives with
  module-specific context (a fitted line list, per-component
  kinematics, stellar population diagnostics) rather than duplicating
  plotting logic.
* ``plotting.stamps.plot_velocity_stack`` is FitSpec's non-interactive
  counterpart to RDGEN's signature stacked common-velocity-scale plot
  (Section 6.1 of the RDGEN manual): several transitions/lines on one
  shared velocity axis relative to a reference redshift, used by both
  the emission and absorption wrappers.
"""
from plotting.spectrum import plot_spectrum_only, plot_spectrum_fit, plot_spectrum_fit_axes
from plotting.residuals import add_residual_panel, plot_residuals
from plotting.stamps import wavelength_to_velocity_kms, plot_line_stamps, plot_velocity_stack
from plotting.diagnostics import plot_residual_diagnostics
from plotting.emission import (
    plot_emission_fit, plot_emission_line_stamps, plot_emission_velocity_stack, plot_emission_line_fluxes,
)
from plotting.absorption import (
    plot_absorption_fit, plot_absorption_line_stamps, plot_absorption_velocity_stack,
    plot_absorption_component_summary,
)

__all__ = [
    "plot_spectrum_only", "plot_spectrum_fit", "plot_spectrum_fit_axes",
    "add_residual_panel", "plot_residuals",
    "wavelength_to_velocity_kms", "plot_line_stamps", "plot_velocity_stack",
    "plot_residual_diagnostics",
    "plot_emission_fit", "plot_emission_line_stamps", "plot_emission_velocity_stack", "plot_emission_line_fluxes",
    "plot_absorption_fit", "plot_absorption_line_stamps", "plot_absorption_velocity_stack",
    "plot_absorption_component_summary",
]
