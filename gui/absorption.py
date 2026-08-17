"""FitSpec absorption-line GUI.

Matplotlib-widget based, in the same style as ``gui.stellar_gui`` and
``gui.emission_gui``. Each component's logN/b_kms/velocity_kms is
edited via ``gui.component_controller.ComponentController``. Unlike
emission, absorption has no per-line amplitude -- a component's
optical depth already sums over every transition in the group -- so
the component controller alone covers every per-component quantity;
when partial coverage is enabled, the single shared covering fraction
gets its own slider (bound to component 0, mirroring how the fitting
code itself treats it, see ``absorption.absorption_model``).

Cross-ion joint fitting, thermal/turbulent linking, abundance-pattern
ties, region shifts, and automatic rejection (Sections on "Advanced
Absorption Features") are code-level constructs, per their own design,
and are not exposed through this single-group interactive GUI -- build
those fits programmatically (``absorption.absorption_fit.fit_joint_absorption_spectrum``,
``absorption.rejection.fit_joint_absorption_spectrum_with_rejection``)
and use ``plotting.absorption`` to inspect the result.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider

from core.parameters import Component, Parameter
from core.results import PosteriorResult
from core.rebinning import compute_display_smoothing

from absorption.atomic import load_atomic_line_list, select_group
from absorption.absorption_model import COVERING_FRACTION_PARAMETER, build_absorption_parameters, make_absorption_model_func
from absorption.absorption_fit import fit_absorption_spectrum
from absorption.inference import run_absorption_inference
from absorption.absorption_results import save_absorption_result, load_absorption_result
from plotting.diagnostics import plot_posterior_corner

from gui.mask_controller import MaskController
from gui.component_controller import ComponentController

__all__ = ["AbsorptionGUI"]


def _get(config, key, default=None):
    return config.get(key, default) if hasattr(config, "get") else default


def _pair(config, key, default):
    value = _get(config, key, default)
    values = list(value) if isinstance(value, (list, tuple, np.ndarray)) else [x.strip() for x in str(value).split(",")]
    return (float(values[0]), float(values[1])) if len(values) >= 2 else tuple(map(float, default))


def _bool(config, key, default=False):
    value = _get(config, key, default)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y", "on"}


class AbsorptionGUI:
    """Interactive single-transition-group absorption fitting session.

    Parameters
    ----------
    spectrum : core.spectrum.Spectrum
        Should already be continuum-normalized.
    config : object supporting ``.get(key, default)``
        See ``config/default_config_absorption.dat``. ``absorption_group``
        selects the transition group unless ``transitions`` is given.
    transitions : list[absorption.atomic.AtomicTransition], optional
        Pre-selected transition group, bypassing ``absorption_group``.
    """

    def __init__(self, spectrum, config, *, transitions=None, result_path="absorption_fit.fits", mask_path=None, state=None):
        self.spectrum = spectrum
        self.config = config
        self.state = state
        if self.state is not None:
            self.state.register_panel("absorption", self)
        self.result_path = Path(result_path)
        self.posterior_path = Path(_get(config, "absorption_inference_output_path", "absorption_posterior.npz"))
        self.posterior = None
        if transitions is None:
            path = _get(config, "absorption_line_list_path", None)
            line_list = load_atomic_line_list(None if not path else path)
            group = _get(config, "absorption_group", None)
            if not group:
                raise ValueError("absorption_group must be set, or pass `transitions` directly.")
            transitions = select_group(line_list, group)
        self.transitions = transitions
        self.partial_coverage = _bool(config, "absorption_partial_coverage", False)
        self.result = None

        n_components = int(_get(config, "absorption_n_components", 1))
        self.logN_bounds = _pair(config, "absorption_logN_bounds", (8.0, 23.0))
        self.b_bounds = _pair(config, "absorption_b_bounds_kms", (1.0, 300.0))
        self.velocity_bounds = _pair(config, "absorption_velocity_bounds_kms", (-500.0, 500.0))
        self.covering_fraction_bounds = _pair(config, "absorption_covering_fraction_bounds", (0.0, 1.0))
        self.model_parameters = build_absorption_parameters(
            n_components,
            logN_initial=float(_get(config, "absorption_logN_initial", 14.0)), logN_bounds=self.logN_bounds,
            b_initial_kms=float(_get(config, "absorption_b_initial_kms", 25.0)), b_bounds_kms=self.b_bounds,
            velocity_initial_kms=float(_get(config, "absorption_velocity_initial_kms", 0.0)),
            velocity_bounds_kms=self.velocity_bounds, partial_coverage=self.partial_coverage,
            covering_fraction_initial=float(_get(config, "absorption_covering_fraction_initial", 1.0)),
            covering_fraction_bounds=self.covering_fraction_bounds,
        )
        self.model_func = make_absorption_model_func(
            transitions, redshift=spectrum.redshift, resolution=spectrum.resolution,
            partial_coverage=self.partial_coverage, subpixel=int(_get(config, "absorption_subpixel", 1)),
        )

        group_label = "+".join(sorted({transition.group for transition in transitions}))
        self.fig, self.ax = plt.subplots(figsize=(13, 8))
        plt.subplots_adjust(left=0.08, right=0.76, bottom=0.42)
        self.data_line, = self.ax.plot([], [], drawstyle="steps-mid", alpha=0.45, label="data")
        self.masked_line, = self.ax.plot([], [], ".", ms=3, alpha=0.18, label="masked")
        self.preview_line, = self.ax.plot([], [], lw=1.5, label="preview")
        self.best_line, = self.ax.plot([], [], lw=2, ls="--", label="best fit")
        self.ax.set_xlabel(r"Wavelength ($\AA$)")
        self.ax.set_ylabel("Transmission")
        coverage_note = "full coverage" if not self.partial_coverage else "partial coverage"
        self.ax.set_title(f"FitSpec absorption fitting [{group_label}, {n_components} component(s), {coverage_note}]")
        self.ax.legend(fontsize=8)

        self.smooth_slider = Slider(plt.axes([0.13, 0.32, 0.28, 0.025]), "Display bin", 1, 50, valinit=1, valstep=1)
        self.smooth_slider.on_changed(self._refresh_data)

        controlled = ["logN", "b_kms", "velocity_kms"]
        self.component_controller = ComponentController(
            self.model_parameters, controlled, make_new_component=self._make_new_component,
            on_change=lambda _mp: self._preview(),
        )
        self.component_controller.connect_matplotlib(
            count_axes=plt.axes([0.13, 0.27, 0.15, 0.03]),
            decrement_axes=plt.axes([0.29, 0.27, 0.03, 0.03]),
            increment_axes=plt.axes([0.33, 0.27, 0.03, 0.03]),
            active_axes=plt.axes([0.13, 0.09, 0.15, 0.15]),
            slider_axes={
                "logN": plt.axes([0.48, 0.32, 0.23, 0.025]),
                "b_kms": plt.axes([0.48, 0.27, 0.23, 0.025]),
                "velocity_kms": plt.axes([0.48, 0.22, 0.23, 0.025]),
            },
        )

        if self.partial_coverage:
            cf_lo, cf_hi = self.covering_fraction_bounds
            initial_cf = self.model_parameters.components[0][COVERING_FRACTION_PARAMETER].value
            self.covering_slider = Slider(plt.axes([0.48, 0.17, 0.23, 0.025]), "covering fraction",
                                           cf_lo, cf_hi, valinit=initial_cf)
            self.covering_slider.on_changed(self._on_covering_change)

        self.fit_button = Button(plt.axes([0.80, 0.09, 0.055, 0.04]), "Fit")
        self.fit_button.on_clicked(self._fit)
        self.load_button = Button(plt.axes([0.86, 0.09, 0.055, 0.04]), "Load Fit")
        self.load_button.on_clicked(self._load_fit)
        self.save_button = Button(plt.axes([0.92, 0.09, 0.055, 0.04]), "Save Fit")
        self.save_button.on_clicked(self._save_fit)

        # Posterior sampling is intentionally a second stage. The deterministic
        # fit defines the accepted component structure and any fixed state first.
        self.posterior_button = Button(plt.axes([0.78, 0.03, 0.047, 0.04]), "Posterior")
        self.posterior_button.on_clicked(self._run_posterior)
        self.posterior_load_button = Button(plt.axes([0.83, 0.03, 0.047, 0.04]), "Load P")
        self.posterior_load_button.on_clicked(self._load_posterior)
        self.posterior_save_button = Button(plt.axes([0.88, 0.03, 0.047, 0.04]), "Save P")
        self.posterior_save_button.on_clicked(self._save_posterior)
        self.posterior_plot_button = Button(plt.axes([0.93, 0.03, 0.047, 0.04]), "Plot P")
        self.posterior_plot_button.on_clicked(self._plot_posterior)

        self.mask_controller = MaskController(
            spectrum, fit_mode="absorption",
            included_intervals=_get(config, "included_intervals", None),
            excluded_intervals=_get(config, "excluded_intervals", None),
            mask_path=mask_path, on_change=lambda _mask, _reason: self._refresh_data(),
        )
        self.mask_controller.connect_matplotlib(
            self.ax, selector_check_axes=plt.axes([0.80, 0.51, 0.17, 0.055]),
            mode_axes=plt.axes([0.80, 0.39, 0.17, 0.10]),
            save_button_axes=plt.axes([0.80, 0.31, 0.075, 0.04]),
            load_button_axes=plt.axes([0.89, 0.31, 0.075, 0.04]),
            reset_button_axes=plt.axes([0.80, 0.25, 0.165, 0.04]),
            status_axes=plt.axes([0.80, 0.18, 0.17, 0.05]),
        )

        self._refresh_data()
        self._preview()

    # -- component construction -------------------------------------------------

    def _make_new_component(self) -> Component:
        logN_lo, logN_hi = self.logN_bounds
        b_lo, b_hi = self.b_bounds
        v_lo, v_hi = self.velocity_bounds
        parameters = [
            Parameter("logN", float(_get(self.config, "absorption_logN_initial", 14.0)), logN_lo, logN_hi),
            Parameter("b_kms", float(_get(self.config, "absorption_b_initial_kms", 25.0)), b_lo, b_hi),
            Parameter("velocity_kms", float(_get(self.config, "absorption_velocity_initial_kms", 0.0)), v_lo, v_hi),
        ]
        if self.partial_coverage:
            cf_lo, cf_hi = self.covering_fraction_bounds
            leader_value = self.model_parameters.components[0][COVERING_FRACTION_PARAMETER].value
            parameters.append(Parameter(COVERING_FRACTION_PARAMETER, leader_value, cf_lo, cf_hi, fixed=True))
        return Component(parameters=parameters)

    def _on_covering_change(self, value: float):
        # Covering fraction is shared: component 0 is the only free holder
        # (see absorption.absorption_model), every other component's copy
        # is fixed and synchronized to it at model-evaluation time.
        self.model_parameters.components[0][COVERING_FRACTION_PARAMETER].value = float(value)
        self._preview()

    # -- plotting -----------------------------------------------------------------

    def _refresh_data(self, *_):
        bins = max(1, int(round(self.smooth_slider.val))) if hasattr(self, "smooth_slider") else 1
        if bins > 1 and self.spectrum.flux_unc is not None:
            wave, flux, _ = compute_display_smoothing(self.spectrum, bins, min_coverage=0.5)
        else:
            wave, flux = np.asarray(self.spectrum.wave, float), np.asarray(self.spectrum.flux, float)
        self.data_line.set_data(wave, flux)
        mask = np.ones(self.spectrum.wave.size, bool) if self.spectrum.mask is None else np.asarray(self.spectrum.mask, bool)
        self.masked_line.set_data(np.asarray(self.spectrum.wave, float)[~mask], np.asarray(self.spectrum.flux, float)[~mask])
        self.ax.relim(); self.ax.autoscale_view(); self.fig.canvas.draw_idle()

    def _preview(self, *_):
        wave = np.asarray(self.spectrum.wave, float)
        self.preview_line.set_data(wave, self.model_func(wave, self.model_parameters))
        self.fig.canvas.draw_idle()

    # -- fit/save/load --------------------------------------------------------------

    def _fit(self, *_):
        self.result = fit_absorption_spectrum(self.spectrum, self.config, transitions=self.transitions)
        if self.state is not None:
            self.state.set_result("absorption", self.result)
        self.best_line.set_data(self.result.fit_result.wave, self.result.fit_result.model)
        self.fig.canvas.draw_idle()
        stats = self.result.fit_result.statistics
        logNs = [measurement.logN for measurement in self.result.measurements]
        print(f"absorption fit: reduced_chi2={stats.reduced_chi_square:.4g}, logN per component={logNs}")

    def _save_fit(self, *_):
        if self.result is None:
            self._fit()
        path = save_absorption_result(self.result_path, self.result, overwrite=True)
        print(f"Saved {path}")

    def _run_posterior(self, *_):
        if self.result is None:
            self._fit()
        self.posterior = run_absorption_inference(
            self.result, self.spectrum, self.config, transitions=self.transitions,
        )
        if self.posterior is not None and self.state is not None:
            self.state.set_posterior("absorption", self.posterior)
        if self.posterior is None:
            print(
                "Absorption posterior sampling is disabled. Set "
                "absorption_inference_method = emcee or dynesty in config.dat."
            )
            return
        summary = self.posterior.summary()
        engine = self.posterior.metadata.get("engine", "posterior")
        print(f"absorption {engine}: {self.posterior.samples.shape[0]} stored posterior samples")
        for name in self.posterior.parameter_names:
            q = summary[name]
            q16, q50, q84 = q["16"], q["50"], q["84"]
            print(f"  {name}: {q50:.6g} -{q50-q16:.3g}/+{q84-q50:.3g}")

    def _load_posterior(self, *_):
        self.posterior = PosteriorResult.load_npz(self.posterior_path)
        if self.state is not None:
            self.state.set_posterior("absorption", self.posterior)
        print(
            f"Loaded {self.posterior_path} "
            f"({self.posterior.samples.shape[0]} posterior samples)"
        )

    def _save_posterior(self, *_):
        if self.posterior is None:
            self._run_posterior()
        if self.posterior is None:
            return
        path = self.posterior.save_npz(self.posterior_path)
        print(f"Saved {path}")

    def _plot_posterior(self, *_):
        if self.posterior is None:
            self._run_posterior()
        if self.posterior is None:
            return
        max_parameters = int(_get(self.config, "absorption_inference_corner_max_parameters", 8))
        figure = plot_posterior_corner(self.posterior, max_parameters=max_parameters)
        figure.suptitle("FitSpec absorption posterior", fontsize=12)
        figure.show()

    def _load_fit(self, *_):
        self.result = load_absorption_result(self.result_path)
        if self.state is not None:
            self.state.set_result("absorption", self.result)
        self.best_line.set_data(self.result.fit_result.wave, self.result.fit_result.model)
        self.fig.canvas.draw_idle()
        print(f"Loaded {self.result_path}")

    def show(self):
        plt.show()
