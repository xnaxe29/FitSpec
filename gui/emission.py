"""FitSpec emission-line GUI.

Matplotlib-widget based, in the same style as ``gui.stellar_gui``: a
single Figure hosting the spectrum plot, sliders/buttons, the mask
controller, and the explicit component controller -- no Tkinter, no
GUI framework dependency beyond Matplotlib itself.

Per component (velocity_kms, sigma_kms) is edited via
``gui.component_controller.ComponentController``, per the FitSpec GUI
architecture doc. Per-line integrated-flux amplitudes are a separate
axis of control from "which component is active" -- a fitted line
list can have many free lines, so rather than one slider per line
(which would not scale), an independent "active line" selector plus a
single amplitude slider edits the active component's active line,
mirroring the same "switching selection never alters other values"
invariant the component controller itself enforces. The active-line
selector itself is a slider (not one radio button per species): a
line list can run into the hundreds of transitions, which individual
radio buttons cannot scale to and a scrollable index slider can. It's
a horizontal slider sharing the same axes "box" style as every other
control here, not a vertical one off to the side.

The velocity_kms/sigma_kms sliders are always draggable, regardless of
``kinematics_mode`` -- that setting controls which parameters are free
in the deterministic *fit*, not what's explorable in the GUI (per the
FitSpec design: "fixed"/"tied"/"free" are useful mainly while fitting,
so the GUI doesn't have to follow them). The GUI's own line/component
scope is instead controlled entirely by the "Line specific"/"Component
specific" checkboxes below.

If a stellar fit (``stellar_fit.fits``) is found in the same working
directory as this panel's own results, its continuum model is loaded
and shown as a faint "ghost" reference curve -- a passive comparison
curve only, never subtracted or fit against.

The continuum this panel actually fits against is a separate,
interactively-editable one (``gui.continuum_controller.ContinuumController``),
ported from the original ``bic_emission_fitting.py`` workflow: an
automated estimate (``continuum.continuum``) is reduced to ~50 sparse
anchor points, which can be added/removed/moved live by clicking the
plot (Add/Remove/Move mode), with the continuum curve rebuilt by spline
on every edit. Fitting subtracts this continuum from a *copy* of the
spectrum (never the shared Spectrum object other panels use) before
calling ``fit_emission_spectrum``; the preview/best-fit curves add it
back for display, which is also what keeps a sparse emission-only model
(correctly ~0 between lines) from reading as a "dropout to zero" now
that the axis-limits fix above shows the true flux scale.

``emission_flux_normalizing_factor``/``emission_flux_reduction`` are
applied exactly once, immediately in ``__init__``, before line
selection, continuum estimation, amplitude-bound defaults, or anything
else -- see ``emission.emission_fit.normalize_emission_spectrum``. From
that point on, ``self.spectrum`` is this panel's own independent,
already-rescaled copy (never the shared Spectrum object other panels
use, same as the continuum-subtracted copy built for fitting). Stellar
fits don't know about this factor (there's no such stellar_* config
key), so the "ghost" reference continuum -- loaded from an unrelated,
unnormalized stellar_fit.fits -- is rescaled the same way before it's
ever plotted, or it would appear on a completely different flux scale
than everything else in this panel.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider

from core.parameters import Component, Parameter
from core.results import PosteriorResult
from core.rebinning import compute_display_smoothing
from gui.axis_limits import compute_sensible_limits

from emission.lines import select_lines
from emission.emission_model import (
    amplitude_parameter_name, velocity_parameter_name, sigma_parameter_name,
    build_emission_parameters, make_emission_model_func,
)
from emission.emission_fit import select_emission_line_list, fit_emission_spectrum, normalize_emission_spectrum
from emission.inference import run_emission_inference
from emission.emission_results import save_emission_result, load_emission_result
from plotting.diagnostics import plot_posterior_corner

from gui.mask_controller import MaskController
from gui.component_controller import ComponentController
from gui.continuum_controller import ContinuumController

__all__ = ["EmissionGUI"]


def _get(config, key, default=None):
    return config.get(key, default) if hasattr(config, "get") else default


def _pair(config, key, default):
    value = _get(config, key, default)
    values = list(value) if isinstance(value, (list, tuple, np.ndarray)) else [x.strip() for x in str(value).split(",")]
    return (float(values[0]), float(values[1])) if len(values) >= 2 else tuple(map(float, default))


class EmissionGUI:
    """Interactive emission-line fitting session for one spectrum.

    Parameters
    ----------
    spectrum : core.spectrum.Spectrum
    config : object supporting ``.get(key, default)``
        See ``config/default_config_emission.dat``.
    line_list : list[emission.lines.EmissionLine], optional
        Pre-selected line list; defaults to
        ``emission.emission_fit.select_emission_line_list(spectrum, config)``
        (which also applies any ``emission_fixed_ratio_species``/
        ``emission_fixed_ratio_value`` overrides).
    """

    def __init__(self, spectrum, config, *, line_list=None, result_path="emission_fit.fits", mask_path=None,
                 continuum_path=None, state=None):
        # Applied exactly once, before anything else touches flux: see
        # emission.emission_fit.normalize_emission_spectrum. self.spectrum
        # from this point on is this panel's own independent, already-
        # rescaled copy -- never the shared Spectrum object other panels use.
        self.spectrum = normalize_emission_spectrum(spectrum, config)
        self.config = config
        self.state = state
        if self.state is not None:
            self.state.register_panel("emission", self)
        self.result_path = Path(result_path)
        self.line_list = line_list if line_list is not None else select_emission_line_list(self.spectrum, config)
        self.free_lines = [line for line in self.line_list if line.tied_to is None]
        if not self.free_lines:
            raise ValueError("Selected line list has no untied lines to fit an amplitude for.")
        self.kinematics_mode = str(_get(config, "emission_kinematics_mode", "tied")).strip().lower()
        self.result = None
        self.posterior = None
        self.posterior_path = Path(_get(config, "emission_inference_output_path", "emission_posterior.npz"))

        n_components = int(_get(config, "emission_n_components", 1))
        self.velocity_bounds = _pair(config, "emission_velocity_bounds_kms", (-500.0, 500.0))
        self.sigma_bounds = _pair(config, "emission_sigma_bounds_kms", (1.0, 1000.0))
        self.amplitude_initial, self.amplitude_bounds = self._default_amplitude_settings(self.spectrum, config)
        self.model_parameters = build_emission_parameters(
            self.line_list, n_components,
            velocity_initial_kms=float(_get(config, "emission_velocity_initial_kms", 0.0)),
            velocity_bounds_kms=self.velocity_bounds,
            sigma_initial_kms=float(_get(config, "emission_sigma_initial_kms", 50.0)),
            sigma_bounds_kms=self.sigma_bounds,
            amplitude_initial=self.amplitude_initial, amplitude_bounds=self.amplitude_bounds,
            kinematics_mode=self.kinematics_mode,
        )
        self.model_func = make_emission_model_func(
            self.line_list, redshift=self.spectrum.redshift, resolution=self.spectrum.resolution,
            kinematics_mode=self.kinematics_mode,
        )

        self.stellar_continuum_wave, self.stellar_continuum_model = self._find_stellar_continuum(result_path)

        self.fig, self.ax = plt.subplots(figsize=(13, 8))
        plt.subplots_adjust(left=0.08, right=0.76, bottom=0.40)
        self.data_line, = self.ax.plot([], [], drawstyle="steps-mid", alpha=0.45, label="data")
        self.masked_line, = self.ax.plot([], [], ".", ms=3, alpha=0.18, label="masked")
        self.ghost_continuum_line, = self.ax.plot(
            [], [], lw=1, ls=":", alpha=0.5, color="0.4", label="stellar continuum (ghost)",
        )
        self.preview_line, = self.ax.plot([], [], lw=1.5, label="preview")
        self.best_line, = self.ax.plot([], [], lw=2, ls="--", label="best fit")
        if self.stellar_continuum_wave is not None:
            self.ghost_continuum_line.set_data(self.stellar_continuum_wave, self.stellar_continuum_model)
        self.ax.set_xlabel(r"Wavelength ($\AA$)")
        self.ax.set_ylabel("Flux")
        self.ax.set_title(
            f"FitSpec emission fitting [{len(self.free_lines)} free line(s), "
            f"{n_components} component(s), kinematics={self.kinematics_mode}]"
        )

        self.continuum_controller = ContinuumController(
            self.spectrum,
            n_points=int(_get(config, "emission_continuum_n_points", 50)),
            method=str(_get(config, "emission_continuum_method", "custom")),
            continuum_path=continuum_path,
            on_change=lambda _continuum, _reason: self._preview(),
        )
        self.continuum_controller.connect_matplotlib(
            self.ax,
            editing_check_axes=plt.axes([0.80, 0.855, 0.17, 0.055]),
            mode_axes=plt.axes([0.80, 0.70, 0.17, 0.13]),
            save_button_axes=plt.axes([0.80, 0.645, 0.075, 0.04]),
            load_button_axes=plt.axes([0.89, 0.645, 0.075, 0.04]),
            reset_button_axes=plt.axes([0.80, 0.585, 0.165, 0.04]),
            status_axes=plt.axes([0.80, 0.92, 0.17, 0.05]),
        )
        self.ax.legend(fontsize=8)

        # Halved from the original 0.28 -- this panel doesn't need a
        # display-bin slider as wide as the kinematics sliders next to it.
        self.smooth_slider = Slider(plt.axes([0.13, 0.30, 0.14, 0.025]), "Display bin", 1, 50, valinit=1, valstep=1)
        self.smooth_slider.on_changed(self._refresh_data)

        self.component_controller = ComponentController(
            self.model_parameters, ["velocity_kms", "sigma_kms"],
            make_new_component=self._make_new_component,
            on_change=lambda _mp: (self._preview(), self._sync_kinematics_slider_state()),
            slider_write_hook=self._on_kinematics_slider_change,
        )
        self.component_controller.connect_matplotlib(
            count_axes=plt.axes([0.13, 0.25, 0.15, 0.03]),
            decrement_axes=plt.axes([0.29, 0.25, 0.03, 0.03]),
            increment_axes=plt.axes([0.33, 0.25, 0.03, 0.03]),
            active_axes=plt.axes([0.13, 0.09, 0.15, 0.13]),
            slider_axes={
                "velocity_kms": plt.axes([0.48, 0.30, 0.23, 0.025]),
                "sigma_kms": plt.axes([0.48, 0.25, 0.23, 0.025]),
            },
        )

        self._active_line_index = 0
        self.amp_slider = Slider(plt.axes([0.48, 0.20, 0.23, 0.025]), "amplitude", 0.0, 1.0, valinit=0.0)
        self.amp_slider.on_changed(self._on_amp_change)
        # Same horizontal-slider "box" style/size used for velocity_kms,
        # sigma_kms, and amplitude above -- a scrollable index slider still
        # scales to a hundreds-strong line list (unlike one radio button
        # per species), but there's no reason for it to be the odd one out
        # oriented vertically while every other control here reads left-to-right.
        self._line_slider_axes = plt.axes([0.48, 0.15, 0.23, 0.025])
        self._build_line_slider()

        # Scope checkboxes for the amp/velocity/sigma sliders -- see
        # _scoped_line_names/_scoped_components and _on_kinematics_slider_change.
        # Both default UNCHECKED: an edit starts out applying to every line
        # and every component, not just the active selection -- check
        # either box to narrow an edit down to just the active line and/or
        # just the active component.
        from matplotlib.widgets import CheckButtons
        self._line_specific = False
        self._component_specific = False
        self._component_specific_check = CheckButtons(
            plt.axes([0.13, 0.02, 0.15, 0.05]), ["Component specific"], [False],
        )
        self._component_specific_check.on_clicked(self._toggle_component_specific)
        self._line_specific_check = CheckButtons(
            plt.axes([0.48, 0.02, 0.23, 0.05]), ["Line specific"], [False],
        )
        self._line_specific_check.on_clicked(self._toggle_line_specific)

        # Whether insignificant-component rejection (see emission.rejection)
        # runs at Fit time -- initialized from emission_reject_insignificant_components
        # so a config default still applies on open, but from here on this
        # checkbox is the live, interactive control: the automatic
        # freeze-if-insignificant mechanism is genuinely useful, but the
        # decision to use it belongs to the person running the fit, not
        # something silently baked into a config file. Unchecked: every
        # requested component stays a genuinely free parameter for the
        # whole fit (n_components is *never* changed either way -- ask for
        # 6 and you get exactly 6 back, checked or not; ask for 1 and you
        # get exactly 1, however poor that fit is).
        self._help_decide_components = bool(_get(config, "emission_reject_insignificant_components", False))
        self._help_decide_components_check = CheckButtons(
            plt.axes([0.48, 0.09, 0.23, 0.05]), ["Help decide components"], [self._help_decide_components],
        )
        self._help_decide_components_check.on_clicked(self._toggle_help_decide_components)

        self._sync_amp_slider()
        self._sync_kinematics_slider_state()

        self.view_button = Button(plt.axes([0.80, 0.14, 0.08, 0.04]), "Reset View")
        self.view_button.on_clicked(self._reset_view)
        self.stamps_button = Button(plt.axes([0.885, 0.14, 0.08, 0.04]), "Plot Stamps")
        self.stamps_button.on_clicked(self._plot_stamps)

        self.fit_button = Button(plt.axes([0.80, 0.09, 0.055, 0.04]), "Fit")
        self.fit_button.on_clicked(self._fit)
        self.load_button = Button(plt.axes([0.86, 0.09, 0.055, 0.04]), "Load Fit")
        self.load_button.on_clicked(self._load_fit)
        self.save_button = Button(plt.axes([0.92, 0.09, 0.055, 0.04]), "Save Fit")
        self.save_button.on_clicked(self._save_fit)

        # Posterior controls are intentionally separate from the deterministic
        # Fit button: the selected deterministic model is the starting point
        # for emcee/dynesty rather than being silently replaced by a sampler.
        self.posterior_button = Button(plt.axes([0.78, 0.03, 0.047, 0.04]), "Posterior")
        self.posterior_button.on_clicked(self._run_posterior)
        self.posterior_load_button = Button(plt.axes([0.83, 0.03, 0.047, 0.04]), "Load P")
        self.posterior_load_button.on_clicked(self._load_posterior)
        self.posterior_save_button = Button(plt.axes([0.88, 0.03, 0.047, 0.04]), "Save P")
        self.posterior_save_button.on_clicked(self._save_posterior)
        self.posterior_plot_button = Button(plt.axes([0.93, 0.03, 0.047, 0.04]), "Plot P")
        self.posterior_plot_button.on_clicked(self._plot_posterior)

        self.mask_controller = MaskController(
            self.spectrum, fit_mode="emission",
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

        self.ax.legend(fontsize=8)
        self._refresh_data(rescale=True)
        self._preview()

    # -- ghost stellar continuum -------------------------------------------------

    def _default_amplitude_settings(self, spectrum, config):
        """Pick a physically-scaled default (initial, (lower, upper)) for
        every per-line amplitude parameter.

        Amplitudes are integrated line flux, in the spectrum's own flux
        units -- a hardcoded ``(1.0, 0.0..inf)`` (the previous default)
        means next to nothing for real data (e.g. HST-COS flux densities
        around 1e-15), and displays as a slider whose range bears no
        relation to the spectrum. ``emission_maximum_amplitude`` still
        wins if the user has set it explicitly (same key the actual fit's
        bounds already respect, see emission.emission_fit); otherwise the
        upper bound defaults to 1.5x the spectrum's own peak flux.
        """
        configured_max = _get(config, "emission_maximum_amplitude", None)
        if configured_max not in (None, "", 0):
            upper = float(configured_max)
        else:
            flux = np.asarray(spectrum.flux, float)
            finite = flux[np.isfinite(flux)]
            peak = float(np.nanmax(finite)) if finite.size else 0.0
            upper = 1.5 * peak if peak > 0 else 1.0
        initial = 0.1 * upper if upper > 0 else 0.0
        return initial, (0.0, upper)

    def _find_stellar_continuum(self, result_path):
        """Auto-detect a stellar fit saved in the same working directory and
        return ``(wave_observed, model)`` for display as a faint reference
        curve, or ``(None, None)`` if none is found/loadable/disabled.

        This is purely a visual reference ("ghost continuum") -- it is not
        subtracted from the data or added into the emission model, so it
        never silently changes fit results. "Same working directory" is
        resolved the same way the rest of the session already organizes
        per-mode outputs: ``state.run_dir`` when this panel was opened via
        ``FitSpecApp``, else the directory holding this panel's own
        ``result_path``. Controlled by the ``ghost_continuum`` config key
        (default False -- an existing base-config key that nothing
        previously read).
        """
        if not bool(_get(self.config, "ghost_continuum", False)):
            return None, None
        run_dir = self.state.run_dir if self.state is not None else Path(result_path).parent
        candidate = Path(run_dir) / "stellar_fit.fits"
        if not candidate.is_file():
            return None, None
        try:
            from stellar.stellar_results import load_stellar_result
            stellar_result = load_stellar_result(candidate)
        except Exception as exc:
            print(f"Found {candidate} but could not load it as a stellar fit ({exc}); skipping ghost continuum.")
            return None, None
        wave_observed = np.asarray(stellar_result.wave, float) * (1.0 + float(self.spectrum.redshift))
        # Stellar fits have no emission_flux_normalizing_factor/reduction
        # concept (no such stellar_* config key) -- rescale the loaded
        # (native-unit) continuum the same way self.spectrum already was,
        # or it would sit on a completely different flux scale than
        # everything else in this panel.
        normalizing_factor = float(_get(self.config, "emission_flux_normalizing_factor", 1.0))
        reduction = float(_get(self.config, "emission_flux_reduction", 0.0))
        model_normalized = (np.asarray(stellar_result.model, float) - reduction) / normalizing_factor
        print(
            f"Detected stellar fit at {candidate}; showing its continuum as a faint ghost "
            "reference curve for comparison, rescaled by emission_flux_normalizing_factor/"
            "emission_flux_reduction to match this panel's units (the stellar fit itself knows "
            "nothing about those keys). This is separate from -- and never used as -- the "
            "actively-estimated/editable continuum this panel fits against (see the continuum "
            "controls); it is never subtracted from the data or folded into the fitted model."
        )
        return wave_observed, model_normalized

    # -- component/line construction -------------------------------------------

    def _make_new_component(self) -> Component:
        v_lo, v_hi = self.velocity_bounds
        s_lo, s_hi = self.sigma_bounds
        a_lo, a_hi = self.amplitude_bounds
        v_init = float(_get(self.config, "emission_velocity_initial_kms", 0.0))
        s_init = float(_get(self.config, "emission_sigma_initial_kms", 50.0))
        # Matches build_emission_parameters: only "tied" fits one value per
        # component (this new one included); "fixed"/"free" hold the
        # component-level parameter fixed (single global value, or a pure
        # tracking source/GUI broadcast target for independent per-line
        # values, respectively).
        component_kinematics_fixed = self.kinematics_mode in ("fixed", "free")
        per_line_kinematics_fixed = self.kinematics_mode != "free"
        parameters = [
            Parameter("velocity_kms", v_init, v_lo, v_hi, fixed=component_kinematics_fixed),
            Parameter("sigma_kms", s_init, s_lo, s_hi, fixed=component_kinematics_fixed),
        ]
        for line in self.free_lines:
            parameters.append(Parameter(amplitude_parameter_name(line.name), self.amplitude_initial, a_lo, a_hi))
            parameters.append(
                Parameter(velocity_parameter_name(line.name), v_init, v_lo, v_hi, fixed=per_line_kinematics_fixed)
            )
            parameters.append(
                Parameter(sigma_parameter_name(line.name), s_init, s_lo, s_hi, fixed=per_line_kinematics_fixed)
            )
        return Component(parameters=parameters)

    # -- amp/velocity/sigma slider write scope (line/component checkboxes) -------

    def _toggle_component_specific(self, _label):
        self._component_specific = not self._component_specific
        self._sync_kinematics_slider_state()

    def _toggle_line_specific(self, _label):
        self._line_specific = not self._line_specific
        self._sync_kinematics_slider_state()

    def _toggle_help_decide_components(self, _label):
        self._help_decide_components = not self._help_decide_components

    def _scoped_line_names(self) -> "list[str]":
        """Which line(s) an amp/velocity/sigma slider edit should write to."""
        if self._line_specific:
            return [self.free_lines[self._active_line_index].name]
        return [line.name for line in self.free_lines]

    def _scoped_components(self) -> "list[Component]":
        """Which component(s) an amp/velocity/sigma slider edit should write to."""
        if self._component_specific:
            return [self.model_parameters.active_component]
        return list(self.model_parameters.components)

    def _on_kinematics_slider_change(self, parameter_name: str, value: float) -> bool:
        """ComponentController.slider_write_hook for velocity_kms/sigma_kms.

        Always writes every scoped line's own per-line kinematics
        parameter directly (exactly like the amplitude slider writes
        every scoped line's amplitude) -- this is what actually reaches
        ``emission_component_flux`` regardless of ``kinematics_mode``, so
        a broadcast always visibly does something even when a line's
        override happens to be ``fixed=True`` (still tracking its
        component, under "fixed"/"tied") or ``fixed=False`` (independent,
        under "free"). Line checkbox checked also marks the written
        line(s) explicitly diverged (``fixed=False``) so they no longer
        track their component after this; unchecked leaves each line's
        ``fixed`` flag alone, and additionally updates the component-level
        "shared" parameter -- purely as a display/tracking-source
        convenience for any *other*, still-tracking line, since
        ``make_emission_model_func`` only reads it for that.

        Always returns True: this hook fully decides the write target
        itself, so ComponentController's own default
        ``active_component[parameter_name].value = value`` never also fires.
        """
        override_name = velocity_parameter_name if parameter_name == "velocity_kms" else sigma_parameter_name
        for component in self._scoped_components():
            for line_name in self._scoped_line_names():
                override = component[override_name(line_name)]
                override.value = value
                if self._line_specific:
                    override.fixed = False
            if not self._line_specific:
                component[parameter_name].value = value
        return True

    def _kinematics_effective_value(self, component: Component, parameter_name: str) -> float:
        """The value the velocity_kms/sigma_kms slider should *display* for
        `component`: the active line's own per-line kinematics value if
        "Line specific" is checked, else the component-level shared value
        (which every still-tracking line follows, and which is what an
        unchecked-scope edit writes to as a display/tracking-source
        convenience -- see ``_on_kinematics_slider_change``)."""
        if not self._line_specific:
            return float(component[parameter_name].value)
        line_name = self.free_lines[self._active_line_index].name
        override_name = (
            velocity_parameter_name(line_name) if parameter_name == "velocity_kms"
            else sigma_parameter_name(line_name)
        )
        return float(component[override_name].value)

    def _sync_kinematics_slider_state(self):
        """Refreshes the velocity_kms/sigma_kms sliders' *displayed* value
        for the current line/component checkbox scope.

        The sliders are always active/draggable -- ``kinematics_mode``
        ("fixed"/"tied"/"free") governs which parameters the deterministic
        *fit* treats as free, not what's explorable here; per FitSpec's
        design, the GUI doesn't need to follow it (that's what lets you
        freely try out different per-component or per-line kinematics by
        eye before committing to what the actual fit should optimize).
        """
        active = self.model_parameters.active_component
        for name in ("velocity_kms", "sigma_kms"):
            slider = self.component_controller._sliders[name]
            slider.eventson = False
            try:
                slider.set_val(self._kinematics_effective_value(active, name))
            finally:
                slider.eventson = True
        self.fig.canvas.draw_idle()

    # -- active-line amplitude control -------------------------------------------

    def _build_line_slider(self):
        # Scrollable index slider over ``self.free_lines`` -- a fitted line
        # list can have hundreds of free transitions, which one radio
        # button per species cannot scale to. The slider's displayed value
        # is overridden to show the active line's name rather than its
        # numeric index (see _update_line_slider_label).
        n_lines = len(self.free_lines)
        # valmax must stay > valmin even for a single-line list, or Matplotlib
        # warns about a singular (zero-range) slider axis transform; a stray
        # extra step above the last line is harmless since the change
        # handler below always clamps back into range.
        slider_max = max(n_lines - 1, 1)
        self._line_slider = Slider(
            self._line_slider_axes, "line", 0, slider_max,
            valinit=self._active_line_index, valstep=1,
        )
        self._line_slider.on_changed(self._on_line_slider_change)
        self._update_line_slider_label()

    def _update_line_slider_label(self):
        name = self.free_lines[self._active_line_index].name
        self._line_slider.valtext.set_text(name)

    def _on_line_slider_change(self, value):
        index = min(int(round(value)), len(self.free_lines) - 1)
        self._on_line_change(self.free_lines[index].name)

    def _active_amp_parameter(self) -> Parameter:
        line = self.free_lines[self._active_line_index]
        return self.model_parameters.active_component[amplitude_parameter_name(line.name)]

    def _sync_amp_slider(self):
        parameter = self._active_amp_parameter()
        upper = parameter.upper if np.isfinite(parameter.upper) else max(parameter.value * 2.0, 1.0)
        self.amp_slider.valmax = upper
        self.amp_slider.ax.set_xlim(self.amp_slider.valmin, upper)
        # Guarded like every other display-only slider refresh in this
        # file (e.g. _line_slider, the velocity_kms/sigma_kms sliders in
        # _sync_kinematics_slider_state) -- without this, set_val() fires
        # _on_amp_change, which (now that the default scope is "every
        # line, every component") would broadcast the *currently
        # displayed* line's value onto every other line/component any
        # time this ran, e.g. on every line-slider switch. That
        # corrupted freshly-fitted/loaded amplitudes down to whatever
        # the first line happened to hold.
        self.amp_slider.eventson = False
        try:
            self.amp_slider.set_val(parameter.value)
        finally:
            self.amp_slider.eventson = True

    def _on_line_change(self, label: str):
        # Stable internal API: called both by the line slider's on_changed
        # callback and directly (e.g. by callers/tests that just want to
        # select a line by name). Guarded with eventson so that keeping the
        # slider's displayed position in sync never re-enters this method.
        self._active_line_index = [line.name for line in self.free_lines].index(label)
        if int(round(self._line_slider.val)) != self._active_line_index:
            self._line_slider.eventson = False
            try:
                self._line_slider.set_val(self._active_line_index)
            finally:
                self._line_slider.eventson = True
        self._update_line_slider_label()
        self._sync_amp_slider()
        self._sync_kinematics_slider_state()

    def _on_amp_change(self, value: float):
        value = float(value)
        for component in self._scoped_components():
            for line_name in self._scoped_line_names():
                component[amplitude_parameter_name(line_name)].value = value
        self._preview()

    # -- plotting -----------------------------------------------------------------

    def _refresh_data(self, *_, rescale=False):
        bins = max(1, int(round(self.smooth_slider.val))) if hasattr(self, "smooth_slider") else 1
        if bins > 1 and self.spectrum.flux_unc is not None:
            wave, flux, _ = compute_display_smoothing(self.spectrum, bins, min_coverage=0.5)
        else:
            wave, flux = np.asarray(self.spectrum.wave, float), np.asarray(self.spectrum.flux, float)
        self.data_line.set_data(wave, flux)
        mask = np.ones(self.spectrum.wave.size, bool) if self.spectrum.mask is None else np.asarray(self.spectrum.mask, bool)
        self.masked_line.set_data(np.asarray(self.spectrum.wave, float)[~mask], np.asarray(self.spectrum.flux, float)[~mask])
        if rescale:
            self._set_sensible_limits()
        self.fig.canvas.draw_idle()

    def _set_sensible_limits(self):
        # Explicit, mask-aware, outlier-robust limits -- ax.relim()/
        # autoscale_view() silently fall back to Matplotlib's default (0, 1)
        # view whenever they can't find finite data (e.g. before any line
        # has data), which is why the panel used to open with wavelength
        # starting at 0 and flux pinned to 0--1. Same fix as gui.stellar,
        # factored out into gui.axis_limits so both panels share it.
        wave = np.asarray(self.spectrum.wave, float)
        flux = np.asarray(self.spectrum.flux, float)
        mask = None if self.spectrum.mask is None else np.asarray(self.spectrum.mask, bool)
        extra_y = [self.continuum_controller.working_continuum]
        if self.stellar_continuum_model is not None:
            extra_y.append(self.stellar_continuum_model)
        limits = compute_sensible_limits(wave, flux, mask, y_widen_frac=0.2, extra_y=extra_y)
        if limits is None:
            return
        xlim, ylim = limits
        self.ax.set_xlim(*xlim)
        if ylim is not None:
            self.ax.set_ylim(*ylim)

    def _reset_view(self, *_):
        self._set_sensible_limits()
        self.fig.canvas.draw_idle()

    def _output_dir(self) -> Path:
        """Where debugging/plot artifacts (stamp data dumps, saved PDFs) go
        -- the session's run directory when opened via FitSpecApp, else
        the folder holding this panel's own result_path, matching the
        same convention _find_stellar_continuum already uses."""
        output_dir = Path(self.state.run_dir) if self.state is not None else self.result_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _save_figure_pdf(self, fig, stem: str) -> Path:
        path = self._output_dir() / f"{stem}.pdf"
        fig.savefig(path)
        print(f"Saved {path}")
        return path

    def _save_stamps_data(self, wave, flux, flux_unc, continuum, total_curve) -> Path:
        """Plain-text dump of exactly what the stamp plot's total-fit curve
        is drawn from, for debugging what a stamp is actually showing
        (e.g. checking the continuum or total fit by eye/by hand outside
        the GUI) -- wave, flux, flux_unc, cont, fit, on the full (not
        per-line-windowed) wave grid, one row per pixel."""
        path = self._output_dir() / "emission_stamps_data.dat"
        data = np.column_stack([
            np.asarray(wave, float), np.asarray(flux, float), np.asarray(flux_unc, float),
            np.asarray(continuum, float), np.asarray(total_curve, float),
        ])
        np.savetxt(path, data, header="wave flux flux_unc cont fit", fmt="%.8g")
        print(f"Saved {path}")
        return path

    def _plot_stamps(self, *_):
        """Per-line velocity-space stamp grid, each with a paired residual
        panel (see ``plotting.stamps.plot_emission_stamps``).

        Uses the current ``self.model_parameters`` -- i.e. whatever's
        currently on the sliders, before or after a completed Fit -- so
        this works as a by-eye initialization aid, not just a post-fit
        diagnostic.

        ``emission_fig_normalized`` toggles between the original tool's
        two display modes (see ``dynesty_mcmc_bic.py``): True subtracts
        the continuum from the data and plots the *line-only* model
        (zero baseline); False (default) plots the raw data against
        model + continuum, with the continuum drawn as its own reference
        curve. Neither is "normalization" in the divide-by-continuum
        sense -- an earlier version of this method did that instead,
        which didn't match the original and was reverted.

        Also saves the figure as a PDF and dumps the exact arrays behind
        the total-fit curve to a plain-text .dat file, both into
        ``_output_dir()``, so what's plotted can be checked independently
        of the GUI.
        """
        from core.parameters import ModelParameters
        from plotting.stamps import plot_emission_stamps

        wave = np.asarray(self.spectrum.wave, float)
        flux = np.asarray(self.spectrum.flux, float)
        flux_unc = np.asarray(self.spectrum.flux_unc, float)
        continuum = self.continuum_controller.continuum_on(wave)

        total_line_model = self.model_func(wave, self.model_parameters)

        component_line_models = []
        for component in self.model_parameters.components:
            single = ModelParameters(n_components=1, components=[component])
            component_line_models.append(self.model_func(wave, single))

        # Which components rejection froze (see emission.rejection), if
        # this GUI's current component count still matches the last
        # completed fit's -- e.g. not after clicking "+"/"-" post-fit
        # without refitting. None (rather than guessing from Parameter.fixed,
        # which is also True under kinematics_mode == "fixed"/"tied" for
        # reasons that have nothing to do with rejection) if there's no
        # fit yet or the counts no longer line up.
        frozen_components = None
        if self.result is not None:
            stored_flags = self.result.fit_result.metadata.get("emission_frozen_components")
            if stored_flags is not None and len(stored_flags) == self.model_parameters.n_components:
                frozen_components = list(stored_flags)

        if bool(_get(self.config, "emission_fig_normalized", False)):
            # Continuum-subtracted, zero-baseline view: the plotting
            # function always does `model + continuum`, so passing an
            # all-zero continuum alongside already-subtracted data gives
            # exactly the original tool's "fig_normalized=True" behavior
            # without plot_emission_stamps needing to know about this
            # toggle at all.
            flux_for_plot = flux - continuum
            continuum_for_plot = np.zeros_like(continuum)
        else:
            flux_for_plot = flux
            continuum_for_plot = continuum

        self._save_stamps_data(wave, flux_for_plot, flux_unc, continuum_for_plot, total_line_model + continuum_for_plot)

        half_width = float(_get(self.config, "emission_fig_velocity_limit_kms", 300.0))
        sigma_limit = float(_get(self.config, "emission_fig_residual_sigma_limit", 3.0))
        x_axis = str(_get(self.config, "emission_fig_stamp_xaxis", "velocity")).strip().lower()
        fig = plot_emission_stamps(
            wave, flux_for_plot, flux_unc, continuum_for_plot, total_line_model, component_line_models,
            self.free_lines, redshift=self.spectrum.redshift, velocity_half_width_kms=half_width,
            x_axis=x_axis, sigma_lines=(-sigma_limit, sigma_limit), frozen_components=frozen_components,
        )
        self._save_figure_pdf(fig, "emission_stamps")
        fig.show()

    def _continuum_baseline(self, wave):
        """The actively-estimated/editable continuum (see ContinuumController),
        interpolated onto ``wave``.

        This is the continuum actually subtracted before fitting (see
        _fit_spectrum), added back here purely for display so the
        preview/best-fit curves sit on a realistic baseline instead of the
        emission-only model's correct-but-misleading ~0 between lines. The
        separate "ghost" stellar continuum is never used here -- it's a
        passive reference curve only, never subtracted or fit against.
        """
        return self.continuum_controller.continuum_on(wave)

    def _preview(self, *_):
        wave = np.asarray(self.spectrum.wave, float)
        line_model = self.model_func(wave, self.model_parameters)
        self.preview_line.set_data(wave, line_model + self._continuum_baseline(wave))
        self.fig.canvas.draw_idle()

    # -- fit/save/load --------------------------------------------------------------

    def _continuum_subtracted_spectrum(self):
        """A copy of ``self.spectrum`` with the active edited continuum
        subtracted from its flux, ready to fit lines against.

        A copy, not an in-place edit: even though ``self.spectrum`` is
        already this panel's own independent, normalized copy (never the
        shared Spectrum object the stellar/absorption panels use -- see
        ``normalize_emission_spectrum`` in ``__init__``), the un-subtracted
        version is still needed for display (data_line shows continuum +
        lines). The continuum itself still lives on ``spectrum.continuum``
        (kept in sync by ContinuumController) for bookkeeping/session
        save-load, same as the original workflow's ``data_with_cont`` file.
        Both the data and this continuum are already in
        emission_flux_normalizing_factor/emission_flux_reduction units, so
        no further rescaling happens here -- fit_emission_spectrum no
        longer applies or undoes that scaling itself (see its docstring).
        """
        from dataclasses import replace
        continuum = self.continuum_controller.continuum_on(self.spectrum.wave)
        flux = np.asarray(self.spectrum.flux, float) - continuum
        return replace(self.spectrum, flux=flux)

    def _fit(self, *_):
        fit_spectrum = self._continuum_subtracted_spectrum()
        self.result = fit_emission_spectrum(
            fit_spectrum, self.config, line_list=self.line_list, use_rejection=self._help_decide_components,
            n_components=self.model_parameters.n_components,
        )
        if self.state is not None:
            self.state.set_result("emission", self.result)
        self._sync_gui_to_fitted_parameters(self.result.fit_result.parameters)
        wave = self.result.fit_result.wave
        self.best_line.set_data(wave, self.result.fit_result.model + self._continuum_baseline(wave))
        self.fig.canvas.draw_idle()
        stats = self.result.fit_result.statistics
        print(f"emission fit (continuum subtracted): reduced_chi2={stats.reduced_chi_square:.4g}, "
              f"velocities={list(self.result.component_velocities_kms)}")

    def _sync_gui_to_fitted_parameters(self, fitted_parameters):
        """Push a completed Fit's (or a just-Loaded fit's) parameters back
        into every GUI control: component count, active-component
        selector, and every slider -- so what's on screen always matches
        what was actually fit/loaded, not stale pre-fit slider positions.

        ``fitted_parameters`` becomes the new ``self.model_parameters``
        outright (not copied value-by-value into the old one), since the
        fit may have picked a different component count than the GUI had
        before (e.g. automatic BIC component-count search) -- the
        component controller is repointed at it and fully rebuilt
        (radio buttons, count text, slider positions) to match. The
        velocity_kms/sigma_kms sliders show a single value per component
        either way (shared across that component's lines when
        kinematics_mode == "tied"/"fixed"; the active line's own value
        when "free" and "Line specific" is checked -- see
        _kinematics_effective_value).
        """
        fitted_parameters.set_active(0)
        self.model_parameters = fitted_parameters
        self.component_controller.model_parameters = fitted_parameters
        self.component_controller._rebuild_radio()
        self.component_controller._update_count_text()
        self.component_controller._sync_sliders_to_active()
        self._on_line_change(self.free_lines[0].name)  # also resets the line slider's position, not just the index
        self._sync_kinematics_slider_state()
        self._preview()

    def _save_fit(self, *_):
        if self.result is None:
            self._fit()
        path = save_emission_result(self.result_path, self.result, overwrite=True)
        print(f"Saved {path}")

    def _run_posterior(self, *_):
        if self.result is None:
            self._fit()
        self.posterior = run_emission_inference(
            self.result, self._continuum_subtracted_spectrum(), self.config, line_list=self.line_list,
        )
        if self.posterior is not None and self.state is not None:
            self.state.set_posterior("emission", self.posterior)
        if self.posterior is None:
            print(
                "Emission posterior sampling is disabled. Set "
                "emission_inference_method = emcee or dynesty in config.dat."
            )
            return
        summary = self.posterior.summary()
        engine = self.posterior.metadata.get("engine", "posterior")
        print(f"emission {engine}: {self.posterior.samples.shape[0]} stored posterior samples")
        for name in self.posterior.parameter_names:
            q = summary[name]
            q16, q50, q84 = q["16"], q["50"], q["84"]
            print(f"  {name}: {q50:.6g} -{q50-q16:.3g}/+{q84-q50:.3g}")

    def _load_posterior(self, *_):
        self.posterior = PosteriorResult.load_npz(self.posterior_path)
        if self.state is not None:
            self.state.set_posterior("emission", self.posterior)
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
        max_parameters = int(_get(self.config, "emission_inference_corner_max_parameters", 8))
        figure = plot_posterior_corner(self.posterior, max_parameters=max_parameters)
        figure.suptitle("FitSpec emission posterior", fontsize=12)
        self._save_figure_pdf(figure, "emission_posterior_corner")
        figure.show()

    def _load_fit(self, *_):
        self.result = load_emission_result(self.result_path)
        if self.state is not None:
            self.state.set_result("emission", self.result)
        self._sync_gui_to_fitted_parameters(self.result.fit_result.parameters)
        wave = self.result.fit_result.wave
        self.best_line.set_data(wave, self.result.fit_result.model + self._continuum_baseline(wave))
        self.fig.canvas.draw_idle()
        print(f"Loaded {self.result_path}")

    def show(self):
        plt.show()
