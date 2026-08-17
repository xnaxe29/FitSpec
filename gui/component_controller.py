"""Reusable "number of components" + "active component" GUI controller.

Implements the Explicit Component Controller described in the FitSpec
GUI architecture doc: a model owns an explicit list of components
(``core.parameters.ModelParameters``, which already implements the
add/remove/active-index logic itself -- see its module docstring); this
module is the thin Matplotlib-widget layer on top of that, shared by
the emission and absorption GUIs (stellar fitting has no explicit
multi-component list and does not use this controller).

Two invariants enforced throughout:

* The component count is always an explicit setting -- changed only by
  the +/- buttons here (or programmatically), never inferred from
  anything else.
* Switching the active component never alters any component's stored
  parameter values. Only the sliders' *displayed* values change to
  reflect whichever component just became active; dragging a slider
  writes back into that (and only that) component.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from matplotlib.widgets import Button, RadioButtons, Slider

from core.parameters import Component, ModelParameters

__all__ = ["ComponentController"]


class ComponentController:
    """Connects a ``ModelParameters`` to Matplotlib widgets for component
    count, active-component selection, and per-parameter sliders that
    always edit only the active component.

    Parameters
    ----------
    model_parameters : ModelParameters
        The model this controller edits in place.
    controlled_parameters : list[str]
        Names of the per-component parameters to expose a slider for
        (e.g. ``["velocity_kms", "sigma_kms"]`` for emission,
        ``["logN", "b_kms", "velocity_kms"]`` for absorption). Every
        component must have a parameter with each of these names.
    make_new_component : callable
        ``() -> Component``, used to build a fresh component (with
        sensible initial values/bounds) when the component count is
        increased -- required since those defaults are science-module-
        specific and cannot be inferred generically.
    on_change : callable, optional
        ``on_change(model_parameters)``, called after any edit (count
        change, active-component change, or a slider drag) so the
        caller can refresh a preview plot.
    slider_write_hook : callable, optional
        ``slider_write_hook(parameter_name, value) -> bool``, called
        before the default single-active-component write on every
        slider drag. Returning ``True`` means the hook fully handled
        the write itself (e.g. broadcasting to several components, or
        redirecting to some other parameter entirely) and the default
        write is skipped; returning ``False``/``None`` falls through to
        the default ``active_component[parameter_name].value = value``.
        Absent by default, so existing callers (e.g. absorption) are
        unaffected.
    """

    def __init__(
        self, model_parameters: ModelParameters, controlled_parameters: "list[str]", *,
        make_new_component: "Callable[[], Component]", on_change: "Callable[[ModelParameters], None] | None" = None,
        slider_write_hook: "Callable[[str, float], bool] | None" = None,
    ):
        for parameter_name in controlled_parameters:
            if parameter_name not in model_parameters.active_component:
                raise ValueError(f"Every component must have a {parameter_name!r} parameter.")
        self.model_parameters = model_parameters
        self.controlled_parameters = list(controlled_parameters)
        self.make_new_component = make_new_component
        self.on_change = on_change
        self.slider_write_hook = slider_write_hook

        self._count_text = None
        self._radio = None
        self._radio_axes = None
        self._sliders: "dict[str, Slider]" = {}
        self._suppress_slider_events = False

    # -- wiring -----------------------------------------------------------------

    def connect_matplotlib(self, *, count_axes, decrement_axes, increment_axes, active_axes, slider_axes: "dict[str, object]"):
        """Create the widgets. ``slider_axes`` maps each controlled
        parameter name to the Matplotlib axes its Slider should occupy.
        """
        missing = set(self.controlled_parameters) - set(slider_axes)
        if missing:
            raise ValueError(f"slider_axes is missing axes for: {sorted(missing)}")

        self._count_axes = count_axes
        count_axes.axis("off")
        self._count_text = count_axes.text(0.5, 0.5, "", ha="center", va="center", fontsize=11, transform=count_axes.transAxes)

        self._decrement_button = Button(decrement_axes, "\u2212")
        self._increment_button = Button(increment_axes, "+")
        self._decrement_button.on_clicked(lambda _event: self._on_count_delta(-1))
        self._increment_button.on_clicked(lambda _event: self._on_count_delta(+1))

        self._radio_axes = active_axes

        for parameter_name in self.controlled_parameters:
            active = self.model_parameters.active_component[parameter_name]
            slider = Slider(slider_axes[parameter_name], parameter_name, active.lower, active.upper, valinit=active.value)
            slider.on_changed(lambda value, name=parameter_name: self._on_slider_change(name, value))
            self._sliders[parameter_name] = slider

        self._rebuild_radio()
        self._update_count_text()

    # -- internal -----------------------------------------------------------------

    def _rebuild_radio(self):
        """RadioButtons' label set can't be changed in place; rebuild it
        whenever the component count changes."""
        if self._radio_axes is None:
            return
        self._radio_axes.clear()
        labels = [f"C{i}" for i in range(self.model_parameters.n_components)]
        self._radio = RadioButtons(self._radio_axes, labels, active=self.model_parameters.active_component_index)
        self._radio.on_clicked(self._on_active_change)

    def _update_count_text(self):
        if self._count_text is not None:
            self._count_text.set_text(f"n components: {self.model_parameters.n_components}")

    def _sync_sliders_to_active(self):
        """Update every slider's displayed value to the newly-active
        component's stored values, without re-triggering their on_changed
        callbacks (which would otherwise overwrite that component's value
        with the slider's just-set display value -- a no-op here, but the
        suppression flag also protects against any transient inconsistency
        during the update)."""
        self._suppress_slider_events = True
        try:
            active = self.model_parameters.active_component
            for parameter_name, slider in self._sliders.items():
                slider.set_val(active[parameter_name].value)
        finally:
            self._suppress_slider_events = False

    def _on_count_delta(self, delta: int):
        if delta > 0:
            self.model_parameters.add_component(self.make_new_component(), make_active=True)
        elif delta < 0:
            if self.model_parameters.n_components <= 1:
                return  # never remove the last component
            self.model_parameters.remove_component(self.model_parameters.active_component_index)
        self._rebuild_radio()
        self._update_count_text()
        self._sync_sliders_to_active()
        if self.on_change is not None:
            self.on_change(self.model_parameters)

    def _on_active_change(self, label: str):
        index = int(label[1:])  # "C3" -> 3
        self.model_parameters.set_active(index)
        self._sync_sliders_to_active()
        if self.on_change is not None:
            self.on_change(self.model_parameters)

    def _on_slider_change(self, parameter_name: str, value: float):
        if self._suppress_slider_events:
            return
        value = float(value)
        handled = False
        if self.slider_write_hook is not None:
            handled = bool(self.slider_write_hook(parameter_name, value))
        if not handled:
            self.model_parameters.active_component[parameter_name].value = value
        if self.on_change is not None:
            self.on_change(self.model_parameters)
