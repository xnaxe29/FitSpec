"""Tests for gui.component_controller.ComponentController.

Widget click/drag events can't be simulated headlessly without a real
display, so tests call the controller's event handlers directly (the
same functions a real click/drag would invoke) and check the resulting
ModelParameters state -- exercising exactly the same code path a live
GUI session would.
"""
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from core.parameters import ModelParameters, Component, Parameter
from gui.component_controller import ComponentController


def _make_component():
    return Component(parameters=[
        Parameter("velocity_kms", 0.0, -500.0, 500.0),
        Parameter("sigma_kms", 50.0, 1.0, 1000.0),
    ])


def _wire(controller, fig=None):
    fig = fig or plt.figure()
    controller.connect_matplotlib(
        count_axes=fig.add_axes([0.1, 0.9, 0.2, 0.05]),
        decrement_axes=fig.add_axes([0.32, 0.9, 0.05, 0.05]),
        increment_axes=fig.add_axes([0.38, 0.9, 0.05, 0.05]),
        active_axes=fig.add_axes([0.1, 0.6, 0.2, 0.25]),
        slider_axes={
            "velocity_kms": fig.add_axes([0.5, 0.8, 0.3, 0.03]),
            "sigma_kms": fig.add_axes([0.5, 0.7, 0.3, 0.03]),
        },
    )
    return fig


def test_constructor_rejects_component_missing_a_controlled_parameter():
    bad_component = Component(parameters=[Parameter("only_this", 1.0, 0.0, 2.0)])
    bad_params = ModelParameters(n_components=1, components=[bad_component])
    with pytest.raises(ValueError):
        ComponentController(bad_params, ["velocity_kms"], make_new_component=_make_component)


def test_switching_active_component_preserves_every_component_value():
    params = ModelParameters(n_components=2, components=[_make_component(), _make_component()])
    params.components[0]["velocity_kms"].value = 10.0
    params.components[1]["velocity_kms"].value = -30.0
    controller = ComponentController(params, ["velocity_kms", "sigma_kms"], make_new_component=_make_component)
    fig = _wire(controller)

    controller._on_active_change("C1")
    assert params.active_component_index == 1
    assert params.active_component["velocity_kms"].value == pytest.approx(-30.0)
    assert controller._sliders["velocity_kms"].val == pytest.approx(-30.0)
    # switching back: component 0's value must be untouched by having been inactive
    controller._on_active_change("C0")
    assert params.active_component["velocity_kms"].value == pytest.approx(10.0)
    plt.close(fig)


def test_slider_drag_only_edits_the_active_component():
    params = ModelParameters(n_components=2, components=[_make_component(), _make_component()])
    controller = ComponentController(params, ["velocity_kms", "sigma_kms"], make_new_component=_make_component)
    fig = _wire(controller)

    controller._on_active_change("C1")
    controller._on_slider_change("velocity_kms", -99.0)
    assert params.components[1]["velocity_kms"].value == pytest.approx(-99.0)
    assert params.components[0]["velocity_kms"].value == pytest.approx(0.0)  # untouched
    plt.close(fig)


def test_increment_adds_component_and_makes_it_active():
    params = ModelParameters(n_components=1, components=[_make_component()])
    controller = ComponentController(params, ["velocity_kms", "sigma_kms"], make_new_component=_make_component)
    fig = _wire(controller)

    controller._on_count_delta(+1)
    assert params.n_components == 2
    assert params.active_component_index == 1
    plt.close(fig)


def test_decrement_removes_active_component():
    params = ModelParameters(n_components=3, components=[_make_component(), _make_component(), _make_component()])
    params.components[1]["velocity_kms"].value = 42.0
    controller = ComponentController(params, ["velocity_kms", "sigma_kms"], make_new_component=_make_component)
    fig = _wire(controller)

    controller._on_active_change("C1")
    controller._on_count_delta(-1)
    assert params.n_components == 2
    # component that was at index 2 is now the survivor at index 1, unaffected
    assert 42.0 not in [c["velocity_kms"].value for c in params.components]
    plt.close(fig)


def test_decrement_never_removes_the_last_component():
    params = ModelParameters(n_components=1, components=[_make_component()])
    controller = ComponentController(params, ["velocity_kms", "sigma_kms"], make_new_component=_make_component)
    fig = _wire(controller)

    controller._on_count_delta(-1)
    assert params.n_components == 1
    plt.close(fig)


def test_on_change_callback_fires_for_every_kind_of_edit():
    params = ModelParameters(n_components=1, components=[_make_component()])
    calls = []
    controller = ComponentController(params, ["velocity_kms", "sigma_kms"],
                                       make_new_component=_make_component, on_change=lambda mp: calls.append(mp))
    fig = _wire(controller)

    controller._on_count_delta(+1)
    controller._on_active_change("C0")
    controller._on_slider_change("velocity_kms", 5.0)
    assert len(calls) == 3
    assert all(call is params for call in calls)
    plt.close(fig)


def test_slider_bounds_come_from_the_parameter_bounds():
    params = ModelParameters(n_components=1, components=[_make_component()])
    controller = ComponentController(params, ["velocity_kms", "sigma_kms"], make_new_component=_make_component)
    fig = _wire(controller)

    assert controller._sliders["velocity_kms"].valmin == pytest.approx(-500.0)
    assert controller._sliders["velocity_kms"].valmax == pytest.approx(500.0)
    plt.close(fig)
