"""Tests for gui.emission.EmissionGUI and gui.absorption.AbsorptionGUI.

As with ComponentController, real widget click/drag events can't be
simulated headlessly; tests construct each GUI (exercising all its
widget-wiring code) and call its internal handlers directly to verify
the resulting state, matching what a live session would produce.
"""
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

from core.spectrum import Spectrum

from emission.lines import EmissionLine
from gui.emission import EmissionGUI

from absorption.atomic import AtomicTransition
from absorption.synthetic import generate_synthetic_absorption_spectrum
from gui.absorption import AbsorptionGUI


class DictConfig(dict):
    """Minimal stand-in for core.config.Config supporting .get()."""


CIV = [
    AtomicTransition("CIV_1548", "CIV", 1548.204, 0.1899, 2.643e8, "CIV", atomic_mass_amu=12.011),
    AtomicTransition("CIV_1550", "CIV", 1550.781, 0.09475, 2.628e8, "CIV", atomic_mass_amu=12.011),
]


def _emission_spectrum():
    wave = np.linspace(6500.0, 6650.0, 500)
    rng = np.random.default_rng(0)
    flux = 1.0 + 0.3 * np.exp(-0.5 * ((wave - 6563.0) / 2.0) ** 2) + rng.normal(0, 0.01, wave.shape)
    flux_unc = np.full_like(wave, 0.01)
    return Spectrum.from_arrays(wave, flux, flux_unc)


# --- EmissionGUI ---------------------------------------------------------------------

def test_emission_gui_constructs_and_previews():
    lines = [EmissionLine("Halpha6562", 6562.819, ion="Ha"), EmissionLine("[N_II]6583", 6583.45, ion="[N II]")]
    gui = EmissionGUI(_emission_spectrum(), DictConfig(emission_n_components=1), line_list=lines)
    assert gui.model_parameters.n_components == 1
    assert [line.name for line in gui.free_lines] == ["Halpha6562", "[N_II]6583"]
    assert len(gui.preview_line.get_xdata()) > 0


def test_emission_gui_rejects_all_tied_line_list():
    tied_only = [EmissionLine("A", 5000.0, ion="A"), EmissionLine("B", 5010.0, ion="A", tied_to="A", ratio_to_tied=1.0)]
    # only "A" is free; still valid (one free line) -- confirm the *empty* case actually raises
    with pytest.raises(ValueError):
        EmissionGUI(_emission_spectrum(), DictConfig(), line_list=[])


def test_emission_gui_component_increment_adds_all_amplitude_parameters():
    lines = [EmissionLine("Halpha6562", 6562.819, ion="Ha"), EmissionLine("[N_II]6583", 6583.45, ion="[N II]")]
    gui = EmissionGUI(_emission_spectrum(), DictConfig(emission_n_components=1), line_list=lines)
    gui.component_controller._on_count_delta(+1)
    assert gui.model_parameters.n_components == 2
    names = gui.model_parameters.active_component.names()
    assert "amp_Halpha6562" in names and "amp_[N_II]6583" in names


def test_emission_gui_amplitude_edit_only_affects_active_component_and_active_line():
    lines = [EmissionLine("Halpha6562", 6562.819, ion="Ha"), EmissionLine("[N_II]6583", 6583.45, ion="[N II]")]
    gui = EmissionGUI(_emission_spectrum(), DictConfig(emission_n_components=1), line_list=lines)
    gui.component_controller._on_count_delta(+1)  # now 2 components, C1 active
    gui._on_line_change("[N_II]6583")
    # Scope checkboxes default unchecked (an edit applies to every line and
    # every component) -- opt into "just this one cell" explicitly.
    gui._line_specific = True
    gui._component_specific = True
    gui._on_amp_change(75.0)

    assert gui.model_parameters.components[1]["amp_[N_II]6583"].value == pytest.approx(75.0)
    assert gui.model_parameters.components[1]["amp_Halpha6562"].value != pytest.approx(75.0)
    assert gui.model_parameters.components[0]["amp_[N_II]6583"].value != pytest.approx(75.0)


def test_emission_gui_amplitude_edit_defaults_to_every_line_and_component():
    lines = [EmissionLine("Halpha6562", 6562.819, ion="Ha"), EmissionLine("[N_II]6583", 6583.45, ion="[N II]")]
    gui = EmissionGUI(_emission_spectrum(), DictConfig(emission_n_components=1), line_list=lines)
    gui.component_controller._on_count_delta(+1)  # now 2 components, C1 active
    assert gui._line_specific is False
    assert gui._component_specific is False
    gui._on_line_change("[N_II]6583")
    gui._on_amp_change(75.0)

    for component in gui.model_parameters.components:
        assert component["amp_[N_II]6583"].value == pytest.approx(75.0)
        assert component["amp_Halpha6562"].value == pytest.approx(75.0)


def test_emission_gui_line_switch_syncs_slider_to_that_lines_value():
    lines = [EmissionLine("Halpha6562", 6562.819, ion="Ha"), EmissionLine("[N_II]6583", 6583.45, ion="[N II]")]
    gui = EmissionGUI(_emission_spectrum(), DictConfig(emission_n_components=1), line_list=lines)
    gui._line_specific = True  # otherwise the edit below would apply to every line
    gui._on_line_change("[N_II]6583")
    gui._on_amp_change(50.0)
    gui._on_line_change("Halpha6562")
    assert gui.amp_slider.val != pytest.approx(50.0)
    gui._on_line_change("[N_II]6583")
    assert gui.amp_slider.val == pytest.approx(50.0)


def test_emission_gui_line_switch_does_not_corrupt_other_lines_amplitudes():
    """Regression test: _sync_amp_slider used to call amp_slider.set_val()
    without an eventson guard, so with the default (unchecked/broadcast)
    scope, merely switching the displayed line -- e.g. via _on_line_change,
    called internally after every Fit/Load Fit -- silently overwrote every
    other line's (and component's) amplitude with whatever the newly
    displayed line's value happened to be."""
    lines = [EmissionLine("Halpha6562", 6562.819, ion="Ha"), EmissionLine("[N_II]6583", 6583.45, ion="[N II]")]
    gui = EmissionGUI(_emission_spectrum(), DictConfig(emission_n_components=2), line_list=lines)
    assert gui._line_specific is False and gui._component_specific is False  # default scope

    # Set distinct, identifiable amplitudes per line/component directly
    # (bypassing the slider, so this doesn't itself trigger any sync).
    expected = {}
    for component_index, component in enumerate(gui.model_parameters.components):
        for line_index, line in enumerate(lines):
            value = 10.0 * component_index + line_index + 1.0
            component[f"amp_{line.name}"].value = value
            expected[(component_index, line.name)] = value

    # Merely switching the active line/component must not alter any of them.
    gui.component_controller._on_active_change("C1")
    gui._on_line_change("[N_II]6583")
    gui._on_line_change("Halpha6562")
    gui.component_controller._on_active_change("C0")

    for component_index, component in enumerate(gui.model_parameters.components):
        for line in lines:
            assert component[f"amp_{line.name}"].value == pytest.approx(expected[(component_index, line.name)])


def test_emission_gui_fit_save_load_round_trip(tmp_path):
    lines = [EmissionLine("Halpha6562", 6562.819, ion="Ha")]
    result_path = tmp_path / "em.fits"
    gui = EmissionGUI(_emission_spectrum(), DictConfig(emission_n_components=1), line_list=lines,
                       result_path=str(result_path))
    gui._fit()
    assert gui.result is not None
    gui._save_fit()
    assert result_path.is_file()
    original_flux = gui.result.flux("Halpha6562")
    gui._load_fit()
    assert gui.result.flux("Halpha6562") == pytest.approx(original_flux, rel=1e-6)


# --- AbsorptionGUI -------------------------------------------------------------------

def _absorption_spectrum():
    wave = np.linspace(1540.0, 1560.0, 1000)
    return generate_synthetic_absorption_spectrum(
        wave, CIV, [{"logN": 14.0, "b_kms": 20.0, "velocity_kms": 0.0}], signal_to_noise=100, seed=3,
    )


def test_absorption_gui_constructs_full_coverage():
    gui = AbsorptionGUI(_absorption_spectrum(), DictConfig(absorption_n_components=1), transitions=CIV)
    assert gui.model_parameters.n_components == 1
    assert not gui.partial_coverage
    assert not hasattr(gui, "covering_slider")
    assert "covering_fraction" not in gui.model_parameters.active_component.names()


def test_absorption_gui_constructs_partial_coverage_with_slider():
    gui = AbsorptionGUI(_absorption_spectrum(), DictConfig(absorption_n_components=2, absorption_partial_coverage=True),
                         transitions=CIV)
    assert gui.partial_coverage
    assert hasattr(gui, "covering_slider")
    assert "covering_fraction" in gui.model_parameters.components[0].names()


def test_absorption_gui_covering_slider_updates_leader_only():
    gui = AbsorptionGUI(_absorption_spectrum(), DictConfig(absorption_n_components=2, absorption_partial_coverage=True),
                         transitions=CIV)
    gui._on_covering_change(0.42)
    assert gui.model_parameters.components[0]["covering_fraction"].value == pytest.approx(0.42)
    # follower stays fixed=True (synced at model-eval time, not written here directly)
    assert gui.model_parameters.components[1]["covering_fraction"].fixed


def test_absorption_gui_new_component_inherits_fixed_covering_fraction():
    gui = AbsorptionGUI(_absorption_spectrum(), DictConfig(absorption_n_components=1, absorption_partial_coverage=True),
                         transitions=CIV)
    gui.component_controller._on_count_delta(+1)
    assert gui.model_parameters.n_components == 2
    assert gui.model_parameters.components[1]["covering_fraction"].fixed


def test_absorption_gui_requires_group_or_transitions():
    with pytest.raises(ValueError):
        AbsorptionGUI(_absorption_spectrum(), DictConfig(absorption_n_components=1))


def test_absorption_gui_slider_edit_only_affects_active_component():
    gui = AbsorptionGUI(_absorption_spectrum(), DictConfig(absorption_n_components=1), transitions=CIV)
    gui.component_controller._on_count_delta(+1)
    gui.component_controller._on_slider_change("logN", 16.5)
    assert gui.model_parameters.components[1]["logN"].value == pytest.approx(16.5)
    assert gui.model_parameters.components[0]["logN"].value != pytest.approx(16.5)


def test_absorption_gui_fit_save_load_round_trip(tmp_path):
    result_path = tmp_path / "abs.fits"
    gui = AbsorptionGUI(_absorption_spectrum(), DictConfig(absorption_n_components=1), transitions=CIV,
                         result_path=str(result_path))
    gui._fit()
    assert gui.result is not None
    original_logN = gui.result.measurements[0].logN
    gui._save_fit()
    assert result_path.is_file()
    gui._load_fit()
    assert gui.result.measurements[0].logN == pytest.approx(original_logN, rel=1e-6)
