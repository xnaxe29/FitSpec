"""Tests for the emission/absorption plotting wrappers (plotting.emission,
plotting.absorption): these run against real fit results (not fakes),
confirming the wrappers correctly pull line lists / component
kinematics / config values into the generic plotting primitives.
"""
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest
import matplotlib.figure
import matplotlib.pyplot as plt

from core.spectrum import Spectrum
from core.parameters import ModelParameters, Component, Parameter

from emission.lines import EmissionLine
from emission.emission_model import make_emission_model_func
from emission.emission_fit import fit_emission_spectrum

from absorption.atomic import AtomicTransition
from absorption.absorption_model import AbsorptionSystem
from absorption.rejection import fit_joint_absorption_spectrum_with_rejection
from absorption.synthetic import generate_synthetic_absorption_spectrum

from plotting.emission import (
    plot_emission_fit, plot_emission_line_stamps, plot_emission_velocity_stack, plot_emission_line_fluxes,
)
from plotting.absorption import (
    plot_absorption_fit, plot_absorption_line_stamps, plot_absorption_velocity_stack,
    plot_absorption_component_summary,
)


class DictConfig(dict):
    pass


@pytest.fixture(scope="module")
def emission_result():
    lines = [EmissionLine("Halpha6562", 6562.819, ion="Ha"), EmissionLine("[N_II]6583", 6583.45, ion="[N II]")]
    wave = np.linspace(6500.0, 6650.0, 600)
    model_func = make_emission_model_func(lines)
    true_params = ModelParameters(n_components=1, components=[Component(parameters=[
        Parameter("velocity_kms", 0.0, -500, 500), Parameter("sigma_kms", 50.0, 1, 1000),
        Parameter("amp_Halpha6562", 200.0, 0, np.inf), Parameter("amp_[N_II]6583", 60.0, 0, np.inf),
    ])])
    flux = model_func(wave, true_params)
    flux_unc = np.full_like(wave, 0.02 * flux.max())
    rng = np.random.default_rng(0)
    flux_noisy = flux + rng.normal(0, flux_unc[0], wave.shape)
    spectrum = Spectrum.from_arrays(wave, flux_noisy, flux_unc)
    return fit_emission_spectrum(spectrum, DictConfig(emission_n_components=1), line_list=lines)


@pytest.fixture(scope="module")
def absorption_result():
    civ = [AtomicTransition("CIV_1548", "CIV", 1548.204, 0.1899, 2.643e8, "CIV", atomic_mass_amu=12.011),
           AtomicTransition("CIV_1550", "CIV", 1550.781, 0.09475, 2.628e8, "CIV", atomic_mass_amu=12.011)]
    wave = np.linspace(1540.0, 1560.0, 1200)
    spectrum = generate_synthetic_absorption_spectrum(
        wave, civ, [{"logN": 14.0, "b_kms": 20.0, "velocity_kms": 0.0}], signal_to_noise=150, seed=7)
    systems = [AbsorptionSystem(civ, n_components=3, label="CIV", logN_bounds=(11.0, 18.0), logN_initial=12.5,
                                 b_initial_kms=20.0, b_bounds_kms=(3.0, 300.0), velocity_bounds_kms=(-100, 100))]
    return fit_joint_absorption_spectrum_with_rejection(spectrum, systems, reject_margin_dex=0.2)


# --- plotting.emission -----------------------------------------------------------------

def test_plot_emission_fit(emission_result):
    fig = plot_emission_fit(emission_result)
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)


def test_plot_emission_line_stamps(emission_result):
    fig = plot_emission_line_stamps(emission_result, velocity_half_width_kms=400.0)
    assert len(fig.axes) >= 2
    plt.close(fig)


def test_plot_emission_velocity_stack_default_lines(emission_result):
    fig = plot_emission_velocity_stack(emission_result, velocity_range_kms=(-400, 400))
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)


def test_plot_emission_velocity_stack_config_plot_lines_subset(emission_result):
    config = DictConfig(emission_fig_plot_lines="Halpha6562", emission_fig_velocity_limit_kms=300.0)
    fig = plot_emission_velocity_stack(emission_result, config=config)
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)


def test_plot_emission_line_fluxes(emission_result):
    fig = plot_emission_line_fluxes(emission_result)
    assert isinstance(fig, matplotlib.figure.Figure)
    # one bar per measured line
    assert len(fig.axes[0].patches) == len(emission_result.measurements)
    plt.close(fig)


# --- plotting.absorption ----------------------------------------------------------------

def test_plot_absorption_fit(absorption_result):
    fig = plot_absorption_fit(absorption_result)
    assert isinstance(fig, matplotlib.figure.Figure)
    assert "Transmission" in fig.axes[0].get_ylabel()
    plt.close(fig)


def test_plot_absorption_line_stamps(absorption_result):
    fig = plot_absorption_line_stamps(absorption_result, velocity_half_width_kms=400.0)
    assert len(fig.axes) >= 2  # doublet: CIV_1548 + CIV_1550
    plt.close(fig)


def test_plot_absorption_velocity_stack_marks_all_components(absorption_result):
    fig = plot_absorption_velocity_stack(absorption_result, velocity_range_kms=(-400, 400))
    assert isinstance(fig, matplotlib.figure.Figure)
    ax = fig.axes[0]
    # one vertical line per component (velocity markers) plus the zero-velocity reference line
    n_components = absorption_result.fit_result.parameters.n_components
    assert len(ax.get_lines()) >= n_components
    plt.close(fig)


def test_plot_absorption_velocity_stack_config_subsets_transitions_and_velocity_limit(absorption_result):
    config = DictConfig(absorption_fig_plot_transitions="CIV_1548", absorption_fig_velocity_limit_kms=200.0)
    fig = plot_absorption_velocity_stack(absorption_result, config=config)
    ax = fig.axes[0]
    assert ax.get_xlim() == pytest.approx((-200.0, 200.0))
    labels = [text.get_text() for text in ax.texts]
    assert labels == ["CIV_1548"]
    plt.close(fig)


def test_plot_absorption_component_summary_distinguishes_upper_limits(absorption_result):
    fig = plot_absorption_component_summary(absorption_result)
    assert len(fig.axes) == 2
    assert any(m.is_upper_limit for m in absorption_result.measurements)  # sanity: fixture actually has one
    plt.close(fig)
