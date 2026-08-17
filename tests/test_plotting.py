"""Tests for the plotting module. Since these produce matplotlib figures
rather than numeric results, tests check that each function runs without
error, returns the expected object type, and (for a few) that basic
structural expectations hold (axis limits, number of panels) -- not
pixel-exact image comparison, which would be brittle across matplotlib
versions/backends.
"""
import matplotlib
matplotlib.use("Agg")  # headless backend for test environments

import numpy as np
import pytest
import matplotlib.pyplot as plt
import matplotlib.figure

from core.results import FitResult
from core.parameters import ModelParameters, Component, Parameter
from core.statistics import FitStatistics

from plotting.spectrum import plot_spectrum_only, plot_spectrum_fit, plot_spectrum_fit_axes
from plotting.residuals import add_residual_panel, plot_residuals
from plotting.stamps import wavelength_to_velocity_kms, plot_line_stamps, plot_velocity_stack
from plotting.diagnostics import plot_residual_diagnostics


class FakeLine:
    def __init__(self, name, rest_wavelength_angstrom):
        self.name = name
        self.rest_wavelength_angstrom = rest_wavelength_angstrom


def _make_fit_result(n=500, seed=0):
    rng = np.random.default_rng(seed)
    wave = np.linspace(6500.0, 6600.0, n)
    model = 1.0 + 0.3 * np.exp(-0.5 * ((wave - 6563.0) / 2.0) ** 2)
    flux_unc = np.full(n, 0.02)
    flux = model + rng.normal(0, 0.02, n)
    mask = np.ones(n, dtype=bool)
    mask[:10] = False
    stats = FitStatistics(n_data=n - 10, n_eff=n - 10, k_params=3, dof=n - 13, chi_square=float(n - 10),
                           reduced_chi_square=1.0, jitter_scale=1.0, neg2_log_likelihood=float(n - 10),
                           bic=520.0, aic=506.0, aicc=506.1)
    params = ModelParameters(n_components=1, components=[Component(parameters=[Parameter("a", 1.0, 0.0, 2.0)])])
    return FitResult(parameters=params, parameter_uncertainties=None, wave=wave, flux=flux, flux_unc=flux_unc,
                      mask=mask, model=model, statistics=stats, method="test")


# --- wavelength_to_velocity_kms -----------------------------------------------------

def test_wavelength_to_velocity_kms_zero_at_rest_wavelength():
    rest = 1548.204
    v = wavelength_to_velocity_kms(np.array([rest]), rest, redshift=0.0)
    assert v[0] == pytest.approx(0.0, abs=1e-6)


def test_wavelength_to_velocity_kms_matches_forward_doppler_formula():
    from absorption.profiles import C_KMS
    rest = 1548.204
    true_velocity = 120.0
    beta = true_velocity / C_KMS
    shifted_wave = rest * np.sqrt((1 + beta) / (1 - beta))
    recovered = wavelength_to_velocity_kms(np.array([shifted_wave]), rest, redshift=0.0)
    assert recovered[0] == pytest.approx(true_velocity, abs=1e-4)


def test_wavelength_to_velocity_kms_accounts_for_redshift():
    rest = 1215.6701
    z = 0.5
    observed_line_center = rest * (1 + z)
    v = wavelength_to_velocity_kms(np.array([observed_line_center]), rest, redshift=z)
    assert v[0] == pytest.approx(0.0, abs=1e-4)


# --- plotting.spectrum ---------------------------------------------------------------

def test_plot_spectrum_only_returns_axes():
    wave = np.linspace(6500, 6600, 200)
    flux = np.ones_like(wave)
    ax = plot_spectrum_only(wave, flux)
    assert ax is not None
    plt.close(ax.figure)


def test_plot_spectrum_only_with_mask_and_model_and_uncertainty():
    wave = np.linspace(6500, 6600, 200)
    flux = np.ones_like(wave)
    flux_unc = np.full_like(wave, 0.05)
    mask = np.ones_like(wave, dtype=bool)
    mask[:20] = False
    model = np.ones_like(wave) * 0.98
    ax = plot_spectrum_only(wave, flux, flux_unc=flux_unc, model=model, mask=mask)
    assert ax is not None
    plt.close(ax.figure)


def test_plot_spectrum_fit_returns_figure_with_two_panels_by_default():
    fit_result = _make_fit_result()
    fig = plot_spectrum_fit(fit_result)
    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(fig.axes) == 2
    plt.close(fig)


def test_plot_spectrum_fit_no_residuals_single_panel():
    fit_result = _make_fit_result()
    fig = plot_spectrum_fit(fit_result, show_residuals=False)
    assert len(fig.axes) == 1
    plt.close(fig)


def test_plot_spectrum_fit_respects_wave_range():
    fit_result = _make_fit_result()
    fig = plot_spectrum_fit(fit_result, wave_range=(6520, 6540))
    assert fig.axes[0].get_xlim() == pytest.approx((6520, 6540))
    plt.close(fig)


def test_plot_spectrum_fit_axes_composable_with_external_layout():
    fit_result = _make_fit_result()
    fig, (ax_main, ax_resid) = plt.subplots(2, 1)
    returned_main, returned_resid = plot_spectrum_fit_axes(fit_result, ax_main, ax_resid)
    assert returned_main is ax_main
    assert returned_resid is ax_resid
    plt.close(fig)


# --- plotting.residuals ---------------------------------------------------------------

def test_add_residual_panel_normalized_by_uncertainty():
    fit_result = _make_fit_result()
    ax = add_residual_panel(fit_result, normalized=True)
    assert ax.get_ylabel().startswith("Residual")
    plt.close(ax.figure)


def test_plot_residuals_standalone_figure():
    fit_result = _make_fit_result()
    fig = plot_residuals(fit_result)
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)


def test_add_residual_panel_without_flux_unc_falls_back_to_raw_residual():
    fit_result = _make_fit_result()
    fit_result.flux_unc = None
    ax = add_residual_panel(fit_result)
    assert ax.get_ylabel() == "Residual"
    plt.close(ax.figure)


# --- plotting.stamps -------------------------------------------------------------------

def test_plot_line_stamps_grid_shape():
    lines = [FakeLine("A", 6500.0), FakeLine("B", 6520.0), FakeLine("C", 6540.0)]
    wave = np.linspace(6480, 6560, 900)
    flux = np.ones_like(wave)
    fig = plot_line_stamps(wave, flux, lines, velocity_half_width_kms=300.0, ncols=2)
    assert len(fig.axes) >= len(lines)
    plt.close(fig)


def test_plot_line_stamps_handles_uncovered_line_gracefully():
    lines = [FakeLine("Covered", 6500.0), FakeLine("NotCovered", 9000.0)]
    wave = np.linspace(6480, 6520, 200)
    flux = np.ones_like(wave)
    fig = plot_line_stamps(wave, flux, lines, velocity_half_width_kms=200.0, ncols=2)
    assert len(fig.axes) >= 2
    plt.close(fig)


def test_plot_line_stamps_rejects_empty_lines():
    wave = np.linspace(6480, 6520, 200)
    flux = np.ones_like(wave)
    with pytest.raises(ValueError):
        plot_line_stamps(wave, flux, [], velocity_half_width_kms=200.0)


def test_plot_velocity_stack_returns_figure():
    lines = [FakeLine("A", 1548.204), FakeLine("B", 1550.781)]
    wave = np.linspace(1540, 1560, 1000)
    flux = np.ones_like(wave)
    flux[400:600] -= 0.3
    fig = plot_velocity_stack(wave, flux, lines, reference_redshift=0.0, velocity_range_kms=(-500, 500))
    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(fig.axes) == 1
    plt.close(fig)


def test_plot_velocity_stack_with_component_markers_and_model():
    lines = [FakeLine("A", 1548.204)]
    wave = np.linspace(1540, 1560, 500)
    flux = np.ones_like(wave)
    model = np.ones_like(wave) * 0.9
    fig = plot_velocity_stack(wave, flux, lines, model=model, reference_redshift=0.0,
                                component_velocities_kms=[0.0, 50.0])
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)


def test_plot_velocity_stack_rejects_empty_lines():
    wave = np.linspace(1540, 1560, 500)
    flux = np.ones_like(wave)
    with pytest.raises(ValueError):
        plot_velocity_stack(wave, flux, [], reference_redshift=0.0)


# --- plotting.diagnostics --------------------------------------------------------------

def test_plot_residual_diagnostics_returns_figure_with_three_axes():
    fit_result = _make_fit_result()
    fig = plot_residual_diagnostics(fit_result)
    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(fig.axes) == 3
    plt.close(fig)


def test_plot_residual_diagnostics_without_flux_unc():
    fit_result = _make_fit_result()
    fit_result.flux_unc = None
    fig = plot_residual_diagnostics(fit_result)
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)
