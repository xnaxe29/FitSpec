"""Numerical tests for FitSpec's flux-conservative rebinning."""
import numpy as np
import pytest

from core.rebinning import rebin_spectrum, apply_permanent_rebinning, compute_display_smoothing
from core.spectrum import Spectrum


def _integrated_flux(wave, flux):
    edges = np.empty(wave.size + 1)
    edges[1:-1] = 0.5 * (wave[:-1] + wave[1:])
    edges[0] = wave[0] - 0.5 * (wave[1] - wave[0])
    edges[-1] = wave[-1] + 0.5 * (wave[-1] - wave[-2])
    return np.nansum(flux * np.diff(edges))


def test_constant_flux_remains_constant_after_rebinning():
    wave = np.arange(100.0, 120.0)
    flux = np.full(wave.size, 7.0)
    err = np.full(wave.size, 2.0)
    result = rebin_spectrum(wave, flux, err, binsize=4)
    assert np.allclose(result.flux, 7.0)
    assert np.allclose(result.coverage_fraction, 1.0)
    assert result.rebin_matrix.shape == (5, 20)


def test_integrated_flux_is_conserved_for_complete_bins():
    wave = np.arange(50.0)
    flux = 1.0 + 0.02 * wave + np.exp(-0.5 * ((wave - 24.0) / 3.0) ** 2)
    err = np.full(wave.size, 0.1)
    result = rebin_spectrum(wave, flux, err, binsize=5)
    assert _integrated_flux(result.wave, result.flux) == pytest.approx(
        _integrated_flux(wave, flux), rel=1e-12, abs=1e-12
    )


def test_independent_uncertainties_propagate_through_linear_operator():
    wave = np.arange(8.0)
    flux = np.arange(8.0)
    err = np.full(8, 2.0)
    result = rebin_spectrum(wave, flux, err, binsize=2)
    # Equal-width two-pixel averages have sigma = 2 / sqrt(2).
    assert np.allclose(result.flux_unc, 2.0 / np.sqrt(2.0))
    assert result.covariance.shape == (4, 4)


def test_gap_below_minimum_coverage_is_nan_not_bridged():
    wave = np.arange(8.0)
    flux = np.ones(8)
    err = np.ones(8)
    flux[2] = np.nan
    result = rebin_spectrum(wave, flux, err, binsize=2, min_coverage=1.0, fill_gaps=False)
    assert np.isnan(result.flux[1])
    assert result.coverage_fraction[1] < 1.0


def test_keep_partial_bin_retains_last_short_bin():
    wave = np.arange(10.0)
    result = rebin_spectrum(wave, np.ones(10), np.ones(10), binsize=4, keep_partial_bin=True)
    assert result.wave.size == 3
    assert np.allclose(result.flux, 1.0)


def test_display_smoothing_does_not_modify_spectrum_but_permanent_rebin_does():
    wave = np.arange(12.0)
    spectrum = Spectrum.from_arrays(wave, 1.0 + wave, np.full(12, 0.5))
    original_wave = spectrum.wave.copy()
    display_wave, display_flux, display_unc = compute_display_smoothing(spectrum, 3)
    assert np.array_equal(spectrum.wave, original_wave)
    assert display_wave.size == display_flux.size == display_unc.size == 4

    rebinned = apply_permanent_rebinning(spectrum, 3)
    assert spectrum.wave.size == 12  # input object is preserved
    assert rebinned.wave.size == 4
    assert not np.array_equal(rebinned.wave, original_wave)
