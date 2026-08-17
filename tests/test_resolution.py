"""Tests for instrumental-resolution conversion and convolution utilities."""
import numpy as np
import pytest

from core.resolution import (
    sigma_angstrom_from_R,
    sigma_angstrom_from_fwhm_kms,
    combine_gaussian_sigma,
    effective_doppler_b_kms,
    ResolutionModel,
    DefaultResolutionWarning,
    convolve_variable_gaussian,
)

C_KMS = 299792.458
FWHM_TO_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))


def test_sigma_from_resolving_power_matches_definition():
    wave = np.array([4000.0, 6000.0])
    R = np.array([2000.0, 3000.0])
    expected = (wave / R) / FWHM_TO_SIGMA
    assert np.allclose(sigma_angstrom_from_R(wave, R), expected)


def test_sigma_from_constant_velocity_fwhm_scales_with_wavelength():
    wave = np.array([4000.0, 8000.0])
    sigma = sigma_angstrom_from_fwhm_kms(wave, 120.0)
    assert sigma[1] == pytest.approx(2.0 * sigma[0])
    assert sigma[0] == pytest.approx((4000.0 * 120.0 / C_KMS) / FWHM_TO_SIGMA)


def test_gaussian_widths_and_doppler_b_add_in_quadrature():
    assert combine_gaussian_sigma(3.0, 4.0) == pytest.approx(5.0)
    expected = np.sqrt(10.0**2 + (np.sqrt(2.0) * 7.0) ** 2)
    assert effective_doppler_b_kms(10.0, 7.0) == pytest.approx(expected)


def test_resolution_model_from_table_interpolates_R():
    model = ResolutionModel.from_table([4000.0, 6000.0], R=[2000.0, 3000.0], source="test")
    assert model.source == "test"
    assert model.R([4000.0, 6000.0])[0] == pytest.approx(2000.0)
    assert model.R([4000.0, 6000.0])[1] == pytest.approx(3000.0)


def test_constant_fwhm_velocity_resolution_has_constant_velocity_width():
    model = ResolutionModel.from_constant_fwhm_kms(100.0)
    wave = np.array([3000.0, 6000.0])
    fwhm = model.fwhm_angstrom(wave)
    assert fwhm[1] == pytest.approx(2.0 * fwhm[0])


def test_variable_gaussian_convolution_preserves_constant_spectrum():
    model_wave = np.linspace(4900.0, 5100.0, 4001)
    model_flux = np.full(model_wave.size, 3.2)
    target_wave = np.linspace(4920.0, 5080.0, 401)
    sigma = np.linspace(0.5, 2.0, target_wave.size)
    convolved = convolve_variable_gaussian(model_wave, model_flux, target_wave, sigma)
    assert np.allclose(convolved, 3.2, rtol=2e-4, atol=2e-4)


def test_default_fallback_warns_and_marks_source(tmp_path):
    path = tmp_path / "resolution.csv"
    path.write_text("lambda_A,R\n4000,2000\n6000,3000\n")
    with pytest.warns(DefaultResolutionWarning):
        model = ResolutionModel.default_fallback(path)
    assert model.is_default
    assert "default_fallback" in model.source


def test_variable_gaussian_tiny_kernel_falls_back_to_interpolation_not_zero():
    # Regression for stellar-template holes: at 1500 A, sigma=1 km/s is only
    # ~0.005 A, far narrower than this 1-A native template sampling.  The old
    # direct-summation path found no native point in many kernel windows and
    # silently returned zero.  The narrow-kernel limit must be interpolation.
    model_wave = np.arange(1200.0, 1801.0, 1.0)
    model_flux = 2.0 + 0.2 * np.sin(model_wave / 17.0)
    target_wave = np.arange(1200.2, 1799.9, 0.2)
    sigma = target_wave * 1.0 / C_KMS
    transformed = convolve_variable_gaussian(model_wave, model_flux, target_wave, sigma)
    expected = np.interp(target_wave, model_wave, model_flux)
    assert np.all(np.isfinite(transformed))
    assert np.all(transformed > 0.0)
    assert np.allclose(transformed, expected, rtol=1e-12, atol=1e-12)


def test_variable_gaussian_tiny_kernel_preserves_constant_positive_template():
    model_wave = np.arange(1000.0, 2001.0, 1.0)
    model_flux = np.full(model_wave.size, 3.2)
    target_wave = np.arange(1100.13, 1900.0, 0.23)
    sigma = target_wave * 1.0 / C_KMS
    transformed = convolve_variable_gaussian(model_wave, model_flux, target_wave, sigma)
    assert np.allclose(transformed, 3.2)
    assert not np.any(transformed == 0.0)
