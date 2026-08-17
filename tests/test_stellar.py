"""Deterministic tests for the unified UV/optical stellar fitter."""
import numpy as np
import pytest

from core.spectrum import Spectrum
from stellar.stellar_models import (
    StellarLibrary,
    classify_spectral_regime,
    attenuation_transmission,
    physical_light_fractions,
    shift_broaden_resample_library,
)
from stellar.stellar_fit import fit_stellar_spectrum, _flux_to_lsun_factor


class Config(dict):
    def get(self, key, default=None):
        return super().get(key, default)


def _toy_library(wave=None):
    if wave is None:
        wave = np.linspace(4000.0, 4100.0, 201)
    x = (wave - wave.mean()) / 50.0
    templates = np.vstack([
        1.0 + 0.10 * x,
        1.0 - 0.12 * x + 0.04 * x * x,
        0.9 + 0.08 * np.exp(-0.5 * ((wave - 4050.0) / 7.0) ** 2),
    ])
    return StellarLibrary(
        wave=np.asarray(wave),
        flux_per_mass=templates,
        flux_reference=templates.copy(),
        ages_myr=np.array([5.0, 50.0, 500.0]),
        metallicity_codes=np.array(["z002", "z002", "z014"]),
        metallicities_solar=np.array([0.14, 0.14, 1.0]),
        reference_masses=np.ones(3),
        labels=("young", "mid", "old"),
        source_paths=("toy", "toy", "toy"),
        family="toy",
        surviving_mass_fractions=np.array([0.95, 0.85, 0.65]),
    )


def _fit_config(**overrides):
    cfg = Config(
        stellar_include_gas=False,
        stellar_fit_velocity=False,
        stellar_fit_sigma=False,
        stellar_minimum_fit_pixels=20,
        stellar_distance_kpc=10.0,
        stellar_ebv_initial=0.08,
        stellar_ebv_bounds="0,0.5",
        stellar_velocity_initial_kms=15.0,
        stellar_velocity_bounds_kms="-100,100",
        stellar_sigma_initial_kms=25.0,
        stellar_sigma_bounds_kms="1,100",
        stellar_mass_solver="bvls",
        stellar_max_nfev=100,
    )
    cfg.update(overrides)
    return cfg


def test_regime_classification_is_based_on_rest_wavelength_coverage():
    assert classify_spectral_regime([1200.0, 1700.0]) == "uv"
    assert classify_spectral_regime([4000.0, 7000.0]) == "optical"
    assert classify_spectral_regime([2500.0, 3200.0]) == "uv"  # larger UV span
    assert classify_spectral_regime([2800.0, 3800.0]) == "optical"


def test_attenuation_transmission_is_unity_at_zero_reddening():
    wave = np.linspace(1200.0, 8000.0, 200)
    assert np.allclose(attenuation_transmission(wave, 0.0), 1.0)


def test_shift_broaden_resample_identity_case_preserves_library_on_same_grid():
    lib = _toy_library()
    transformed = shift_broaden_resample_library(
        lib, lib.wave, ebv=0.0, velocity_kms=0.0, sigma_kms=0.0, resolution=None
    )
    assert transformed.shape == lib.flux_per_mass.shape
    assert np.allclose(transformed, lib.flux_per_mass, rtol=1e-10, atol=1e-12)


def test_physical_light_fractions_are_normalized_and_nonnegative():
    lib = _toy_library()
    fractions = physical_light_fractions(lib, np.array([2.0, 1.0, 0.0]))
    assert np.all(fractions >= 0.0)
    assert fractions.sum() == pytest.approx(1.0)
    assert fractions[2] == pytest.approx(0.0)


def test_unified_stellar_fitter_recovers_nonnegative_synthetic_mixture():
    lib = _toy_library()
    config = _fit_config(stellar_fit_velocity=True, stellar_fit_sigma=True)
    true_coeff = np.array([2.5, 0.7, 0.0])
    true_ebv, true_velocity, true_sigma = 0.10, 20.0, 30.0
    factor = _flux_to_lsun_factor(0.0, config)
    transformed = shift_broaden_resample_library(
        lib, lib.wave, ebv=true_ebv, velocity_kms=true_velocity, sigma_kms=true_sigma
    )
    luminosity = transformed.T @ true_coeff
    observed_flux = luminosity / factor
    observed_unc = np.full(lib.wave.size, 2e-3 / factor)
    spectrum = Spectrum.from_arrays(lib.wave, observed_flux, observed_unc, redshift=0.0)

    result = fit_stellar_spectrum(spectrum, config, library=lib, regime="optical")
    assert result.success
    assert np.all(result.coefficients >= 0.0)
    assert result.coefficients[0] == pytest.approx(true_coeff[0], rel=0.03, abs=0.03)
    assert result.coefficients[1] == pytest.approx(true_coeff[1], rel=0.08, abs=0.05)
    assert result.coefficients[2] == pytest.approx(0.0, abs=0.05)
    assert result.ebv == pytest.approx(true_ebv, abs=0.03)
    assert result.velocity_kms == pytest.approx(true_velocity, abs=8.0)
    assert result.sigma_kms == pytest.approx(true_sigma, abs=12.0)


def test_stellar_fitter_respects_input_mask():
    lib = _toy_library()
    config = _fit_config()
    factor = _flux_to_lsun_factor(0.0, config)
    transformed = shift_broaden_resample_library(
        lib, lib.wave, ebv=0.10, velocity_kms=20.0, sigma_kms=30.0
    )
    observed = (transformed.T @ np.array([1.0, 0.5, 0.0])) / factor
    unc = np.full(lib.wave.size, 2e-3 / factor)
    spectrum = Spectrum.from_arrays(lib.wave, observed, unc)
    spectrum.mask = np.ones(lib.wave.size, dtype=bool)
    spectrum.mask[:30] = False
    result = fit_stellar_spectrum(spectrum, config, library=lib, regime="optical")
    assert np.array_equal(result.mask, spectrum.mask)
    assert result.degrees_of_freedom < np.count_nonzero(spectrum.mask)


def test_stellar_fitter_requires_uncertainties():
    lib = _toy_library()
    spectrum = Spectrum.from_arrays(lib.wave, np.ones(lib.wave.size))
    with pytest.raises(ValueError, match="requires flux_unc"):
        fit_stellar_spectrum(spectrum, _fit_config(), library=lib, regime="optical")


def test_stellar_fitter_defaults_to_no_gas_nuisance_when_unspecified():
    lib = _toy_library()
    config = _fit_config()
    config.pop("stellar_include_gas", None)
    config.pop("stellar_fit_velocity", None)
    config.pop("stellar_fit_sigma", None)
    factor = _flux_to_lsun_factor(0.0, config)
    transformed = shift_broaden_resample_library(
        lib, lib.wave, ebv=0.08, velocity_kms=15.0, sigma_kms=25.0
    )
    observed = (transformed.T @ np.array([1.0, 0.3, 0.0])) / factor
    unc = np.full(lib.wave.size, 2e-3 / factor)
    spectrum = Spectrum.from_arrays(lib.wave, observed, unc)
    result = fit_stellar_spectrum(spectrum, config, library=lib, regime="optical")
    assert result.metadata["simultaneous_gas"] is False
    assert result.metadata["fit_velocity"] is False
    assert result.metadata["fit_sigma"] is False
    assert result.parameter_names == ("ebv",)
    assert result.velocity_kms == pytest.approx(config["stellar_velocity_initial_kms"])
    assert result.sigma_kms == pytest.approx(config["stellar_sigma_initial_kms"])


def test_stellar_velocity_and_sigma_are_independent_opt_in_parameters():
    lib = _toy_library()
    factor = _flux_to_lsun_factor(0.0, _fit_config())
    transformed = shift_broaden_resample_library(
        lib, lib.wave, ebv=0.08, velocity_kms=15.0, sigma_kms=25.0
    )
    observed = (transformed.T @ np.array([1.0, 0.3, 0.0])) / factor
    unc = np.full(lib.wave.size, 2e-3 / factor)
    spectrum = Spectrum.from_arrays(lib.wave, observed, unc)

    velocity_only = fit_stellar_spectrum(
        spectrum, _fit_config(stellar_fit_velocity=True, stellar_fit_sigma=False),
        library=lib, regime="optical",
    )
    assert velocity_only.parameter_names == ("ebv", "velocity_kms")
    assert velocity_only.metadata["fit_velocity"] is True
    assert velocity_only.metadata["fit_sigma"] is False
    assert velocity_only.sigma_kms == pytest.approx(25.0)

    sigma_only = fit_stellar_spectrum(
        spectrum, _fit_config(stellar_fit_velocity=False, stellar_fit_sigma=True),
        library=lib, regime="optical",
    )
    assert sigma_only.parameter_names == ("ebv", "sigma_kms")
    assert sigma_only.metadata["fit_velocity"] is False
    assert sigma_only.metadata["fit_sigma"] is True
    assert sigma_only.velocity_kms == pytest.approx(15.0)


def test_candidate_basis_caps_expensive_nnls_and_preserves_full_result_indexing():
    wave = np.linspace(4000.0, 4100.0, 201)
    x = (wave - wave.mean()) / 50.0
    n_models = 60
    templates = []
    for i in range(n_models):
        phase = i / max(1, n_models - 1)
        templates.append(
            1.0 + (0.02 + 0.15 * phase) * x
            + 0.04 * np.sin((1.0 + 3.0 * phase) * np.pi * x)
        )
    templates = np.asarray(templates)
    lib = StellarLibrary(
        wave=wave, flux_per_mass=templates, flux_reference=templates.copy(),
        ages_myr=np.linspace(1.0, 1000.0, n_models),
        metallicity_codes=np.asarray(["z014"] * n_models),
        metallicities_solar=np.ones(n_models), reference_masses=np.ones(n_models),
        labels=tuple(f"ssp{i}" for i in range(n_models)),
        source_paths=tuple("toy" for _ in range(n_models)), family="toy",
        surviving_mass_fractions=np.ones(n_models),
    )
    config = _fit_config(
        stellar_template_selection="candidate", stellar_candidate_max=5,
        stellar_ebv_initial=0.0, stellar_ebv_bounds="0,0.01",
    )
    factor = _flux_to_lsun_factor(0.0, config)
    observed = (templates[17] * 2.0 + templates[41] * 0.3) / factor
    unc = np.full(wave.size, 1e-3 / factor)
    spectrum = Spectrum.from_arrays(wave, observed, unc)
    result = fit_stellar_spectrum(spectrum, config, library=lib, regime="optical")

    assert result.success
    assert result.metadata["template_selection"] == "candidate"
    assert result.metadata["candidate_basis_size"] == 5
    assert result.coefficients.shape == (n_models,)
    assert np.count_nonzero(result.coefficients > 0) <= 5
    assert result.diagnostics.single_ssp_chi_square.shape == (n_models,)
    assert result.diagnostics.single_ssp_delta_chi_square.shape == (n_models,)


def test_full_template_selection_remains_explicit_opt_in():
    lib = _toy_library()
    config = _fit_config(stellar_template_selection="full")
    factor = _flux_to_lsun_factor(0.0, config)
    observed = lib.flux_per_mass[0] / factor
    unc = np.full(lib.wave.size, 2e-3 / factor)
    result = fit_stellar_spectrum(
        Spectrum.from_arrays(lib.wave, observed, unc), config, library=lib, regime="optical"
    )
    assert result.metadata["template_selection"] == "full"
    assert result.metadata["candidate_basis_size"] == lib.n_models


def test_prepare_stellar_library_slug_does_not_duplicate_family_keyword(monkeypatch):
    """SLUG's internal track-family selector must not collide with wrapper family."""
    from types import SimpleNamespace
    import stellar.stellar_fit as sf

    captured = {}

    def fake_loader(family, filename=None, *, wave_range=None, **kwargs):
        captured["family"] = family
        captured["filename"] = filename
        captured["wave_range"] = wave_range
        captured["kwargs"] = kwargs
        return SimpleNamespace()

    monkeypatch.setattr(sf, "load_stellar_library", fake_loader)
    spectrum = SimpleNamespace(
        wave=np.linspace(1200.0, 1700.0, 50),
        redshift=0.0,
    )
    config = {
        "stellar_library": "slug",
        "slug_h5_path": "/tmp/slug.h5",
        "slug_family": "MIST",
        "slug_subfamily": "vvcrit000",
        "slug_specsyn": "sb99hruv",
        "stellar_age_range_myr": [1.0, 40.0],
    }
    regime, _ = sf.prepare_stellar_library_from_config(spectrum, config)
    assert regime == "uv"
    assert captured["family"] == "slug"
    assert captured["kwargs"]["slug_family"] == "MIST"
    assert "family" not in captured["kwargs"]


def test_unified_loader_translates_slug_family_keyword(monkeypatch):
    """Unified loader maps slug_family -> load_slug_library(family=...)."""
    import stellar.stellar_models as sm

    captured = {}

    def fake_slug(path, *, wave_range=None, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(sm, "load_slug_library", fake_slug)
    # The dispatch dictionary is constructed at call time, so the monkeypatch
    # is picked up by load_stellar_library.
    sm.load_stellar_library(
        "slug", "/tmp/slug.h5", wave_range=(1200.0, 1700.0),
        slug_family="MIST", subfamily="vvcrit000", specsyn="sb99hruv",
    )
    assert captured["family"] == "MIST"
    assert captured["subfamily"] == "vvcrit000"
    assert captured["specsyn"] == "sb99hruv"
    assert "slug_family" not in captured


def test_slug_auto_branch_discovery_selects_available_branch(tmp_path):
    import h5py
    import numpy as np
    from stellar.stellar_models import load_slug_library

    path = tmp_path / "slug.h5"
    with h5py.File(path, "w") as h5:
        grp = h5.create_group("models/deterministic/ActualFamily/ActualSub/ActualSyn/Z020")
        grp.create_dataset("wavelength_A", data=np.array([1200., 1300., 1400.]))
        grp.create_dataset("age_Myr", data=np.array([1., 5.]))
        grp.create_dataset("flux", data=np.ones((2, 3)) * 1e36)
        grp.attrs["cluster_mass_Msun"] = 1e6
        grp.attrs["metallicity_Zsun"] = 1.0

    lib = load_slug_library(path, family="auto", subfamily="auto", specsyn="auto")
    assert lib.library_metadata["family"] == "ActualFamily"
    assert lib.library_metadata["subfamily"] == "ActualSub"
    assert lib.library_metadata["specsyn"] == "ActualSyn"


def test_slug_missing_explicit_branch_lists_available_branches(tmp_path):
    import h5py
    import numpy as np
    import pytest
    from stellar.stellar_models import load_slug_library

    path = tmp_path / "slug.h5"
    with h5py.File(path, "w") as h5:
        grp = h5.create_group("models/deterministic/ActualFamily/ActualSub/ActualSyn/Z020")
        grp.create_dataset("wavelength_A", data=np.array([1200., 1300., 1400.]))
        grp.create_dataset("age_Myr", data=np.array([1.]))
        grp.create_dataset("flux", data=np.ones((1, 3)) * 1e36)

    with pytest.raises(ValueError, match="Available deterministic branches") as exc:
        load_slug_library(path, family="MIST", subfamily="vvcrit000", specsyn="sb99hruv")
    assert "/models/deterministic/ActualFamily/ActualSub/ActualSyn" in str(exc.value)


def _write_toy_sb99_h5(path, *, physical_scale_attr=None, include_provenance=True):
    import h5py
    import numpy as np
    with h5py.File(path, "w") as h5:
        grp = h5.create_group("models/GENEC/nonrot/PoWR/MW")
        grp.create_dataset("wavelength_A", data=np.array([1200., 1300., 1400.]))
        grp.create_dataset("age_Myr", data=np.array([1., 5.]))
        # Native pySB99 linear high-resolution population flux.  A physical
        # luminosity density of 1e40 erg/s/A is represented as 1e20 because
        # specsyn_hires carries the historical /1e20 scale.
        dflux = grp.create_dataset("flux", data=np.ones((2, 3)) * 1.0e20)
        grp.attrs["reference_mass_Msun"] = 1.0e6
        grp.attrs["metallicity_Z"] = 0.014
        if include_provenance:
            grp.attrs["flux_representation"] = (
                "native linear pySB99 integrated high-resolution stellar population flux"
            )
            grp.attrs["source_saved_transform"] = (
                "pySB99 saved log10(linear_flux + 1e-35) + 20; builder inverted this transform"
            )
        if physical_scale_attr is not None:
            dflux.attrs["physical_flux_scale_to_erg_s_A"] = float(physical_scale_attr)


def test_sb99_loader_infers_historical_1e20_physical_flux_scale(tmp_path):
    from astropy.constants import L_sun
    from stellar.stellar_models import load_sb99_library

    path = tmp_path / "sb99.h5"
    _write_toy_sb99_h5(path)
    lib = load_sb99_library(path, metallicity_labels=["MW"], ages_myr=[1.0])

    expected_physical = 1.0e40
    expected_per_mass_lsun = expected_physical / (1.0e6 * L_sun.cgs.value)
    assert np.allclose(lib.flux_reference, expected_physical)
    assert np.allclose(lib.flux_per_mass, expected_per_mass_lsun)
    assert lib.library_metadata["physical_flux_scale_to_erg_s_A"] == 1.0e20
    assert lib.library_metadata["physical_flux_unit"] == "erg/s/A"


def test_sb99_loader_prefers_explicit_hdf5_physical_flux_scale(tmp_path):
    from astropy.constants import L_sun
    from stellar.stellar_models import load_sb99_library

    path = tmp_path / "sb99_explicit.h5"
    _write_toy_sb99_h5(path, physical_scale_attr=2.5e19, include_provenance=False)
    lib = load_sb99_library(path, metallicity_labels=["MW"], ages_myr=[1.0])
    expected = 1.0e20 * 2.5e19 / (1.0e6 * L_sun.cgs.value)
    assert np.allclose(lib.flux_per_mass, expected)
    assert lib.library_metadata["physical_flux_scale_to_erg_s_A"] == 2.5e19


def test_sb99_loader_rejects_ambiguous_old_library_instead_of_guessing_units(tmp_path):
    import pytest
    from stellar.stellar_models import load_sb99_library

    path = tmp_path / "sb99_ambiguous.h5"
    _write_toy_sb99_h5(path, include_provenance=False)
    with pytest.raises(ValueError, match="Cannot determine the physical normalization"):
        load_sb99_library(path, metallicity_labels=["MW"], ages_myr=[1.0])


def test_prepare_sb99_library_does_not_pass_user_flux_unit(monkeypatch):
    from types import SimpleNamespace
    import stellar.stellar_fit as sf

    captured = {}
    def fake_loader(family, filename=None, *, wave_range=None, **kwargs):
        captured["family"] = family
        captured["kwargs"] = kwargs
        return SimpleNamespace()

    monkeypatch.setattr(sf, "load_stellar_library", fake_loader)
    spectrum = SimpleNamespace(wave=np.linspace(1200.0, 1700.0, 50), redshift=0.0)
    config = {
        "stellar_library": "sb99",
        "sb99_h5_path": "/tmp/sb99.h5",
        "sb99_track": "GENEC",
        "sb99_rotation": "nonrot",
        "sb99_spectra_library": "PoWR",
        # Deliberately include the obsolete key: the controller must ignore it.
        "sb99_flux_unit": "garbage",
    }
    regime, _ = sf.prepare_stellar_library_from_config(spectrum, config)
    assert regime == "uv"
    assert captured["family"] == "sb99"
    assert "native_flux_unit" not in captured["kwargs"]
