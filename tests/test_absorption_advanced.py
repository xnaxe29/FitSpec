"""Tests for the VPFIT-inspired absorption features: general parameter
tying/fixing, cross-ion joint fitting, thermal/turbulent Doppler
linking, fixed-ratio and common-pattern column-density ties, subpixel
oversampling, region velocity shifts, in-fit continuum adjustments,
automatic component rejection, and column-density upper limits.

Joint multi-*file* fitting is explicitly out of scope (see
absorption/absorption_model.py's module docstring) and is not tested
here.
"""
import numpy as np
import pytest

from core.spectrum import Spectrum
from core.parameters import Parameter, ParameterTie, apply_ties

from absorption.atomic import AtomicTransition
from absorption.absorption_model import (
    AbsorptionSystem, build_joint_absorption_parameters, make_joint_absorption_model_func,
    thermal_b_kms, mass_scaled_b_transform, ThermalTurbulentLink, apply_thermal_turbulent_links,
    fixed_log_ratio_transform, AbundancePatternGroup, apply_abundance_pattern_ties,
    RegionShift, ContinuumAdjustment, apply_region_shifts, apply_continuum_adjustments,
)
from absorption.absorption_fit import fit_joint_absorption_spectrum
from absorption.rejection import fit_joint_absorption_spectrum_with_rejection
from absorption.upper_limits import estimate_column_density_upper_limit
from absorption.synthetic import generate_synthetic_absorption_spectrum

CIV = [
    AtomicTransition("CIV_1548", "CIV", 1548.204, 0.1899, 2.643e8, "CIV", atomic_mass_amu=12.011),
    AtomicTransition("CIV_1550", "CIV", 1550.781, 0.09475, 2.628e8, "CIV", atomic_mass_amu=12.011),
]
SiIV = [
    AtomicTransition("SiIV_1393", "SiIV", 1393.755, 0.513, 8.80e8, "SiIV", atomic_mass_amu=28.0855),
    AtomicTransition("SiIV_1402", "SiIV", 1402.770, 0.255, 8.62e8, "SiIV", atomic_mass_amu=28.0855),
]
OVI = [
    AtomicTransition("OVI_1031", "OVI", 1031.9261, 0.1330, 4.149e8, "OVI", atomic_mass_amu=15.9994),
    AtomicTransition("OVI_1037", "OVI", 1037.6167, 0.0658, 4.076e8, "OVI", atomic_mass_amu=15.9994),
]


# --- core.parameters.ParameterTie -------------------------------------------------

def test_parameter_tie_identity_transform():
    civ_system = AbsorptionSystem(CIV, n_components=1, label="CIV")
    siiv_system = AbsorptionSystem(SiIV, n_components=1, label="SiIV")
    params, _ = build_joint_absorption_parameters([civ_system, siiv_system])
    params.components[1]["velocity_kms"].fixed = True
    ties = [ParameterTie(leader=(0, "velocity_kms"), follower=(1, "velocity_kms"))]

    params.components[0]["velocity_kms"].value = 77.0
    apply_ties(params, ties)
    assert params.components[1]["velocity_kms"].value == pytest.approx(77.0)


def test_parameter_tie_custom_transform():
    params, _ = build_joint_absorption_parameters([
        AbsorptionSystem(CIV, n_components=1, label="CIV"),
        AbsorptionSystem(SiIV, n_components=1, label="SiIV"),
    ])
    params.components[1]["logN"].fixed = True
    ties = [ParameterTie(leader=(0, "logN"), follower=(1, "logN"), transform=lambda v: v - 0.5)]
    params.components[0]["logN"].value = 14.2
    apply_ties(params, ties)
    assert params.components[1]["logN"].value == pytest.approx(13.7)


def test_apply_ties_noop_on_empty():
    params, _ = build_joint_absorption_parameters([AbsorptionSystem(CIV, n_components=1)])
    apply_ties(params, None)
    apply_ties(params, [])
    # no error means success


# --- Cross-ion joint fitting with tied velocity -----------------------------------

def test_joint_fit_recovers_shared_tied_velocity_across_ions():
    wave_civ = np.linspace(1540.0, 1560.0, 900)
    wave_siiv = np.linspace(1385.0, 1410.0, 900)
    spec_civ = generate_synthetic_absorption_spectrum(
        wave_civ, CIV, [{"logN": 14.0, "b_kms": 20.0, "velocity_kms": 45.0}], signal_to_noise=150, seed=1)
    spec_siiv = generate_synthetic_absorption_spectrum(
        wave_siiv, SiIV, [{"logN": 13.4, "b_kms": 20.0, "velocity_kms": 45.0}], signal_to_noise=150, seed=2)
    wave = np.concatenate([wave_siiv, wave_civ])
    flux = np.concatenate([spec_siiv.flux, spec_civ.flux])
    flux_unc = np.concatenate([spec_siiv.flux_unc, spec_civ.flux_unc])
    spectrum = Spectrum.from_arrays(wave, flux, flux_unc)

    systems = [AbsorptionSystem(CIV, n_components=1, label="CIV"), AbsorptionSystem(SiIV, n_components=1, label="SiIV")]
    ties = [ParameterTie(leader=(0, "velocity_kms"), follower=(1, "velocity_kms"))]
    result = fit_joint_absorption_spectrum(spectrum, systems, ties=ties)

    assert len(result.measurements) == 2
    civ_m, siiv_m = result.measurements
    assert civ_m.system_label == "CIV"
    assert siiv_m.system_label == "SiIV"
    assert civ_m.logN == pytest.approx(14.0, abs=0.1)
    assert siiv_m.logN == pytest.approx(13.4, abs=0.1)
    # the tie forces these to be numerically equal, not just close
    assert siiv_m.velocity_kms == pytest.approx(civ_m.velocity_kms, abs=1e-6)
    assert civ_m.velocity_kms == pytest.approx(45.0, abs=5.0)


# --- Thermal / turbulent Doppler-parameter linking ---------------------------------

def test_thermal_b_kms_matches_vpfit_constant():
    # b_thermal = 12.85 * sqrt(T/1e4/m); at T=1e4K, m=1amu this is exactly 12.85
    assert thermal_b_kms(1.0e4, 1.0) == pytest.approx(12.85)
    assert thermal_b_kms(0.0, 5.0) == pytest.approx(0.0)


def test_thermal_b_kms_rejects_invalid_input():
    with pytest.raises(ValueError):
        thermal_b_kms(-1.0, 1.0)
    with pytest.raises(ValueError):
        thermal_b_kms(1e4, 0.0)


def test_mass_scaled_b_transform_lighter_ion_gets_larger_b():
    transform = mass_scaled_b_transform(leader_mass_amu=12.011, follower_mass_amu=1.00794)
    assert transform(20.0) > 20.0  # H is lighter than C -> larger thermal b at the same T


def test_thermal_turbulent_link_combines_in_quadrature():
    params, _ = build_joint_absorption_parameters([
        AbsorptionSystem(CIV, n_components=1, label="CIV"),
        AbsorptionSystem(OVI, n_components=1, label="OVI"),
    ])
    params.components[1]["b_kms"].fixed = True
    turbulent = 10.0
    temperature = 2.0e4
    links = [ThermalTurbulentLink(follower=(1, "b_kms"), turbulent_kms=turbulent,
                                   temperature_K=temperature, follower_mass_amu=15.9994)]
    apply_thermal_turbulent_links(params, links)
    expected = np.hypot(turbulent, thermal_b_kms(temperature, 15.9994))
    assert params.components[1]["b_kms"].value == pytest.approx(expected)


def test_thermal_turbulent_link_pure_turbulent_is_mass_independent():
    params, _ = build_joint_absorption_parameters([
        AbsorptionSystem(CIV, n_components=1), AbsorptionSystem(OVI, n_components=1),
    ])
    params.components[1]["b_kms"].fixed = True
    links = [ThermalTurbulentLink(follower=(1, "b_kms"), turbulent_kms=12.0, temperature_K=0.0, follower_mass_amu=15.9994)]
    apply_thermal_turbulent_links(params, links)
    assert params.components[1]["b_kms"].value == pytest.approx(12.0)


# --- Fixed-ratio and common-pattern column-density ties ---------------------------

def test_fixed_log_ratio_transform():
    transform = fixed_log_ratio_transform(-0.3)
    assert transform(14.0) == pytest.approx(13.7)


def test_abundance_pattern_group_recovers_common_pattern_and_free_ratio():
    wave_civ = np.linspace(1540.0, 1560.0, 1000)
    wave_siiv = np.linspace(1385.0, 1405.0, 1000)
    true_civ = [{"logN": 14.0, "b_kms": 15.0, "velocity_kms": -20.0}, {"logN": 13.7, "b_kms": 15.0, "velocity_kms": 20.0}]
    true_siiv = [{"logN": 13.5, "b_kms": 15.0, "velocity_kms": -20.0}, {"logN": 13.2, "b_kms": 15.0, "velocity_kms": 20.0}]
    spec_civ = generate_synthetic_absorption_spectrum(wave_civ, CIV, true_civ, signal_to_noise=200, seed=3)
    spec_siiv = generate_synthetic_absorption_spectrum(wave_siiv, SiIV, true_siiv, signal_to_noise=200, seed=4)
    wave = np.concatenate([wave_siiv, wave_civ])
    flux = np.concatenate([spec_siiv.flux, spec_civ.flux])
    flux_unc = np.concatenate([spec_siiv.flux_unc, spec_civ.flux_unc])
    spectrum = Spectrum.from_arrays(wave, flux, flux_unc)

    systems = [
        AbsorptionSystem(CIV, n_components=2, label="CIV", velocity_bounds_kms=(-100, 100)),
        AbsorptionSystem(SiIV, n_components=2, label="SiIV", velocity_bounds_kms=(-100, 100)),
    ]
    from absorption.absorption_model import build_joint_absorption_parameters as _build
    params, component_transitions = _build(systems)
    # start close to truth (multi-component Voigt fitting is prone to local minima
    # far from a poor initial guess -- this is a property of the physics, not the
    # tying machinery under test here)
    params.components[0]["velocity_kms"].value = -18.0
    params.components[1]["velocity_kms"].value = 18.0
    params.components[2]["velocity_kms"].value = -18.0
    params.components[3]["velocity_kms"].value = 18.0

    ties = [ParameterTie(leader=(0, "velocity_kms"), follower=(2, "velocity_kms")),
            ParameterTie(leader=(1, "velocity_kms"), follower=(3, "velocity_kms"))]
    groups = [AbundancePatternGroup(leader_components=[0, 1], follower_components=[2, 3], ratio_holder=(2, "logN_ratio"))]
    params.components[2]["logN"].fixed = True
    params.components[3]["logN"].fixed = True
    params.components[2].parameters.append(Parameter("logN_ratio", 0.0, -5.0, 5.0))

    model_func = make_joint_absorption_model_func(component_transitions, ties=ties)
    inner = model_func

    def wrapped(wave, model_parameters):
        apply_abundance_pattern_ties(model_parameters, groups)
        return inner(wave, model_parameters)

    from core.fitting import fit_deterministic
    fit_result = fit_deterministic(spectrum.wave, spectrum.flux, spectrum.flux_unc, params, wrapped)

    c = fit_result.parameters.components
    assert c[0]["logN"].value == pytest.approx(14.0, abs=0.1)
    assert c[1]["logN"].value == pytest.approx(13.7, abs=0.1)
    assert c[2]["logN"].value == pytest.approx(13.5, abs=0.15)
    assert c[3]["logN"].value == pytest.approx(13.2, abs=0.15)
    assert c[2]["logN_ratio"].value == pytest.approx(-0.5, abs=0.1)
    # the tie forces these ratios to be numerically identical, by construction
    assert c[2]["logN"].value - c[0]["logN"].value == pytest.approx(c[2]["logN_ratio"].value, abs=1e-6)
    assert c[3]["logN"].value - c[1]["logN"].value == pytest.approx(c[2]["logN_ratio"].value, abs=1e-6)


# --- Subpixel oversampling ---------------------------------------------------------

def test_subpixel_grid_and_resample_round_trip_on_smooth_function():
    from absorption.absorption_model import _subpixel_grid, _resample_to_data
    wave = np.linspace(1000.0, 1010.0, 20)
    fine = _subpixel_grid(wave, 5)
    assert fine.size > wave.size
    assert fine.min() == pytest.approx(wave.min())
    assert fine.max() == pytest.approx(wave.max())
    smooth_fine = np.sin(fine / 3.0)
    resampled = _resample_to_data(fine, smooth_fine, wave, 5)
    assert resampled.shape == wave.shape
    assert np.allclose(resampled, np.sin(wave / 3.0), atol=1e-3)


def test_subpixel_disabled_is_identity():
    from absorption.absorption_model import _subpixel_grid, _resample_to_data
    wave = np.linspace(1000.0, 1010.0, 20)
    assert np.array_equal(_subpixel_grid(wave, 1), wave)


def test_joint_model_func_runs_with_subpixel_oversampling():
    systems = [AbsorptionSystem(CIV, n_components=1, label="CIV")]
    params, component_transitions = build_joint_absorption_parameters(systems)
    wave = np.linspace(1540.0, 1560.0, 300)
    model_func = make_joint_absorption_model_func(component_transitions, subpixel=5)
    flux = model_func(wave, params)
    assert flux.shape == wave.shape
    assert np.all(flux <= 1.0 + 1e-9) and np.all(flux > 0)


# --- Region velocity shift and in-fit continuum adjustment -------------------------

def test_region_shift_only_affects_its_window():
    wave = np.linspace(1540.0, 1560.0, 200)
    in_region = (wave >= 1545) & (wave <= 1550)
    shift = RegionShift(wave_min=1545, wave_max=1550, parameter=Parameter("shift", 100.0, -1000, 1000))
    shifted = apply_region_shifts(wave, [shift])
    assert np.allclose(shifted[~in_region], wave[~in_region])
    assert np.all(shifted[in_region] > wave[in_region])  # positive velocity -> redward shift


def test_region_shift_noop_on_empty():
    wave = np.linspace(1540.0, 1560.0, 50)
    assert np.array_equal(apply_region_shifts(wave, None), wave)


def test_continuum_adjustment_applies_level_and_slope_only_in_window():
    wave = np.linspace(1540.0, 1560.0, 200)
    transmission = np.ones_like(wave)
    in_region = (wave >= 1545) & (wave <= 1555)
    adjustment = ContinuumAdjustment(
        wave_min=1545, wave_max=1555, reference_wavelength=1550,
        level=Parameter("level", 0.9, 0.0, 2.0), slope=Parameter("slope", 0.0, -1.0, 1.0),
    )
    adjusted = apply_continuum_adjustments(wave, transmission, [adjustment])
    assert np.allclose(adjusted[in_region], 0.9)
    assert np.allclose(adjusted[~in_region], 1.0)


# --- Automatic component rejection (freeze, don't drop) ---------------------------

def test_rejection_preserves_requested_component_count_and_total_column_density():
    """Insignificant components must be frozen (near-negligible, marked as an
    upper limit) rather than removed, so the caller always gets back
    exactly the number of components they asked for, and the summed
    column density across components tracks the true total."""
    wave = np.linspace(1540.0, 1560.0, 1500)
    spectrum = generate_synthetic_absorption_spectrum(
        wave, CIV, [{"logN": 14.0, "b_kms": 20.0, "velocity_kms": 0.0}], signal_to_noise=150, seed=7)
    systems = [AbsorptionSystem(CIV, n_components=3, label="CIV", logN_bounds=(11.0, 18.0),
                                 logN_initial=12.5, b_initial_kms=20.0, b_bounds_kms=(3.0, 300.0),
                                 velocity_bounds_kms=(-100, 100))]
    result = fit_joint_absorption_spectrum_with_rejection(spectrum, systems, reject_margin_dex=0.2)

    assert result.fit_result.parameters.n_components == 3
    assert len(result.measurements) == 3
    assert sum(m.is_upper_limit for m in result.measurements) >= 1  # at least one flagged insignificant
    assert any(not m.is_upper_limit for m in result.measurements)  # the real one is not flagged

    total_linear = sum(10 ** m.logN for m in result.measurements)
    assert np.log10(total_linear) == pytest.approx(14.0, abs=0.05)


def test_rejection_keeps_genuinely_significant_components_unflagged():
    wave = np.linspace(1540.0, 1560.0, 1500)
    spectrum = generate_synthetic_absorption_spectrum(
        wave, CIV, [{"logN": 14.0, "b_kms": 15.0, "velocity_kms": -30.0},
                    {"logN": 13.8, "b_kms": 15.0, "velocity_kms": 30.0}], signal_to_noise=200, seed=9)
    systems = [AbsorptionSystem(CIV, n_components=2, label="CIV", logN_bounds=(11.0, 18.0),
                                 logN_initial=13.5, b_initial_kms=15.0,
                                 velocity_bounds_kms=(-100, 100), velocity_initial_kms=-25.0)]
    result = fit_joint_absorption_spectrum_with_rejection(spectrum, systems, reject_margin_dex=0.2)
    assert result.fit_result.parameters.n_components == 2
    assert all(not m.is_upper_limit for m in result.measurements)


def test_rejection_never_freezes_every_component_in_a_system():
    wave = np.linspace(1540.0, 1560.0, 800)
    # essentially pure noise, no real line: even so, at least 1 component must stay free
    rng = np.random.default_rng(0)
    flux = np.ones_like(wave) + rng.normal(0, 0.01, wave.shape)
    flux_unc = np.full_like(wave, 0.01)
    spectrum = Spectrum.from_arrays(wave, flux, flux_unc)
    systems = [AbsorptionSystem(CIV, n_components=2, label="CIV", logN_bounds=(11.0, 18.0), logN_initial=11.5)]
    result = fit_joint_absorption_spectrum_with_rejection(spectrum, systems, max_rejection_passes=2)
    assert result.fit_result.parameters.n_components == 2  # never removed
    assert sum(not m.is_upper_limit for m in result.measurements) >= 1  # at least one stays free


# --- Column-density upper limits ----------------------------------------------------

def test_upper_limit_non_detection_gives_finite_modest_limit():
    wave = np.linspace(1025.0, 1045.0, 3000)
    rng = np.random.default_rng(0)
    flux = np.ones_like(wave) + rng.normal(0, 0.02, wave.shape)
    flux_unc = np.full_like(wave, 0.02)
    result = estimate_column_density_upper_limit(
        wave, flux, flux_unc, OVI,
        reference_b_kms=15.0, reference_b_uncertainty_kms=2.0,
        reference_mass_amu=12.011, test_mass_amu=15.9994, redshift=0.0,
    )
    assert np.isfinite(result.logN_limit)
    assert 10.0 < result.logN_limit < 16.0
    assert result.b_grid_kms.size >= 2
    assert result.logN_limit_per_b.size == result.b_grid_kms.size


def test_upper_limit_search_is_conservative_across_b_grid():
    """The adopted limit must be the MAXIMUM over the b grid, per Schaye et al. 2007."""
    wave = np.linspace(1025.0, 1045.0, 3000)
    rng = np.random.default_rng(1)
    flux = np.ones_like(wave) + rng.normal(0, 0.03, wave.shape)
    flux_unc = np.full_like(wave, 0.03)
    result = estimate_column_density_upper_limit(
        wave, flux, flux_unc, OVI,
        reference_b_kms=20.0, reference_b_uncertainty_kms=3.0,
        reference_mass_amu=1.00794, test_mass_amu=15.9994, redshift=0.0,
        b_grid_size=5,
    )
    assert result.logN_limit == pytest.approx(np.max(result.logN_limit_per_b))


def test_upper_limit_detects_presence_of_a_real_strong_line():
    wave = np.linspace(1025.0, 1045.0, 3000)
    spectrum = generate_synthetic_absorption_spectrum(
        wave, OVI, [{"logN": 14.5, "b_kms": 15.0, "velocity_kms": 0.0}], noise_sigma=0.02, seed=3)
    result = estimate_column_density_upper_limit(
        wave, spectrum.flux, spectrum.flux_unc, OVI,
        reference_b_kms=15.0, reference_b_uncertainty_kms=2.0,
        reference_mass_amu=12.011, test_mass_amu=15.9994, redshift=0.0,
    )
    # a real logN=14.5 line present must push the "upper limit" well above the
    # non-detection case's modest value -- the search correctly reflects that
    # weak/no absorption is no longer consistent with the data.
    assert result.logN_limit > 14.0


def test_upper_limit_rejects_mismatched_array_shapes():
    with pytest.raises(ValueError):
        estimate_column_density_upper_limit(
            np.array([1.0, 2.0]), np.array([1.0]), np.array([0.1, 0.1]), OVI,
            reference_b_kms=15.0, reference_mass_amu=12.0, test_mass_amu=16.0, redshift=0.0,
        )


# --- Synthetic spectrum generator ---------------------------------------------------

def test_generate_synthetic_absorption_spectrum_shapes_and_noise_level():
    wave = np.linspace(1540.0, 1560.0, 500)
    spectrum = generate_synthetic_absorption_spectrum(
        wave, CIV, [{"logN": 14.0, "b_kms": 20.0, "velocity_kms": 0.0}], signal_to_noise=50, seed=0,
    )
    assert spectrum.wave.shape == wave.shape
    assert spectrum.flux.shape == wave.shape
    assert np.allclose(spectrum.flux_unc, 1.0 / 50.0)
    assert np.min(spectrum.flux) < 0.9  # real absorption feature present


def test_generate_synthetic_absorption_spectrum_partial_coverage():
    wave = np.linspace(1540.0, 1560.0, 500)
    spectrum = generate_synthetic_absorption_spectrum(
        wave, CIV, [{"logN": 16.0, "b_kms": 10.0, "velocity_kms": 0.0}],
        covering_fraction=0.5, noise_sigma=1e-6, seed=0,
    )
    # even a saturated, high-N line can never go below 1 - C_f in transmission
    assert np.min(spectrum.flux) > 0.5 - 1e-3


def test_generate_synthetic_absorption_spectrum_requires_exactly_one_noise_spec():
    wave = np.linspace(1540.0, 1560.0, 100)
    with pytest.raises(ValueError):
        generate_synthetic_absorption_spectrum(wave, CIV, [{"logN": 14.0, "b_kms": 20.0}])
    with pytest.raises(ValueError):
        generate_synthetic_absorption_spectrum(wave, CIV, [{"logN": 14.0, "b_kms": 20.0}],
                                                 signal_to_noise=50, noise_sigma=0.01)
