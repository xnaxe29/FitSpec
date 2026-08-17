import numpy as np
import pytest

from core.spectrum import Spectrum
from core.parameters import ModelParameters, Component, Parameter

from absorption.atomic import AtomicTransition, load_atomic_line_list, select_group, list_groups
from absorption.profiles import optical_depth_voigt, transmission_voigt, apply_partial_covering, voigt_hjerting
from absorption.absorption_model import build_absorption_parameters, make_absorption_model_func, COVERING_FRACTION_PARAMETER
from absorption.absorption_fit import fit_absorption_spectrum
from absorption.absorption_results import save_absorption_result, load_absorption_result


class DictConfig(dict):
    """Minimal stand-in for core.config.Config supporting .get()."""


def test_load_default_atomic_line_list_and_groups():
    transitions = load_atomic_line_list()
    assert len(transitions) == 1320
    groups = list_groups(transitions)
    assert len(groups) == 128
    assert "CIV" in groups
    assert "HI" in groups
    civ = select_group(transitions, "CIV")
    assert {t.name for t in civ} == {"CIV_1548", "CIV_1550"}
    assert all(t.group == "CIV" for t in civ)


def test_atomic_line_list_covers_isotopes_and_molecules():
    """The full VPFIT-derived catalog includes isotopes (D I, C13 series)
    and molecular lines (H2, CO, HD rotational bands), not just atomic
    resonance lines."""
    transitions = load_atomic_line_list()
    groups = set(list_groups(transitions))
    assert "DI" in groups
    assert any(g.startswith("H2J") for g in groups)
    assert any(g.startswith("COJ") for g in groups)
    assert any(g.startswith("HDJ") for g in groups)


def test_atomic_line_list_preserves_provenance_reference():
    transitions = load_atomic_line_list()
    by_name = {t.name: t for t in transitions}
    assert "Morton(03)" in by_name["HI_1215"].reference


def test_atomic_line_list_allows_zero_damping_constant():
    """Some transitions in the source compilation have no measured
    radiative damping constant (Gamma = 0); this must load, not raise."""
    transitions = load_atomic_line_list()
    zero_gamma = [t for t in transitions if t.damping_constant_s == 0.0]
    assert len(zero_gamma) > 0


def test_excited_fine_structure_state_is_a_separate_group_from_ground_state():
    """Si II* (excited fine-structure level) must not be silently merged
    into the Si II ground-state group -- they represent different
    populations with, in general, different column densities."""
    transitions = load_atomic_line_list()
    groups = set(list_groups(transitions))
    assert "SiII" in groups
    assert "SiII*" in groups
    ground = {t.name for t in select_group(transitions, "SiII")}
    excited = {t.name for t in select_group(transitions, "SiII*")}
    assert ground.isdisjoint(excited)


def test_select_group_unknown_raises():
    transitions = load_atomic_line_list()
    with pytest.raises(ValueError):
        select_group(transitions, "NotARealGroup")


def test_optical_depth_increases_with_column_density():
    wave = np.linspace(1540.0, 1560.0, 2000)
    tau_low = optical_depth_voigt(wave, 1548.204, 0.1899, 2.643e8, 10 ** 13.0, 20.0)
    tau_high = optical_depth_voigt(wave, 1548.204, 0.1899, 2.643e8, 10 ** 15.0, 20.0)
    assert np.max(tau_high) > np.max(tau_low)
    assert np.max(tau_low) >= 0
    T = transmission_voigt(wave, 1548.204, 0.1899, 2.643e8, 10 ** 13.0, 20.0)
    assert np.all(T <= 1.0) and np.all(T > 0)


def test_apply_partial_covering_bounds_and_identity():
    T_full = np.array([0.0, 0.5, 1.0])
    # full coverage (C_f=1) is the identity transform
    assert np.allclose(apply_partial_covering(T_full, 1.0), T_full)
    # C_f=0 means no absorption at all is observed (unabsorbed continuum)
    assert np.allclose(apply_partial_covering(T_full, 0.0), np.ones_like(T_full))
    with pytest.raises(ValueError):
        apply_partial_covering(T_full, 1.5)


def test_build_absorption_parameters_explicit_component_count():
    params = build_absorption_parameters(3)
    assert params.n_components == 3
    for component in params.components:
        assert "logN" in component
        assert "b_kms" in component
        assert "velocity_kms" in component
        assert COVERING_FRACTION_PARAMETER not in component  # off by default


def test_build_absorption_parameters_partial_coverage_only_leader_free():
    params = build_absorption_parameters(3, partial_coverage=True)
    assert not params.components[0][COVERING_FRACTION_PARAMETER].fixed
    for component in params.components[1:]:
        assert component[COVERING_FRACTION_PARAMETER].fixed


def _synthetic_doublet_spectrum(true_logN=14.0, true_b=20.0, true_velocity=0.0,
                                 covering_fraction=1.0, redshift=0.0):
    transitions = [
        AtomicTransition("CIV_1548", "CIV", 1548.204, 0.1899, 2.643e8, "CIV"),
        AtomicTransition("CIV_1550", "CIV", 1550.781, 0.09475, 2.628e8, "CIV"),
    ]
    wave = np.linspace(1540.0, 1560.0, 1200)
    model_func = make_absorption_model_func(
        transitions, redshift=redshift, resolution=None, partial_coverage=(covering_fraction != 1.0),
    )
    parameters = [
        Parameter("logN", true_logN, 8.0, 23.0),
        Parameter("b_kms", true_b, 1.0, 300.0),
        Parameter("velocity_kms", true_velocity, -500.0, 500.0),
    ]
    if covering_fraction != 1.0:
        parameters.append(Parameter(COVERING_FRACTION_PARAMETER, covering_fraction, 0.0, 1.0))
    true_params = ModelParameters(n_components=1, components=[Component(parameters=parameters)])
    flux = model_func(wave, true_params)

    rng = np.random.default_rng(0)
    flux_unc = np.full(wave.shape, 0.01)
    flux_noisy = flux + rng.normal(0, 0.01, size=wave.shape)
    spectrum = Spectrum.from_arrays(wave, flux_noisy, flux_unc, redshift=redshift)
    return spectrum, transitions


def test_fit_recovers_full_coverage_column_density_and_b():
    spectrum, transitions = _synthetic_doublet_spectrum(true_logN=14.0, true_b=20.0)
    config = DictConfig(absorption_n_components=1, absorption_partial_coverage=False,
                         absorption_logN_initial=13.5, absorption_b_initial_kms=15.0)
    result = fit_absorption_spectrum(spectrum, config, transitions=transitions)

    assert not result.partial_coverage
    assert result.covering_fraction is None
    measurement = result.measurements[0]
    assert measurement.logN == pytest.approx(14.0, abs=0.1)
    assert measurement.b_kms == pytest.approx(20.0, abs=8.0)


def test_fit_recovers_partial_coverage_fraction():
    spectrum, transitions = _synthetic_doublet_spectrum(
        true_logN=15.0, true_b=25.0, covering_fraction=0.6,
    )
    config = DictConfig(absorption_n_components=1, absorption_partial_coverage=True,
                         absorption_logN_initial=14.5, absorption_b_initial_kms=20.0,
                         absorption_covering_fraction_initial=0.9)
    result = fit_absorption_spectrum(spectrum, config, transitions=transitions)

    assert result.partial_coverage
    assert result.covering_fraction == pytest.approx(0.6, abs=0.1)
    assert result.measurements[0].logN == pytest.approx(15.0, abs=0.3)


def test_default_full_coverage_does_not_add_free_parameter():
    """Fitting the same data with partial_coverage off must use exactly
    3 free parameters per component (logN, b_kms, velocity_kms), never 4."""
    spectrum, transitions = _synthetic_doublet_spectrum(true_logN=14.0, true_b=20.0)
    config = DictConfig(absorption_n_components=1, absorption_partial_coverage=False)
    result = fit_absorption_spectrum(spectrum, config, transitions=transitions)
    assert result.fit_result.parameters.to_vector().size == 3


def test_save_and_load_round_trip_with_partial_coverage(tmp_path):
    spectrum, transitions = _synthetic_doublet_spectrum(
        true_logN=14.5, true_b=22.0, covering_fraction=0.7,
    )
    config = DictConfig(absorption_n_components=1, absorption_partial_coverage=True)
    result = fit_absorption_spectrum(spectrum, config, transitions=transitions)

    path = tmp_path / "absorption_result.fits"
    save_absorption_result(path, result)
    loaded = load_absorption_result(path)

    assert loaded.partial_coverage
    assert loaded.covering_fraction == pytest.approx(result.covering_fraction, rel=1e-6)
    assert loaded.measurements[0].logN == pytest.approx(result.measurements[0].logN, rel=1e-6)
    assert np.allclose(loaded.fit_result.wave, result.fit_result.wave)
    assert np.allclose(loaded.fit_result.model, result.fit_result.model, rtol=1e-5)
    assert {t.name for t in loaded.transitions} == {t.name for t in result.transitions}


def test_voigt_hjerting_matches_reference_tepper_garcia():
    def H_reference(a, x):
        P = x ** 2
        H0 = np.exp(-x ** 2)
        Q = 1.5 / x ** 2
        return H0 - a / np.sqrt(np.pi) / P * (H0 * H0 * (4. * P * P + 7. * P + 4. + Q) - Q - 1)

    a = 0.01
    x = np.linspace(0.1, 5.0, 50)
    assert np.allclose(voigt_hjerting(a, x), H_reference(a, x), rtol=1e-10)


def test_missing_group_raises_clear_error():
    spectrum, transitions = _synthetic_doublet_spectrum()
    config = DictConfig(absorption_n_components=1)  # no absorption_group set
    with pytest.raises(ValueError):
        fit_absorption_spectrum(spectrum, config)
