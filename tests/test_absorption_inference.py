"""Integration tests for absorption -> shared posterior inference."""
from __future__ import annotations

import numpy as np
import pytest

from core.parameters import Component, ModelParameters, Parameter, ParameterTie
from core.spectrum import Spectrum
from absorption.atomic import AtomicTransition
from absorption.absorption_model import (
    COVERING_FRACTION_PARAMETER,
    AbsorptionSystem,
    build_absorption_parameters,
    make_absorption_model_func,
)
from absorption.absorption_fit import fit_absorption_spectrum, fit_joint_absorption_spectrum
from absorption.absorption_results import (
    AbsorptionComponentMeasurement, save_absorption_result, load_absorption_result,
)
from absorption.inference import (
    build_absorption_prior_set,
    build_absorption_posterior_problem,
    build_joint_absorption_posterior_problem,
    run_absorption_inference,
)


class DictConfig(dict):
    pass


CIV = [
    AtomicTransition("CIV_1548", "CIV", 1548.204, 0.1899, 2.643e8, "CIV"),
    AtomicTransition("CIV_1550", "CIV", 1550.781, 0.09475, 2.628e8, "CIV"),
]
SiIV = [
    AtomicTransition("SiIV_1393", "SiIV", 1393.755, 0.513, 8.80e8, "SiIV"),
    AtomicTransition("SiIV_1402", "SiIV", 1402.770, 0.254, 8.63e8, "SiIV"),
]


def _single_spectrum(partial=False):
    wave = np.linspace(1542.0, 1557.0, 700)
    params = build_absorption_parameters(
        1, logN_initial=14.1, b_initial_kms=22.0, velocity_initial_kms=18.0,
        partial_coverage=partial, covering_fraction_initial=0.72,
    )
    model = make_absorption_model_func(CIV, partial_coverage=partial)(wave, params)
    unc = np.full_like(wave, 0.01)
    return Spectrum.from_arrays(wave, model, unc)


def test_absorption_prior_uses_finite_physical_bounds_and_excludes_fixed():
    spectrum = _single_spectrum(partial=True)
    config = DictConfig(
        absorption_n_components=1,
        absorption_partial_coverage=True,
        absorption_logN_initial=14.0,
        absorption_b_initial_kms=20.0,
        absorption_velocity_initial_kms=10.0,
        absorption_covering_fraction_initial=0.8,
    )
    result = fit_absorption_spectrum(spectrum, config, transitions=CIV)
    priors = build_absorption_prior_set(result.fit_result, config)
    assert priors.parameter_names == [
        "c0_logN", "c0_b_kms", "c0_velocity_kms", "c0_covering_fraction"
    ]
    assert len(priors) == 4


def test_single_absorption_problem_is_finite_and_preserves_partial_coverage():
    spectrum = _single_spectrum(partial=True)
    config = DictConfig(
        absorption_n_components=1,
        absorption_partial_coverage=True,
        absorption_covering_fraction_initial=0.75,
        absorption_inference_method="emcee",
    )
    result = fit_absorption_spectrum(spectrum, config, transitions=CIV)
    problem = build_absorption_posterior_problem(result, spectrum, config)
    assert "c0_covering_fraction" in problem.parameter_names
    assert np.isfinite(problem.log_probability(problem.initial_position))
    assert problem.metadata["science_module"] == "absorption"
    assert problem.metadata["partial_coverage"] is True


def test_joint_problem_reapplies_velocity_tie_and_excludes_follower_from_sampling():
    # Two disjoint wavelength regions in one spectrum, one system per ion.
    wave = np.concatenate([
        np.linspace(1388.0, 1407.0, 500),
        np.linspace(1542.0, 1557.0, 500),
    ])
    flux = np.ones_like(wave)
    unc = np.full_like(wave, 0.02)
    spectrum = Spectrum.from_arrays(wave, flux, unc)
    systems = [
        AbsorptionSystem(CIV, label="CIV", velocity_bounds_kms=(-150.0, 150.0)),
        AbsorptionSystem(SiIV, label="SiIV", velocity_bounds_kms=(-150.0, 150.0)),
    ]
    ties = [ParameterTie(leader=(0, "velocity_kms"), follower=(1, "velocity_kms"))]
    result = fit_joint_absorption_spectrum(spectrum, systems, ties=ties)
    problem = build_joint_absorption_posterior_problem(
        result, spectrum, DictConfig(), systems=systems, ties=ties,
    )
    assert "c0_velocity_kms" in problem.parameter_names
    assert "c1_velocity_kms" not in problem.parameter_names
    assert np.isfinite(problem.log_probability(problem.initial_position))
    assert problem.metadata["has_parameter_ties"] is True


def test_frozen_rejection_component_remains_out_of_posterior_vector():
    spectrum = _single_spectrum(partial=False)
    config = DictConfig(absorption_n_components=1)
    result = fit_absorption_spectrum(spectrum, config, transitions=CIV)
    # Mimic a two-component finalized rejection result: component 0 remains
    # detected/free while component 1 is frozen at a negligible column density.
    result.fit_result.parameters.add_component(Component(parameters=[
        Parameter("logN", 8.0, 8.0, 23.0, fixed=True),
        Parameter("b_kms", 20.0, 1.0, 300.0, fixed=True),
        Parameter("velocity_kms", 100.0, -500.0, 500.0, fixed=True),
    ]), make_active=False)
    result.measurements.append(AbsorptionComponentMeasurement(
        logN=8.0, logN_uncertainty=np.nan,
        b_kms=20.0, b_kms_uncertainty=np.nan,
        velocity_kms=100.0, velocity_kms_uncertainty=np.nan,
        is_upper_limit=True,
    ))
    problem = build_absorption_posterior_problem(result, spectrum, config)
    assert problem.parameter_names == ["c0_logN", "c0_b_kms", "c0_velocity_kms"]
    assert problem.metadata["n_frozen_upper_limit_components"] == 1
    assert np.isfinite(problem.log_probability(problem.initial_position))


def test_saved_absorption_result_preserves_bounds_and_fixed_state_for_inference(tmp_path):
    spectrum = _single_spectrum(partial=True)
    config = DictConfig(absorption_n_components=1, absorption_partial_coverage=True)
    result = fit_absorption_spectrum(spectrum, config, transitions=CIV)
    result.fit_result.parameters.components[0]["velocity_kms"].fixed = True

    path = tmp_path / "absorption_inference_ready.fits"
    save_absorption_result(path, result)
    loaded = load_absorption_result(path)
    component = loaded.fit_result.parameters.components[0]

    assert component["logN"].lower == pytest.approx(8.0)
    assert component["logN"].upper == pytest.approx(23.0)
    assert component["b_kms"].lower == pytest.approx(1.0)
    assert component["velocity_kms"].fixed is True
    assert component[COVERING_FRACTION_PARAMETER].lower == pytest.approx(0.0)
    assert component[COVERING_FRACTION_PARAMETER].upper == pytest.approx(1.0)
    assert loaded.fit_result.metadata["absorption_fit_mode"] == "single"
    assert loaded.fit_result.metadata["absorption_component_transition_names"][0] == [
        "CIV_1548", "CIV_1550"
    ]


def test_absorption_inference_disabled_by_default_returns_none():
    spectrum = _single_spectrum(partial=False)
    config = DictConfig(absorption_n_components=1)
    result = fit_absorption_spectrum(spectrum, config, transitions=CIV)
    assert run_absorption_inference(result, spectrum, config, transitions=CIV) is None
