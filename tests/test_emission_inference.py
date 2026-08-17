"""Emission-module integration tests for the shared inference layer."""
import numpy as np
import pytest

from core.parameters import Component, ModelParameters, Parameter
from core.results import FitResult, PosteriorResult
from core.spectrum import Spectrum
from core.statistics import FitStatistics

from emission.lines import EmissionLine
from emission.emission_results import EmissionFitResult
from emission.inference import (
    build_emission_prior_set,
    build_emission_posterior_problem,
    run_emission_inference,
)


class DictConfig(dict):
    pass


def _result(*, amplitude_upper=np.inf, reduction=0.0):
    line = EmissionLine("Halpha6562", 6562.819, ion="Hα")
    wave = np.linspace(6555.0, 6570.0, 80)
    unc = np.full_like(wave, 0.1)
    params = ModelParameters(
        n_components=1,
        components=[Component(parameters=[
            Parameter("velocity_kms", 0.0, -250.0, 250.0),
            Parameter("sigma_kms", 60.0, 10.0, 1500.0),
            Parameter("amp_Halpha6562", 5.0, 0.0, amplitude_upper),
        ])],
    )
    # The exact model is not important for prior tests.  For posterior-problem
    # tests construct the data from the real emission model below.
    from emission.emission_model import make_emission_model_func
    physical = make_emission_model_func([line], redshift=0.0, resolution=None)
    model = physical(wave, params) + reduction
    fit = FitResult(
        parameters=params,
        parameter_uncertainties=None,
        wave=wave,
        flux=model.copy(),
        flux_unc=unc,
        mask=np.ones(wave.size, bool),
        model=model.copy(),
        statistics=FitStatistics(
            n_data=80, n_eff=80.0, k_params=3, dof=77, chi_square=0.0,
            reduced_chi_square=0.0, jitter_scale=1.0, neg2_log_likelihood=0.0,
            bic=0.0, aic=0.0, aicc=0.0,
        ),
        redshift=0.0,
        metadata={
            "emission_kinematics_mode": "free",
            "emission_flux_reduction": reduction,
            "emission_flux_normalizing_factor": 1.0,
        },
    )
    emission = EmissionFitResult(
        fit_result=fit,
        line_list=[line],
        measurements=[],
        component_velocities_kms=np.asarray([0.0]),
        component_sigmas_kms=np.asarray([60.0]),
    )
    spectrum = Spectrum.from_arrays(wave, model, unc)
    return emission, spectrum


def test_emission_prior_requires_finite_amplitude_prior():
    result, _ = _result(amplitude_upper=np.inf)
    with pytest.raises(ValueError, match="amplitude prior"):
        build_emission_prior_set(result.fit_result, DictConfig())


def test_emission_prior_config_supplies_finite_amplitude_limit():
    result, _ = _result(amplitude_upper=np.inf)
    priors = build_emission_prior_set(
        result.fit_result,
        DictConfig(emission_inference_amplitude_prior_max=20.0),
    )
    assert len(priors) == 3
    assert np.isfinite(priors.log_probability([0.0, 60.0, 5.0]))
    assert not np.isfinite(priors.log_probability([0.0, 60.0, 25.0]))


def test_emission_posterior_problem_preserves_flux_reduction():
    result, spectrum = _result(amplitude_upper=20.0, reduction=3.5)
    problem = build_emission_posterior_problem(result, spectrum, DictConfig())
    # Data were generated exactly at the deterministic parameter vector, so
    # the normalized log likelihood must be finite at that reference point.
    assert np.isfinite(problem.log_probability(problem.initial_position))
    assert problem.metadata["science_module"] == "emission"
    assert problem.metadata["n_components"] == 1


def test_disabled_emission_inference_does_not_require_sampler_dependency():
    result, spectrum = _result(amplitude_upper=20.0)
    posterior = run_emission_inference(
        result, spectrum, DictConfig(emission_inference_method="deterministic")
    )
    assert posterior is None


def test_emission_inference_dispatches_emcee(monkeypatch):
    result, spectrum = _result(amplitude_upper=20.0)
    captured = {}

    def fake_run(problem, **kwargs):
        captured.update(kwargs)
        theta = problem.initial_position[None, :]
        return PosteriorResult(
            samples=theta,
            log_probability=np.asarray([problem.log_probability(theta[0])]),
            parameter_names=problem.parameter_names,
            deterministic_result=kwargs.get("deterministic_result"),
            metadata={"engine": "emcee"},
        )

    monkeypatch.setattr("emission.inference.run_emcee", fake_run)
    posterior = run_emission_inference(
        result,
        spectrum,
        DictConfig(
            emission_inference_method="emcee",
            emission_inference_emcee_nwalkers=40,
            emission_inference_emcee_nsteps=100,
            emission_inference_emcee_burn=20,
            emission_inference_emcee_thin=2,
            emission_inference_amplitude_prior_max=20.0,
        ),
    )
    assert posterior.metadata["engine"] == "emcee"
    assert captured["nwalkers"] == 40
    assert captured["nsteps"] == 100
    assert captured["burn"] == 20
    assert captured["thin"] == 2
