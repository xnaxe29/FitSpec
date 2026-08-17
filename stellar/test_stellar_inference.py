"""Tests for the stellar posterior adapter and basis-selection semantics."""
from types import SimpleNamespace

import numpy as np

from core.results import PosteriorResult
from stellar.stellar_models import StellarLibrary
from stellar.stellar_results import StellarFitDiagnostics, StellarFitResult
from stellar.inference import (
    select_stellar_inference_basis,
    build_stellar_posterior_problem,
    run_stellar_inference,
    stellar_population_samples,
)


class Config(dict):
    def get(self, key, default=None):
        return super().get(key, default)


def _library_and_result():
    wave = np.linspace(4000.0, 4100.0, 101)
    x = (wave - wave.mean()) / 50.0
    basis = np.vstack([
        1.0 + 0.04*x,
        0.9 - 0.03*x + 0.02*x*x,
        1.1 + 0.02*np.sin(3*x),
        0.8 + 0.06*x*x,
    ])
    lib = StellarLibrary(
        wave=wave, flux_per_mass=basis, flux_reference=basis.copy(),
        ages_myr=np.array([5.0, 10.0, 50.0, 500.0]),
        metallicity_codes=np.array(["z1", "z1", "z2", "z2"]),
        metallicities_solar=np.array([0.2, 0.2, 1.0, 1.0]),
        reference_masses=np.ones(4), labels=("a", "b", "c", "d"),
        source_paths=("a", "b", "c", "d"), family="test",
        metallicities_dex=np.array([-0.7, -0.7, 0.0, 0.0]),
    )
    coeff = np.array([2.0, 0.0, 1.0, 0.0])
    flux = basis.T @ coeff
    unc = np.full(wave.size, 0.05)
    diag = StellarFitDiagnostics(single_ssp_delta_chi_square=np.array([0.0, 4.0, 2.0, 80.0]))
    result = StellarFitResult(
        regime="optical", library_family="test", wave=wave, flux=flux,
        flux_unc=unc, mask=np.ones(wave.size, bool), model=flux.copy(),
        stellar_model=flux.copy(), gas_model=np.zeros_like(flux),
        coefficients=coeff, ages_myr=lib.ages_myr,
        metallicity_codes=lib.metallicity_codes,
        metallicities_solar=lib.metallicities_solar,
        ebv=0.0, velocity_kms=0.0, sigma_kms=0.0,
        diagnostics=diag, metadata={"flux_to_lsun_factor": 1.0, "simultaneous_gas": False},
    )
    spectrum = SimpleNamespace(redshift=0.0, resolution=None)
    return lib, result, spectrum


def test_candidate_basis_adds_degenerate_zero_weight_ssps():
    _, result, _ = _library_and_result()
    cfg = Config(stellar_inference_basis_mode="candidate",
                 stellar_inference_candidate_delta_chi2_max=5.0,
                 stellar_inference_max_basis_templates=4)
    assert np.array_equal(select_stellar_inference_basis(result, cfg), np.array([0, 1, 2]))


def test_full_basis_guard_refuses_accidental_high_dimension():
    _, result, _ = _library_and_result()
    cfg = Config(stellar_inference_basis_mode="full", stellar_inference_max_basis_templates=3)
    try:
        select_stellar_inference_basis(result, cfg)
    except ValueError as exc:
        assert "exceeding" in str(exc)
    else:
        raise AssertionError("full-basis dimensionality guard should fail")


def test_conditional_problem_samples_masses_and_has_finite_initial_posterior():
    lib, result, spectrum = _library_and_result()
    cfg = Config(
        stellar_inference_posterior_mode="conditional",
        stellar_inference_basis_mode="candidate",
        stellar_inference_candidate_delta_chi2_max=5.0,
        stellar_inference_max_basis_templates=4,
        stellar_inference_log10_mass_bounds="-6,6",
        stellar_ebv_bounds="0,1", stellar_velocity_bounds_kms="-200,200",
        stellar_sigma_bounds_kms="0,300",
    )
    problem = build_stellar_posterior_problem(result, spectrum, cfg, library=lib)
    assert problem.metadata["posterior_kind"] == "conditional_full"
    assert problem.metadata["ssp_indices"] == [0, 1, 2]
    assert problem.ndim == 6  # 3 nonlinear + 3 SSP masses
    assert np.isfinite(problem.log_probability(problem.initial_position))


def test_profile_problem_is_explicitly_not_marginalized():
    lib, result, spectrum = _library_and_result()
    cfg = Config(
        stellar_inference_posterior_mode="profile",
        stellar_ebv_bounds="0,1", stellar_velocity_bounds_kms="-200,200",
        stellar_sigma_bounds_kms="0,300", stellar_mass_solver="bvls",
    )
    problem = build_stellar_posterior_problem(result, spectrum, cfg, library=lib)
    assert problem.ndim == 3
    assert problem.metadata["posterior_kind"] == "profile_not_marginalized"
    assert np.isfinite(problem.log_probability(problem.initial_position))


def test_population_samples_recover_total_and_fractions():
    _, result, _ = _library_and_result()
    posterior = PosteriorResult(
        samples=np.array([[0.0, np.log10(2.0), np.log10(1.0)], [0.1, np.log10(4.0), np.log10(2.0)]]),
        log_probability=np.array([-1.0, -2.0]),
        parameter_names=["ebv", "log10_mass_ssp_0000", "log10_mass_ssp_0002"],
    )
    pop = stellar_population_samples(posterior, result)
    assert np.allclose(pop["total_formed_mass_msun"], [3.0, 6.0])
    assert np.allclose(pop["mass_fractions"], [[2/3, 1/3], [2/3, 1/3]])
    assert np.array_equal(pop["ssp_indices"], [0, 2])


def test_run_stellar_inference_dispatches_and_leaves_deterministic_disabled(monkeypatch):
    import stellar.inference as si
    lib, result, spectrum = _library_and_result()
    disabled = Config(stellar_inference_method="deterministic")
    assert run_stellar_inference(result, spectrum, disabled, library=lib) is None

    sentinel = object()
    monkeypatch.setattr(si, "run_emcee", lambda problem, **kwargs: sentinel)
    cfg = Config(
        stellar_inference_method="emcee", stellar_inference_posterior_mode="conditional",
        stellar_inference_basis_mode="active", stellar_inference_max_basis_templates=4,
        stellar_inference_log10_mass_bounds="-6,6",
        stellar_ebv_bounds="0,1", stellar_velocity_bounds_kms="-200,200",
        stellar_sigma_bounds_kms="0,300", stellar_inference_emcee_nsteps=20,
        stellar_inference_emcee_burn=5,
    )
    assert run_stellar_inference(result, spectrum, cfg, library=lib) is sentinel


def test_conditional_active_gas_mode_does_not_sample_unconstrained_gas_kinematics():
    lib, result, spectrum = _library_and_result()
    result.metadata["simultaneous_gas"] = True
    result.gas_names = ("Hbeta",)
    result.gas_rest_wavelengths = np.array([4861.333])
    result.gas_amplitudes = np.array([0.0])
    cfg = Config(
        stellar_inference_posterior_mode="conditional",
        stellar_inference_basis_mode="active", stellar_inference_max_basis_templates=4,
        stellar_inference_gas_basis_mode="active", stellar_inference_log10_mass_bounds="-6,6",
        stellar_ebv_bounds="0,1", stellar_velocity_bounds_kms="-200,200",
        stellar_sigma_bounds_kms="0,300", stellar_gas_velocity_bounds_kms="-200,200",
        stellar_gas_sigma_bounds_kms="1,300",
    )
    problem = build_stellar_posterior_problem(result, spectrum, cfg, library=lib)
    assert "gas_velocity_kms" not in problem.parameter_names
    assert "gas_sigma_kms" not in problem.parameter_names
    assert problem.metadata["n_gas_amplitudes_sampled"] == 0
