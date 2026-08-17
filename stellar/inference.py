"""Bayesian posterior integration for FitSpec stellar-population fitting.

The deterministic stellar engine uses variable projection: nonlinear dust and
kinematic parameters are optimized while non-negative SSP/gas amplitudes are
re-solved by bounded linear least squares.  That remains the fast deterministic
stage, but it is not by itself a marginalized Bayesian posterior.

This module therefore exposes two explicitly different inference modes:

``conditional`` (default)
    Sample nonlinear parameters *and* the non-negative stellar/gas amplitudes.
    The sampled SSP basis can be the deterministic active set, a broader
    diagnostic candidate set, or the complete loaded library.  This is a full
    posterior conditional on that chosen basis.

``profile``
    Sample only nonlinear parameters and re-solve linear amplitudes at every
    likelihood evaluation.  This is useful as a fast diagnostic but is a
    profile posterior, not marginalization over population weights.
"""
from __future__ import annotations

import numpy as np

from inference import PosteriorProblem, PriorSet, UniformPrior, gaussian_log_likelihood, run_emcee, run_dynesty
from stellar.stellar_fit import (
    DEFAULT_UV_EMISSION_LINES,
    DEFAULT_OPTICAL_EMISSION_LINES,
    _flux_to_lsun_factor,
    _solve_linear,
    build_emission_line_templates,
)
from stellar.stellar_models import StellarLibrary, shift_broaden_resample_library

__all__ = [
    "select_stellar_inference_basis",
    "build_stellar_posterior_problem",
    "run_stellar_inference",
    "stellar_population_samples",
]


def _get(config, key, default=None):
    return config.get(key, default) if hasattr(config, "get") else default


def _bool(config, key, default=False):
    value = _get(config, key, default)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y", "on"}


def _pair(config, key, default):
    value = _get(config, key, default)
    values = list(value) if isinstance(value, (list, tuple, np.ndarray)) else [x.strip() for x in str(value).split(",")]
    if len(values) < 2:
        return tuple(map(float, default))
    return float(values[0]), float(values[1])


def _seed(config):
    value = int(_get(config, "stellar_inference_random_seed", -1))
    return None if value < 0 else value


def _nonlinear_definition(result, config, *, include_gas=None):
    if include_gas is None:
        include_gas = bool(result.gas_names) or bool(result.metadata.get("simultaneous_gas", False))
    include_gas = bool(include_gas)
    names = ["ebv", "velocity_kms", "sigma_kms"]
    initial = [float(result.ebv), float(result.velocity_kms), float(result.sigma_kms)]
    bounds = [
        _pair(config, "stellar_ebv_bounds", (0.0, 2.0)),
        _pair(config, "stellar_velocity_bounds_kms", (-500.0, 500.0)),
        _pair(config, "stellar_sigma_bounds_kms", (0.0, 1000.0)),
    ]
    if include_gas:
        names += ["gas_velocity_kms", "gas_sigma_kms"]
        initial += [float(result.gas_velocity_kms), float(result.gas_sigma_kms)]
        bounds += [
            _pair(config, "stellar_gas_velocity_bounds_kms", (-500.0, 500.0)),
            _pair(config, "stellar_gas_sigma_bounds_kms", (1.0, 1000.0)),
        ]
    priors = []
    for name, (lower, upper) in zip(names, bounds):
        if not (np.isfinite(lower) and np.isfinite(upper) and lower < upper):
            raise ValueError(f"Stellar inference requires finite bounds for {name!r}.")
        priors.append(UniformPrior(lower, upper))
    return names, np.asarray(initial, float), priors, include_gas


def select_stellar_inference_basis(result, config) -> np.ndarray:
    """Select SSP indices sampled by the conditional posterior.

    ``candidate`` is the recommended default: retain all deterministic active
    populations and add low-delta-chi-square single-SSP alternatives from the
    already-computed stellar diagnostics.  ``active`` conditions only on the
    deterministic non-zero basis.  ``full`` samples every loaded SSP and is
    guarded by ``stellar_inference_max_basis_templates`` for tractability.
    """
    mode = str(_get(config, "stellar_inference_basis_mode", "candidate")).strip().lower()
    if mode not in {"active", "candidate", "full"}:
        raise ValueError("stellar_inference_basis_mode must be active, candidate, or full.")

    coeff = np.asarray(result.coefficients, float)
    n_models = coeff.size
    if n_models == 0:
        raise ValueError("Stellar result contains no SSP coefficients.")
    maximum = float(np.nanmax(coeff)) if np.any(np.isfinite(coeff)) else 0.0
    relative = float(_get(config, "stellar_inference_active_relative_threshold", 1e-8))
    floor = max(1e-12, relative * maximum)
    active = np.flatnonzero(np.isfinite(coeff) & (coeff > floor))
    if active.size == 0:
        active = np.asarray([int(np.nanargmax(np.where(np.isfinite(coeff), coeff, -np.inf)))])

    max_basis = int(_get(config, "stellar_inference_max_basis_templates", 32))
    if max_basis < 1:
        raise ValueError("stellar_inference_max_basis_templates must be >= 1.")

    if mode == "full":
        indices = np.arange(n_models, dtype=int)
        if indices.size > max_basis:
            raise ValueError(
                f"Full stellar posterior requests {indices.size} SSP templates, exceeding "
                f"stellar_inference_max_basis_templates={max_basis}. Increase the cap "
                "deliberately or use candidate/active basis mode."
            )
        return indices

    if active.size > max_basis:
        raise ValueError(
            f"The deterministic stellar solution has {active.size} active SSPs, exceeding "
            f"stellar_inference_max_basis_templates={max_basis}. Increase the cap deliberately."
        )
    if mode == "active":
        return np.sort(active)

    selected = list(map(int, active))
    diagnostics = getattr(result, "diagnostics", None)
    delta = None if diagnostics is None else diagnostics.single_ssp_delta_chi_square
    if delta is not None:
        delta = np.asarray(delta, float)
        threshold = float(_get(config, "stellar_inference_candidate_delta_chi2_max", 25.0))
        candidates = np.flatnonzero(np.isfinite(delta) & (delta <= threshold))
        candidates = candidates[np.argsort(delta[candidates])]
        for index in candidates:
            index = int(index)
            if index not in selected:
                selected.append(index)
            if len(selected) >= max_basis:
                break
    return np.asarray(sorted(selected), dtype=int)


def _gas_indices(result, config):
    amplitudes = np.asarray(result.gas_amplitudes, float)
    if amplitudes.size == 0:
        return np.empty(0, dtype=int)
    mode = str(_get(config, "stellar_inference_gas_basis_mode", "active")).strip().lower()
    if mode not in {"active", "all", "none"}:
        raise ValueError("stellar_inference_gas_basis_mode must be active, all, or none.")
    if mode == "none":
        return np.empty(0, dtype=int)
    if mode == "all":
        return np.arange(amplitudes.size, dtype=int)
    maximum = float(np.nanmax(amplitudes)) if np.any(np.isfinite(amplitudes)) else 0.0
    relative = float(_get(config, "stellar_inference_gas_active_relative_threshold", 1e-8))
    floor = max(0.0, relative * maximum)
    return np.flatnonzero(np.isfinite(amplitudes) & (amplitudes > floor))


def _covariance_for_mask(covariance, mask):
    if covariance is None:
        return None
    cov = np.asarray(covariance, float)
    n = mask.size
    m = int(np.count_nonzero(mask))
    if cov.shape == (n, n):
        return cov[np.ix_(mask, mask)]
    if cov.shape == (m, m):
        return cov
    raise ValueError("covariance must describe either the full or masked stellar spectrum.")


def _line_list_from_result(result, regime):
    if result.gas_names:
        return list(zip(result.gas_names, np.asarray(result.gas_rest_wavelengths, float)))
    return list(DEFAULT_UV_EMISSION_LINES if str(regime).lower() == "uv" else DEFAULT_OPTICAL_EMISSION_LINES)


def build_stellar_posterior_problem(
    result, spectrum, config, *, library: StellarLibrary, regime=None, line_list=None, covariance=None,
):
    """Construct the sampler-independent posterior for a stellar fit.

    The numerical data/mask come from ``result``.  The loaded ``library`` and
    the spectrum's ResolutionModel are required to regenerate transformed SSPs
    continuously as dust and kinematics vary.
    """
    if result.flux_unc is None and covariance is None:
        raise ValueError("Stellar posterior sampling requires flux_unc or covariance.")
    if library.n_models != np.asarray(result.coefficients).size:
        raise ValueError("The supplied stellar library does not match the deterministic result basis size.")

    inference_mode = str(_get(config, "stellar_inference_posterior_mode", "conditional")).strip().lower()
    if inference_mode not in {"conditional", "profile"}:
        raise ValueError("stellar_inference_posterior_mode must be conditional or profile.")
    regime = str(result.regime if regime is None else regime).lower()
    wave = np.asarray(result.wave, float)
    flux = np.asarray(result.flux, float)
    unc = None if result.flux_unc is None else np.asarray(result.flux_unc, float)
    mask = np.asarray(result.mask, bool)
    resolution = getattr(spectrum, "resolution", None)
    attenuation_law = str(_get(config, "stellar_attenuation_law", "calzetti00"))
    deterministic_has_gas = bool(result.gas_names) or bool(result.metadata.get("simultaneous_gas", False))
    gas_indices = _gas_indices(result, config) if deterministic_has_gas and inference_mode == "conditional" else np.empty(0, dtype=int)
    include_gas = deterministic_has_gas if inference_mode == "profile" else bool(gas_indices.size)
    nonlinear_names, nonlinear_initial, nonlinear_priors, include_gas = _nonlinear_definition(result, config, include_gas=include_gas)
    cov_used = _covariance_for_mask(covariance, mask)
    lines = _line_list_from_result(result, regime) if line_list is None else list(line_list)

    metadata = {
        "science_module": "stellar",
        "posterior_kind": "conditional_full" if inference_mode == "conditional" else "profile_not_marginalized",
        "library_family": str(result.library_family),
        "regime": regime,
        "attenuation_law": attenuation_law,
    }

    if inference_mode == "profile":
        priors = PriorSet(nonlinear_priors, parameter_names=nonlinear_names)
        flux_factor = float(result.metadata.get("flux_to_lsun_factor", _flux_to_lsun_factor(float(getattr(spectrum, "redshift", 0.0)), config)))
        flux_internal = flux * flux_factor
        unc_internal = None if unc is None else unc * flux_factor
        solver = str(_get(config, "stellar_mass_solver", "bvls")).lower()
        tolerance = float(_get(config, "stellar_mass_tolerance", 1e-8))
        maxiter = int(_get(config, "stellar_mass_max_iterations", 3000))
        gas_peak = float(_get(config, "stellar_gas_peak_max_factor", 2.0))

        def log_likelihood(theta):
            try:
                ebv, vel, sig = map(float, theta[:3])
                stellar = shift_broaden_resample_library(
                    library, wave, ebv=ebv, velocity_kms=vel, sigma_kms=sig,
                    attenuation_law=attenuation_law, resolution=resolution,
                )
                G = np.empty((wave.size, 0), float)
                if include_gas:
                    G, _, _ = build_emission_line_templates(
                        wave, float(theta[3]), float(theta[4]), resolution, lines,
                        full_wave=wave, fit_mask=mask,
                    )
                design = np.column_stack([stellar.T, G]) if G.shape[1] else stellar.T
                _, model_internal, _ = _solve_linear(
                    design, flux_internal, unc_internal, mask,
                    n_stellar=library.n_models, gas_peak_max_factor=gas_peak,
                    method=solver, tolerance=tolerance, max_iter=maxiter,
                )
                model = model_internal / flux_factor
                return gaussian_log_likelihood(
                    flux[mask], model[mask], None if unc is None else unc[mask], covariance=cov_used,
                )
            except (ValueError, FloatingPointError, OverflowError, np.linalg.LinAlgError):
                return -np.inf

        metadata.update({"basis_mode": "variable_projection_all", "n_ssp_sampled": 0, "n_gas_amplitudes_sampled": 0})
        return PosteriorProblem(
            parameter_names=nonlinear_names, log_likelihood=log_likelihood, priors=priors,
            initial_position=nonlinear_initial, metadata=metadata,
        )

    basis = select_stellar_inference_basis(result, config)
    log_mass_bounds = _pair(config, "stellar_inference_log10_mass_bounds", (-6.0, 12.0))
    if not (np.isfinite(log_mass_bounds[0]) and np.isfinite(log_mass_bounds[1]) and log_mass_bounds[0] < log_mass_bounds[1]):
        raise ValueError("stellar_inference_log10_mass_bounds must be finite lower,upper in log10(Msun).")

    names = list(nonlinear_names)
    initial = list(nonlinear_initial)
    priors = list(nonlinear_priors)
    coeff = np.asarray(result.coefficients, float)
    low_mass = 10.0 ** log_mass_bounds[0]
    for index in basis:
        index = int(index)
        names.append(f"log10_mass_ssp_{index:04d}")
        if coeff[index] > 0:
            log_initial = float(np.log10(coeff[index]))
            if not (log_mass_bounds[0] <= log_initial <= log_mass_bounds[1]):
                raise ValueError(
                    f"Deterministic SSP mass at index {index} (log10 M={log_initial:.4g}) lies outside "
                    "stellar_inference_log10_mass_bounds. Widen the configured proper prior."
                )
        else:
            log_initial = float(log_mass_bounds[0] + 0.1 * (log_mass_bounds[1] - log_mass_bounds[0]))
        initial.append(log_initial)
        priors.append(UniformPrior(*log_mass_bounds))

    gas_names = tuple(result.gas_names)
    gas_amplitudes = np.asarray(result.gas_amplitudes, float)
    data_peak = max(float(np.nanmax(np.abs(flux[mask]))), np.finfo(float).tiny)
    gas_upper_default = data_peak * float(_get(config, "stellar_gas_peak_max_factor", 2.0))
    gas_upper = float(_get(config, "stellar_inference_gas_amplitude_prior_max", 0.0))
    if not (np.isfinite(gas_upper) and gas_upper > 0):
        gas_upper = gas_upper_default
    for index in gas_indices:
        index = int(index)
        names.append(f"gas_amp_{gas_names[index]}")
        initial.append(float(np.clip(gas_amplitudes[index], 0.0, gas_upper)))
        priors.append(UniformPrior(0.0, gas_upper))

    prior_set = PriorSet(priors, parameter_names=names)
    initial = np.asarray(initial, float)
    n_nonlin = len(nonlinear_names)
    n_mass = basis.size
    sampled_lines = [lines[int(i)] for i in gas_indices]

    def log_likelihood(theta):
        try:
            theta = np.asarray(theta, float)
            ebv, vel, sig = map(float, theta[:3])
            stellar = shift_broaden_resample_library(
                library, wave, ebv=ebv, velocity_kms=vel, sigma_kms=sig,
                attenuation_law=attenuation_law, resolution=resolution,
            )
            masses = 10.0 ** theta[n_nonlin:n_nonlin + n_mass]
            if include_gas and gas_indices.size:
                gv, gs = float(theta[3]), float(theta[4])
                G, built_names, _ = build_emission_line_templates(
                    wave, gv, gs, resolution, sampled_lines, full_wave=wave, fit_mask=mask,
                )
                if G.shape[1] != gas_indices.size or tuple(built_names) != tuple(name for name, _ in sampled_lines):
                    return -np.inf
                gas_amp = theta[n_nonlin + n_mass:]
            flux_factor = float(result.metadata.get("flux_to_lsun_factor", _flux_to_lsun_factor(float(getattr(spectrum, "redshift", 0.0)), config)))
            stellar_native = (stellar[basis].T @ masses) / flux_factor
            if include_gas and gas_indices.size:
                model_native = stellar_native + G @ gas_amp
            else:
                model_native = stellar_native
            if not np.all(np.isfinite(model_native[mask])):
                return -np.inf
            return gaussian_log_likelihood(
                flux[mask], model_native[mask], None if unc is None else unc[mask], covariance=cov_used,
            )
        except (ValueError, FloatingPointError, OverflowError, np.linalg.LinAlgError):
            return -np.inf

    metadata.update({
        "basis_mode": str(_get(config, "stellar_inference_basis_mode", "candidate")).strip().lower(),
        "ssp_indices": basis.tolist(),
        "n_ssp_sampled": int(basis.size),
        "gas_indices": gas_indices.tolist(),
        "n_gas_amplitudes_sampled": int(gas_indices.size),
        "log10_mass_prior_bounds": list(map(float, log_mass_bounds)),
        "gas_amplitude_prior_max": float(gas_upper),
        "evidence_prior_note": "gas amplitude upper bound is data-scaled when no explicit maximum is configured" if float(_get(config, "stellar_inference_gas_amplitude_prior_max", 0.0)) <= 0 and gas_indices.size else "explicit/fixed priors",
    })
    return PosteriorProblem(
        parameter_names=names, log_likelihood=log_likelihood, priors=prior_set,
        initial_position=initial, metadata=metadata,
    )


def run_stellar_inference(result, spectrum, config, *, library: StellarLibrary, regime=None, line_list=None, covariance=None):
    """Run emcee or dynesty for the configured stellar posterior."""
    method = str(_get(config, "stellar_inference_method", "deterministic")).strip().lower()
    if method in {"", "none", "off", "deterministic"}:
        return None
    if method not in {"emcee", "dynesty"}:
        raise ValueError("stellar_inference_method must be deterministic, emcee, or dynesty.")
    problem = build_stellar_posterior_problem(
        result, spectrum, config, library=library, regime=regime, line_list=line_list, covariance=covariance,
    )
    seed = _seed(config)
    progress = _bool(config, "stellar_inference_progress", False)
    if method == "emcee":
        configured_walkers = int(_get(config, "stellar_inference_emcee_nwalkers", 0))
        return run_emcee(
            problem,
            nwalkers=None if configured_walkers <= 0 else configured_walkers,
            nsteps=int(_get(config, "stellar_inference_emcee_nsteps", 4000)),
            burn=int(_get(config, "stellar_inference_emcee_burn", 1000)),
            thin=int(_get(config, "stellar_inference_emcee_thin", 5)),
            initial_scale=float(_get(config, "stellar_inference_emcee_initial_scale", 1e-3)),
            random_seed=seed, progress=progress, deterministic_result=None,
        )
    return run_dynesty(
        problem,
        dynamic=_bool(config, "stellar_inference_dynesty_dynamic", True),
        nlive=int(_get(config, "stellar_inference_dynesty_nlive", 500)),
        dlogz=float(_get(config, "stellar_inference_dynesty_dlogz", 0.1)),
        sample=str(_get(config, "stellar_inference_dynesty_sample", "auto")),
        bound=str(_get(config, "stellar_inference_dynesty_bound", "multi")),
        random_seed=seed, progress=progress, deterministic_result=None,
    )


def stellar_population_samples(posterior, result):
    """Return derived total/mass-fraction samples from a conditional posterior.

    Returns a dictionary with ``total_formed_mass_msun``, ``ssp_indices`` and
    ``mass_fractions``.  Profile posteriors have no population-weight samples
    and therefore raise ``ValueError``.
    """
    indices = []
    columns = []
    for j, name in enumerate(posterior.parameter_names):
        if name.startswith("log10_mass_ssp_"):
            indices.append(int(name.rsplit("_", 1)[1]))
            columns.append(j)
    if not columns:
        raise ValueError("This posterior does not sample SSP masses (profile mode).")
    masses = 10.0 ** posterior.samples[:, columns]
    total = np.sum(masses, axis=1)
    fractions = masses / total[:, None]
    return {
        "total_formed_mass_msun": total,
        "ssp_indices": np.asarray(indices, dtype=int),
        "mass_fractions": fractions,
        "ages_myr": np.asarray(result.ages_myr, float)[indices],
        "metallicity_codes": np.asarray(result.metallicity_codes).astype(str)[indices],
    }
