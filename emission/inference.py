"""Bayesian posterior integration for FitSpec emission-line models.

The deterministic emission fit remains the model-construction and
component-selection stage.  This module adapts that selected model to the
shared :mod:`inference` layer and runs either emcee or dynesty without
reading mutable GUI state.
"""
from __future__ import annotations

import numpy as np

from inference import PriorSet, UniformPrior, model_parameters_problem, run_emcee, run_dynesty
from emission.emission_model import make_emission_model_func

__all__ = [
    "build_emission_prior_set",
    "build_emission_posterior_problem",
    "run_emission_inference",
]


def _get(config, key, default=None):
    return config.get(key, default) if hasattr(config, "get") else default


def _seed(config):
    value = int(_get(config, "emission_inference_random_seed", -1))
    return None if value < 0 else value


def build_emission_prior_set(fit_result, config) -> PriorSet:
    """Construct proper independent uniform priors for the selected model.

    Velocity and dispersion priors use their fitted Parameter bounds.
    Emission amplitudes use those bounds when finite.  If the deterministic
    amplitude upper bound is infinite, ``emission_inference_amplitude_prior_max``
    must supply a finite positive upper limit.  FitSpec deliberately refuses
    to turn an optimizer's infinite bound into an improper Bayesian prior.
    """
    amplitude_prior_max = float(_get(config, "emission_inference_amplitude_prior_max", 0.0))
    names = fit_result.parameters.parameter_names()
    priors = []
    for _, parameter in fit_result.parameters.free_parameters():
        lower = float(parameter.lower)
        upper = float(parameter.upper)
        if parameter.name.startswith("amp_") and not np.isfinite(upper):
            if not (np.isfinite(amplitude_prior_max) and amplitude_prior_max > lower):
                raise ValueError(
                    "Emission posterior sampling requires a proper finite amplitude prior. "
                    "Set emission_inference_amplitude_prior_max to a finite value larger "
                    "than every allowed amplitude, or set emission_maximum_amplitude."
                )
            upper = amplitude_prior_max
        if not (np.isfinite(lower) and np.isfinite(upper) and lower < upper):
            raise ValueError(
                f"Cannot construct a proper uniform prior for emission parameter "
                f"{parameter.name!r}: bounds are ({lower}, {upper})."
            )
        priors.append(UniformPrior(lower, upper))
    return PriorSet(priors=priors, parameter_names=names)


def build_emission_posterior_problem(
    emission_result, spectrum, config, *, line_list=None, covariance=None,
):
    """Create the shared :class:`inference.PosteriorProblem` for an emission fit.

    ``emission_result`` is the already-selected deterministic model.  The
    spectrum supplies the live ResolutionModel needed to reproduce instrumental
    convolution; the numerical data and mask themselves are taken from the
    deterministic FitResult so posterior sampling exactly follows what was fit.
    """
    fit_result = emission_result.fit_result
    resolved_lines = emission_result.line_list if line_list is None else line_list
    kinematics_mode = str(
        fit_result.metadata.get(
            "emission_kinematics_mode", _get(config, "emission_kinematics_mode", "tied")
        )
    ).strip().lower()
    physical_model_func = make_emission_model_func(
        resolved_lines,
        redshift=fit_result.redshift,
        resolution=spectrum.resolution,
        kinematics_mode=kinematics_mode,
    )
    # fit_result.flux/.model are in whatever units the spectrum passed to
    # fit_emission_spectrum was already in (see
    # emission.emission_fit.normalize_emission_spectrum) -- the
    # deterministic fitter no longer applies or undoes
    # emission_flux_normalizing_factor/emission_flux_reduction itself, so
    # nothing needs reproducing or re-offsetting here either; the physical
    # model already lives in the same domain as fit_result.flux.
    priors = build_emission_prior_set(fit_result, config)
    metadata = {
        "science_module": "emission",
        "n_components": int(fit_result.parameters.n_components),
        "kinematics_mode": kinematics_mode,
        "line_names": [line.name for line in resolved_lines],
    }
    return model_parameters_problem(
        fit_result.wave,
        fit_result.flux,
        fit_result.flux_unc,
        fit_result.parameters,
        physical_model_func,
        mask=fit_result.mask,
        covariance=covariance,
        priors=priors,
        metadata=metadata,
    )


def run_emission_inference(
    emission_result, spectrum, config, *, line_list=None, covariance=None,
):
    """Run the emission posterior backend configured for this FitSpec run.

    Recognized methods are ``emcee`` and ``dynesty``.  ``deterministic``/``none``
    intentionally return ``None`` so a common configuration can leave posterior
    sampling disabled without importing optional sampler packages.
    """
    method = str(_get(config, "emission_inference_method", "deterministic")).strip().lower()
    if method in ("", "none", "deterministic", "off"):
        return None
    if method not in ("emcee", "dynesty"):
        raise ValueError("emission_inference_method must be deterministic, emcee, or dynesty.")

    problem = build_emission_posterior_problem(
        emission_result, spectrum, config, line_list=line_list, covariance=covariance,
    )
    deterministic_result = emission_result.fit_result
    progress = bool(_get(config, "emission_inference_progress", False))
    random_seed = _seed(config)

    if method == "emcee":
        nwalkers = int(_get(config, "emission_inference_emcee_nwalkers", 64))
        return run_emcee(
            problem,
            nwalkers=nwalkers,
            nsteps=int(_get(config, "emission_inference_emcee_nsteps", 4000)),
            burn=int(_get(config, "emission_inference_emcee_burn", 1000)),
            thin=int(_get(config, "emission_inference_emcee_thin", 5)),
            initial_scale=float(_get(config, "emission_inference_emcee_initial_scale", 1e-3)),
            random_seed=random_seed,
            progress=progress,
            deterministic_result=deterministic_result,
        )

    return run_dynesty(
        problem,
        dynamic=bool(_get(config, "emission_inference_dynesty_dynamic", True)),
        nlive=int(_get(config, "emission_inference_dynesty_nlive", 500)),
        dlogz=float(_get(config, "emission_inference_dynesty_dlogz", 0.1)),
        sample=str(_get(config, "emission_inference_dynesty_sample", "auto")),
        bound=str(_get(config, "emission_inference_dynesty_bound", "multi")),
        random_seed=random_seed,
        progress=progress,
        deterministic_result=deterministic_result,
    )
