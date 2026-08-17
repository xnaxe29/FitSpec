"""Bayesian posterior integration for FitSpec absorption-line models.

The deterministic absorption fit remains responsible for choosing the
component structure, applying optional rejection/freeze logic, and defining
all fixed/tied state.  This module samples the posterior of that finalized
model with the shared :mod:`inference` layer.

Single-group fits can be reconstructed from an ``AbsorptionFitResult`` plus
the live spectrum/configuration.  Joint cross-ion fits additionally require
the original ``AbsorptionSystem``/tie objects because arbitrary Python tie
transforms are intentionally not serialized into FITS products.
"""
from __future__ import annotations

import numpy as np

from inference import PriorSet, UniformPrior, model_parameters_problem, run_emcee, run_dynesty
from absorption.absorption_model import (
    make_absorption_model_func,
    make_joint_absorption_model_func,
    apply_abundance_pattern_ties,
)

__all__ = [
    "build_absorption_prior_set",
    "build_absorption_posterior_problem",
    "build_joint_absorption_posterior_problem",
    "run_absorption_inference",
]


def _get(config, key, default=None):
    return config.get(key, default) if hasattr(config, "get") else default


def _bool(config, key, default=False):
    value = _get(config, key, default)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y", "on"}


def _seed(config):
    value = int(_get(config, "absorption_inference_random_seed", -1))
    return None if value < 0 else value


def build_absorption_prior_set(fit_result, config=None) -> PriorSet:
    """Construct proper independent priors for every free absorption parameter.

    Absorption parameters are already defined with finite physical bounds by
    the deterministic model builders (logN, b, velocity, covering fraction,
    and any abundance-ratio holder), so the default posterior prior is uniform
    over those exact bounds.  Fixed/tied/frozen parameters are excluded from
    the sampled vector by ``ModelParameters`` itself.

    Infinite bounds are rejected rather than silently converted into an
    improper prior.  This is especially important when loading legacy result
    files that pre-date persistence of parameter bounds.
    """
    names = fit_result.parameters.parameter_names()
    priors = []
    for component_index, parameter in fit_result.parameters.free_parameters():
        lower = float(parameter.lower)
        upper = float(parameter.upper)
        if not (np.isfinite(lower) and np.isfinite(upper) and lower < upper):
            raise ValueError(
                "Absorption posterior sampling requires proper finite bounds. "
                f"Parameter c{component_index}_{parameter.name!s} has bounds "
                f"({lower}, {upper}). Refit with finite bounds or supply a newer "
                "FitSpec result file that preserves them."
            )
        priors.append(UniformPrior(lower, upper))
    return PriorSet(priors=priors, parameter_names=names)


def _metadata_for_result(absorption_result, *, joint: bool, extra=None):
    fit_result = absorption_result.fit_result
    frozen = [bool(measurement.is_upper_limit) for measurement in absorption_result.measurements]
    metadata = {
        "science_module": "absorption",
        "fit_mode": "joint" if joint else "single",
        "n_components": int(fit_result.parameters.n_components),
        "partial_coverage": bool(absorption_result.partial_coverage),
        "transition_names": [transition.name for transition in absorption_result.transitions],
        "system_labels": [measurement.system_label for measurement in absorption_result.measurements],
        "frozen_upper_limit_components": frozen,
        "n_frozen_upper_limit_components": int(sum(frozen)),
    }
    if extra:
        metadata.update(extra)
    return metadata


def build_absorption_posterior_problem(
    absorption_result, spectrum, config, *, transitions=None, covariance=None,
):
    """Build a posterior problem for a finalized single-group absorption fit.

    Frozen parameters remain fixed because the posterior starts from the exact
    ``ModelParameters`` stored in the deterministic result.  Partial covering
    is reconstructed from the result itself; subpixel oversampling is taken
    from saved fit metadata when available, otherwise from config.
    """
    fit_result = absorption_result.fit_result
    resolved_transitions = absorption_result.transitions if transitions is None else list(transitions)
    if not resolved_transitions:
        raise ValueError("Single-group absorption inference requires at least one transition.")

    partial_coverage = bool(absorption_result.partial_coverage)
    subpixel = int(
        (fit_result.metadata or {}).get(
            "absorption_subpixel", _get(config, "absorption_subpixel", 1)
        )
    )
    model_func = make_absorption_model_func(
        resolved_transitions,
        redshift=fit_result.redshift,
        resolution=spectrum.resolution,
        partial_coverage=partial_coverage,
        subpixel=subpixel,
    )
    priors = build_absorption_prior_set(fit_result, config)
    metadata = _metadata_for_result(
        absorption_result,
        joint=False,
        extra={"subpixel": subpixel},
    )
    return model_parameters_problem(
        fit_result.wave,
        fit_result.flux,
        fit_result.flux_unc,
        fit_result.parameters,
        model_func,
        mask=fit_result.mask,
        covariance=covariance,
        priors=priors,
        metadata=metadata,
    )


def _component_transitions_from_systems(systems):
    component_transitions = []
    for system in systems:
        component_transitions.extend([list(system.transitions)] * int(system.n_components))
    return component_transitions


def build_joint_absorption_posterior_problem(
    absorption_result,
    spectrum,
    config,
    *,
    systems,
    ties=None,
    thermal_turbulent_links=None,
    abundance_pattern_groups=None,
    covariance=None,
    partial_coverage=None,
    subpixel=None,
):
    """Build a posterior problem for a finalized joint/tied absorption fit.

    The original code-level tie objects are required because ``ParameterTie``
    may contain arbitrary Python transforms and therefore cannot be safely or
    generally reconstructed from a FITS product.  The fitted parameter values,
    bounds, and fixed/free state come from ``absorption_result``; ``systems``
    supplies only the physical transition assignment for each component.
    """
    if not systems:
        raise ValueError("Joint absorption inference requires the original non-empty systems list.")

    fit_result = absorption_result.fit_result
    posterior_parameters = fit_result.parameters.copy()
    # Protect inference reconstructed from older deterministic results that may
    # have serialized a tied follower as free: ties define derived parameters,
    # never independent posterior dimensions.
    for tie in (ties or []):
        follower_component, follower_name = tie.follower
        posterior_parameters.components[follower_component][follower_name].fixed = True
    for link in (thermal_turbulent_links or []):
        follower_component, follower_name = link.follower
        posterior_parameters.components[follower_component][follower_name].fixed = True
    for group in (abundance_pattern_groups or []):
        for follower_component in group.follower_components:
            posterior_parameters.components[follower_component]["logN"].fixed = True

    component_transitions = _component_transitions_from_systems(systems)
    if len(component_transitions) != fit_result.parameters.n_components:
        raise ValueError(
            "Joint posterior context does not match the deterministic fit: "
            f"systems define {len(component_transitions)} components but the result has "
            f"{fit_result.parameters.n_components}."
        )

    if partial_coverage is None:
        partial_coverage = bool(absorption_result.partial_coverage)
    if subpixel is None:
        subpixel = int(
            (fit_result.metadata or {}).get(
                "absorption_subpixel", _get(config, "absorption_subpixel", 1)
            )
        )

    physical_model = make_joint_absorption_model_func(
        component_transitions,
        redshift=fit_result.redshift,
        resolution=spectrum.resolution,
        partial_coverage=bool(partial_coverage),
        ties=ties,
        thermal_turbulent_links=thermal_turbulent_links,
        subpixel=int(subpixel),
    )

    if abundance_pattern_groups:
        def model_func(wave, model_parameters):
            apply_abundance_pattern_ties(model_parameters, abundance_pattern_groups)
            return physical_model(wave, model_parameters)
    else:
        model_func = physical_model

    # Priors must reflect the tie-adjusted posterior parameter state.
    class _PosteriorFitView:
        parameters = posterior_parameters
    priors = build_absorption_prior_set(_PosteriorFitView(), config)
    metadata = _metadata_for_result(
        absorption_result,
        joint=True,
        extra={
            "subpixel": int(subpixel),
            "n_systems": len(systems),
            "has_parameter_ties": bool(ties),
            "has_thermal_turbulent_links": bool(thermal_turbulent_links),
            "has_abundance_pattern_groups": bool(abundance_pattern_groups),
        },
    )
    return model_parameters_problem(
        fit_result.wave,
        fit_result.flux,
        fit_result.flux_unc,
        posterior_parameters,
        model_func,
        mask=fit_result.mask,
        covariance=covariance,
        priors=priors,
        metadata=metadata,
    )


def run_absorption_inference(
    absorption_result,
    spectrum,
    config,
    *,
    transitions=None,
    systems=None,
    ties=None,
    thermal_turbulent_links=None,
    abundance_pattern_groups=None,
    covariance=None,
    partial_coverage=None,
    subpixel=None,
):
    """Run emcee or dynesty for the finalized absorption model.

    ``systems=None`` selects the single-group adapter.  Pass ``systems`` (and
    the same tie/link/group objects used by the deterministic fit) for joint
    cross-ion inference.  ``deterministic``/``none`` leaves inference disabled
    and returns ``None``.
    """
    method = str(_get(config, "absorption_inference_method", "deterministic")).strip().lower()
    if method in ("", "none", "deterministic", "off"):
        return None
    if method not in ("emcee", "dynesty"):
        raise ValueError("absorption_inference_method must be deterministic, emcee, or dynesty.")

    if systems is None:
        problem = build_absorption_posterior_problem(
            absorption_result,
            spectrum,
            config,
            transitions=transitions,
            covariance=covariance,
        )
    else:
        problem = build_joint_absorption_posterior_problem(
            absorption_result,
            spectrum,
            config,
            systems=systems,
            ties=ties,
            thermal_turbulent_links=thermal_turbulent_links,
            abundance_pattern_groups=abundance_pattern_groups,
            covariance=covariance,
            partial_coverage=partial_coverage,
            subpixel=subpixel,
        )

    deterministic_result = absorption_result.fit_result
    progress = _bool(config, "absorption_inference_progress", False)
    random_seed = _seed(config)

    if method == "emcee":
        return run_emcee(
            problem,
            nwalkers=int(_get(config, "absorption_inference_emcee_nwalkers", 64)),
            nsteps=int(_get(config, "absorption_inference_emcee_nsteps", 5000)),
            burn=int(_get(config, "absorption_inference_emcee_burn", 1500)),
            thin=int(_get(config, "absorption_inference_emcee_thin", 5)),
            initial_scale=float(_get(config, "absorption_inference_emcee_initial_scale", 1e-3)),
            random_seed=random_seed,
            progress=progress,
            deterministic_result=deterministic_result,
        )

    return run_dynesty(
        problem,
        dynamic=_bool(config, "absorption_inference_dynesty_dynamic", True),
        nlive=int(_get(config, "absorption_inference_dynesty_nlive", 600)),
        dlogz=float(_get(config, "absorption_inference_dynesty_dlogz", 0.1)),
        sample=str(_get(config, "absorption_inference_dynesty_sample", "auto")),
        bound=str(_get(config, "absorption_inference_dynesty_bound", "multi")),
        random_seed=random_seed,
        progress=progress,
        deterministic_result=deterministic_result,
    )
