"""Absorption-line fitting entry points.

``fit_absorption_spectrum`` builds an explicit-component
``ModelParameters`` for one transition group (see
``absorption.atomic.select_group``), wires it to the generic
``core.fitting.fit_deterministic`` engine via
``absorption.absorption_model.make_absorption_model_func``, and returns
the derived ``AbsorptionFitResult``.

``fit_joint_absorption_spectrum`` is the multi-system counterpart: it
fits several transition groups simultaneously (e.g. several ions),
optionally linked to each other (shared/tied redshifts, thermally- or
turbulently-linked Doppler parameters, fixed-ratio or common-pattern
column densities). Because these linkages are inherently code-level
constructs -- VPFIT itself requires a hand-crafted start file for the
same reason -- this entry point is programmatic (built from
``absorption.absorption_model.AbsorptionSystem`` objects and tie
specifications) rather than config-string-driven like the single-group
entry point.
"""
from __future__ import annotations

import numpy as np

from core.fitting import fit_deterministic
from core.parameters import ModelParameters, Component, Parameter, ParameterTie

from absorption.absorption_model import (
    build_absorption_parameters, make_absorption_model_func,
    AbsorptionSystem, build_joint_absorption_parameters, make_joint_absorption_model_func,
    ThermalTurbulentLink, AbundancePatternGroup,
)
from absorption.absorption_results import AbsorptionFitResult, summarize_absorption_fit
from absorption.atomic import load_atomic_line_list, select_group

__all__ = ["fit_absorption_spectrum", "fit_joint_absorption_spectrum"]


def _get(config, key, default=None):
    return config.get(key, default) if hasattr(config, "get") else default


def _pair(config, key, default):
    value = _get(config, key, default)
    if isinstance(value, (list, tuple, np.ndarray)):
        vals = list(value)
    else:
        vals = [x.strip() for x in str(value).split(",")]
    if len(vals) < 2:
        return tuple(map(float, default))
    return float(vals[0]), float(vals[1])


def _bool(config, key, default=False):
    value = _get(config, key, default)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y", "on"}


def fit_absorption_spectrum(spectrum, config, *, transitions=None) -> AbsorptionFitResult:
    """Fit a continuum-normalized spectrum with N velocity components.

    By default the fit assumes the absorbing gas fully covers the
    background continuum source. Setting ``absorption_partial_coverage``
    (config) to true switches on the partial-covering model, in which
    the covering fraction becomes an additional fit variable shared
    across every velocity component -- see
    ``absorption.absorption_model.build_absorption_parameters``.

    Parameters
    ----------
    spectrum : core.spectrum.Spectrum
        Must have ``flux_unc`` set and should already be
        continuum-normalized (flux ~ 1 in unabsorbed regions), since the
        absorption model itself has no separate continuum term.
    config : object supporting ``.get(key, default)``
        Recognized keys: ``absorption_line_list_path``,
        ``absorption_group`` (which transition group to fit, e.g.
        ``"CIV"``), ``absorption_n_components``,
        ``absorption_logN_initial``/``absorption_logN_bounds``,
        ``absorption_b_initial_kms``/``absorption_b_bounds_kms``,
        ``absorption_velocity_initial_kms``/``absorption_velocity_bounds_kms``,
        ``absorption_partial_coverage`` (bool, default False),
        ``absorption_covering_fraction_initial``/``absorption_covering_fraction_bounds``,
        ``absorption_subpixel`` (int, default 1; oversampling factor for
        narrow-line profile evaluation, see
        ``absorption.absorption_model.make_absorption_model_func``),
        ``absorption_minimum_fit_pixels``, ``absorption_max_function_evaluations``.
    transitions : list[absorption.atomic.AtomicTransition], optional
        Pre-loaded/pre-selected transition group, bypassing
        ``absorption_line_list_path``/``absorption_group``.

    Returns
    -------
    AbsorptionFitResult
    """
    if spectrum.flux_unc is None:
        raise ValueError("Absorption-line chi-square fitting requires flux_unc.")

    if transitions is None:
        path = _get(config, "absorption_line_list_path", None)
        line_list = load_atomic_line_list(None if not path else path)
        group = _get(config, "absorption_group", None)
        if not group:
            raise ValueError(
                "absorption_group must be set (e.g. 'CIV') to select which transitions to fit, "
                "or pass a pre-selected `transitions` list directly."
            )
        transitions = select_group(line_list, group)

    n_components = int(_get(config, "absorption_n_components", 1))
    partial_coverage = _bool(config, "absorption_partial_coverage", False)

    model_parameters = build_absorption_parameters(
        n_components,
        logN_initial=float(_get(config, "absorption_logN_initial", 14.0)),
        logN_bounds=_pair(config, "absorption_logN_bounds", (8.0, 23.0)),
        b_initial_kms=float(_get(config, "absorption_b_initial_kms", 25.0)),
        b_bounds_kms=_pair(config, "absorption_b_bounds_kms", (1.0, 300.0)),
        velocity_initial_kms=float(_get(config, "absorption_velocity_initial_kms", 0.0)),
        velocity_bounds_kms=_pair(config, "absorption_velocity_bounds_kms", (-500.0, 500.0)),
        partial_coverage=partial_coverage,
        covering_fraction_initial=float(_get(config, "absorption_covering_fraction_initial", 1.0)),
        covering_fraction_bounds=_pair(config, "absorption_covering_fraction_bounds", (0.0, 1.0)),
    )

    wave = np.asarray(spectrum.wave, dtype=float)
    flux = np.asarray(spectrum.flux, dtype=float)
    flux_unc = np.asarray(spectrum.flux_unc, dtype=float)
    mask = spectrum.mask
    n_valid = int(np.count_nonzero(mask)) if mask is not None else wave.size
    minimum_pixels = int(_get(config, "absorption_minimum_fit_pixels", 10))
    if n_valid < minimum_pixels:
        raise ValueError(f"Too few valid pixels ({n_valid}) for absorption fitting (minimum {minimum_pixels}).")

    model_func = make_absorption_model_func(
        transitions, redshift=spectrum.redshift, resolution=spectrum.resolution, partial_coverage=partial_coverage,
        subpixel=int(_get(config, "absorption_subpixel", 1)),
    )
    resolution_source = None if spectrum.resolution is None else getattr(spectrum.resolution, "source", str(spectrum.resolution))

    fit_result = fit_deterministic(
        wave, flux, flux_unc, model_parameters, model_func,
        mask=mask, redshift=spectrum.redshift, resolution_source=resolution_source,
        max_function_evaluations=int(_get(config, "absorption_max_function_evaluations", 20000)),
    )
    fit_result.metadata.update({
        "absorption_fit_mode": "single",
        "absorption_partial_coverage": bool(partial_coverage),
        "absorption_subpixel": int(_get(config, "absorption_subpixel", 1)),
        "absorption_component_transition_names": [
            [transition.name for transition in transitions]
            for _ in range(model_parameters.n_components)
        ],
    })

    return summarize_absorption_fit(fit_result, transitions, partial_coverage=partial_coverage)


def fit_joint_absorption_spectrum(
    spectrum, systems: "list[AbsorptionSystem]", *,
    ties: "list[ParameterTie] | None" = None,
    thermal_turbulent_links: "list[ThermalTurbulentLink] | None" = None,
    abundance_pattern_groups: "list[AbundancePatternGroup] | None" = None,
    partial_coverage: bool = False,
    covering_fraction_initial: float = 1.0, covering_fraction_bounds=(0.0, 1.0),
    subpixel: int = 1,
    minimum_fit_pixels: int = 10, max_function_evaluations: int = 20000,
) -> AbsorptionFitResult:
    """Fit several transition groups (e.g. several ions) simultaneously, with optional cross-system linking.

    Component indices for building ``ties``/``thermal_turbulent_links``/
    ``abundance_pattern_groups`` follow ``systems`` order: system 0
    contributes components ``0 .. systems[0].n_components - 1``, system
    1 the next block, and so on -- exactly the numbering
    ``build_joint_absorption_parameters`` establishes.

    For an ``AbundancePatternGroup``, the caller must add its
    ``ratio_holder`` Parameter to the built ``ModelParameters`` before
    fitting and mark every follower component's ``logN`` as
    ``fixed=True`` (since ``apply_abundance_pattern_ties`` overwrites
    it every evaluation); this function does that bookkeeping
    automatically from ``abundance_pattern_groups`` for the common case
    of a single free ``"logN_ratio"`` parameter per group, appended to
    the first follower component.

    Parameters
    ----------
    spectrum : core.spectrum.Spectrum
        Shared by every system in the joint fit (same wave/flux/mask/
        redshift/resolution) -- this is what "joint" means here; fitting
        several *separate* spectra simultaneously is out of scope (see
        the module-level architecture notes).
    systems : list[AbsorptionSystem]

    Returns
    -------
    AbsorptionFitResult
        ``measurements`` are in joint component order, each tagged with
        its originating system's ``label``.
    """
    if spectrum.flux_unc is None:
        raise ValueError("Absorption-line chi-square fitting requires flux_unc.")
    if not systems:
        raise ValueError("systems must be non-empty.")

    model_parameters, component_transitions = build_joint_absorption_parameters(
        systems, partial_coverage=partial_coverage,
        covering_fraction_initial=covering_fraction_initial, covering_fraction_bounds=covering_fraction_bounds,
    )

    # A tied follower is not an independent degree of freedom. Enforce that
    # invariant here so callers cannot accidentally leave a follower free and
    # make the deterministic covariance / posterior dimensionality singular.
    for tie in (ties or []):
        follower_component, follower_name = tie.follower
        model_parameters.components[follower_component][follower_name].fixed = True
    for link in (thermal_turbulent_links or []):
        follower_component, follower_name = link.follower
        model_parameters.components[follower_component][follower_name].fixed = True

    # Component index offsets per system, for building the label list and for
    # automatically wiring up AbundancePatternGroup ratio-holder bookkeeping.
    offsets = []
    running = 0
    for system in systems:
        offsets.append(running)
        running += system.n_components
    system_labels = []
    for system in systems:
        system_labels.extend([system.label] * system.n_components)

    for group in (abundance_pattern_groups or []):
        follower_component_0 = group.follower_components[0]
        component = model_parameters.components[follower_component_0]
        ratio_name = group.ratio_holder[1]
        if ratio_name not in component:
            component.parameters.append(Parameter(ratio_name, 0.0, -5.0, 5.0))
        for follower_index in group.follower_components:
            model_parameters.components[follower_index]["logN"].fixed = True

    wave = np.asarray(spectrum.wave, dtype=float)
    flux = np.asarray(spectrum.flux, dtype=float)
    flux_unc = np.asarray(spectrum.flux_unc, dtype=float)
    mask = spectrum.mask
    n_valid = int(np.count_nonzero(mask)) if mask is not None else wave.size
    if n_valid < minimum_fit_pixels:
        raise ValueError(f"Too few valid pixels ({n_valid}) for absorption fitting (minimum {minimum_fit_pixels}).")

    model_func = make_joint_absorption_model_func(
        component_transitions, redshift=spectrum.redshift, resolution=spectrum.resolution,
        partial_coverage=partial_coverage, ties=ties, thermal_turbulent_links=thermal_turbulent_links,
        subpixel=subpixel,
    )
    # apply_abundance_pattern_ties is not part of make_joint_absorption_model_func's
    # own tie application (it needs the *group* objects, not just Parameter pairs),
    # so wrap the returned model_func to layer it in.
    from absorption.absorption_model import apply_abundance_pattern_ties
    if abundance_pattern_groups:
        inner_model_func = model_func

        def model_func(wave, model_parameters, _inner=inner_model_func, _groups=abundance_pattern_groups):
            apply_abundance_pattern_ties(model_parameters, _groups)
            return _inner(wave, model_parameters)

    resolution_source = None if spectrum.resolution is None else getattr(spectrum.resolution, "source", str(spectrum.resolution))

    fit_result = fit_deterministic(
        wave, flux, flux_unc, model_parameters, model_func,
        mask=mask, redshift=spectrum.redshift, resolution_source=resolution_source,
        max_function_evaluations=max_function_evaluations,
    )
    fit_result.metadata.update({
        "absorption_fit_mode": "joint",
        "absorption_partial_coverage": bool(partial_coverage),
        "absorption_subpixel": int(subpixel),
        "absorption_component_transition_names": [
            [transition.name for transition in transitions]
            for transitions in component_transitions
        ],
        "absorption_system_labels": list(system_labels),
        "absorption_has_parameter_ties": bool(ties),
        "absorption_has_thermal_turbulent_links": bool(thermal_turbulent_links),
        "absorption_has_abundance_pattern_groups": bool(abundance_pattern_groups),
    })

    all_transitions = []
    seen_names = set()
    for system in systems:
        for transition in system.transitions:
            if transition.name not in seen_names:
                all_transitions.append(transition)
                seen_names.add(transition.name)

    return summarize_absorption_fit(
        fit_result, all_transitions, partial_coverage=partial_coverage, system_labels=system_labels,
    )
