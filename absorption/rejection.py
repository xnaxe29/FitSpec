"""Automatic rejection of insignificant kinematic components.

A simplified analogue of VPFIT's ``dropmode`` (Section 11.7 of the
VPFIT manual): fit, identify components the data don't actually
require, and -- rather than deleting them, which would change the
returned component count -- freeze each one at its current (already
negligible) column density, excluding it from further optimization,
then refit the remaining free components. This is repeated until
nothing more needs freezing or a pass limit is reached.

Freezing instead of deleting means:

* the returned fit always has exactly the number of components the
  caller asked for;
* the summed column density across components is essentially
  unaffected, since a frozen component's contribution was already
  negligible at the point it was frozen;
* a frozen component's ``logN`` should be read as an upper limit on its
  true column density, not a detection (see
  ``AbsorptionComponentMeasurement.is_upper_limit``).

Scope: this operates independently per system (only within
``systems``, never across a cross-system tie), because renumbering an
arbitrary set of ``ParameterTie``/``ThermalTurbulentLink``/
``AbundancePatternGroup`` objects after a component's role changes is
not generally well-defined without knowing the caller's intent. Use
``fit_joint_absorption_spectrum`` directly (without rejection) for tied
joint fits.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from core.fitting import fit_deterministic

from absorption.absorption_model import (
    AbsorptionSystem, build_joint_absorption_parameters, make_joint_absorption_model_func,
)
from absorption.absorption_results import AbsorptionFitResult, summarize_absorption_fit

__all__ = ["fit_joint_absorption_spectrum_with_rejection"]


def _flatten_labels(systems: "list[AbsorptionSystem]") -> "list[str]":
    labels = []
    for system in systems:
        labels.extend([system.label] * system.n_components)
    return labels


def _system_offsets(systems: "list[AbsorptionSystem]") -> "list[int]":
    offsets = []
    running = 0
    for system in systems:
        offsets.append(running)
        running += system.n_components
    return offsets


def fit_joint_absorption_spectrum_with_rejection(
    spectrum, systems: "list[AbsorptionSystem]", *,
    reject_margin_dex: float = 0.1, reject_max_uncertainty_dex: float = 1.0,
    max_rejection_passes: int = 3,
    partial_coverage: bool = False,
    covering_fraction_initial: float = 1.0, covering_fraction_bounds=(0.0, 1.0),
    subpixel: int = 1, minimum_fit_pixels: int = 10, max_function_evaluations: int = 20000,
) -> AbsorptionFitResult:
    """Fit, freeze insignificant components at their current (negligible) value, refit.

    A component is frozen if either:

    * its fitted ``logN`` is within ``reject_margin_dex`` of that
      system's ``logN_bounds[0]`` -- the optimizer pushed it to the
      allowed floor, the clearest sign of no real detection there; or
    * its ``logN`` uncertainty is non-finite or larger than
      ``reject_max_uncertainty_dex`` -- the fit found nowhere
      meaningful to put it, so the covariance is degenerate even though
      the best-fit value itself didn't land exactly on the bound (a
      narrow, negligible-amplitude "phantom" component is the typical
      symptom).

    Once frozen, a component's ``logN``, ``b_kms``, and ``velocity_kms``
    are all held fixed at their current values (excluded from the free-
    parameter vector) for every subsequent pass, so it keeps
    contributing its already-negligible optical depth to the model
    without being re-optimized. Every *unfrozen* component's fitted
    value from the previous pass is carried forward as the next pass's
    starting point, so freezing only removes degrees of freedom without
    disturbing the fit of the components that remain free.

    A system's components are never all frozen at once: if every
    component in a system would be frozen, the single most significant
    one (highest logN) is kept free.

    Returns
    -------
    AbsorptionFitResult
        From the final pass, with exactly the number of components
        originally requested across ``systems``. Frozen components are
        marked ``AbsorptionComponentMeasurement.is_upper_limit = True``.
    """
    if spectrum.flux_unc is None:
        raise ValueError("Absorption-line chi-square fitting requires flux_unc.")
    if not systems:
        raise ValueError("systems must be non-empty.")

    current_systems = [replace(system) for system in systems]
    model_parameters, component_transitions = build_joint_absorption_parameters(
        current_systems, partial_coverage=partial_coverage,
        covering_fraction_initial=covering_fraction_initial, covering_fraction_bounds=covering_fraction_bounds,
    )
    frozen_flags = [False] * model_parameters.n_components

    wave = np.asarray(spectrum.wave, dtype=float)
    flux = np.asarray(spectrum.flux, dtype=float)
    flux_unc = np.asarray(spectrum.flux_unc, dtype=float)
    mask = spectrum.mask
    n_valid = int(np.count_nonzero(mask)) if mask is not None else wave.size
    if n_valid < minimum_fit_pixels:
        raise ValueError(f"Too few valid pixels ({n_valid}) for absorption fitting (minimum {minimum_fit_pixels}).")
    resolution_source = None if spectrum.resolution is None else getattr(spectrum.resolution, "source", str(spectrum.resolution))

    result = None
    for _ in range(max(1, max_rejection_passes)):
        model_func = make_joint_absorption_model_func(
            component_transitions, redshift=spectrum.redshift, resolution=spectrum.resolution,
            partial_coverage=partial_coverage, subpixel=subpixel,
        )
        fit_result = fit_deterministic(
            wave, flux, flux_unc, model_parameters, model_func,
            mask=mask, redshift=spectrum.redshift, resolution_source=resolution_source,
            max_function_evaluations=max_function_evaluations,
        )
        all_transitions = []
        seen_names = set()
        for transitions in component_transitions:
            for transition in transitions:
                if transition.name not in seen_names:
                    all_transitions.append(transition)
                    seen_names.add(transition.name)
        system_labels = _flatten_labels(current_systems)
        fit_result.metadata.update({
            "absorption_fit_mode": "rejection",
            "absorption_partial_coverage": bool(partial_coverage),
            "absorption_subpixel": int(subpixel),
            "absorption_component_transition_names": [
                [transition.name for transition in transitions]
                for transitions in component_transitions
            ],
            "absorption_system_labels": list(system_labels),
            "absorption_frozen_components": list(frozen_flags),
        })
        result = summarize_absorption_fit(
            fit_result, all_transitions, partial_coverage=partial_coverage,
            system_labels=system_labels, upper_limit_flags=frozen_flags,
        )

        offsets = _system_offsets(current_systems)
        newly_frozen = [False] * len(result.measurements)
        for system_index, system in enumerate(current_systems):
            start = offsets[system_index]
            floor = system.logN_bounds[0]
            local_indices = range(start, start + system.n_components)
            already_free = [i for i in local_indices if not frozen_flags[i]]
            if not already_free:
                continue  # every component in this system is already frozen

            def _insignificant(index):
                measurement = result.measurements[index]
                if measurement.logN <= floor + reject_margin_dex:
                    return True
                unc = measurement.logN_uncertainty
                return (not np.isfinite(unc)) or unc > reject_max_uncertainty_dex

            candidates = [i for i in already_free if _insignificant(i)]
            if len(candidates) == len(already_free):
                # Never freeze every free component in a system: keep the most significant one.
                best_index = max(already_free, key=lambda i: result.measurements[i].logN)
                candidates = [i for i in already_free if i != best_index]

            for index in candidates:
                newly_frozen[index] = True

        if not any(newly_frozen):
            break

        # Freeze: hold logN/b_kms/velocity_kms at their current fitted values,
        # excluded from the free-parameter vector, for every subsequent pass.
        for component, is_newly_frozen in zip(fit_result.parameters.components, newly_frozen):
            if is_newly_frozen:
                for parameter_name in ("logN", "b_kms", "velocity_kms"):
                    component[parameter_name].fixed = True

        frozen_flags = [old or new for old, new in zip(frozen_flags, newly_frozen)]
        model_parameters = fit_result.parameters  # warm start: carry every value forward as-is

    if result is not None:
        result.fit_result.metadata["absorption_frozen_components"] = list(frozen_flags)
    return result
