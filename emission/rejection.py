"""Automatic freezing of insignificant kinematic components.

The emission analogue of ``absorption.rejection`` (see that module's
docstring for the full VPFIT-``dropmode``-inspired rationale): fit,
identify components the data don't actually require, and -- rather than
deleting them, which would change the returned component count --
freeze each one at its current (already negligible) amplitude,
excluding it from further optimization, then refit the remaining free
components. Repeated until nothing more needs freezing or a pass limit
is reached.

Freezing instead of deleting means:

* the returned fit always has exactly the number of components the
  caller asked for;
* a frozen component's contribution to the model is essentially
  unaffected, since it was already negligible at the point it was
  frozen;
* this constrains a fit in both directions at once: too many components
  requested -> the extra ones freeze out at a negligible amplitude
  instead of fighting over the same real signal; and it's the same
  mechanism ``emission.emission_fit.fit_emission_spectrum`` reuses for
  each candidate N in automatic BIC component-count search, so a
  spuriously-large N is penalized doubly -- both by BIC's parameter
  penalty and by however many of its components immediately freeze out
  as insignificant.

Where absorption tests a single quantity per component (``logN``),
a component here can carry several free lines' amplitudes at once
(everything sharing that component's kinematics) -- so a component is
judged insignificant only if *every* line assigned to it is
individually insignificant; if even one line shows a real detection,
the whole component (and thus every line it carries, weak ones
included) stays free.
"""
from __future__ import annotations

import numpy as np

from core.fitting import fit_deterministic

from emission.emission_model import amplitude_parameter_name, velocity_parameter_name, sigma_parameter_name

__all__ = ["fit_with_rejection"]


def _line_insignificant(parameter, uncertainty, *, margin_fraction, snr_threshold) -> bool:
    value = parameter.value
    upper = parameter.upper
    near_floor = (value <= margin_fraction * upper) if (np.isfinite(upper) and upper > 0) else (value <= 0.0)
    if near_floor:
        return True
    snr = (value / uncertainty) if (np.isfinite(uncertainty) and uncertainty > 0) else 0.0
    return snr < snr_threshold


def fit_with_rejection(
    model_parameters, model_func, *, wave, flux, flux_unc, mask, redshift, resolution_source,
    max_function_evaluations, free_line_names, snr_threshold=3.0, margin_fraction=0.01, max_passes=3,
):
    """Fit ``model_parameters``, freezing insignificant components between
    passes, using the generic ``core.fitting.fit_deterministic`` engine.

    A component is frozen if every line in ``free_line_names`` has, in
    that component, either an amplitude within ``margin_fraction`` of its
    upper bound's floor at 0 (the optimizer pushed it to essentially
    nothing) or a detection significance (amplitude / uncertainty) below
    ``snr_threshold`` (including a non-finite uncertainty -- the fit
    found nowhere meaningful to put it). Once frozen, a component's
    ``velocity_kms``, ``sigma_kms``, and every line's amplitude/per-line
    kinematics override are held fixed at their current (negligible)
    values for every subsequent pass. A component's *previous*-pass
    fitted values are always carried forward as the next pass's starting
    point, so freezing only removes degrees of freedom without
    disturbing the components that remain free. At least one component
    is always left free, even if every one of them would otherwise be
    judged insignificant -- the single most significant (highest total
    amplitude across its lines) is kept.

    Returns
    -------
    (FitResult, list[bool])
        The final pass's result, and which of ``model_parameters``'
        components ended up frozen, in component order.
    """
    n_components = model_parameters.n_components
    frozen_flags = [False] * n_components

    fit_result = None
    for _ in range(max(1, max_passes)):
        fit_result = fit_deterministic(
            wave, flux, flux_unc, model_parameters, model_func,
            mask=mask, redshift=redshift, resolution_source=resolution_source,
            max_function_evaluations=max_function_evaluations,
        )

        already_free = [i for i in range(n_components) if not frozen_flags[i]]
        if len(already_free) <= 1:
            break  # never freeze the last remaining free component

        def _component_insignificant(index):
            component = fit_result.parameters.components[index]
            for name in free_line_names:
                parameter = component[amplitude_parameter_name(name)]
                uncertainty = fit_result.parameter_uncertainties.get(
                    f"c{index}_{amplitude_parameter_name(name)}", np.nan,
                )
                if not _line_insignificant(
                    parameter, uncertainty, margin_fraction=margin_fraction, snr_threshold=snr_threshold,
                ):
                    return False  # at least one line is a real detection
            return True

        candidates = [i for i in already_free if _component_insignificant(i)]
        if len(candidates) == len(already_free):
            def _total_amplitude(index):
                component = fit_result.parameters.components[index]
                return sum(component[amplitude_parameter_name(name)].value for name in free_line_names)
            best = max(already_free, key=_total_amplitude)
            candidates = [i for i in candidates if i != best]

        if not candidates:
            break

        for index in candidates:
            component = fit_result.parameters.components[index]
            component["velocity_kms"].fixed = True
            component["sigma_kms"].fixed = True
            for name in free_line_names:
                component[amplitude_parameter_name(name)].fixed = True
                # Also freeze any per-line kinematics override -- under
                # kinematics_mode == "free" these (not the component-level
                # velocity_kms/sigma_kms above) are the parameters the
                # optimizer actually varies, so freezing only the shared
                # ones would leave this "frozen" component's kinematics
                # still moving.
                component[velocity_parameter_name(name)].fixed = True
                component[sigma_parameter_name(name)].fixed = True

        frozen_flags = [old or (i in candidates) for i, old in enumerate(frozen_flags)]
        model_parameters = fit_result.parameters  # warm start: carry every value forward as-is

    fit_result.metadata["emission_frozen_components"] = list(frozen_flags)
    return fit_result, frozen_flags
