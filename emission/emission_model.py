"""Multi-component, multi-line emission-model construction and evaluation.

Per the FitSpec design principles, the number of kinematic components is
an explicit user/model property (``n_components`` on
``core.parameters.ModelParameters``), never inferred from whether a
config value happens to be scalar or list-valued.

``kinematics_mode`` controls which velocity/sigma values are free
parameters *in the deterministic fit* -- it has no bearing on what the
GUI lets you drag for preview (see ``gui.emission.EmissionGUI``, which
always leaves every slider draggable regardless of mode; mode only
matters once you click Fit):

* ``"fixed"`` -- one velocity, one sigma, for the whole model: no
  per-component or per-line variation at all. Neither the per-component
  nor the per-line kinematics parameters are ever fit.
* ``"tied"`` -- every line *within* a component shares that component's
  velocity/sigma (this is what "tied" ties -- lines to each other, not
  components to each other), but each component is independently fit.
  This is the natural default for well-separated kinematic systems
  (e.g. narrow vs. broad components) where every line in a given system
  moves together.
* ``"free"`` -- every line, in every component, is independently fit --
  no sharing at all, even within a component.

Lines declared as ratio-tied in the line list (see ``emission.lines``)
are, regardless of ``kinematics_mode``, always forced to a fixed
multiple of another line's amplitude (and follow that line's
kinematics too) and never get their own free parameters.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from core.parameters import Component, ModelParameters, Parameter
from core.models import sum_components
from core.resolution import ResolutionModel

from emission.lines import EmissionLine
from emission.profiles import doppler_shifted_wavelength, gaussian_line_flux, velocity_dispersion_to_angstrom

__all__ = [
    "amplitude_parameter_name", "velocity_parameter_name", "sigma_parameter_name",
    "build_emission_parameters", "emission_component_flux", "make_emission_model_func",
]


def amplitude_parameter_name(line_name: str) -> str:
    return f"amp_{line_name}"


def velocity_parameter_name(line_name: str) -> str:
    return f"velocity_kms_{line_name}"


def sigma_parameter_name(line_name: str) -> str:
    return f"sigma_kms_{line_name}"


def _free_lines(line_list: "list[EmissionLine]") -> "list[EmissionLine]":
    return [emission_line for emission_line in line_list if emission_line.tied_to is None]


def build_emission_parameters(
    line_list: "list[EmissionLine]", n_components: int, *,
    velocity_initial_kms=0.0, velocity_bounds_kms=(-500.0, 500.0),
    sigma_initial_kms=50.0, sigma_bounds_kms=(1.0, 1000.0),
    amplitude_initial=1.0, amplitude_bounds=(0.0, np.inf),
    kinematics_mode: str = "tied",
) -> ModelParameters:
    """Build one Component per kinematic component, each holding shared
    velocity/sigma parameters, one amplitude parameter per untied line,
    and one per-line velocity/sigma *override* parameter per untied line.

    All components share the same initial guesses/bounds; per-component
    customization (e.g. a narrower velocity range for a "broad" component)
    can be done by editing the returned ModelParameters' components in
    place before fitting.

    Parameters
    ----------
    kinematics_mode : {"fixed", "tied", "free"}
        Which velocity/sigma values are free parameters in the fit --
        see the module docstring for the full definitions. In short:
        ``"fixed"`` fits none of them (single global value), ``"tied"``
        fits one value per component (shared across that component's
        lines -- the default), ``"free"`` fits one value per line.

        The per-component "shared" velocity_kms/sigma_kms parameter is
        only itself a free fit parameter under ``"tied"``; under
        ``"fixed"``/``"free"`` it's held fixed (never optimized) but
        still exists, as the value every still-tracking line's
        per-line override numerically follows every evaluation (see
        ``make_emission_model_func``) and as the GUI's broadcast-write
        target when "Line specific" is unchecked.

        Per-line velocity/sigma override parameters (named via
        :func:`velocity_parameter_name`/:func:`sigma_parameter_name`)
        are only themselves free fit parameters under ``"free"``; under
        ``"fixed"``/``"tied"`` they're held fixed and numerically track
        their component's shared value every evaluation, contributing
        zero extra fit dimensionality.
    """
    if n_components < 1:
        raise ValueError("n_components must be >= 1.")
    if kinematics_mode not in ("fixed", "tied", "free"):
        raise ValueError("kinematics_mode must be 'fixed', 'tied', or 'free'.")
    free_lines = _free_lines(line_list)
    if not free_lines:
        raise ValueError("line_list has no untied lines to fit an amplitude for.")

    v_lo, v_hi = map(float, velocity_bounds_kms)
    s_lo, s_hi = map(float, sigma_bounds_kms)
    a_lo, a_hi = map(float, amplitude_bounds)

    # Only "tied" fits one shared value per component; "fixed" wants a
    # single global value (never fit), "free" wants one value per line
    # (the shared per-component value is then just a tracking source/
    # GUI broadcast target, not itself fit).
    component_kinematics_fixed = kinematics_mode in ("fixed", "free")
    # Only "free" gives genuine per-line freedom; both "fixed" and
    # "tied" have every line track its component's shared value.
    per_line_kinematics_fixed = kinematics_mode != "free"

    components = []
    for _component_index in range(n_components):
        parameters = [
            Parameter("velocity_kms", float(velocity_initial_kms), v_lo, v_hi, fixed=component_kinematics_fixed),
            Parameter("sigma_kms", float(sigma_initial_kms), s_lo, s_hi, fixed=component_kinematics_fixed),
        ]
        for emission_line in free_lines:
            parameters.append(
                Parameter(amplitude_parameter_name(emission_line.name), float(amplitude_initial), a_lo, a_hi)
            )
            parameters.append(
                Parameter(velocity_parameter_name(emission_line.name), float(velocity_initial_kms), v_lo, v_hi,
                           fixed=per_line_kinematics_fixed)
            )
            parameters.append(
                Parameter(sigma_parameter_name(emission_line.name), float(sigma_initial_kms), s_lo, s_hi,
                           fixed=per_line_kinematics_fixed)
            )
        components.append(Component(parameters=parameters))

    return ModelParameters(n_components=n_components, components=components)


def emission_component_flux(
    wave, *, velocity_kms, sigma_kms, line_list: "list[EmissionLine]",
    redshift: float = 0.0, resolution: "ResolutionModel | None" = None, **amplitudes,
) -> np.ndarray:
    """One kinematic component's total flux: sum of every line assigned to it.

    Called once per component by ``core.models.sum_components`` (via
    ``make_emission_model_func``), with ``velocity_kms``/``sigma_kms`` and
    every free amplitude/per-line kinematics-override of this component
    supplied as keyword arguments by name. Tied lines are resolved here
    from the tied-to line's amplitude value (and kinematics), which is
    available in ``amplitudes`` because tied-to lines are always untied
    (never themselves tied further).

    Each line's *effective* velocity/sigma is its per-line override
    value if present in ``amplitudes`` (by the time this runs, that
    value is already resolved -- either tracking the component's shared
    value, or an explicit divergence under ``kinematics_mode == "free"``
    -- by ``make_emission_model_func``), falling back to the
    component-shared ``velocity_kms``/``sigma_kms`` only if the override
    key is entirely absent (e.g. a caller-built ``ModelParameters`` that
    predates per-line overrides).
    """
    wave = np.asarray(wave, dtype=float)
    total = np.zeros_like(wave)
    for emission_line in line_list:
        if emission_line.tied_to is None:
            key = amplitude_parameter_name(emission_line.name)
            if key not in amplitudes:
                continue  # line not part of this fit's active set
            integrated_flux = amplitudes[key]
            kinematics_name = emission_line.name
        else:
            tied_key = amplitude_parameter_name(emission_line.tied_to)
            if tied_key not in amplitudes:
                continue
            integrated_flux = amplitudes[tied_key] * emission_line.ratio_to_tied
            kinematics_name = emission_line.tied_to

        line_velocity_kms = amplitudes.get(velocity_parameter_name(kinematics_name), velocity_kms)
        line_sigma_kms = amplitudes.get(sigma_parameter_name(kinematics_name), sigma_kms)

        center = doppler_shifted_wavelength(emission_line.rest_wavelength_angstrom, line_velocity_kms, redshift)
        sigma = velocity_dispersion_to_angstrom(center, line_sigma_kms, resolution)
        total = total + gaussian_line_flux(wave, integrated_flux, center, float(sigma))
    return total


def make_emission_model_func(
    line_list: "list[EmissionLine]", *, redshift: float = 0.0, resolution: "ResolutionModel | None" = None,
    kinematics_mode: str = "tied",
) -> "Callable[[np.ndarray, ModelParameters], np.ndarray]":
    """Close over the fixed (non-fitted) context and return a bare model_func(wave, params).

    The returned callable matches the signature required by
    ``core.fitting.fit_deterministic``. Every untied line's per-line
    velocity/sigma override is resynced to its own component's
    (possibly just-updated) shared value whenever that override is
    still ``fixed=True`` -- i.e. hasn't been explicitly diverged under
    ``kinematics_mode == "free"`` -- so a line with no per-line override
    always tracks its component's kinematics exactly, and only a
    genuinely free (``fixed=False``) line's override keeps its own
    independent value. There is deliberately no cross-*component*
    syncing here at all: unlike an earlier (incorrect) reading of
    "tied", nothing about ``kinematics_mode`` ever forces one
    component's kinematics to match another's -- "tied" ties lines to
    their own component, not components to each other. A component
    built without per-line overrides at all (e.g. a hand-built
    ``Component`` bypassing ``build_emission_parameters``) is left
    alone -- this is an additive refinement, not a requirement every
    caller's ``ModelParameters`` has to satisfy.
    """
    free_line_names = [emission_line.name for emission_line in line_list if emission_line.tied_to is None]

    def model_func(wave, model_parameters: ModelParameters) -> np.ndarray:
        for component in model_parameters.components:
            shared_velocity = component["velocity_kms"].value
            shared_sigma = component["sigma_kms"].value
            for line_name in free_line_names:
                velocity_name = velocity_parameter_name(line_name)
                if velocity_name in component:
                    velocity_override = component[velocity_name]
                    if velocity_override.fixed:
                        velocity_override.value = shared_velocity
                sigma_name = sigma_parameter_name(line_name)
                if sigma_name in component:
                    sigma_override = component[sigma_name]
                    if sigma_override.fixed:
                        sigma_override.value = shared_sigma
        return sum_components(
            wave, model_parameters, emission_component_flux,
            line_list=line_list, redshift=redshift, resolution=resolution,
        )
    return model_func
