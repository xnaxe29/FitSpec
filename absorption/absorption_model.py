"""Multi-component, multi-system absorption-model construction and evaluation.

Per the "explicit state beats implicit inference" design principle, the
number of kinematic (velocity) components is always an explicit
``n_components`` on ``core.parameters.ModelParameters``. Each component
represents one absorbing "cloud": a single column density, Doppler
parameter, and velocity applied identically to every transition in its
group (all transitions of one ion/multiplet, since they share one
physical origin -- see ``absorption.atomic.select_group``).

Two entry points are provided:

* ``build_absorption_parameters``/``make_absorption_model_func`` -- the
  original single-group API: every component shares one transition
  list. Kept exactly as before for backward compatibility.
* ``AbsorptionSystem``/``build_joint_absorption_parameters``/
  ``make_joint_absorption_model_func`` -- fits several transition
  groups (e.g. several different ions) *simultaneously* in one
  nonlinear problem, each with its own component block, optionally
  linked to each other via ``core.parameters.ParameterTie`` (shared/tied
  redshifts, thermally-linked Doppler parameters, fixed-ratio or
  common-pattern column densities). This is what makes cross-ion tying
  possible; it does not require, and does not implement, joint fitting
  across separate spectra/files -- every system here still shares one
  wavelength/flux/mask array.

Partial covering (``absorption_partial_coverage``) is optional and off
by default (full coverage, i.e. the standard assumption that the
absorbing gas fully covers the background continuum source). When
enabled, one shared covering-fraction parameter is fit alongside the
column density/Doppler/velocity parameters -- shared across every
kinematic component (the covering geometry is a property of the
background source and absorber alignment, not of any one velocity
component), following the reference partial-covering formalism
``T_obs = (1 - C_f) + C_f * T_full``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from core.parameters import Component, ModelParameters, Parameter, ParameterTie, apply_ties
from core.resolution import ResolutionModel, convolve_variable_gaussian

from absorption.atomic import AtomicTransition
from absorption.profiles import apply_partial_covering, optical_depth_voigt

__all__ = [
    "COVERING_FRACTION_PARAMETER",
    "build_absorption_parameters", "absorption_component_optical_depth", "make_absorption_model_func",
    "AbsorptionSystem", "build_joint_absorption_parameters", "make_joint_absorption_model_func",
    "thermal_b_kms", "mass_scaled_b_transform", "ThermalTurbulentLink", "apply_thermal_turbulent_links",
    "fixed_log_ratio_transform",
    "AbundancePatternGroup", "apply_abundance_pattern_ties",
    "RegionShift", "ContinuumAdjustment", "apply_region_shifts", "apply_continuum_adjustments",
]

COVERING_FRACTION_PARAMETER = "covering_fraction"


# ---------------------------------------------------------------------------
# Single-group API (unchanged; kept for backward compatibility)
# ---------------------------------------------------------------------------

def build_absorption_parameters(
    n_components: int, *,
    logN_initial=14.0, logN_bounds=(8.0, 23.0),
    b_initial_kms=25.0, b_bounds_kms=(1.0, 300.0),
    velocity_initial_kms=0.0, velocity_bounds_kms=(-500.0, 500.0),
    partial_coverage: bool = False,
    covering_fraction_initial: float = 1.0, covering_fraction_bounds=(0.0, 1.0),
) -> ModelParameters:
    """Build one Component per kinematic (velocity) component.

    Each component holds ``logN`` (log10 column density [cm^-2]),
    ``b_kms`` (Doppler parameter), and ``velocity_kms``. If
    ``partial_coverage`` is True, component 0 additionally holds a free
    ``covering_fraction`` parameter; every other component gets the same
    parameter held ``fixed=True`` and synchronized to component 0's
    value at every model evaluation by ``make_absorption_model_func``,
    since one global covering fraction is the physically standard
    assumption.

    Parameters
    ----------
    logN_initial, logN_bounds : float, (float, float)
        Initial guess and bounds for log10(column density / cm^-2).
        Column density itself (not its log) is what enters the Voigt
        optical depth; fitting in log space is standard practice since
        N spans many orders of magnitude and is highly asymmetric in
        its uncertainty.
    partial_coverage : bool, default False
        If False (default), the model assumes full coverage of the
        background source (equivalent to fixing covering_fraction = 1
        without adding it as a free parameter at all). If True, the
        covering fraction becomes a fit variable per the description
        above.

    Note
    ----
    As with any Voigt-profile fit, a poor initial guess can converge to
    a spurious, very narrow, saturated local minimum near
    ``b_bounds_kms[0]`` rather than the true, broader solution
    (multi-transition groups are more prone to this than a single
    line). Supplying a reasonable ``b_initial_kms`` from the line's
    apparent width, and/or a physically-motivated ``b_bounds_kms``
    lower bound above the pixel-scale regime, avoids this -- exactly as
    a sensible starting guess is required for VPFIT itself.
    """
    if n_components < 1:
        raise ValueError("n_components must be >= 1.")
    logN_lo, logN_hi = map(float, logN_bounds)
    b_lo, b_hi = map(float, b_bounds_kms)
    v_lo, v_hi = map(float, velocity_bounds_kms)
    cf_lo, cf_hi = map(float, covering_fraction_bounds)
    if partial_coverage and not (0.0 <= covering_fraction_initial <= 1.0):
        raise ValueError("covering_fraction_initial must be in [0, 1].")

    components = []
    for component_index in range(n_components):
        parameters = [
            Parameter("logN", float(logN_initial), logN_lo, logN_hi),
            Parameter("b_kms", float(b_initial_kms), b_lo, b_hi),
            Parameter("velocity_kms", float(velocity_initial_kms), v_lo, v_hi),
        ]
        if partial_coverage:
            parameters.append(Parameter(
                COVERING_FRACTION_PARAMETER, float(covering_fraction_initial), cf_lo, cf_hi,
                fixed=(component_index > 0),
            ))
        components.append(Component(parameters=parameters))

    return ModelParameters(n_components=n_components, components=components)


def absorption_component_optical_depth(
    wave, *, logN, b_kms, velocity_kms, transitions: "list[AtomicTransition]", redshift: float = 0.0,
    **_ignored,
) -> np.ndarray:
    """One kinematic component's total optical depth: sum over every transition in its group.

    ``**_ignored`` swallows any parameter present on the component but
    not consumed here (``covering_fraction``, since covering is applied
    to the combined transmission, not per-component optical depth).
    """
    wave = np.asarray(wave, dtype=float)
    column_density_cm2 = 10.0 ** float(logN)
    total_tau = np.zeros_like(wave)
    for transition in transitions:
        total_tau = total_tau + optical_depth_voigt(
            wave, transition.rest_wavelength_angstrom, transition.oscillator_strength,
            transition.damping_constant_s, column_density_cm2, b_kms,
            velocity_kms=velocity_kms, redshift=redshift,
        )
    return total_tau


def make_absorption_model_func(
    transitions: "list[AtomicTransition]", *, redshift: float = 0.0,
    resolution: "ResolutionModel | None" = None, partial_coverage: bool = False,
    subpixel: int = 1,
) -> "Callable[[np.ndarray, ModelParameters], np.ndarray]":
    """Close over the fixed (non-fitted) context and return a bare model_func(wave, params).

    Evaluation order (physically required, not arbitrary): (1) sum every
    component's optical depth over every transition in the group and
    exponentiate to get the intrinsic, unconvolved, full-coverage
    transmission; (2) if partial coverage is enabled, mix with the
    unabsorbed continuum via ``absorption.profiles.apply_partial_covering``
    -- this mixing is a property of the source geometry and must happen
    before convolution; (3) convolve with the spectrum's instrumental
    resolution, which is a purely instrumental effect applied last. When
    ``partial_coverage`` is True, every component after the first has
    its ``covering_fraction`` value synchronized to component 0's
    current value before each evaluation.

    subpixel : int, default 1
        Evaluate the optical depth on a grid oversampled by this factor
        before convolving/resampling back onto ``wave`` (see
        ``_subpixel_grid``/``_resample_to_data``). 1 disables
        oversampling (evaluate directly on ``wave``, the previous
        behavior). Use > 1 for lines narrow relative to the pixel scale,
        where evaluating directly on the data grid under-resolves the
        profile.

    The returned callable matches the signature required by
    ``core.fitting.fit_deterministic``.
    """
    def model_func(wave, model_parameters: ModelParameters) -> np.ndarray:
        wave = np.asarray(wave, dtype=float)

        if partial_coverage and model_parameters.n_components > 1:
            leader = model_parameters.components[0]
            for component in model_parameters.components[1:]:
                component[COVERING_FRACTION_PARAMETER].value = leader[COVERING_FRACTION_PARAMETER].value

        eval_wave = _subpixel_grid(wave, subpixel)
        total_tau = np.zeros_like(eval_wave)
        for component in model_parameters.components:
            parameter_values = {parameter.name: parameter.value for parameter in component.parameters}
            total_tau = total_tau + absorption_component_optical_depth(
                eval_wave, transitions=transitions, redshift=redshift, **parameter_values,
            )
        transmission = np.exp(-total_tau)

        if partial_coverage:
            covering_fraction = model_parameters.components[0][COVERING_FRACTION_PARAMETER].value
            transmission = apply_partial_covering(transmission, covering_fraction)

        if resolution is not None:
            sigma_angstrom = resolution.sigma_angstrom(eval_wave)
            transmission = convolve_variable_gaussian(eval_wave, transmission, eval_wave, sigma_angstrom)

        return _resample_to_data(eval_wave, transmission, wave, subpixel)

    return model_func


# ---------------------------------------------------------------------------
# Subpixel oversampling
# ---------------------------------------------------------------------------

def _subpixel_grid(wave: np.ndarray, subpixel: int) -> np.ndarray:
    """Insert ``subpixel - 1`` evenly-spaced points between every pair of
    input wavelengths (linear-in-wavelength oversampling of a possibly
    non-uniform grid), so narrow lines are resolved before convolution
    and rebinning back onto the data grid. ``subpixel <= 1`` returns
    ``wave`` unchanged.
    """
    if subpixel <= 1 or wave.size < 2:
        return wave
    fine = np.linspace(wave[:-1], wave[1:], subpixel, endpoint=False, axis=1).ravel()
    return np.concatenate([fine, wave[-1:]])


def _resample_to_data(eval_wave: np.ndarray, eval_flux: np.ndarray, wave: np.ndarray, subpixel: int) -> np.ndarray:
    """Undo ``_subpixel_grid``: average the oversampled evaluation back
    onto the original data pixels (box-average per data pixel, the
    standard "evaluate fine, integrate/average to coarse" resampling for
    subpixel model evaluation). No-op if no oversampling was applied.
    """
    if subpixel <= 1 or wave.size < 2:
        return eval_flux
    return np.interp(wave, eval_wave, eval_flux)


# ---------------------------------------------------------------------------
# Multi-system (cross-ion) joint fitting
# ---------------------------------------------------------------------------

@dataclass
class AbsorptionSystem:
    """One transition group's kinematic-component block within a joint fit.

    Attributes
    ----------
    transitions : list[AtomicTransition]
        The transition group this system's components apply to (see
        ``absorption.atomic.select_group``).
    n_components : int
        Number of kinematic components for this system.
    label : str
        Human-readable identifier (e.g. the group name), used only for
        error messages and result bookkeeping.
    """

    transitions: "list[AtomicTransition]"
    n_components: int = 1
    label: str = ""
    logN_initial: float = 14.0
    logN_bounds: "tuple[float, float]" = (8.0, 23.0)
    b_initial_kms: float = 25.0
    b_bounds_kms: "tuple[float, float]" = (1.0, 300.0)
    velocity_initial_kms: float = 0.0
    velocity_bounds_kms: "tuple[float, float]" = (-500.0, 500.0)

    def __post_init__(self):
        if self.n_components < 1:
            raise ValueError("n_components must be >= 1.")
        if not self.label:
            groups = {transition.group for transition in self.transitions}
            self.label = "+".join(sorted(groups)) if groups else "system"


def build_joint_absorption_parameters(
    systems: "list[AbsorptionSystem]", *,
    partial_coverage: bool = False,
    covering_fraction_initial: float = 1.0, covering_fraction_bounds=(0.0, 1.0),
) -> "tuple[ModelParameters, list[list[AtomicTransition]]]":
    """Concatenate several systems' kinematic components into one joint ModelParameters.

    Returns ``(model_parameters, component_transitions)`` where
    ``component_transitions[i]`` is the transition list that applies to
    ``model_parameters.components[i]`` -- i.e. which system that
    (flattened, joint-numbered) component belongs to. Component indices
    are assigned in the order ``systems`` are given, each system
    contributing ``system.n_components`` consecutive components; use
    this numbering when building ``ParameterTie``/``ThermalTurbulentLink``
    entries that reference specific components.

    Cross-system ties (shared redshift, thermally-linked b, fixed or
    common-pattern column-density ratios) are NOT applied here -- build
    them separately (see ``core.parameters.ParameterTie``,
    ``ThermalTurbulentLink``, ``AbundancePatternGroup``,
    ``fixed_log_ratio_transform``) referencing the joint component
    indices this function establishes, and pass them to
    ``make_joint_absorption_model_func``.
    """
    if not systems:
        raise ValueError("systems must be non-empty.")
    cf_lo, cf_hi = map(float, covering_fraction_bounds)

    components: "list[Component]" = []
    component_transitions: "list[list[AtomicTransition]]" = []
    for system in systems:
        logN_lo, logN_hi = map(float, system.logN_bounds)
        b_lo, b_hi = map(float, system.b_bounds_kms)
        v_lo, v_hi = map(float, system.velocity_bounds_kms)
        for _ in range(system.n_components):
            parameters = [
                Parameter("logN", float(system.logN_initial), logN_lo, logN_hi),
                Parameter("b_kms", float(system.b_initial_kms), b_lo, b_hi),
                Parameter("velocity_kms", float(system.velocity_initial_kms), v_lo, v_hi),
            ]
            components.append(Component(parameters=parameters))
            component_transitions.append(system.transitions)

    if partial_coverage:
        components[0].parameters.append(
            Parameter(COVERING_FRACTION_PARAMETER, float(covering_fraction_initial), cf_lo, cf_hi)
        )
        for component in components[1:]:
            component.parameters.append(
                Parameter(COVERING_FRACTION_PARAMETER, float(covering_fraction_initial), cf_lo, cf_hi, fixed=True)
            )

    model_parameters = ModelParameters(n_components=len(components), components=components)
    return model_parameters, component_transitions


def make_joint_absorption_model_func(
    component_transitions: "list[list[AtomicTransition]]", *, redshift: float = 0.0,
    resolution: "ResolutionModel | None" = None, partial_coverage: bool = False,
    ties: "list[ParameterTie] | None" = None,
    thermal_turbulent_links: "list[ThermalTurbulentLink] | None" = None,
    subpixel: int = 1,
) -> "Callable[[np.ndarray, ModelParameters], np.ndarray]":
    """Like ``make_absorption_model_func``, but each component uses its own
    transition list (``component_transitions[i]``) and, before the model
    is evaluated, every requested tie is applied in order: ``ties``
    first (generic value ties -- shared redshift, fixed-ratio or
    common-pattern column densities), then ``thermal_turbulent_links``
    (mass-scaled Doppler-parameter linking, which needs its own
    mechanism since it combines two source values -- see
    ``ThermalTurbulentLink``), then the covering-fraction sync if
    partial coverage is enabled.
    """
    def model_func(wave, model_parameters: ModelParameters) -> np.ndarray:
        wave = np.asarray(wave, dtype=float)
        apply_ties(model_parameters, ties)
        apply_thermal_turbulent_links(model_parameters, thermal_turbulent_links)

        if partial_coverage and model_parameters.n_components > 1:
            leader = model_parameters.components[0]
            for component in model_parameters.components[1:]:
                if COVERING_FRACTION_PARAMETER in component:
                    component[COVERING_FRACTION_PARAMETER].value = leader[COVERING_FRACTION_PARAMETER].value

        eval_wave = _subpixel_grid(wave, subpixel)
        total_tau = np.zeros_like(eval_wave)
        for component, transitions in zip(model_parameters.components, component_transitions):
            parameter_values = {parameter.name: parameter.value for parameter in component.parameters}
            total_tau = total_tau + absorption_component_optical_depth(
                eval_wave, transitions=transitions, redshift=redshift, **parameter_values,
            )
        transmission = np.exp(-total_tau)

        if partial_coverage:
            covering_fraction = model_parameters.components[0][COVERING_FRACTION_PARAMETER].value
            transmission = apply_partial_covering(transmission, covering_fraction)

        if resolution is not None:
            sigma_angstrom = resolution.sigma_angstrom(eval_wave)
            transmission = convolve_variable_gaussian(eval_wave, transmission, eval_wave, sigma_angstrom)

        return _resample_to_data(eval_wave, transmission, wave, subpixel)

    return model_func


# ---------------------------------------------------------------------------
# Thermal / turbulent Doppler-parameter linking
# ---------------------------------------------------------------------------

# VPFIT's own constant relating temperature and thermal Doppler width:
# b_thermal = 12.85 * sqrt(T / 1e4 K / m[amu]) km/s.
_THERMAL_B_COEFFICIENT_KMS = 12.85


def thermal_b_kms(temperature_K: float, mass_amu: float) -> float:
    """Pure-thermal Doppler parameter for a Maxwellian at ``temperature_K`` [K]."""
    if temperature_K < 0:
        raise ValueError("temperature_K must be non-negative.")
    if mass_amu <= 0:
        raise ValueError("mass_amu must be positive.")
    return _THERMAL_B_COEFFICIENT_KMS * np.sqrt(float(temperature_K) / 1.0e4 / float(mass_amu))


def mass_scaled_b_transform(leader_mass_amu: float, follower_mass_amu: float) -> "Callable[[float], float]":
    """``ParameterTie.transform`` for a purely-thermal (no turbulent component) b-link.

    follower_b = leader_b * sqrt(leader_mass / follower_mass) -- the
    lighter ion has the larger Doppler parameter for a shared
    temperature. Use this directly as a ``ParameterTie.transform`` when
    you want two ions' Doppler parameters locked to a common
    temperature with zero turbulent broadening (VPFIT's "temperature
    estimation" mode collapsed to the fully-thermal limit).
    """
    if leader_mass_amu <= 0 or follower_mass_amu <= 0:
        raise ValueError("Both masses must be positive.")
    ratio = np.sqrt(float(leader_mass_amu) / float(follower_mass_amu))
    return lambda leader_b_kms: float(leader_b_kms) * ratio


@dataclass
class ThermalTurbulentLink:
    """Link a follower component's Doppler parameter to a shared turbulent
    velocity plus a mass-scaled thermal contribution at a fixed temperature.

    ``b_follower = sqrt(b_turbulent_kms**2 + thermal_b_kms(temperature_K, follower_mass_amu)**2)``

    Unlike ``ParameterTie`` (one source parameter, one transform), this
    combines two independent physical inputs -- a shared turbulent
    velocity and a shared temperature -- so it is applied via its own
    ``apply_thermal_turbulent_links`` rather than the generic
    single-leader tie mechanism.

    Attributes
    ----------
    follower : (int, str)
        ``(component_index, parameter_name)`` of the b_kms parameter
        this computes (must already be ``fixed=True``).
    turbulent_kms : float
        Assumed shared turbulent Doppler-parameter contribution [km/s].
        0.0 for a purely thermal link.
    temperature_K : float
        Assumed shared temperature [K]. 0.0 for a purely turbulent link
        (follower_b = turbulent_kms exactly, mass-independent).
    follower_mass_amu : float
        Atomic/molecular mass of the follower's species [amu] (see
        ``AtomicTransition.atomic_mass_amu``).
    """

    follower: "tuple[int, str]"
    turbulent_kms: float
    temperature_K: float
    follower_mass_amu: float


def apply_thermal_turbulent_links(model_parameters: ModelParameters, links: "list[ThermalTurbulentLink] | None") -> None:
    """Set every linked follower's b_kms from its assumed turbulent+thermal budget. No-op if ``links`` is falsy."""
    if not links:
        return
    for link in links:
        component_index, parameter_name = link.follower
        thermal_component = thermal_b_kms(link.temperature_K, link.follower_mass_amu) if link.temperature_K > 0 else 0.0
        b_value = float(np.hypot(link.turbulent_kms, thermal_component))
        model_parameters.components[component_index][parameter_name].value = b_value


# ---------------------------------------------------------------------------
# Column-density ratio ties (isotope/molecular patterns, common abundance patterns)
# ---------------------------------------------------------------------------

def fixed_log_ratio_transform(delta_log10: float) -> "Callable[[float], float]":
    """``ParameterTie.transform`` for a fixed column-density ratio in log space.

    follower_logN = leader_logN + delta_log10, i.e. a fixed multiplicative
    ratio N_follower/N_leader = 10**delta_log10 -- e.g. a fixed isotope
    ratio, or an assumed fixed molecular level-population pattern
    (VPFIT's ``vp_abund.dat`` mechanism).
    """
    offset = float(delta_log10)
    return lambda leader_logN: float(leader_logN) + offset


@dataclass
class AbundancePatternGroup:
    """Tie a whole block of a follower ion's per-component logN values to a
    leader ion's per-component logN values via one shared, *free* ratio.

    Unlike ``fixed_log_ratio_transform`` (a caller-chosen constant),
    here the ratio itself is a fit parameter: every follower component's
    logN is forced to track ``leader_logN[i] + logN_ratio``, and
    ``logN_ratio`` is free to vary, all followers sharing the same
    value -- VPFIT's "common pattern relative ion abundances": the
    velocity-component-by-component pattern is locked between two ions,
    but their overall relative normalization is fit.

    Attributes
    ----------
    leader_components, follower_components : list[int]
        Paired, same-length, same-order lists of joint component
        indices: ``follower_components[i]``'s logN tracks
        ``leader_components[i]``'s logN plus the shared ratio.
    ratio_holder : (int, str)
        ``(component_index, parameter_name)`` of a genuine free
        Parameter (e.g. named ``"logN_ratio"``) holding the shared,
        fitted log10 ratio. The caller is responsible for having added
        this Parameter to ``model_parameters`` (typically appended to
        one of the follower components) before fitting.
    """

    leader_components: "list[int]"
    follower_components: "list[int]"
    ratio_holder: "tuple[int, str]"

    def __post_init__(self):
        if len(self.leader_components) != len(self.follower_components):
            raise ValueError("leader_components and follower_components must be the same length.")
        if not self.leader_components:
            raise ValueError("leader_components/follower_components must be non-empty.")


def apply_abundance_pattern_ties(model_parameters: ModelParameters, groups: "list[AbundancePatternGroup] | None") -> None:
    """Set every follower component's logN from its paired leader plus the shared ratio. No-op if ``groups`` is falsy."""
    if not groups:
        return
    for group in groups:
        ratio_component, ratio_name = group.ratio_holder
        ratio = model_parameters.components[ratio_component][ratio_name].value
        for leader_index, follower_index in zip(group.leader_components, group.follower_components):
            leader_logN = model_parameters.components[leader_index]["logN"].value
            model_parameters.components[follower_index]["logN"].value = leader_logN + ratio


# ---------------------------------------------------------------------------
# Region velocity shift and in-fit continuum adjustment (nuisance parameters)
# ---------------------------------------------------------------------------

@dataclass
class RegionShift:
    """A small, fittable velocity offset applied to one wavelength sub-range.

    Reproduces VPFIT's ">>" region-wavelength-shift mechanism: models a
    possible wavelength-calibration mismatch between two parts of a
    spectrum (e.g. two orders/exposures, or two members of a doublet
    whose relative rest wavelengths are uncertain) as a fittable
    velocity shift applied only within ``[wave_min, wave_max]``.

    Attributes
    ----------
    wave_min, wave_max : float
        Observed-frame wavelength range this shift applies to.
    parameter : Parameter
        The free (or fixed) shift parameter itself, in km/s. Not stored
        inside any ``Component`` -- ``RegionShift`` objects are passed
        directly to ``apply_region_shifts``, which reads
        ``parameter.value`` each evaluation.
    """

    wave_min: float
    wave_max: float
    parameter: Parameter


def apply_region_shifts(wave: np.ndarray, shifts: "list[RegionShift] | None") -> np.ndarray:
    """Return an effective wavelength array with each RegionShift's Doppler
    shift applied only within its wavelength window. No-op if ``shifts``
    is falsy.
    """
    if not shifts:
        return wave
    from absorption.profiles import C_KMS
    shifted = np.array(wave, dtype=float, copy=True)
    for shift in shifts:
        in_region = (wave >= shift.wave_min) & (wave <= shift.wave_max)
        beta = shift.parameter.value / C_KMS
        shifted[in_region] = wave[in_region] * np.sqrt((1 + beta) / (1 - beta))
    return shifted


@dataclass
class ContinuumAdjustment:
    """A small, fittable multiplicative correction to the input continuum
    level (and optionally its slope) within one wavelength window.

    Reproduces VPFIT's "<>" continuum-adjustment mechanism: the model's
    transmission within ``[wave_min, wave_max]`` is multiplied by
    ``level.value + slope.value * (wave / reference_wavelength - 1)``,
    for cases where the pre-determined continuum in a crowded region
    (e.g. the Lyman-alpha forest) is suspected to be slightly off.
    Intended for small final adjustments, not as a substitute for
    proper continuum fitting (see the ``continuum`` module) -- a
    continuum-level parameter can trade off against a broad, shallow
    absorption feature if left too free.

    Attributes
    ----------
    wave_min, wave_max : float
        Observed-frame window this adjustment applies to.
    reference_wavelength : float
        Wavelength at which ``level`` applies exactly (the slope term
        vanishes there).
    level, slope : Parameter
        Multiplicative level (usually near 1) and, optionally, its
        linear wavelength dependence (usually fixed at 0 for a flat
        rescaling).
    """

    wave_min: float
    wave_max: float
    reference_wavelength: float
    level: Parameter
    slope: Parameter


def apply_continuum_adjustments(wave: np.ndarray, transmission: np.ndarray,
                                 adjustments: "list[ContinuumAdjustment] | None") -> np.ndarray:
    """Multiply ``transmission`` by each ContinuumAdjustment's level+slope
    correction within its wavelength window. No-op if ``adjustments`` is
    falsy.
    """
    if not adjustments:
        return transmission
    adjusted = np.array(transmission, dtype=float, copy=True)
    for adjustment in adjustments:
        in_region = (wave >= adjustment.wave_min) & (wave <= adjustment.wave_max)
        factor = adjustment.level.value + adjustment.slope.value * (
            wave[in_region] / adjustment.reference_wavelength - 1.0
        )
        adjusted[in_region] = adjusted[in_region] * factor
    return adjusted
