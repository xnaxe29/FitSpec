"""FitSpec absorption-line fitting: independent science module.

Shares core infrastructure, masking, resolution, plotting, and
inference with the other science modules. By default assumes full
coverage of the background source; partial covering (a shared,
fittable covering-fraction parameter) is available via
``absorption_partial_coverage`` in config, or the ``partial_coverage``
argument to the lower-level builders.

Cross-ion/cross-system parameter tying (shared redshifts, thermally- or
turbulently-linked Doppler parameters, fixed-ratio or common-pattern
column densities), region velocity shifts, in-fit continuum
adjustments, subpixel profile oversampling, automatic tied-component
rejection, and column-density upper-limit estimation are all available
-- see ``absorption.absorption_model``, ``absorption.rejection``, and
``absorption.upper_limits``. Joint fitting across *separate* spectra/
files is out of scope; every system in a joint fit shares one spectrum.
"""
from absorption.atomic import AtomicTransition, load_atomic_line_list, select_group, list_groups
from absorption.profiles import voigt_hjerting, optical_depth_voigt, transmission_voigt, apply_partial_covering
from absorption.absorption_model import (
    COVERING_FRACTION_PARAMETER,
    build_absorption_parameters, absorption_component_optical_depth, make_absorption_model_func,
    AbsorptionSystem, build_joint_absorption_parameters, make_joint_absorption_model_func,
    thermal_b_kms, mass_scaled_b_transform, ThermalTurbulentLink, apply_thermal_turbulent_links,
    fixed_log_ratio_transform, AbundancePatternGroup, apply_abundance_pattern_ties,
    RegionShift, ContinuumAdjustment, apply_region_shifts, apply_continuum_adjustments,
)
from absorption.absorption_fit import fit_absorption_spectrum, fit_joint_absorption_spectrum
from absorption.absorption_results import (
    AbsorptionComponentMeasurement, AbsorptionFitResult,
    save_absorption_result, load_absorption_result, summarize_absorption_fit,
)
from absorption.rejection import fit_joint_absorption_spectrum_with_rejection
from absorption.upper_limits import estimate_column_density_upper_limit, ColumnDensityUpperLimit
from absorption.synthetic import generate_synthetic_absorption_spectrum
from absorption.inference import (
    build_absorption_prior_set, build_absorption_posterior_problem,
    build_joint_absorption_posterior_problem, run_absorption_inference,
)

__all__ = [
    "AtomicTransition", "load_atomic_line_list", "select_group", "list_groups",
    "voigt_hjerting", "optical_depth_voigt", "transmission_voigt", "apply_partial_covering",
    "COVERING_FRACTION_PARAMETER",
    "build_absorption_parameters", "absorption_component_optical_depth", "make_absorption_model_func",
    "AbsorptionSystem", "build_joint_absorption_parameters", "make_joint_absorption_model_func",
    "thermal_b_kms", "mass_scaled_b_transform", "ThermalTurbulentLink", "apply_thermal_turbulent_links",
    "fixed_log_ratio_transform", "AbundancePatternGroup", "apply_abundance_pattern_ties",
    "RegionShift", "ContinuumAdjustment", "apply_region_shifts", "apply_continuum_adjustments",
    "fit_absorption_spectrum", "fit_joint_absorption_spectrum",
    "AbsorptionComponentMeasurement", "AbsorptionFitResult",
    "save_absorption_result", "load_absorption_result", "summarize_absorption_fit",
    "fit_joint_absorption_spectrum_with_rejection",
    "estimate_column_density_upper_limit", "ColumnDensityUpperLimit",
    "generate_synthetic_absorption_spectrum",
    "build_absorption_prior_set", "build_absorption_posterior_problem",
    "build_joint_absorption_posterior_problem", "run_absorption_inference",
]
