"""FitSpec emission-line fitting: independent science module.

Shares core infrastructure, continuum, masking, resolution, plotting,
and inference with the other science modules; does not alter or depend
on the unified stellar-fitting architecture.
"""
from emission.lines import (
    EmissionLine, load_emission_line_list, select_lines, select_lines_in_wavelength_range,
    apply_fixed_ratio_overrides,
)
from emission.profiles import doppler_shifted_wavelength, gaussian_line_flux, velocity_dispersion_to_angstrom
from emission.emission_model import build_emission_parameters, emission_component_flux, make_emission_model_func
from emission.emission_fit import fit_emission_spectrum, select_emission_line_list
from emission.inference import (
    build_emission_prior_set, build_emission_posterior_problem, run_emission_inference,
)
from emission.emission_results import (
    EmissionLineMeasurement, EmissionFitResult, save_emission_result, load_emission_result,
    summarize_emission_fit,
)

__all__ = [
    "EmissionLine", "load_emission_line_list", "select_lines", "select_lines_in_wavelength_range",
    "apply_fixed_ratio_overrides",
    "doppler_shifted_wavelength", "gaussian_line_flux", "velocity_dispersion_to_angstrom",
    "build_emission_parameters", "emission_component_flux", "make_emission_model_func",
    "fit_emission_spectrum", "select_emission_line_list",
    "build_emission_prior_set", "build_emission_posterior_problem", "run_emission_inference",
    "EmissionLineMeasurement", "EmissionFitResult", "save_emission_result", "load_emission_result",
    "summarize_emission_fit",
]
