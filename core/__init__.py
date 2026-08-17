"""FitSpec core infrastructure.

Spectrum representation, I/O, configuration, rebinning, resolution,
masking, wavelength (air/vacuum) conventions, explicit component
parameters, generic deterministic fitting, statistics/model-selection,
and common result objects -- the shared foundation every science module
(stellar, emission, absorption) is built on.
"""
from core.masking import (
    region_mask, initial_mask_from_intervals, velocity_window_mask,
    FitMaskState, save_mask_file, load_mask_file,
    combine_masks, MaskComponents,
)
from core.spectrum import Spectrum, clean_spectrum, CleanedSpectrum
from core.io import load_spectrum, load_text_spectrum, load_fits_spectrum
from core.rebinning import rebin_spectrum, apply_permanent_rebinning, compute_display_smoothing, RebinResult
from core.wavelengths import to_vacuum, to_air, standard_medium, air_to_vacuum, vacuum_to_air
from core.resolution import (
    ResolutionModel, convolve_variable_gaussian, effective_doppler_b_kms,
    combine_gaussian_sigma, DefaultResolutionWarning,
)
from core.statistics import compute_fit_statistics, FitStatistics
from core.config import load_config, Config, ConfigError
from core.parameters import Parameter, Component, ModelParameters, ParameterTie, apply_ties
from core.models import gaussian, voigt_profile, sum_components
from core.results import FitResult, PosteriorResult
from core.fitting import fit_deterministic

__all__ = [
    "region_mask", "initial_mask_from_intervals", "velocity_window_mask",
    "FitMaskState", "save_mask_file", "load_mask_file",
    "combine_masks", "MaskComponents",
    "Spectrum", "clean_spectrum", "CleanedSpectrum",
    "load_spectrum", "load_text_spectrum", "load_fits_spectrum",
    "rebin_spectrum", "apply_permanent_rebinning", "compute_display_smoothing", "RebinResult",
    "to_vacuum", "to_air", "standard_medium", "air_to_vacuum", "vacuum_to_air",
    "ResolutionModel", "convolve_variable_gaussian", "effective_doppler_b_kms",
    "combine_gaussian_sigma", "DefaultResolutionWarning",
    "compute_fit_statistics", "FitStatistics",
    "load_config", "Config", "ConfigError",
    "Parameter", "Component", "ModelParameters", "ParameterTie", "apply_ties",
    "gaussian", "voigt_profile", "sum_components",
    "FitResult", "PosteriorResult",
    "fit_deterministic",
]
