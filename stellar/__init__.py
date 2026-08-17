"""Public exports for the unified FitSpec stellar section (v3 draft).

When merged into the repository, copy these exports into ``stellar/__init__.py``.
"""
from stellar.stellar_models import StellarLibrary, classify_spectral_regime, load_stellar_library
from stellar.stellar_fit import prepare_stellar_library_from_config, fit_stellar_spectrum, evaluate_stellar_result_on_grid, calculate_stellar_diagnostics
from stellar.stellar_results import StellarFitDiagnostics, StellarFitResult, save_stellar_result, load_stellar_result
from stellar.inference import select_stellar_inference_basis, build_stellar_posterior_problem, run_stellar_inference, stellar_population_samples

__all__=[
    "StellarLibrary","classify_spectral_regime","load_stellar_library",
    "prepare_stellar_library_from_config","fit_stellar_spectrum","evaluate_stellar_result_on_grid","calculate_stellar_diagnostics",
    "StellarFitDiagnostics","StellarFitResult","save_stellar_result","load_stellar_result",
    "select_stellar_inference_basis","build_stellar_posterior_problem","run_stellar_inference","stellar_population_samples",
]
