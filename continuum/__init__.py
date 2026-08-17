"""Continuum estimation.

A set of interchangeable continuum-estimation methods, selected by name
through `estimate_continuum`. See continuum.continuum for details on
each method and the bug fixes applied relative to the original
continuum_function_module.py.

`continuum.continuum_points` builds on top of these methods with the
interactive, anchor-point-based continuum editing workflow (estimate,
then add/remove/move points on the fly), ported from the original
`bic_emission_fitting.py`.
"""
from continuum.continuum import (
    ContinuumNoOpWarning,
    continuum_finder,
    poly_fit_continuum,
    spline_fit_continuum,
    pca_continuum,
    lowess_continuum,
    gaussian_fit_continuum,
    peak_find_continuum,
    median_filter_continuum,
    continuum_using_fof,
    estimate_continuum,
)
from continuum.continuum_points import (
    continuum_from_points,
    default_anchor_points,
    ContinuumPointsState,
    save_continuum_points_file,
    load_continuum_points_file,
)

__all__ = [
    "ContinuumNoOpWarning",
    "continuum_finder",
    "poly_fit_continuum",
    "spline_fit_continuum",
    "pca_continuum",
    "lowess_continuum",
    "gaussian_fit_continuum",
    "peak_find_continuum",
    "median_filter_continuum",
    "continuum_using_fof",
    "estimate_continuum",
    "continuum_from_points",
    "default_anchor_points",
    "ContinuumPointsState",
    "save_continuum_points_file",
    "load_continuum_points_file",
]

