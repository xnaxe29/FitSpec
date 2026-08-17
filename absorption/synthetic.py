"""Synthetic absorption-spectrum generation.

Builds a model spectrum from a transition list/system and a known set
of parameters, then adds noise -- the RDGEN ``gp``/``noise`` workflow
(generate Voigt profiles, then add Gaussian noise) reduced to a single
function call against FitSpec's own model machinery, for testing and
for exploring detectability/recoverability before committing telescope
time.
"""
from __future__ import annotations

import numpy as np

from core.parameters import ModelParameters, Component, Parameter
from core.spectrum import Spectrum

from absorption.absorption_model import COVERING_FRACTION_PARAMETER, make_absorption_model_func
from absorption.atomic import AtomicTransition

__all__ = ["generate_synthetic_absorption_spectrum"]


def generate_synthetic_absorption_spectrum(
    wave, transitions: "list[AtomicTransition]", components: "list[dict]", *,
    covering_fraction: "float | None" = None,
    redshift: float = 0.0, resolution=None,
    signal_to_noise: "float | None" = None, noise_sigma: "float | None" = None,
    seed: "int | None" = None,
) -> Spectrum:
    """Generate a noisy synthetic absorption spectrum for one transition group.

    Parameters
    ----------
    wave : array-like
        Observed-frame wavelength grid.
    transitions : list[AtomicTransition]
        The transition group to include (see
        ``absorption.atomic.select_group``).
    components : list[dict]
        One dict per kinematic component, each with keys ``logN``,
        ``b_kms``, ``velocity_kms``.
    covering_fraction : float, optional
        If given, applies the partial-covering model
        (``T_obs = (1-C_f) + C_f*T_full``) with this fixed fraction.
        None (default) means full coverage.
    signal_to_noise : float, optional
        Uniform S/N relative to the (unit) continuum -- sets a flat
        ``noise_sigma = 1 / signal_to_noise``. Mutually exclusive with
        ``noise_sigma``.
    noise_sigma : float, optional
        Explicit uniform 1-sigma noise level (in continuum-normalized
        flux units). Mutually exclusive with ``signal_to_noise``.
    seed : int, optional
        Random seed for reproducible noise.

    Returns
    -------
    core.spectrum.Spectrum
        ``flux`` is the noisy transmission, ``flux_unc`` the (uniform)
        noise level, ``continuum`` left at None since the spectrum is
        already continuum-normalized (flux ~ 1 in unabsorbed regions).
    """
    if not components:
        raise ValueError("components must be non-empty.")
    if (signal_to_noise is None) == (noise_sigma is None):
        raise ValueError("Provide exactly one of signal_to_noise or noise_sigma.")
    if noise_sigma is None:
        if signal_to_noise <= 0:
            raise ValueError("signal_to_noise must be positive.")
        noise_sigma = 1.0 / float(signal_to_noise)

    wave = np.asarray(wave, dtype=float)
    partial_coverage = covering_fraction is not None

    parameter_components = []
    for component_index, values in enumerate(components):
        parameters = [
            Parameter("logN", float(values["logN"]), -np.inf, np.inf),
            Parameter("b_kms", float(values["b_kms"]), 1e-6, np.inf),
            Parameter("velocity_kms", float(values.get("velocity_kms", 0.0)), -np.inf, np.inf),
        ]
        if partial_coverage:
            parameters.append(Parameter(
                COVERING_FRACTION_PARAMETER, float(covering_fraction), 0.0, 1.0, fixed=(component_index > 0),
            ))
        parameter_components.append(Component(parameters=parameters))

    model_parameters = ModelParameters(n_components=len(components), components=parameter_components)
    model_func = make_absorption_model_func(
        transitions, redshift=redshift, resolution=resolution, partial_coverage=partial_coverage,
    )
    true_transmission = model_func(wave, model_parameters)

    rng = np.random.default_rng(seed)
    flux_unc = np.full(wave.shape, float(noise_sigma))
    noisy_flux = true_transmission + rng.normal(0.0, noise_sigma, size=wave.shape)

    return Spectrum.from_arrays(wave, noisy_flux, flux_unc, redshift=redshift, resolution=resolution)
