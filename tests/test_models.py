"""Unit tests for shared numerical models and parameter containers."""
import numpy as np
import pytest

from core.models import gaussian, voigt_profile, sum_components
from core.parameters import Parameter, Component, ModelParameters, ParameterTie, apply_ties


def _two_component_parameters():
    return ModelParameters(
        n_components=2,
        components=[
            Component([
                Parameter("amplitude", 2.0, 0.0, 10.0),
                Parameter("mean", -1.0, -5.0, 5.0),
                Parameter("sigma", 0.5, 0.01, 5.0),
            ]),
            Component([
                Parameter("amplitude", 1.0, 0.0, 10.0),
                Parameter("mean", 1.0, -5.0, 5.0),
                Parameter("sigma", 0.75, 0.01, 5.0),
            ]),
        ],
    )


def test_gaussian_is_peak_normalized_and_symmetric():
    wave = np.linspace(-5.0, 5.0, 1001)
    model = gaussian(wave, amplitude=3.5, mean=0.0, sigma=1.2)
    assert model[wave.size // 2] == pytest.approx(3.5)
    assert np.allclose(model, model[::-1])


def test_gaussian_rejects_nonpositive_sigma():
    with pytest.raises(ValueError, match="sigma must be positive"):
        gaussian([0.0, 1.0], 1.0, 0.0, 0.0)


def test_voigt_is_finite_and_peak_normalized():
    wave = np.linspace(-10.0, 10.0, 2001)
    profile = voigt_profile(wave, amplitude=4.0, mean=0.0, sigma=1.0, gamma=0.3)
    assert np.all(np.isfinite(profile))
    assert profile.max() == pytest.approx(4.0, rel=1e-12)
    assert np.allclose(profile, profile[::-1], rtol=0.0, atol=1e-12)


def test_modelparameters_vector_bounds_and_round_trip():
    params = _two_component_parameters()
    params.components[1]["mean"].fixed = True
    vector = params.to_vector()
    lower, upper = params.bounds()
    assert vector.size == 5
    assert np.all(vector >= lower)
    assert np.all(vector <= upper)

    changed = vector.copy()
    changed[0] = 3.0
    params.from_vector(changed)
    assert params.components[0]["amplitude"].value == pytest.approx(3.0)
    assert params.components[1]["mean"].value == pytest.approx(1.0)

    restored = ModelParameters.from_dict(params.to_dict())
    assert restored.to_dict() == params.to_dict()


def test_parameter_tie_updates_fixed_follower_with_transform():
    params = _two_component_parameters()
    params.components[1]["sigma"].fixed = True
    tie = ParameterTie(
        leader=(0, "sigma"),
        follower=(1, "sigma"),
        transform=lambda value: 2.0 * value,
    )
    params.components[0]["sigma"].value = 0.8
    apply_ties(params, [tie])
    assert params.components[1]["sigma"].value == pytest.approx(1.6)


def test_sum_components_matches_manual_sum():
    wave = np.linspace(-5.0, 5.0, 401)
    params = _two_component_parameters()

    def component_model(w, amplitude, mean, sigma):
        return gaussian(w, amplitude, mean, sigma)

    combined = sum_components(wave, params, component_model)
    manual = sum(
        component_model(
            wave,
            component["amplitude"].value,
            component["mean"].value,
            component["sigma"].value,
        )
        for component in params.components
    )
    assert np.allclose(combined, manual)
