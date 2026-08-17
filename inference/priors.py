"""Prior distributions shared by FitSpec posterior samplers."""
from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

from core.parameters import ModelParameters

__all__ = [
    "Prior", "UniformPrior", "LogUniformPrior", "GaussianPrior",
    "TruncatedGaussianPrior", "PriorSet",
]


class Prior:
    """Minimal scalar-prior protocol."""
    def log_probability(self, value: float) -> float:  # pragma: no cover - abstract contract
        raise NotImplementedError

    def transform(self, unit_value: float) -> float:  # pragma: no cover - abstract contract
        raise NotImplementedError


@dataclass(frozen=True)
class UniformPrior(Prior):
    lower: float
    upper: float

    def __post_init__(self):
        if not (np.isfinite(self.lower) and np.isfinite(self.upper) and self.lower < self.upper):
            raise ValueError("UniformPrior requires finite lower < upper.")

    def log_probability(self, value):
        if self.lower <= value <= self.upper:
            return float(-np.log(self.upper - self.lower))
        return -np.inf

    def transform(self, unit_value):
        u = _unit(unit_value)
        return float(self.lower + u * (self.upper - self.lower))


@dataclass(frozen=True)
class LogUniformPrior(Prior):
    lower: float
    upper: float

    def __post_init__(self):
        if not (np.isfinite(self.lower) and np.isfinite(self.upper) and 0 < self.lower < self.upper):
            raise ValueError("LogUniformPrior requires finite 0 < lower < upper.")

    def log_probability(self, value):
        if self.lower <= value <= self.upper:
            return float(-np.log(value) - np.log(np.log(self.upper / self.lower)))
        return -np.inf

    def transform(self, unit_value):
        u = _unit(unit_value)
        return float(np.exp(np.log(self.lower) + u * np.log(self.upper / self.lower)))


@dataclass(frozen=True)
class GaussianPrior(Prior):
    mean: float
    sigma: float

    def __post_init__(self):
        if not (np.isfinite(self.mean) and np.isfinite(self.sigma) and self.sigma > 0):
            raise ValueError("GaussianPrior requires finite mean and sigma > 0.")

    def log_probability(self, value):
        z = (value - self.mean) / self.sigma
        return float(-0.5 * z*z - np.log(self.sigma * np.sqrt(2.0*np.pi)))

    def transform(self, unit_value):
        u = _unit_open(unit_value)
        return float(NormalDist(mu=self.mean, sigma=self.sigma).inv_cdf(u))


@dataclass(frozen=True)
class TruncatedGaussianPrior(Prior):
    mean: float
    sigma: float
    lower: float
    upper: float

    def __post_init__(self):
        if self.sigma <= 0 or not self.lower < self.upper:
            raise ValueError("TruncatedGaussianPrior requires sigma > 0 and lower < upper.")
        dist = NormalDist(mu=self.mean, sigma=self.sigma)
        if dist.cdf(self.upper) <= dist.cdf(self.lower):
            raise ValueError("Truncation interval has zero Gaussian probability.")

    def log_probability(self, value):
        if not self.lower <= value <= self.upper:
            return -np.inf
        dist = NormalDist(mu=self.mean, sigma=self.sigma)
        norm = dist.cdf(self.upper) - dist.cdf(self.lower)
        z = (value - self.mean) / self.sigma
        return float(-0.5*z*z - np.log(self.sigma*np.sqrt(2*np.pi)) - np.log(norm))

    def transform(self, unit_value):
        u = _unit(unit_value)
        dist = NormalDist(mu=self.mean, sigma=self.sigma)
        lo, hi = dist.cdf(self.lower), dist.cdf(self.upper)
        p = np.clip(lo + u*(hi-lo), np.finfo(float).eps, 1.0-np.finfo(float).eps)
        return float(dist.inv_cdf(float(p)))


def _unit(value):
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError("unit-cube values must lie in [0, 1].")
    return value


def _unit_open(value):
    return float(np.clip(_unit(value), np.finfo(float).eps, 1.0-np.finfo(float).eps))


@dataclass
class PriorSet:
    priors: list[Prior]
    parameter_names: "list[str] | None" = None

    def __post_init__(self):
        self.priors = list(self.priors)
        if not self.priors:
            raise ValueError("PriorSet requires at least one prior.")
        if self.parameter_names is not None:
            self.parameter_names = list(self.parameter_names)
            if len(self.parameter_names) != len(self.priors):
                raise ValueError("parameter_names must have one entry per prior.")

    def __len__(self):
        return len(self.priors)

    def log_probability(self, values) -> float:
        values = np.asarray(values, dtype=float)
        if values.shape != (len(self),):
            raise ValueError(f"values must have shape ({len(self)},).")
        terms = [prior.log_probability(float(value)) for prior, value in zip(self.priors, values)]
        return float(np.sum(terms)) if np.all(np.isfinite(terms)) else -np.inf

    def transform(self, unit_cube) -> np.ndarray:
        unit_cube = np.asarray(unit_cube, dtype=float)
        if unit_cube.shape != (len(self),):
            raise ValueError(f"unit_cube must have shape ({len(self)},).")
        return np.asarray([p.transform(u) for p, u in zip(self.priors, unit_cube)], dtype=float)

    def sample(self, rng=None, size=1) -> np.ndarray:
        rng = np.random.default_rng(rng)
        cubes = rng.random((int(size), len(self)))
        return np.asarray([self.transform(row) for row in cubes], dtype=float)

    @classmethod
    def from_model_parameters(cls, model_parameters: ModelParameters) -> "PriorSet":
        names = model_parameters.parameter_names()
        priors = []
        for _, parameter in model_parameters.free_parameters():
            if not (np.isfinite(parameter.lower) and np.isfinite(parameter.upper)):
                raise ValueError(
                    f"Free parameter {parameter.name!r} has an infinite bound. Posterior sampling "
                    "requires a proper finite prior; pass an explicit PriorSet."
                )
            if parameter.lower == parameter.upper:
                raise ValueError(f"Free parameter {parameter.name!r} has zero-width bounds.")
            priors.append(UniformPrior(float(parameter.lower), float(parameter.upper)))
        return cls(priors=priors, parameter_names=names)
