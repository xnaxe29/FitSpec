"""Common fit-result objects.

FitResult bundles everything downstream code (plotting, saving,
posterior sampling) needs from a deterministic fit: the fitted
parameters, their uncertainties, the exact spectrum data used (wave,
flux, flux_unc, mask) -- so that per the "posterior sampling should be
reproducible without the GUI" design principle, an MCMC/dynesty run can
be initialized from a saved FitResult alone, from the terminal, without
touching the GUI or re-deriving what was actually fit -- the best-fit
model, and the generic statistics from core.statistics.

PosteriorResult is the analogous container for posterior samples,
optionally linked back to the FitResult it was seeded from.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np

from core.parameters import ModelParameters
from core.statistics import FitStatistics


__all__ = ["FitResult", "PosteriorResult"]


@dataclass
class FitResult:
    """Result of a deterministic fit.

    Attributes
    ----------
    parameters : ModelParameters
        The fitted parameters (values already updated to the best fit).
    parameter_uncertainties : dict[str, float] or None
        Keyed by the same names as ``parameters.parameter_names()``.
    wave, flux, flux_unc : np.ndarray
        The *full* (unmasked) spectrum arrays actually used for this fit
        -- kept alongside the result so it is fully reproducible later
        without needing to reload or re-derive the input spectrum.
    mask : np.ndarray of bool
        Which points were actually included in the fit.
    model : np.ndarray
        Best-fit model, evaluated on the full ``wave`` grid.
    statistics : FitStatistics
        Goodness-of-fit/model-selection statistics (see core.statistics).
    redshift : float
        Systemic redshift used for this fit.
    resolution_source : str or None
        Provenance string from the ResolutionModel used (see
        core.resolution), if any.
    method : str
        Which optimizer produced this result (e.g. "curve_fit").
    metadata : dict
        Anything else worth carrying along.
    """

    parameters: ModelParameters
    parameter_uncertainties: "dict[str, float] | None"
    wave: np.ndarray
    flux: np.ndarray
    flux_unc: "np.ndarray | None"
    mask: np.ndarray
    model: np.ndarray
    statistics: FitStatistics
    redshift: float = 0.0
    resolution_source: "str | None" = None
    method: str = "unknown"
    metadata: dict = field(default_factory=dict)

    @property
    def residuals(self) -> np.ndarray:
        return self.flux - self.model

    def to_dict(self) -> dict:
        """Serialize to a plain, JSON-able dict."""
        return {
            "parameters": self.parameters.to_dict(),
            "parameter_uncertainties": self.parameter_uncertainties,
            "wave": self.wave.tolist(),
            "flux": self.flux.tolist(),
            "flux_unc": None if self.flux_unc is None else self.flux_unc.tolist(),
            "mask": self.mask.tolist(),
            "model": self.model.tolist(),
            "statistics": vars(self.statistics),
            "redshift": self.redshift,
            "resolution_source": self.resolution_source,
            "method": self.method,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FitResult":
        return cls(
            parameters=ModelParameters.from_dict(data["parameters"]),
            parameter_uncertainties=data.get("parameter_uncertainties"),
            wave=np.asarray(data["wave"], dtype=float),
            flux=np.asarray(data["flux"], dtype=float),
            flux_unc=None if data.get("flux_unc") is None else np.asarray(data["flux_unc"], dtype=float),
            mask=np.asarray(data["mask"], dtype=bool),
            model=np.asarray(data["model"], dtype=float),
            statistics=FitStatistics(**data["statistics"]),
            redshift=data.get("redshift", 0.0),
            resolution_source=data.get("resolution_source"),
            method=data.get("method", "unknown"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class PosteriorResult:
    """Posterior samples from Dynesty/emcee, optionally seeded from a FitResult."""

    samples: np.ndarray  # shape (n_samples, n_free_params)
    log_probability: np.ndarray
    parameter_names: "list[str]"
    deterministic_result: "FitResult | None" = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.samples = np.asarray(self.samples, dtype=float)
        self.log_probability = np.asarray(self.log_probability, dtype=float)
        self.parameter_names = list(self.parameter_names)
        if self.samples.ndim != 2:
            raise ValueError("samples must be a 2-D array (n_samples, n_free_params).")
        if self.log_probability.ndim != 1:
            raise ValueError("log_probability must be a 1-D array.")
        if self.samples.shape[1] != len(self.parameter_names):
            raise ValueError("samples.shape[1] must equal len(parameter_names).")
        if self.samples.shape[0] != self.log_probability.shape[0]:
            raise ValueError("samples and log_probability must have the same number of rows.")
        if self.samples.shape[0] == 0:
            raise ValueError("PosteriorResult requires at least one posterior sample.")

    def percentiles(self, percentiles=(16, 50, 84)) -> np.ndarray:
        """Posterior percentiles, shape ``(len(percentiles), n_free_params)``."""
        return np.percentile(self.samples, percentiles, axis=0)

    def summary(self, percentiles=(16, 50, 84)) -> dict:
        """Return named percentile summaries for every sampled parameter."""
        values = self.percentiles(percentiles)
        return {name: {str(p): float(values[i, j]) for i, p in enumerate(percentiles)}
                for j, name in enumerate(self.parameter_names)}

    @property
    def maximum_posterior_sample(self) -> np.ndarray:
        """Sample with the largest stored log posterior probability."""
        return self.samples[int(np.nanargmax(self.log_probability))].copy()

    def save_npz(self, path) -> Path:
        """Save posterior arrays and JSON metadata without pickling Python objects.

        The linked deterministic result is serialized through ``FitResult.to_dict``
        when present, so the product remains portable and does not depend on live
        GUI objects or sampler classes.
        """
        path = Path(path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        deterministic = None if self.deterministic_result is None else self.deterministic_result.to_dict()
        payload = {"metadata": self.metadata, "deterministic_result": deterministic}
        np.savez_compressed(
            path, samples=self.samples, log_probability=self.log_probability,
            parameter_names=np.asarray(self.parameter_names, dtype=str),
            payload_json=np.asarray(json.dumps(payload, default=_json_default)),
        )
        return path

    @classmethod
    def load_npz(cls, path) -> "PosteriorResult":
        """Load a product written by :meth:`save_npz`."""
        with np.load(Path(path).expanduser(), allow_pickle=False) as data:
            payload = json.loads(str(data["payload_json"].item()))
            deterministic_data = payload.get("deterministic_result")
            deterministic = None if deterministic_data is None else FitResult.from_dict(deterministic_data)
            return cls(
                samples=np.asarray(data["samples"], dtype=float),
                log_probability=np.asarray(data["log_probability"], dtype=float),
                parameter_names=[str(x) for x in data["parameter_names"].tolist()],
                deterministic_result=deterministic, metadata=payload.get("metadata", {}),
            )


def _json_default(value):
    """JSON conversion for numpy scalar/array metadata."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
