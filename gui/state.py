"""Shared FitSpec GUI session state.

The session owns the one universal :class:`core.spectrum.Spectrum` instance
used by every science panel, the layered mode configurations, and the most
recent deterministic/posterior result produced by each mode.  It deliberately
contains no Matplotlib objects; figures/panels register themselves separately
and can be recreated without changing the scientific state.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json

import numpy as np

from core.spectrum import Spectrum

__all__ = ["SessionState"]

_MODES = ("stellar", "emission", "absorption")


def _json_safe(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return repr(value)


@dataclass
class SessionState:
    """State shared by the common shell and all science panels."""

    spectrum: Spectrum | None = None
    spectrum_path: Path | None = None
    run_dir: Path = field(default_factory=lambda: Path.cwd())
    config_dir: Path | None = None
    configs: dict = field(default_factory=dict)
    results: dict = field(default_factory=dict)
    posteriors: dict = field(default_factory=dict)
    panels: dict = field(default_factory=dict, repr=False)
    active_mode: str | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.run_dir = Path(self.run_dir)
        if self.config_dir is not None:
            self.config_dir = Path(self.config_dir)
        if self.spectrum_path is not None:
            self.spectrum_path = Path(self.spectrum_path)

    def require_spectrum(self) -> Spectrum:
        if self.spectrum is None:
            raise RuntimeError("No spectrum is loaded in this FitSpec session.")
        return self.spectrum

    def set_spectrum(self, spectrum: Spectrum, *, path=None) -> Spectrum:
        self.spectrum = spectrum
        if path is not None:
            self.spectrum_path = Path(path)
        elif spectrum.metadata.get("source_path"):
            self.spectrum_path = Path(spectrum.metadata["source_path"])
        # Results from a previous target are never valid for a new spectrum.
        self.results.clear()
        self.posteriors.clear()
        self.panels.clear()
        return spectrum

    def set_config(self, mode: str, config):
        self._check_mode(mode)
        self.configs[mode] = config
        return config

    def config_for(self, mode: str):
        self._check_mode(mode)
        if mode not in self.configs:
            raise RuntimeError(f"No {mode!r} configuration is loaded in this session.")
        return self.configs[mode]

    def set_result(self, mode: str, result):
        self._check_mode(mode)
        self.results[mode] = result
        return result

    def set_posterior(self, mode: str, posterior):
        self._check_mode(mode)
        self.posteriors[mode] = posterior
        return posterior

    def register_panel(self, mode: str, panel):
        self._check_mode(mode)
        self.panels[mode] = panel
        self.active_mode = mode
        return panel

    def summary(self) -> dict:
        spectrum = self.spectrum
        return {
            "spectrum_path": None if self.spectrum_path is None else str(self.spectrum_path),
            "n_pixels": None if spectrum is None else int(spectrum.wave.size),
            "redshift": None if spectrum is None else float(spectrum.redshift),
            "configs": sorted(self.configs),
            "results": sorted(self.results),
            "posteriors": sorted(self.posteriors),
            "active_mode": self.active_mode,
        }

    def save(self, path) -> Path:
        """Save the portable session core to one compressed NPZ file.

        Scientific result objects keep their own FITS/NPZ serializers.  The
        session file stores the common spectrum/mask plus enough provenance to
        reconstruct the shell, and records which modes had live results.
        """
        spectrum = self.require_spectrum()
        path = Path(path)
        payload = {
            "wave": np.asarray(spectrum.wave, float),
            "flux": np.asarray(spectrum.flux, float),
            "mask": np.asarray(spectrum.mask if spectrum.mask is not None else np.ones(spectrum.wave.size, bool), bool),
            "redshift": np.asarray([float(spectrum.redshift)]),
            "has_flux_unc": np.asarray([spectrum.flux_unc is not None], bool),
            "flux_unc": np.asarray([] if spectrum.flux_unc is None else spectrum.flux_unc, float),
            "has_continuum": np.asarray([spectrum.continuum is not None], bool),
            "continuum": np.asarray([] if spectrum.continuum is None else spectrum.continuum, float),
            "has_model": np.asarray([spectrum.model is not None], bool),
            "model": np.asarray([] if spectrum.model is None else spectrum.model, float),
        }
        manifest = {
            "format": "FitSpecSession-v1",
            "spectrum_path": None if self.spectrum_path is None else str(self.spectrum_path),
            "run_dir": str(self.run_dir),
            "config_dir": None if self.config_dir is None else str(self.config_dir),
            "spectrum_metadata": _json_safe(spectrum.metadata),
            "session_metadata": _json_safe(self.metadata),
            "configured_modes": sorted(self.configs),
            "result_modes": sorted(self.results),
            "posterior_modes": sorted(self.posteriors),
            "active_mode": self.active_mode,
        }
        payload["manifest_json"] = np.asarray(json.dumps(manifest))
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **payload)
        return path

    @classmethod
    def load(cls, path) -> "SessionState":
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            manifest = json.loads(str(data["manifest_json"].item()))
            def optional_array(name, flag):
                return np.asarray(data[name], float) if bool(data[flag][0]) else None
            spectrum = Spectrum(
                wave=np.asarray(data["wave"], float),
                flux=np.asarray(data["flux"], float),
                flux_unc=optional_array("flux_unc", "has_flux_unc"),
                continuum=optional_array("continuum", "has_continuum"),
                model=optional_array("model", "has_model"),
                mask=np.asarray(data["mask"], bool),
                redshift=float(data["redshift"][0]),
                metadata=dict(manifest.get("spectrum_metadata", {})),
            )
        state = cls(
            spectrum=spectrum,
            spectrum_path=manifest.get("spectrum_path"),
            run_dir=manifest.get("run_dir", path.parent),
            config_dir=manifest.get("config_dir"),
            active_mode=manifest.get("active_mode"),
            metadata=dict(manifest.get("session_metadata", {})),
        )
        # Config/result objects are intentionally reconstructed by the app from
        # their authoritative files; the manifest keeps their presence only.
        return state

    @staticmethod
    def _check_mode(mode: str):
        if mode not in _MODES:
            raise ValueError(f"Unknown FitSpec mode {mode!r}; expected one of {_MODES}.")
