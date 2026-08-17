"""Top-level FitSpec GUI application tying all three science panels together."""
from __future__ import annotations

from pathlib import Path

from core.config import load_configs
from core.io import load_spectrum
from gui.base_gui import BaseGUI
from gui.state import SessionState

__all__ = ["FitSpecApp", "build_session"]

_MODES = ("stellar", "emission", "absorption")


def build_session(spectrum_path, *, config_dir, run_dir=None, redshift=None, cli_overrides=None) -> SessionState:
    """Load one spectrum and all three layered mode configurations."""
    spectrum_path = Path(spectrum_path)
    run_dir = Path(run_dir) if run_dir is not None else spectrum_path.parent
    config_dir = Path(config_dir)
    spectrum = load_spectrum(spectrum_path)
    state = SessionState(spectrum=spectrum, spectrum_path=spectrum_path, run_dir=run_dir, config_dir=config_dir)
    if cli_overrides:
        raise ValueError("Per-mode config CLI overrides are not supported by build_session; use config.dat or the top-level --redshift option.")
    configs = load_configs(_MODES, config_dir, run_dir=run_dir)
    for mode, config in configs.items():
        state.set_config(mode, config)
    # A top-level explicit redshift wins; otherwise use the shared base z.  The
    # three config loads must agree because z is a base keyword.
    z_values = {float(state.configs[m].get("z", 0.0)) for m in _MODES}
    if redshift is None:
        if len(z_values) != 1:
            raise ValueError(f"Mode configurations disagree on systemic redshift: {sorted(z_values)}")
        redshift = z_values.pop()
    spectrum.redshift = float(redshift)
    spectrum.metadata.setdefault("source_path", str(spectrum_path))
    return state


class FitSpecApp(BaseGUI):
    """Common FitSpec shell with lazy science-panel launch."""

    PANEL_CLASSES = {}

    def __init__(self, state: SessionState, *, title="FitSpec — universal spectral fitting"):
        super().__init__(state, title=title)

    @classmethod
    def _panel_class(cls, mode):
        # Lazy imports keep startup light and prevent optional science-library
        # dependencies from being imported until that panel is requested.
        if mode in cls.PANEL_CLASSES:
            return cls.PANEL_CLASSES[mode]
        if mode == "stellar":
            from gui.stellar import StellarGUI
            return StellarGUI
        if mode == "emission":
            from gui.emission import EmissionGUI
            return EmissionGUI
        if mode == "absorption":
            from gui.absorption import AbsorptionGUI
            return AbsorptionGUI
        raise ValueError(f"Unknown FitSpec mode {mode!r}.")

    def open_mode(self, mode: str):
        spectrum = self.state.require_spectrum()
        config = self.state.config_for(mode)
        cls = self._panel_class(mode)
        kwargs = {
            "state": self.state,
            "result_path": self.state.run_dir / f"{mode}_fit.fits",
            "mask_path": self.state.run_dir / f"{mode}_mask.npz",
        }
        if mode == "emission":
            # Only EmissionGUI currently has the interactive continuum
            # feature; StellarGUI/AbsorptionGUI don't accept this kwarg.
            kwargs["continuum_path"] = self.state.run_dir / "emission_continuum.npz"
        panel = cls(spectrum, config, **kwargs)
        self.state.register_panel(mode, panel)
        self.refresh()
        panel.fig.show()
        return panel
