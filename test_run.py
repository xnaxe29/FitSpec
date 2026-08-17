#!/usr/bin/env python3
"""Lightweight end-to-end FitSpec smoke test.

This is deliberately not a replacement for the pytest suite.  It checks the
installed source tree the way a user reaches it: load a spectrum, build the
unified session, instantiate the common application shell, and verify that the
stellar/emission/absorption panels all receive the same Spectrum object.

Run from the FitSpec directory with::

    python test_run.py

No sampler is started and no external stellar-library file is required.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gui.app import FitSpecApp, build_session


def run_smoke_test() -> None:
    project_dir = Path(__file__).resolve().parent
    config_dir = project_dir / "config"

    with tempfile.TemporaryDirectory(prefix="fitspec_smoke_") as tmp:
        run_dir = Path(tmp)
        spectrum_path = run_dir / "synthetic_spectrum.dat"
        wave = np.linspace(4000.0, 4100.0, 301)
        flux = np.ones_like(wave)
        unc = np.full_like(wave, 0.02)
        np.savetxt(spectrum_path, np.column_stack([wave, flux, unc]))

        state = build_session(
            spectrum_path,
            config_dir=config_dir,
            run_dir=run_dir,
            redshift=0.0,
        )
        assert state.spectrum is not None
        assert set(state.configs) == {"stellar", "emission", "absorption"}

        created = []

        class SmokePanel:
            def __init__(self, spectrum, config, **kwargs):
                self.spectrum = spectrum
                self.config = config
                self.kwargs = kwargs
                self.fig = plt.figure()
                created.append(self)

        original = FitSpecApp.PANEL_CLASSES
        FitSpecApp.PANEL_CLASSES = {
            "stellar": SmokePanel,
            "emission": SmokePanel,
            "absorption": SmokePanel,
        }
        try:
            app = FitSpecApp(state)
            panels = [app.open_mode(mode) for mode in ("stellar", "emission", "absorption")]
            assert all(panel.spectrum is state.spectrum for panel in panels)
            assert set(state.panels) == {"stellar", "emission", "absorption"}

            session_path = state.save(run_dir / "fitspec_session.npz")
            assert session_path.exists()
        finally:
            FitSpecApp.PANEL_CLASSES = original
            plt.close("all")

    print("FitSpec smoke test: PASS")
    print("  spectrum loading:       OK")
    print("  unified config loading: OK")
    print("  shared session state:   OK")
    print("  three-panel shell:      OK")
    print("  session save:           OK")


if __name__ == "__main__":
    run_smoke_test()
