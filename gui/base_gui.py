"""Common Matplotlib shell used by the FitSpec application."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np

from gui.state import SessionState

__all__ = ["BaseGUI"]


class BaseGUI:
    """A lightweight shell around a shared :class:`SessionState`.

    Science panels remain responsible for their specialized controls.  This
    shell owns target-level navigation and displays the common spectrum once;
    opening a panel passes the exact same Spectrum object to that panel.
    """

    def __init__(self, state: SessionState, *, title="FitSpec"):
        self.state = state
        self.fig, self.ax = plt.subplots(figsize=(13, 7))
        plt.subplots_adjust(left=0.08, right=0.80, bottom=0.14, top=0.90)
        self.data_line, = self.ax.plot([], [], drawstyle="steps-mid", alpha=0.65, label="spectrum")
        self.masked_line, = self.ax.plot([], [], ".", ms=2.5, alpha=0.20, label="masked")
        self.ax.set_xlabel(r"Wavelength ($\AA$)")
        self.ax.set_ylabel("Flux")
        self.ax.legend(fontsize=8)
        self.title = title

        self.mode_buttons = {}
        for label, mode, y in (("Stellar", "stellar", .73), ("Emission", "emission", .65), ("Absorption", "absorption", .57)):
            button = Button(plt.axes([.83, y, .13, .055]), label)
            button.on_clicked(lambda _event, m=mode: self.open_mode(m))
            self.mode_buttons[mode] = button

        self.save_session_button = Button(plt.axes([.83, .43, .13, .05]), "Save Session")
        self.save_session_button.on_clicked(self._save_session)
        self.status_ax = plt.axes([.82, .15, .16, .22])
        self.status_ax.axis("off")
        self.status_text = self.status_ax.text(0, 1, "", va="top", ha="left", fontsize=9, wrap=True)
        self.refresh()

    def refresh(self):
        spectrum = self.state.spectrum
        if spectrum is None:
            self.data_line.set_data([], [])
            self.masked_line.set_data([], [])
            self.ax.set_title(f"{self.title} — no spectrum loaded")
        else:
            self.data_line.set_data(spectrum.wave, spectrum.flux)
            mask = np.ones(spectrum.wave.size, bool) if spectrum.mask is None else np.asarray(spectrum.mask, bool)
            self.masked_line.set_data(spectrum.wave[~mask], spectrum.flux[~mask])
            source = self.state.spectrum_path.name if self.state.spectrum_path is not None else "in-memory spectrum"
            self.ax.set_title(f"{self.title} — {source} — z={spectrum.redshift:g}")
            self.ax.relim(); self.ax.autoscale_view()
        summary = self.state.summary()
        self.status_text.set_text(
            f"pixels: {summary['n_pixels']}\n"
            f"active: {summary['active_mode'] or '-'}\n"
            f"fits: {', '.join(summary['results']) or '-'}\n"
            f"posteriors: {', '.join(summary['posteriors']) or '-'}"
        )
        self.fig.canvas.draw_idle()

    def open_mode(self, mode: str):
        raise NotImplementedError("BaseGUI.open_mode is implemented by FitSpecApp.")

    def _save_session(self, _event=None):
        path = self.state.run_dir / "fitspec_session.npz"
        saved = self.state.save(path)
        print(f"FitSpec session saved: {saved}")
        self.refresh()

    def show(self):
        plt.show()
