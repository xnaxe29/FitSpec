"""Reusable interactive point-based continuum controller for FitSpec GUIs.

This module contains no line-fitting logic. It connects a Spectrum-like
object (``wave``, ``flux``) to :class:`continuum.continuum_points.ContinuumPointsState`
and optionally to Matplotlib widgets, following the same working/saved
pattern and Matplotlib-wiring style as :class:`gui.mask_controller.MaskController`.

On every edit, the evaluated continuum is written to ``spectrum.continuum``
(a field ``core.spectrum.Spectrum`` already carries for exactly this
purpose) so it travels with the spectrum through ``gui.state.SessionState``
save/load -- but ``spectrum.flux`` itself is never touched here. Actually
subtracting the continuum before fitting is the caller's job (e.g.
``gui.emission.EmissionGUI._fit``), since a shared ``Spectrum`` is also
used, unmodified, by the stellar/absorption panels.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from continuum.continuum import estimate_continuum
from continuum.continuum_points import (
    ContinuumPointsState,
    default_anchor_points,
    load_continuum_points_file,
    save_continuum_points_file,
)

__all__ = ["ContinuumController"]


class ContinuumController:
    """Own working/saved continuum anchor-point state for one spectrum."""

    def __init__(
        self,
        spectrum,
        *,
        n_points: int = 50,
        method: str = "custom",
        method_kwargs: "dict | None" = None,
        continuum_path=None,
        on_change: "Callable[[np.ndarray, str], None] | None" = None,
    ):
        self.spectrum = spectrum
        self.n_points = int(n_points)
        self.method = str(method)
        self.method_kwargs = dict(method_kwargs or {})
        # Deliberately not set to `on_change` yet: the constructor's own
        # initial sync below must not call back into the caller, which
        # (e.g. in EmissionGUI) is often still in the middle of assigning
        # `self.continuum_controller = ContinuumController(...)` at this
        # point and would see an AttributeError reaching back for it.
        self.on_change = None

        if continuum_path is None:
            source_path = getattr(spectrum, "metadata", {}).get("source_path")
            if source_path:
                source = Path(source_path).expanduser()
                self.continuum_path = source.with_name(source.name + ".fitspec_continuum.npz")
            else:
                self.continuum_path = Path("fitspec_continuum.npz")
        else:
            self.continuum_path = Path(continuum_path).expanduser()

        wave_points, flux_points = self._auto_estimate_points()
        self.state = ContinuumPointsState.from_points(wave_points, flux_points)

        # Optional Matplotlib state, initialized by connect_matplotlib().
        # Must be set before the first _sync() call below, since _sync()
        # calls _update_status(), which reads self._status_text.
        self._continuum_line = None
        self._points_line = None
        self._figure = None
        self._axes = None
        self._enabled_check = None
        self._mode_radio = None
        self._status_text = None
        self._edit_mode = "add"
        self._editing_enabled = False
        self._cid_click = None

        self._sync("initialized")
        self.on_change = on_change

    @property
    def working_continuum(self) -> np.ndarray:
        return self.continuum_on(self.spectrum.wave)

    def continuum_on(self, wave) -> np.ndarray:
        return self.state.continuum_on(np.asarray(wave, dtype=float))

    def _auto_estimate_points(self):
        wave = np.asarray(self.spectrum.wave, dtype=float)
        flux = np.asarray(self.spectrum.flux, dtype=float)
        good = np.isfinite(wave) & np.isfinite(flux)
        try:
            estimate = estimate_continuum(wave[good], flux[good], method=self.method, **self.method_kwargs)
            # "custom" evaluates only at the finite subset (see
            # continuum.continuum.estimate_continuum); every method here
            # is re-expressed on the full wave grid by interpolation so
            # anchor points can be picked anywhere, including across
            # any masked/non-finite gaps.
            full_estimate = np.interp(wave, wave[good], np.asarray(estimate, dtype=float))
        except Exception as exc:
            print(
                f"Automated continuum estimation (method={self.method!r}) failed ({exc}); "
                "falling back to a flat median continuum. You can still edit points by hand."
            )
            full_estimate = np.full_like(wave, float(np.nanmedian(flux[good])) if np.any(good) else 0.0)
        return default_anchor_points(wave, full_estimate, n_points=self.n_points)

    def _sync(self, reason: str):
        self.spectrum.continuum = self.working_continuum
        if self.on_change is not None:
            self.on_change(self.spectrum.continuum, reason)
        self._update_display()
        self._update_status()
        if self._figure is not None:
            self._figure.canvas.draw_idle()

    def _update_display(self):
        if self._points_line is not None:
            order = np.argsort(self.state.working_wave)
            self._points_line.set_data(self.state.working_wave[order], self.state.working_flux[order])
        if self._continuum_line is not None:
            self._continuum_line.set_data(np.asarray(self.spectrum.wave, dtype=float), self.spectrum.continuum)

    # ------------------------------------------------------------------
    # Explicit point operations
    # ------------------------------------------------------------------
    def add_point(self, wave_value, flux_value):
        self.state.add_point(wave_value, flux_value)
        self._sync("add_point")

    def remove_nearest(self, wave_value):
        self.state.remove_nearest(wave_value)
        self._sync("remove_point")

    def move_nearest(self, wave_value, new_wave, new_flux):
        self.state.move_nearest(wave_value, new_wave, new_flux)
        self._sync("move_point")

    def reset_to_auto(self):
        """Discard working points and re-run automated continuum estimation."""
        wave_points, flux_points = self._auto_estimate_points()
        self.state.reset_to(wave_points, flux_points)
        self._sync("reset_to_auto")

    # ------------------------------------------------------------------
    # Explicit Save / Load semantics
    # ------------------------------------------------------------------
    def save_continuum(self, path=None):
        """Commit the working anchor points; persist them to disk."""
        self.state.save()
        target = self.continuum_path if path is None else Path(path).expanduser()
        save_continuum_points_file(
            target, self.state, metadata={"method": self.method, "n_points": self.n_points},
        )
        self.continuum_path = target
        return self._sync("save_continuum")

    def load_continuum(self, path=None):
        """Restore the last saved anchor points, from memory or a file."""
        target = self.continuum_path if path is None else Path(path).expanduser()
        if target is not None and target.is_file():
            self.state = load_continuum_points_file(target)
            self.continuum_path = target
        else:
            self.state.load()
        return self._sync("load_continuum")

    # ------------------------------------------------------------------
    # Optional generic Matplotlib controls for the emission (or absorption) GUI
    # ------------------------------------------------------------------
    def connect_matplotlib(
        self,
        spectrum_axes,
        *,
        continuum_line_kwargs=None,
        points_line_kwargs=None,
        editing_check_axes=None,
        mode_axes=None,
        save_button_axes=None,
        load_button_axes=None,
        reset_button_axes=None,
        status_axes=None,
    ):
        """Attach the continuum/points display lines and edit controls.

        The caller owns layout and simply supplies axes for the desired
        controls, matching gui.mask_controller.MaskController's pattern.
        """
        from matplotlib.widgets import Button, CheckButtons, RadioButtons

        self._axes = spectrum_axes
        self._figure = spectrum_axes.figure

        continuum_kwargs = dict(lw=1.3, ls="--", color="tab:green", label="continuum")
        continuum_kwargs.update(continuum_line_kwargs or {})
        self._continuum_line, = spectrum_axes.plot([], [], **continuum_kwargs)

        points_kwargs = dict(marker="o", linestyle="none", ms=6, color="gold", mec="k", mew=0.5,
                              label="continuum points", zorder=5)
        points_kwargs.update(points_line_kwargs or {})
        self._points_line, = spectrum_axes.plot([], [], **points_kwargs)

        if editing_check_axes is not None:
            self._enabled_check = CheckButtons(editing_check_axes, ["Edit continuum"], [False])
            self._enabled_check.on_clicked(self._toggle_editing)

        if mode_axes is not None:
            self._mode_radio = RadioButtons(mode_axes, ["Add", "Remove", "Move"], active=0)
            self._mode_radio.on_clicked(self._set_edit_mode)

        if save_button_axes is not None:
            button = Button(save_button_axes, "Save Cont.")
            button.on_clicked(lambda _event: self.save_continuum())
            self._save_button = button
        if load_button_axes is not None:
            button = Button(load_button_axes, "Load Cont.")
            button.on_clicked(lambda _event: self.load_continuum())
            self._load_button = button
        if reset_button_axes is not None:
            button = Button(reset_button_axes, "Re-estimate")
            button.on_clicked(lambda _event: self.reset_to_auto())
            self._reset_button = button
        if status_axes is not None:
            status_axes.axis("off")
            self._status_text = status_axes.text(0.0, 0.5, "", va="center", ha="left")

        self._cid_click = self._figure.canvas.mpl_connect("button_press_event", self._on_click)
        self._update_display()
        self._update_status()
        if self._figure is not None:
            self._figure.canvas.draw_idle()
        return self

    def _toggle_editing(self, _label):
        self._editing_enabled = not self._editing_enabled

    def _set_edit_mode(self, label):
        self._edit_mode = str(label).strip().lower()

    def _on_click(self, event):
        if not self._editing_enabled or self._axes is None:
            return
        if event.inaxes is not self._axes:
            return
        if event.xdata is None or event.ydata is None:
            return
        if self._edit_mode == "add":
            self.add_point(event.xdata, event.ydata)
        elif self._edit_mode == "remove":
            self.remove_nearest(event.xdata)
        elif self._edit_mode == "move":
            self.move_nearest(event.xdata, event.xdata, event.ydata)

    def _update_status(self):
        if self._status_text is None:
            return
        status = "MODIFIED (unsaved)" if self.state.is_modified else "SAVED"
        self._status_text.set_text(f"Continuum: {status} | {self.state.n_points} point(s)")
