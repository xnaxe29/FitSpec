"""Reusable interactive mask controller for FitSpec GUIs.

This module contains no science-model logic.  It connects a Spectrum-like
object (``wave``, ``flux``, ``mask``) to :class:`core.masking.FitMaskState`
and optionally to Matplotlib widgets.

Velocity-window operations are deliberately guarded: they are available only
when ``fit_mode`` is ``'emission'`` or ``'absorption'``.  Stellar fitting has
no line-centred velocity-window mask.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from core.masking import (
    FitMaskState,
    initial_mask_from_intervals,
    load_mask_file,
    save_mask_file,
)

__all__ = ["MaskController"]


class MaskController:
    """Own working/saved mask behaviour and synchronize ``spectrum.mask``."""

    _LINE_MODES = {"emission", "absorption"}

    def __init__(
        self,
        spectrum,
        *,
        fit_mode: str,
        included_intervals=None,
        excluded_intervals=None,
        invalid_mask=None,
        mask_path=None,
        on_change: "Callable[[np.ndarray, str], None] | None" = None,
    ):
        self.spectrum = spectrum
        self.fit_mode = str(fit_mode).strip().lower()
        if self.fit_mode not in {"stellar", "emission", "absorption"}:
            raise ValueError("fit_mode must be 'stellar', 'emission', or 'absorption'.")

        wave = np.asarray(spectrum.wave, dtype=float)
        flux = np.asarray(spectrum.flux, dtype=float)
        if wave.shape != flux.shape or wave.ndim != 1:
            raise ValueError("spectrum.wave and spectrum.flux must be equal-length 1-D arrays.")

        if invalid_mask is None:
            valid = np.isfinite(wave) & np.isfinite(flux)
            flux_unc = getattr(spectrum, "flux_unc", None)
            if flux_unc is not None:
                unc = np.asarray(flux_unc, dtype=float)
                valid &= np.isfinite(unc) & (unc > 0)
        else:
            invalid = np.asarray(invalid_mask, dtype=bool)
            if invalid.shape != wave.shape:
                raise ValueError("invalid_mask must match the spectrum length.")
            valid = ~invalid

        initial = initial_mask_from_intervals(
            wave,
            valid_mask=valid,
            included_intervals=included_intervals,
            excluded_intervals=excluded_intervals,
        )
        self.state = FitMaskState.from_initial_mask(initial, valid_mask=valid)
        if mask_path is None:
            source_path = getattr(spectrum, "metadata", {}).get("source_path")
            if source_path:
                source = Path(source_path).expanduser()
                self.mask_path = source.with_name(source.name + ".fitspec_mask.npz")
            else:
                self.mask_path = Path("fitspec_mask.npz")
        else:
            self.mask_path = Path(mask_path).expanduser()
        self.on_change = on_change

        # Optional Matplotlib state, initialized by connect_matplotlib().
        # Must be set before the first _sync() call below, since _sync()
        # calls _update_status(), which reads self._status_text.
        self._selector = None
        self._mode_radio = None
        self._enabled_check = None
        self._status_text = None
        self._figure = None
        self._axes = None
        self._selection_enabled = False
        self._edit_mode = "unset"

        self._sync("initialized")

    @property
    def working_mask(self):
        return self.state.working_mask

    @property
    def saved_mask(self):
        return self.state.saved_mask

    @property
    def is_modified(self):
        return self.state.is_modified

    def _sync(self, reason: str):
        # Spectrum.mask remains the universal science-facing mask.
        self.spectrum.mask = self.state.working_mask.copy()
        if self.on_change is not None:
            self.on_change(self.spectrum.mask, reason)
        self._update_status()
        if self._figure is not None:
            self._figure.canvas.draw_idle()
        return self.spectrum.mask

    # ------------------------------------------------------------------
    # Generic SET/UNSET operations (all fitting modes)
    # ------------------------------------------------------------------
    def set_points(self, selection):
        self.state.set_selection(selection)
        return self._sync("set_points")

    def unset_points(self, selection):
        self.state.unset_selection(selection)
        return self._sync("unset_points")

    def apply_rectangle(self, xmin, xmax, ymin, ymax, *, mode=None):
        use_mode = self._edit_mode if mode is None else mode
        self.state.apply_rectangle(
            self.spectrum.wave,
            self.spectrum.flux,
            (xmin, xmax, ymin, ymax),
            mode=use_mode,
        )
        return self._sync(f"rectangle_{str(use_mode).lower()}")

    def reset_to_initial(self):
        self.state.reset_to_initial()
        return self._sync("reset_to_initial")

    # ------------------------------------------------------------------
    # Velocity-window operations (emission/absorption ONLY)
    # ------------------------------------------------------------------
    def set_velocity_window(
        self,
        line_centers,
        velocity_range_kms=None,
        *,
        velocity_min_kms=None,
        velocity_max_kms=None,
        redshift=None,
    ):
        """Immediately REPLACE the working mask with the requested windows.

        This is the live-slider behaviour requested for emission/absorption:
        changing (for example) ±500 to ±650 km/s immediately redraws the
        working mask around every active transition, but does not touch the
        saved mask until ``save_mask()`` is explicitly called.
        """
        if self.fit_mode not in self._LINE_MODES:
            raise RuntimeError(
                "Velocity-window masking is only available for emission and absorption fitting."
            )
        if redshift is None:
            redshift = float(getattr(self.spectrum, "redshift", 0.0))
        self.state.apply_velocity_window(
            self.spectrum.wave,
            line_centers,
            velocity_range_kms,
            velocity_min_kms=velocity_min_kms,
            velocity_max_kms=velocity_max_kms,
            redshift=redshift,
            mode="replace",
        )
        return self._sync("velocity_window")

    # ------------------------------------------------------------------
    # Explicit Save / Load semantics
    # ------------------------------------------------------------------
    def save_mask(self, path=None):
        """Commit the working mask; optionally persist it to disk."""
        self.state.save()
        target = self.mask_path if path is None else Path(path).expanduser()
        if target is not None:
            save_mask_file(
                target,
                self.spectrum.wave,
                self.state,
                metadata={"fit_mode": self.fit_mode},
            )
            self.mask_path = target
        return self._sync("save_mask")

    def load_mask(self, path=None):
        """Restore the last saved mask, from memory or an explicit mask file."""
        target = self.mask_path if path is None else Path(path).expanduser()
        if target is not None and target.is_file():
            self.state = load_mask_file(
                target,
                self.spectrum.wave,
                valid_mask=self.state.valid_mask,
            )
            self.mask_path = target
        else:
            self.state.load()
        return self._sync("load_mask")

    # ------------------------------------------------------------------
    # Optional generic Matplotlib controls for any FitSpec GUI
    # ------------------------------------------------------------------
    def connect_matplotlib(
        self,
        spectrum_axes,
        *,
        selector_check_axes=None,
        mode_axes=None,
        save_button_axes=None,
        load_button_axes=None,
        reset_button_axes=None,
        status_axes=None,
    ):
        """Attach rectangle SET/UNSET and Save/Load controls to a plot.

        The caller owns layout and simply supplies axes for the desired
        controls.  This keeps the controller reusable across stellar,
        emission, and absorption GUIs.
        """
        from matplotlib.widgets import Button, CheckButtons, RadioButtons, RectangleSelector

        self._axes = spectrum_axes
        self._figure = spectrum_axes.figure

        self._selector = RectangleSelector(
            spectrum_axes,
            self._on_rectangle,
            useblit=True,
            button=[1],
            minspanx=0,
            minspany=0,
            spancoords="data",
            interactive=False,
        )
        self._selector.set_active(False)

        if selector_check_axes is not None:
            self._enabled_check = CheckButtons(selector_check_axes, ["Region selection"], [False])
            self._enabled_check.on_clicked(self._toggle_selection)

        if mode_axes is not None:
            self._mode_radio = RadioButtons(mode_axes, ["SET", "UNSET"], active=1)
            self._mode_radio.on_clicked(self._set_edit_mode)

        if save_button_axes is not None:
            button = Button(save_button_axes, "Save Mask")
            button.on_clicked(lambda _event: self.save_mask())
            self._save_button = button
        if load_button_axes is not None:
            button = Button(load_button_axes, "Load Mask")
            button.on_clicked(lambda _event: self.load_mask())
            self._load_button = button
        if reset_button_axes is not None:
            button = Button(reset_button_axes, "Reset Mask")
            button.on_clicked(lambda _event: self.reset_to_initial())
            self._reset_button = button
        if status_axes is not None:
            status_axes.axis("off")
            self._status_text = status_axes.text(0.0, 0.5, "", va="center", ha="left")
            self._update_status()

        return self

    def _toggle_selection(self, _label):
        self._selection_enabled = not self._selection_enabled
        if self._selector is not None:
            self._selector.set_active(self._selection_enabled)

    def _set_edit_mode(self, label):
        self._edit_mode = str(label).strip().lower()

    def _on_rectangle(self, press_event, release_event):
        if not self._selection_enabled:
            return
        if None in (press_event.xdata, release_event.xdata, press_event.ydata, release_event.ydata):
            return
        self.apply_rectangle(
            press_event.xdata,
            release_event.xdata,
            press_event.ydata,
            release_event.ydata,
            mode=self._edit_mode,
        )

    def _update_status(self):
        if self._status_text is None:
            return
        status = "MODIFIED (unsaved)" if self.state.is_modified else "SAVED"
        self._status_text.set_text(
            f"Mask: {status} | selected {self.state.n_selected}/{self.state.valid_mask.size}"
        )
