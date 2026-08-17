"""Universal fitting-mask primitives and state management for FitSpec.

FitSpec uses one convention everywhere::

    True  = use this pixel in the fit
    False = do not use this pixel in the fit

There are two separate ideas in this module:

1. *Mask builders* (wavelength intervals and line-centred velocity windows)
   create boolean selections.
2. :class:`FitMaskState` owns the user-facing mask lifecycle:

   ``initial_mask``
       The mask established when the fitting session starts.  It is built
       from valid pixels plus optional config ``included_intervals`` and
       ``excluded_intervals``.

   ``working_mask``
       The mask currently active in the GUI and therefore the mask used by
       fitting.  Interactive point/rectangle edits and, for emission or
       absorption fitting only, velocity-window updates modify this mask
       immediately.

   ``saved_mask``
       The last mask explicitly accepted by the user with *Save Mask*.
       Unsaved GUI changes never alter it.  *Load Mask* replaces the working
       mask with this saved state.

Invalid pixels are a hard constraint: no operation can restore them.
Velocity-window helpers are intentionally generic numerical primitives, but
FitSpec's GUI/controller exposes them only in emission and absorption modes.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy import constants as const

_C_KMS = const.c.to("km/s").value

__all__ = [
    "region_mask",
    "initial_mask_from_intervals",
    "velocity_window_mask",
    "FitMaskState",
    "save_mask_file",
    "load_mask_file",
    # Backwards compatibility with the early FitSpec core API.
    "MaskComponents",
    "combine_masks",
]


def _validate_bool_mask(mask, n_pixels: int, name: str) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (n_pixels,):
        raise ValueError(f"{name} must be a boolean array of length {n_pixels}.")
    return mask


def region_mask(wave, intervals, *, include: bool = True) -> np.ndarray:
    """Return a boolean mask from wavelength intervals.

    Empty include intervals select nothing; empty exclude intervals exclude
    nothing.  Interval bounds are inclusive.
    """
    wave = np.asarray(wave, dtype=float)
    if wave.ndim != 1:
        raise ValueError("wave must be one-dimensional.")

    intervals = np.asarray(intervals, dtype=float)
    if intervals.size == 0:
        inside = np.zeros(wave.size, dtype=bool)
        return inside if include else ~inside
    if intervals.ndim != 2 or intervals.shape[1] != 2:
        raise ValueError("intervals must have shape (N, 2).")

    inside = np.zeros(wave.size, dtype=bool)
    for lower, upper in intervals:
        if not (np.isfinite(lower) and np.isfinite(upper)):
            raise ValueError("Interval bounds must be finite.")
        if upper <= lower:
            raise ValueError(
                f"Interval upper bound must exceed lower bound; got ({lower}, {upper})."
            )
        inside |= (wave >= lower) & (wave <= upper)
    return inside if include else ~inside


def initial_mask_from_intervals(
    wave,
    *,
    valid_mask=None,
    included_intervals=None,
    excluded_intervals=None,
) -> np.ndarray:
    """Build the initial FitSpec mask from config-style interval lists.

    Rules
    -----
    * With no included list, all valid pixels start selected.
    * With an included list, only pixels inside those intervals start selected.
    * Excluded intervals are then subtracted.
    * ``valid_mask`` is applied last and can never be overridden.
    """
    wave = np.asarray(wave, dtype=float)
    if wave.ndim != 1:
        raise ValueError("wave must be one-dimensional.")
    n = wave.size

    if valid_mask is None:
        valid = np.ones(n, dtype=bool)
    else:
        valid = _validate_bool_mask(valid_mask, n, "valid_mask")

    has_include = included_intervals is not None and np.asarray(included_intervals).size > 0
    if has_include:
        selected = region_mask(wave, included_intervals, include=True)
    else:
        selected = np.ones(n, dtype=bool)

    has_exclude = excluded_intervals is not None and np.asarray(excluded_intervals).size > 0
    if has_exclude:
        selected &= region_mask(wave, excluded_intervals, include=False)

    selected &= valid
    return selected


def velocity_window_mask(
    wave,
    line_centers,
    velocity_range_kms=None,
    *,
    velocity_min_kms=None,
    velocity_max_kms=None,
    redshift: float = 0.0,
) -> np.ndarray:
    """Select pixels in velocity windows around one or more transitions.

    Either provide the legacy symmetric ``velocity_range_kms`` half-width or
    provide ``velocity_min_kms`` and ``velocity_max_kms`` for an asymmetric
    interval.  This helper is used only by emission/absorption controllers.
    """
    wave = np.asarray(wave, dtype=float)
    if wave.ndim != 1:
        raise ValueError("wave must be one-dimensional.")

    centers = np.atleast_1d(np.asarray(line_centers, dtype=float))
    if centers.size == 0 or np.any(centers <= 0) or not np.all(np.isfinite(centers)):
        raise ValueError("line_centers must contain finite, positive wavelengths.")

    if velocity_range_kms is not None:
        half = float(velocity_range_kms)
        if not np.isfinite(half) or half <= 0:
            raise ValueError("velocity_range_kms must be finite and positive.")
        if velocity_min_kms is not None or velocity_max_kms is not None:
            raise ValueError(
                "Use either velocity_range_kms or velocity_min_kms/velocity_max_kms, not both."
            )
        vmin, vmax = -half, half
    else:
        if velocity_min_kms is None or velocity_max_kms is None:
            raise ValueError(
                "Provide velocity_range_kms, or both velocity_min_kms and velocity_max_kms."
            )
        vmin = float(velocity_min_kms)
        vmax = float(velocity_max_kms)
        if not (np.isfinite(vmin) and np.isfinite(vmax)) or vmax <= vmin:
            raise ValueError("Require finite velocity limits with velocity_max_kms > velocity_min_kms.")

    if not np.isfinite(redshift) or redshift <= -1:
        raise ValueError("redshift must be finite and greater than -1.")

    selected = np.zeros(wave.size, dtype=bool)
    for rest_wave in centers:
        shifted_center = rest_wave * (1.0 + float(redshift))
        velocity = _C_KMS * (wave / shifted_center - 1.0)
        selected |= (velocity >= vmin) & (velocity <= vmax)
    return selected


@dataclass
class FitMaskState:
    """Initial, working, and explicitly saved FitSpec masks."""

    valid_mask: np.ndarray
    initial_mask: np.ndarray
    working_mask: np.ndarray
    saved_mask: np.ndarray

    def __post_init__(self):
        self.valid_mask = np.asarray(self.valid_mask, dtype=bool).copy()
        n = self.valid_mask.size
        if self.valid_mask.ndim != 1:
            raise ValueError("valid_mask must be one-dimensional.")
        self.initial_mask = _validate_bool_mask(self.initial_mask, n, "initial_mask").copy()
        self.working_mask = _validate_bool_mask(self.working_mask, n, "working_mask").copy()
        self.saved_mask = _validate_bool_mask(self.saved_mask, n, "saved_mask").copy()
        self._enforce_validity_all()

    @classmethod
    def from_initial_mask(cls, initial_mask, *, valid_mask=None) -> "FitMaskState":
        initial = np.asarray(initial_mask, dtype=bool)
        if initial.ndim != 1:
            raise ValueError("initial_mask must be one-dimensional.")
        valid = np.ones(initial.size, dtype=bool) if valid_mask is None else _validate_bool_mask(
            valid_mask, initial.size, "valid_mask"
        )
        initial = initial & valid
        return cls(valid, initial, initial.copy(), initial.copy())

    @classmethod
    def from_spectrum(
        cls,
        wave,
        *,
        invalid_mask=None,
        included_intervals=None,
        excluded_intervals=None,
    ) -> "FitMaskState":
        wave = np.asarray(wave, dtype=float)
        if invalid_mask is None:
            valid = np.ones(wave.size, dtype=bool)
        else:
            invalid = _validate_bool_mask(invalid_mask, wave.size, "invalid_mask")
            valid = ~invalid
        initial = initial_mask_from_intervals(
            wave,
            valid_mask=valid,
            included_intervals=included_intervals,
            excluded_intervals=excluded_intervals,
        )
        return cls.from_initial_mask(initial, valid_mask=valid)

    @property
    def is_modified(self) -> bool:
        """True when the working state differs from the last explicit save."""
        return not np.array_equal(self.working_mask, self.saved_mask)

    @property
    def n_selected(self) -> int:
        return int(np.count_nonzero(self.working_mask))

    def _enforce_validity_all(self) -> None:
        self.initial_mask &= self.valid_mask
        self.working_mask &= self.valid_mask
        self.saved_mask &= self.valid_mask

    def _selection(self, selection, name="selection") -> np.ndarray:
        return _validate_bool_mask(selection, self.valid_mask.size, name)

    def set_selection(self, selection) -> np.ndarray:
        """Add selected valid pixels to the working mask."""
        selection = self._selection(selection)
        self.working_mask |= selection & self.valid_mask
        return self.working_mask

    def unset_selection(self, selection) -> np.ndarray:
        """Remove selected pixels from the working mask."""
        selection = self._selection(selection)
        self.working_mask &= ~selection
        self.working_mask &= self.valid_mask
        return self.working_mask

    def replace_working(self, selection) -> np.ndarray:
        """Replace the working mask, always enforcing valid pixels."""
        selection = self._selection(selection)
        self.working_mask = selection.copy() & self.valid_mask
        return self.working_mask

    def apply_rectangle(self, wave, flux, bounds, *, mode: str) -> np.ndarray:
        """SET or UNSET points lying inside an x-y rectangle."""
        wave = np.asarray(wave, dtype=float)
        flux = np.asarray(flux, dtype=float)
        if wave.shape != self.working_mask.shape or flux.shape != self.working_mask.shape:
            raise ValueError("wave and flux must match the mask length.")
        xmin, xmax, ymin, ymax = map(float, bounds)
        if xmax < xmin:
            xmin, xmax = xmax, xmin
        if ymax < ymin:
            ymin, ymax = ymax, ymin
        selection = (
            np.isfinite(wave) & np.isfinite(flux)
            & (wave >= xmin) & (wave <= xmax)
            & (flux >= ymin) & (flux <= ymax)
        )
        return self.apply_selection(selection, mode=mode)

    def apply_selection(self, selection, *, mode: str) -> np.ndarray:
        mode = str(mode).strip().lower()
        if mode == "set":
            return self.set_selection(selection)
        if mode == "unset":
            return self.unset_selection(selection)
        if mode == "replace":
            return self.replace_working(selection)
        raise ValueError("mode must be 'set', 'unset', or 'replace'.")

    def apply_velocity_window(
        self,
        wave,
        line_centers,
        velocity_range_kms=None,
        *,
        velocity_min_kms=None,
        velocity_max_kms=None,
        redshift: float = 0.0,
        mode: str = "replace",
    ) -> np.ndarray:
        selection = velocity_window_mask(
            wave,
            line_centers,
            velocity_range_kms,
            velocity_min_kms=velocity_min_kms,
            velocity_max_kms=velocity_max_kms,
            redshift=redshift,
        )
        return self.apply_selection(selection, mode=mode)

    def save(self) -> np.ndarray:
        """Commit the working mask as the new explicitly saved state."""
        self.saved_mask = self.working_mask.copy() & self.valid_mask
        return self.saved_mask

    def load(self) -> np.ndarray:
        """Discard unsaved edits and restore the explicitly saved state."""
        self.working_mask = self.saved_mask.copy() & self.valid_mask
        return self.working_mask

    def reset_to_initial(self) -> np.ndarray:
        """Restore config/startup state without changing the saved state."""
        self.working_mask = self.initial_mask.copy() & self.valid_mask
        return self.working_mask

    def select_all_valid(self) -> np.ndarray:
        self.working_mask = self.valid_mask.copy()
        return self.working_mask

    def clear_all(self) -> np.ndarray:
        self.working_mask = np.zeros_like(self.valid_mask)
        return self.working_mask


def save_mask_file(path, wave, state: FitMaskState, *, metadata=None) -> Path:
    """Persist the *saved* mask state to a compressed NPZ file.

    The current working mask is not silently committed.  Call ``state.save()``
    first (the GUI Save Mask action does this) and then write the file.
    """
    path = Path(path).expanduser()
    wave = np.asarray(wave, dtype=float)
    if wave.shape != state.saved_mask.shape:
        raise ValueError("wave must have the same length as the mask state.")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        wavelength=wave,
        valid_mask=state.valid_mask,
        initial_mask=state.initial_mask,
        saved_mask=state.saved_mask,
        metadata=np.asarray([repr({} if metadata is None else dict(metadata))]),
    )
    return path


def load_mask_file(path, wave, *, valid_mask=None, wavelength_rtol=1e-10, wavelength_atol=1e-8) -> FitMaskState:
    """Load a persisted mask and make it both saved and working state.

    A wavelength-grid mismatch is fatal.  This prevents accidentally loading
    a mask made before a different permanent rebinning operation.
    """
    path = Path(path).expanduser()
    wave = np.asarray(wave, dtype=float)
    with np.load(path, allow_pickle=False) as data:
        stored_wave = np.asarray(data["wavelength"], dtype=float)
        stored_saved = np.asarray(data["saved_mask"], dtype=bool)
        stored_initial = np.asarray(data["initial_mask"], dtype=bool)
        stored_valid = np.asarray(data["valid_mask"], dtype=bool)

    if stored_wave.shape != wave.shape or not np.allclose(
        stored_wave, wave, rtol=wavelength_rtol, atol=wavelength_atol, equal_nan=False
    ):
        raise ValueError(
            "Saved mask wavelength grid does not match the current spectrum. "
            "A mask cannot be reused across a different permanent rebinning/grid."
        )
    if valid_mask is not None:
        current_valid = _validate_bool_mask(valid_mask, wave.size, "valid_mask")
        stored_valid &= current_valid

    state = FitMaskState(
        valid_mask=stored_valid,
        initial_mask=stored_initial,
        working_mask=stored_saved,
        saved_mask=stored_saved,
    )
    return state


# ---------------------------------------------------------------------------
# Backwards-compatible early-core composition API.
# New GUI/science code should use FitMaskState instead.
# ---------------------------------------------------------------------------
@dataclass
class MaskComponents:
    region: "np.ndarray | None" = None
    velocity: "np.ndarray | None" = None
    interactive_add: "np.ndarray | None" = None
    interactive_remove: "np.ndarray | None" = None
    invalid: "np.ndarray | None" = None
    combined: "np.ndarray | None" = None


def combine_masks(
    n_pixels: int, *, region=None, velocity=None,
    interactive_add=None, interactive_remove=None, invalid=None,
) -> MaskComponents:
    """Legacy one-shot mask composition retained for compatibility.

    New FitSpec interfaces must not use this function to arbitrate live GUI
    state; use :class:`FitMaskState`.  The historical OR behaviour is kept
    here only so existing early-core callers do not break.
    """
    def _optional(mask, name):
        return None if mask is None else _validate_bool_mask(mask, n_pixels, name)

    region = _optional(region, "region")
    velocity = _optional(velocity, "velocity")
    interactive_add = _optional(interactive_add, "interactive_add")
    interactive_remove = _optional(interactive_remove, "interactive_remove")
    invalid = _optional(invalid, "invalid")

    if region is not None and velocity is not None:
        selected = region | velocity
    elif region is not None:
        selected = region.copy()
    elif velocity is not None:
        selected = velocity.copy()
    else:
        selected = np.ones(n_pixels, dtype=bool)
    if interactive_add is not None:
        selected |= interactive_add
    if interactive_remove is not None:
        selected &= ~interactive_remove
    if invalid is not None:
        selected &= ~invalid

    return MaskComponents(region, velocity, interactive_add, interactive_remove, invalid, selected)
