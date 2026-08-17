import numpy as np
import pytest

from core.masking import (
    FitMaskState,
    initial_mask_from_intervals,
    load_mask_file,
    save_mask_file,
)
from core.spectrum import Spectrum
from gui.mask_controller import MaskController


def test_initial_include_then_exclude_and_invalid():
    wave = np.arange(1000.0, 1010.0)
    valid = np.ones(wave.size, dtype=bool)
    valid[4] = False
    mask = initial_mask_from_intervals(
        wave,
        valid_mask=valid,
        included_intervals=[[1002, 1008]],
        excluded_intervals=[[1006, 1007]],
    )
    expected = np.array([False, False, True, True, False, True, False, False, True, False])
    assert np.array_equal(mask, expected)


def test_working_saved_load_semantics():
    state = FitMaskState.from_initial_mask(np.array([True, True, False, True]))
    state.unset_selection(np.array([False, True, False, False]))
    assert state.is_modified
    assert np.array_equal(state.working_mask, [True, False, False, True])
    state.load()
    assert not state.is_modified
    assert np.array_equal(state.working_mask, [True, True, False, True])
    state.unset_selection(np.array([False, True, False, False]))
    state.save()
    assert not state.is_modified
    state.set_selection(np.array([False, True, False, False]))
    state.load()
    assert np.array_equal(state.working_mask, [True, False, False, True])


def test_invalid_pixel_cannot_be_restored():
    valid = np.array([True, False, True])
    state = FitMaskState.from_initial_mask(valid, valid_mask=valid)
    state.set_selection(np.ones(3, dtype=bool))
    assert np.array_equal(state.working_mask, [True, False, True])


def test_rectangle_uses_both_wave_and_flux():
    wave = np.arange(5.0)
    flux = np.array([0.0, 10.0, 2.0, 3.0, 4.0])
    state = FitMaskState.from_initial_mask(np.ones(5, dtype=bool))
    state.apply_rectangle(wave, flux, (0.5, 3.5, 0.0, 5.0), mode="unset")
    # x selects indices 1,2,3, but y excludes the high-flux index 1.
    assert np.array_equal(state.working_mask, [True, True, False, False, True])


def test_velocity_window_replaces_working_but_not_saved():
    wave = np.linspace(4990.0, 5010.0, 1001)
    spec = Spectrum.from_arrays(wave, np.ones_like(wave), np.ones_like(wave))
    ctl = MaskController(spec, fit_mode="emission")
    original_saved = ctl.saved_mask.copy()
    ctl.set_velocity_window([5000.0], 300.0)
    assert ctl.is_modified
    assert np.count_nonzero(ctl.working_mask) < wave.size
    assert np.array_equal(ctl.saved_mask, original_saved)
    ctl.load_mask()
    assert np.array_equal(ctl.working_mask, original_saved)


def test_stellar_rejects_velocity_window():
    wave = np.linspace(4990.0, 5010.0, 100)
    spec = Spectrum.from_arrays(wave, np.ones_like(wave), np.ones_like(wave))
    ctl = MaskController(spec, fit_mode="stellar")
    with pytest.raises(RuntimeError, match="only available for emission and absorption"):
        ctl.set_velocity_window([5000.0], 300.0)


def test_save_file_and_load_file_are_grid_safe(tmp_path):
    wave = np.arange(10.0)
    state = FitMaskState.from_initial_mask(np.ones(10, dtype=bool))
    state.unset_selection(np.arange(10) % 2 == 0)
    state.save()
    path = tmp_path / "mask.npz"
    save_mask_file(path, wave, state)
    loaded = load_mask_file(path, wave)
    assert np.array_equal(loaded.working_mask, state.saved_mask)
    assert not loaded.is_modified
    with pytest.raises(ValueError, match="wavelength grid"):
        load_mask_file(path, wave + 0.1)
