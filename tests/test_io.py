"""Tests for universal spectrum input/output loading semantics."""
import numpy as np
import pytest

from core.io import load_spectrum, load_text_spectrum, load_fits_spectrum


def test_headerless_two_column_text_is_valid(tmp_path):
    path = tmp_path / "two_column.dat"
    np.savetxt(path, np.column_stack(([3.0, 1.0, 2.0], [30.0, 10.0, 20.0])))
    spectrum = load_spectrum(path)
    assert np.allclose(spectrum.wave, [1.0, 2.0, 3.0])
    assert np.allclose(spectrum.flux, [10.0, 20.0, 30.0])
    assert spectrum.flux_unc is None


def test_header_aliases_and_optional_columns_are_recognized(tmp_path):
    path = tmp_path / "spectrum.csv"
    path.write_text(
        "wavelength,flam,error,continuum,model\n"
        "1000,1.0,0.1,0.9,1.1\n"
        "1001,2.0,0.2,1.8,2.1\n"
    )
    spectrum = load_text_spectrum(path)
    assert np.allclose(spectrum.wave, [1000.0, 1001.0])
    assert np.allclose(spectrum.flux_unc, [0.1, 0.2])
    assert np.allclose(spectrum.continuum, [0.9, 1.8])
    assert np.allclose(spectrum.model, [1.1, 2.1])


def test_structural_cleaning_sorts_and_removes_duplicate_wavelengths(tmp_path):
    path = tmp_path / "dirty.dat"
    np.savetxt(path, np.array([
        [1002.0, 3.0, 0.1],
        [1000.0, 1.0, 0.1],
        [1001.0, 2.0, 0.1],
        [1001.0, 99.0, 0.1],
    ]))
    spectrum = load_spectrum(path)
    assert np.allclose(spectrum.wave, [1000.0, 1001.0, 1002.0])
    assert np.allclose(spectrum.flux, [1.0, 2.0, 3.0])


def test_bad_header_without_wave_or_flux_is_rejected(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("x,y,z\n1,2,3\n")
    with pytest.raises(ValueError, match="could not identify wave/flux"):
        load_text_spectrum(path)


def test_too_many_headerless_columns_are_rejected(tmp_path):
    path = tmp_path / "ambiguous.dat"
    np.savetxt(path, np.ones((3, 6)))
    with pytest.raises(ValueError, match="headerless columns"):
        load_text_spectrum(path)


def test_fits_binary_table_aliases_round_trip(tmp_path):
    fits = pytest.importorskip("astropy.io.fits")
    if not all(hasattr(fits, name) for name in ("Column", "PrimaryHDU", "BinTableHDU", "HDUList")):
        pytest.skip("Astropy FITS I/O is not available in this test environment.")
    path = tmp_path / "spectrum.fits"
    columns = [
        fits.Column(name="WAVELENGTH", array=np.array([1000.0, 1001.0, 1002.0]), format="D"),
        fits.Column(name="FLUX", array=np.array([1.0, 2.0, 3.0]), format="D"),
        fits.Column(name="ERROR", array=np.array([0.1, 0.2, 0.3]), format="D"),
    ]
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(columns)]).writeto(path)
    spectrum = load_fits_spectrum(path)
    assert np.allclose(spectrum.wave, [1000.0, 1001.0, 1002.0])
    assert np.allclose(spectrum.flux_unc, [0.1, 0.2, 0.3])
