"""Regression tests for interactive stellar-GUI runtime behaviour."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from core.spectrum import Spectrum
from gui.stellar import StellarGUI


def test_stellar_gui_sensible_limits_follow_loaded_spectrum_scale():
    wave = np.linspace(1200.0, 1700.0, 501)
    flux = 1.0e-12 * (1.0 + 0.08 * np.sin((wave - 1200.0) / 50.0))
    unc = np.full_like(wave, 2.0e-14)
    spectrum = Spectrum.from_arrays(wave, flux, unc)

    gui = StellarGUI.__new__(StellarGUI)
    gui.spectrum = spectrum
    gui.wave_rest = wave.copy()
    gui.fig, gui.ax = plt.subplots()
    gui._set_sensible_limits()

    xmin, xmax = gui.ax.get_xlim()
    ymin, ymax = gui.ax.get_ylim()
    assert xmin < 1200.0 < xmax
    assert xmin > 1100.0
    assert xmax > 1700.0
    assert xmax < 1800.0
    assert 0.5e-12 < ymin < 1.2e-12
    assert 0.8e-12 < ymax < 1.5e-12
    plt.close(gui.fig)


def test_stellar_gui_plot_fit_requires_result(capsys):
    gui = StellarGUI.__new__(StellarGUI)
    gui.result = None
    out = gui._plot_fit()
    assert out is None
    assert "no deterministic stellar fit is available" in capsys.readouterr().out


def test_stellar_gui_plot_fit_saves_all_products(monkeypatch, tmp_path):
    import gui.stellar as stellar_gui_module

    sentinel_result = object()
    fits_path = tmp_path / "stellar_fit.fits"
    expected = {
        "main_pdf": tmp_path / "stellar_fit.pdf",
        "summary_pdf": tmp_path / "stellar_fit_summary.pdf",
        "diag_pdf": tmp_path / "stellar_fit_diag.pdf",
        "diag_txt": tmp_path / "stellar_fit_diag.txt",
    }
    seen = {}

    def fake_products(result, path):
        seen["products"] = (result, path)
        return expected

    monkeypatch.setattr(stellar_gui_module, "save_stellar_plot_products", fake_products)
    gui = StellarGUI.__new__(StellarGUI)
    gui.result = sentinel_result
    gui.result_path = fits_path
    products = gui._plot_fit()
    assert seen["products"] == (sentinel_result, fits_path)
    assert products == expected


def test_stellar_gui_save_fit_writes_complete_product_set(monkeypatch, tmp_path):
    import gui.stellar as stellar_gui_module

    sentinel_result = object()
    fits_path = tmp_path / "stellar_fit.fits"
    seen = {}

    def fake_save(path, result, overwrite=True):
        seen["save"] = (path, result, overwrite)
        return path

    expected = {
        "main_pdf": tmp_path / "stellar_fit.pdf",
        "summary_pdf": tmp_path / "stellar_fit_summary.pdf",
        "diag_pdf": tmp_path / "stellar_fit_diag.pdf",
        "diag_txt": tmp_path / "stellar_fit_diag.txt",
    }

    def fake_products(result, path):
        seen["products"] = (result, path)
        return expected

    monkeypatch.setattr(stellar_gui_module, "save_stellar_result", fake_save)
    monkeypatch.setattr(stellar_gui_module, "save_stellar_plot_products", fake_products)

    gui = StellarGUI.__new__(StellarGUI)
    gui.result = sentinel_result
    gui.result_path = fits_path
    returned = gui._save_fit()

    assert returned == fits_path
    assert seen["save"] == (fits_path, sentinel_result, True)
    assert seen["products"] == (sentinel_result, fits_path)


def test_stellar_gui_slider_groups_have_horizontal_separation(monkeypatch):
    """Population and kinematic slider groups should not crowd each other."""
    import gui.stellar as stellar_gui

    # Layout constants are intentionally checked from source-level widget
    # positions because the GUI constructor requires a real stellar library.
    source = __import__('inspect').getsource(stellar_gui.StellarGUI.__init__)
    assert 'plt.axes([.14,.27,.17,.025])' in source
    assert 'plt.axes([.54,.27,.13,.025])' in source
    # Left sliders end at x=0.31; right sliders begin at x=0.54.
    assert 0.14 + 0.17 < 0.54
