from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.spectrum import Spectrum
from gui.state import SessionState
from gui.app import FitSpecApp, build_session


def test_session_roundtrip_preserves_common_spectrum_and_mask(tmp_path):
    spectrum=Spectrum.from_arrays([1,2,3],[4,5,6],[.1,.2,.3],redshift=.012)
    spectrum.mask=np.array([True,False,True])
    spectrum.metadata["instrument"]="TEST"
    state=SessionState(spectrum=spectrum,spectrum_path=tmp_path/"spec.dat",run_dir=tmp_path)
    out=state.save(tmp_path/"session.npz")
    restored=SessionState.load(out)
    assert np.allclose(restored.spectrum.wave,spectrum.wave)
    assert np.allclose(restored.spectrum.flux_unc,spectrum.flux_unc)
    assert np.array_equal(restored.spectrum.mask,spectrum.mask)
    assert restored.spectrum.redshift==spectrum.redshift
    assert restored.spectrum.metadata["instrument"]=="TEST"


def test_replacing_spectrum_invalidates_old_science_state():
    a=Spectrum.from_arrays([1,2],[1,1],[.1,.1])
    b=Spectrum.from_arrays([3,4],[2,2],[.2,.2])
    state=SessionState(spectrum=a)
    state.results["stellar"]=object(); state.posteriors["stellar"]=object(); state.panels["stellar"]=object()
    state.set_spectrum(b)
    assert state.spectrum is b
    assert state.results=={} and state.posteriors=={} and state.panels=={}


def test_app_passes_exact_shared_spectrum_to_panel(tmp_path):
    spectrum=Spectrum.from_arrays([1,2,3],[1,1,1],[.1,.1,.1])
    state=SessionState(spectrum=spectrum,run_dir=tmp_path)
    for mode in ("stellar","emission","absorption"):
        state.configs[mode]={"mode":mode}

    made=[]
    class DummyPanel:
        def __init__(self,spectrum,config,**kwargs):
            self.spectrum=spectrum; self.config=config; self.kwargs=kwargs
            self.fig=plt.figure()
            made.append(self)

    old=FitSpecApp.PANEL_CLASSES
    FitSpecApp.PANEL_CLASSES={m:DummyPanel for m in ("stellar","emission","absorption")}
    try:
        app=FitSpecApp(state)
        panels=[app.open_mode(m) for m in ("stellar","emission","absorption")]
    finally:
        FitSpecApp.PANEL_CLASSES=old
        plt.close("all")
    assert all(panel.spectrum is spectrum for panel in panels)
    assert state.active_mode=="absorption"
    assert set(state.panels)=={"stellar","emission","absorption"}


def test_build_session_loads_all_mode_configs_and_shared_redshift(tmp_path):
    spectrum_path=tmp_path/"spectrum.dat"
    np.savetxt(spectrum_path,np.column_stack(([1200.,1201.,1202.],[1.,1.1,.9],[.1,.1,.1])))
    config_dir=Path(__file__).resolve().parents[1]/"config"
    state=build_session(spectrum_path,config_dir=config_dir,run_dir=tmp_path,redshift=.03)
    assert state.spectrum.redshift==.03
    assert set(state.configs)=={"stellar","emission","absorption"}
    assert state.spectrum_path==spectrum_path


def test_unified_config_dat_accepts_keys_from_multiple_science_modes(tmp_path):
    spectrum_path=tmp_path/"spectrum.dat"
    np.savetxt(spectrum_path,np.column_stack(([1200.,1201.,1202.],[1.,1.1,.9],[.1,.1,.1])))
    (tmp_path/"config.dat").write_text(
        "stellar_inference_method = emcee\n"
        "emission_n_components = 2\n"
        "absorption_n_components = 3\n"
    )
    config_dir=Path(__file__).resolve().parents[1]/"config"
    state=build_session(spectrum_path,config_dir=config_dir,run_dir=tmp_path)
    assert state.configs["stellar"].get("stellar_inference_method")=="emcee"
    assert state.configs["emission"].get("emission_n_components")==2
    assert state.configs["absorption"].get("absorption_n_components")==3


def test_unified_config_dat_still_rejects_truly_unknown_key(tmp_path):
    import pytest
    from core.config import ConfigError
    spectrum_path=tmp_path/"spectrum.dat"
    np.savetxt(spectrum_path,np.column_stack(([1200.,1201.],[1.,1.],[.1,.1])))
    (tmp_path/"config.dat").write_text("this_key_does_not_exist = 1\n")
    config_dir=Path(__file__).resolve().parents[1]/"config"
    with pytest.raises(ConfigError):
        build_session(spectrum_path,config_dir=config_dir,run_dir=tmp_path)


def test_launcher_can_take_spectrum_redshift_and_mode_from_config(tmp_path):
    from fitspec import _parser, resolve_launch_settings
    spectrum_path = tmp_path / "target.dat"
    np.savetxt(spectrum_path, np.column_stack(([1200.,1201.],[1.,1.],[.1,.1])))
    (tmp_path / "config.dat").write_text(
        "input_spectrum = target.dat\n"
        "redshift = 0.001721\n"
        "mode = stellar\n"
    )
    config_dir = Path(__file__).resolve().parents[1] / "config"
    args = _parser().parse_args(["--run-dir", str(tmp_path), "--no-show"])
    settings = resolve_launch_settings(args, config_dir=config_dir)
    assert settings.spectrum == spectrum_path.resolve()
    assert settings.run_dir == tmp_path.resolve()
    assert settings.redshift == 0.001721
    assert settings.mode == "stellar"


def test_launcher_cli_overrides_config_values(tmp_path):
    from fitspec import _parser, resolve_launch_settings
    config_spectrum = tmp_path / "config_target.dat"
    cli_spectrum = tmp_path / "cli_target.dat"
    for path in (config_spectrum, cli_spectrum):
        np.savetxt(path, np.column_stack(([1200.,1201.],[1.,1.],[.1,.1])))
    (tmp_path / "config.dat").write_text(
        f"input_spectrum = {config_spectrum.name}\n"
        "redshift = 0.001721\n"
        "mode = stellar\n"
    )
    config_dir = Path(__file__).resolve().parents[1] / "config"
    args = _parser().parse_args([
        str(cli_spectrum), "--run-dir", str(tmp_path),
        "--redshift", "0.02", "--mode", "emission", "--no-show",
    ])
    settings = resolve_launch_settings(args, config_dir=config_dir)
    assert settings.spectrum == cli_spectrum.resolve()
    assert settings.redshift == 0.02
    assert settings.mode == "emission"
