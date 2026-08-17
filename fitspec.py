#!/usr/bin/env python3
"""FitSpec command-line entry point.

Launcher precedence
-------------------
Target/session launcher values may be supplied either in the run-directory
``config.dat`` or explicitly on the command line.  Explicit command-line
arguments always win::

    CLI  >  config.dat  >  shared defaults

With no positional spectrum and no ``--run-dir``, FitSpec looks for
``config.dat`` in the current working directory.  This enables the compact
workflow::

    cd /path/to/target_run
    fitspec

when ``config.dat`` defines ``input_spectrum``, ``redshift``, and optionally
``mode``.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from core.config import load_configs
from gui.app import FitSpecApp, build_session
from gui.state import SessionState

_MODES = ("stellar", "emission", "absorption")


@dataclass(frozen=True)
class LaunchSettings:
    """Resolved launcher settings after applying CLI-over-config precedence."""

    spectrum: Path | None
    run_dir: Path
    redshift: float | None
    mode: str | None


def _parser():
    p = argparse.ArgumentParser(description="FitSpec: universal stellar, emission, and absorption spectral fitting GUI")
    p.add_argument("spectrum", nargs="?", help="Input text/CSV/FITS spectrum; overrides config.dat input_spectrum")
    p.add_argument("--config-dir", default=str(Path(__file__).resolve().parent / "config"), help="Directory containing default_config*.dat")
    p.add_argument("--run-dir", default=None, help="Directory containing optional config.dat and receiving outputs")
    p.add_argument("--redshift", type=float, default=None, help="Override config.dat redshift (and legacy shared z) for this session")
    p.add_argument("--session", default=None, help="Load a previously saved fitspec_session.npz instead of an input spectrum")
    p.add_argument("--mode", choices=_MODES, default=None, help="Override config.dat mode and open one science panel immediately after launch")
    p.add_argument("--no-show", action="store_true", help="Build the application without entering Matplotlib's blocking event loop")
    return p


def _bootstrap_run_dir(args) -> Path:
    """Choose where to read config.dat before the input spectrum is resolved."""
    if args.run_dir is not None:
        return Path(args.run_dir).expanduser().resolve()
    if args.spectrum:
        return Path(args.spectrum).expanduser().resolve().parent
    return Path.cwd().resolve()


def _resolve_spectrum_path(raw_value, *, from_config: bool, run_dir: Path) -> Path | None:
    if raw_value in (None, ""):
        return None
    path = Path(str(raw_value)).expanduser()
    # Relative paths written inside config.dat are relative to that config's
    # run directory, not to the installation directory of the executable.
    if from_config and not path.is_absolute():
        path = run_dir / path
    return path.resolve()


def resolve_launch_settings(args, *, config_dir) -> LaunchSettings:
    """Resolve input spectrum, redshift, and mode from CLI + config.dat.

    Explicit CLI values win.  If no positional spectrum is supplied,
    ``input_spectrum`` is read from the run-directory config.  ``redshift``
    is preferred over the legacy shared ``z`` keyword when both are present.
    """
    config_dir = Path(config_dir)
    run_dir = _bootstrap_run_dir(args)
    configs = load_configs(_MODES, config_dir, run_dir=run_dir)
    common = configs["stellar"]  # all base launcher keys are shared by every mode

    if args.spectrum:
        spectrum = _resolve_spectrum_path(args.spectrum, from_config=False, run_dir=run_dir)
    else:
        spectrum = _resolve_spectrum_path(common.get("input_spectrum"), from_config=True, run_dir=run_dir)

    if args.redshift is not None:
        redshift = float(args.redshift)
    elif common.get("redshift") is not None:
        redshift = float(common.get("redshift"))
    else:
        redshift = float(common.get("z", 0.0))

    mode = args.mode if args.mode is not None else common.get("mode")
    if mode in ("", "none", "None"):
        mode = None
    if mode is not None:
        mode = str(mode).strip().lower()
        if mode not in _MODES:
            raise ValueError(f"config.dat mode must be one of {_MODES}, got {mode!r}.")

    return LaunchSettings(spectrum=spectrum, run_dir=run_dir, redshift=redshift, mode=mode)


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    config_dir = Path(args.config_dir)
    launch_mode = args.mode

    if args.session:
        state = SessionState.load(args.session)
        if args.run_dir is not None:
            state.run_dir = Path(args.run_dir).expanduser().resolve()
        state.config_dir = config_dir
        # Session files intentionally do not pickle Config objects: rebuild
        # them from the authoritative defaults + current run/config.dat.
        configs = load_configs(_MODES, config_dir, run_dir=state.run_dir)
        for mode, config in configs.items():
            state.set_config(mode, config)
        if launch_mode is None:
            configured_mode = configs["stellar"].get("mode")
            if configured_mode not in (None, ""):
                launch_mode = str(configured_mode).strip().lower()
    else:
        settings = resolve_launch_settings(args, config_dir=config_dir)
        if settings.spectrum is None:
            parser.error(
                "provide a spectrum path, --session, or set input_spectrum in the run-directory config.dat"
            )
        if not settings.spectrum.is_file():
            parser.error(f"input spectrum not found: {settings.spectrum}")
        state = build_session(
            settings.spectrum,
            config_dir=config_dir,
            run_dir=settings.run_dir,
            redshift=settings.redshift,
        )
        launch_mode = settings.mode

    app = FitSpecApp(state)
    if launch_mode:
        app.open_mode(launch_mode)
    if not args.no_show:
        app.show()
    return app


if __name__ == "__main__":
    main()
