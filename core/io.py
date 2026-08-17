"""Load a spectrum from disk into a universal core.spectrum.Spectrum.

Supports two families of input, matching the FitSpec fundamentals
requirements:

- **Delimited text** (comma-, tab-, or whitespace-separated), either
  with a header row naming columns, or headerless with columns in the
  positional order wave, flux, flux_unc, continuum, model. A plain
  two-column wavelength/flux file is valid; missing optional columns
  never cause a load failure.
- **FITS binary tables**, matching common column-name variants
  case-insensitively. Multi-row array-valued columns (e.g. one row per
  spectral order/segment, as in HST/COS x1d products) are flattened and
  handed to ``Spectrum.from_arrays``, which sorts and removes duplicate
  wavelengths -- exactly what overlapping/out-of-order segments need.

FITS images (e.g. IFU cubes) are not yet supported; that is a
substantially different loading path (WCS-based wavelength axis,
spatial extraction) left for later.

Wavelength-medium (air/vacuum) conversion and permanent rebinning are
deliberately *not* performed here -- those are config-driven decisions
(see ``core.wavelengths`` and ``core.rebinning``) applied by the caller
immediately after loading, not baked into the loader itself.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table

from core.spectrum import Spectrum


__all__ = ["load_spectrum", "load_text_spectrum", "load_fits_spectrum"]


#: Recognized column-name aliases (case-insensitive), by canonical field.
_COLUMN_ALIASES = {
    "wave": ("wave", "wavelength", "lambda", "lam", "wave_ang"),
    "flux": ("flux", "flux_density", "flam", "f"),
    "flux_unc": ("flux_unc", "flux_err", "flux_error", "fluxerr", "error", "err", "sigma", "unc"),
    "continuum": ("cont", "continuum"),
    "model": ("fit", "model", "fitted_flux", "total_model_flux"),
}

#: Positional column order used for headerless delimited text files.
_POSITIONAL_ORDER = ("wave", "flux", "flux_unc", "continuum", "model")

_FITS_EXTENSIONS = {".fits", ".fit", ".fts"}


def _match_column_aliases(column_names) -> dict:
    """Map canonical field -> actual column name, case-insensitively."""
    lookup = {name.lower(): name for name in column_names}
    matched = {}
    for field, aliases in _COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in lookup:
                matched[field] = lookup[alias]
                break
    return matched


def _sniff_delimiter(path: Path) -> "str | None":
    """Return ',' if the first non-comment line looks comma-separated,
    else None (meaning: split on any run of whitespace, handling both
    space- and tab-delimited files uniformly).
    """
    with open(path, "r") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            return "," if "," in stripped else None
    raise ValueError(f"{path}: file contains no data lines.")


def _first_data_line(path: Path, delimiter: "str | None"):
    with open(path, "r") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            return stripped.split(delimiter) if delimiter else stripped.split()
    raise ValueError(f"{path}: file contains no data lines.")


def _is_float(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


def load_text_spectrum(path, *, delimiter: "str | None" = None) -> Spectrum:
    """Load a delimited-text spectrum (comma, tab, or whitespace).

    Parameters
    ----------
    path : str or Path
        File to load.
    delimiter : str, optional
        Force a delimiter (e.g. ','). If not given, it is auto-detected
        from the first data line.

    Returns
    -------
    Spectrum
    """
    path = Path(path)
    if delimiter is None:
        delimiter = _sniff_delimiter(path)

    first_tokens = _first_data_line(path, delimiter)
    header_present = not all(_is_float(token) for token in first_tokens)

    if header_present:
        column_names = [token.strip().lstrip("#").strip() for token in first_tokens]
        table = np.genfromtxt(
            path, delimiter=delimiter, names=column_names, skip_header=0,
            comments="#", dtype=float, encoding=None,
        )
        matched = _match_column_aliases(table.dtype.names)
        if "wave" not in matched or "flux" not in matched:
            raise ValueError(
                f"{path}: could not identify wave/flux columns among header {column_names!r}."
            )
        arrays = {field: np.asarray(table[name], dtype=float) for field, name in matched.items()}
    else:
        data = np.genfromtxt(path, delimiter=delimiter, comments="#", dtype=float, encoding=None)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        n_columns = data.shape[1]
        if n_columns < 2:
            raise ValueError(f"{path}: expected at least wave and flux columns, found {n_columns}.")
        if n_columns > len(_POSITIONAL_ORDER):
            raise ValueError(
                f"{path}: {n_columns} headerless columns exceeds the known positional "
                f"order {_POSITIONAL_ORDER}; add a header row to disambiguate."
            )
        arrays = {field: data[:, i] for i, field in enumerate(_POSITIONAL_ORDER[:n_columns])}

    return Spectrum.from_arrays(
        wave=arrays["wave"], flux=arrays["flux"], flux_unc=arrays.get("flux_unc"),
        continuum=arrays.get("continuum"), model=arrays.get("model"),
        metadata={"source_path": str(path)},
    )


def load_fits_spectrum(path, *, extension=None) -> Spectrum:
    """Load a spectrum from a FITS binary table.

    Parameters
    ----------
    path : str or Path
        File to load.
    extension : int or str, optional
        Which HDU to read. If not given, the first ``BinTableHDU`` in the
        file is used.

    Returns
    -------
    Spectrum
    """
    path = Path(path)
    with fits.open(path) as hdulist:
        if extension is not None:
            hdu = hdulist[extension]
        else:
            hdu = next((h for h in hdulist if isinstance(h, fits.BinTableHDU)), None)
            if hdu is None:
                raise ValueError(f"{path}: no binary table extension found.")
        table = Table.read(hdu)

    matched = _match_column_aliases(table.colnames)
    if "wave" not in matched or "flux" not in matched:
        raise ValueError(
            f"{path}: could not identify wave/flux columns among {table.colnames!r}."
        )

    def _flattened(field):
        if field not in matched:
            return None
        column = np.asarray(table[matched[field]])
        # Multi-row array-valued columns (e.g. one row per echelle/COS
        # segment): flatten across rows. clean_spectrum handles the
        # resulting sort + duplicate-wavelength removal.
        return column.ravel().astype(float)

    arrays = {field: _flattened(field) for field in _COLUMN_ALIASES}
    arrays = {field: value for field, value in arrays.items() if value is not None}

    return Spectrum.from_arrays(
        wave=arrays["wave"], flux=arrays["flux"], flux_unc=arrays.get("flux_unc"),
        continuum=arrays.get("continuum"), model=arrays.get("model"),
        metadata={"source_path": str(path), "extension": hdu.name},
    )


def load_spectrum(path, **kwargs) -> Spectrum:
    """Load a spectrum, dispatching on file extension.

    ``.fits``/``.fit``/``.fts`` are read as FITS binary tables; anything
    else is treated as delimited text. See :func:`load_fits_spectrum` /
    :func:`load_text_spectrum` for accepted keyword arguments.
    """
    path = Path(path)
    if path.suffix.lower() in _FITS_EXTENSIONS:
        return load_fits_spectrum(path, **kwargs)
    return load_text_spectrum(path, **kwargs)
