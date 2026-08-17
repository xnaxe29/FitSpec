"""Unified deterministic stellar-fit results for FitSpec.

One result schema is used for UV and optical fits.  The fitting algorithm is
identical in both regimes; only the selected stellar library and line-list
metadata may differ.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import numpy as np
from astropy.io import fits

__all__ = ["StellarFitDiagnostics", "StellarFitResult", "save_stellar_result", "load_stellar_result"]


@dataclass
class StellarFitDiagnostics:
    correlation_matrix: np.ndarray | None = None
    singular_values: np.ndarray | None = None
    effective_rank: int | None = None
    condition_number: float | None = None
    single_ssp_chi_square: np.ndarray | None = None
    single_ssp_delta_chi_square: np.ndarray | None = None
    dominant_ssp_distance: np.ndarray | None = None
    transformed_model_fluxes: np.ndarray | None = None


@dataclass
class StellarFitResult:
    regime: str
    library_family: str
    wave: np.ndarray
    flux: np.ndarray
    flux_unc: np.ndarray | None
    mask: np.ndarray
    model: np.ndarray
    stellar_model: np.ndarray
    gas_model: np.ndarray
    coefficients: np.ndarray
    ages_myr: np.ndarray
    metallicity_codes: np.ndarray
    metallicities_solar: np.ndarray
    ebv: float
    velocity_kms: float
    sigma_kms: float
    gas_velocity_kms: float = 0.0
    gas_sigma_kms: float = 0.0
    gas_names: tuple[str, ...] = ()
    gas_rest_wavelengths: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    gas_amplitudes: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    chi_square: float = np.nan
    reduced_chi_square: float = np.nan
    degrees_of_freedom: int = 0
    success: bool = False
    message: str = ""
    n_function_evaluations: int = 0
    parameter_names: tuple[str, ...] = ()
    parameter_uncertainties: np.ndarray | None = None
    nonlinear_covariance: np.ndarray | None = None
    coefficient_kind: str = "formed_mass_Msun"
    surviving_mass_fractions: np.ndarray | None = None
    light_fractions: np.ndarray | None = None
    diagnostics: StellarFitDiagnostics | None = None
    resolution_source: str | None = None
    metadata: dict = field(default_factory=dict)
    optimizer_result: object | None = field(default=None, repr=False)

    @property
    def dominant_index(self):
        values = np.where(np.isfinite(self.coefficients), self.coefficients, -np.inf)
        return int(np.nanargmax(values))

    @property
    def total_coefficient(self):
        return float(np.nansum(self.coefficients))

    @property
    def current_mass_coefficients(self):
        if self.surviving_mass_fractions is None:
            return np.asarray(self.coefficients, dtype=float)
        return np.asarray(self.coefficients, dtype=float) * np.asarray(self.surviving_mass_fractions, dtype=float)


def _jsonable_metadata(metadata):
    out = {}
    for key, value in (metadata or {}).items():
        if value is None or isinstance(value, (str, int, float, bool)):
            out[str(key)] = value
        elif isinstance(value, np.generic):
            out[str(key)] = value.item()
        else:
            out[str(key)] = str(value)
    return out


def save_stellar_result(path, result: StellarFitResult, *, overwrite=True):
    """Persist a unified stellar fit to FITS without regime-specific formats."""
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)

    ph = fits.PrimaryHDU()
    h = ph.header
    h["REGIME"] = str(result.regime).upper()
    h["LIBRARY"] = str(result.library_family)[:68]
    h["METHOD"] = "unified_variable_projection"
    h["EBV"] = float(result.ebv)
    h["VSTAR"] = float(result.velocity_kms)
    h["SIGSTAR"] = float(result.sigma_kms)
    h["VGAS"] = float(result.gas_velocity_kms)
    h["SIGGAS"] = float(result.gas_sigma_kms)
    h["CHI2"] = float(result.chi_square)
    h["RCHI2"] = float(result.reduced_chi_square)
    h["DOF"] = int(result.degrees_of_freedom)
    h["SUCCESS"] = bool(result.success)
    h["NFEV"] = int(result.n_function_evaluations)
    h["DOMINDEX"] = int(result.dominant_index)
    h["COEFKIND"] = str(result.coefficient_kind)[:68]
    if result.resolution_source is not None:
        h["RESSRC"] = str(result.resolution_source)[:68]
    for key, value in _jsonable_metadata(result.metadata).items():
        if value is None:
            continue
        try:
            h[str(key).upper()[:8]] = value
        except Exception:
            pass

    n = result.wave.size
    unc = np.full(n, np.nan) if result.flux_unc is None else np.asarray(result.flux_unc, float)
    spec = fits.BinTableHDU.from_columns([
        fits.Column(name="WAVELENGTH", format="D", unit="Angstrom", array=np.asarray(result.wave, float)),
        fits.Column(name="FLUX", format="D", array=np.asarray(result.flux, float)),
        fits.Column(name="FLUX_UNC", format="D", array=unc),
        fits.Column(name="MODEL", format="D", array=np.asarray(result.model, float)),
        fits.Column(name="STELLAR_MODEL", format="D", array=np.asarray(result.stellar_model, float)),
        fits.Column(name="GAS_MODEL", format="D", array=np.asarray(result.gas_model, float)),
        fits.Column(name="FIT_MASK", format="L", array=np.asarray(result.mask, bool)),
    ], name="SPECTRUM")

    ns = result.coefficients.size
    codes = np.asarray(result.metallicity_codes).astype(str)
    width = max(1, max(map(len, codes))) if codes.size else 1
    surv = np.full(ns, np.nan) if result.surviving_mass_fractions is None else np.asarray(result.surviving_mass_fractions, float)
    light = np.full(ns, np.nan) if result.light_fractions is None else np.asarray(result.light_fractions, float)
    pop = fits.BinTableHDU.from_columns([
        fits.Column(name="AGE_MYR", format="D", array=np.asarray(result.ages_myr, float)),
        fits.Column(name="METALLICITY", format=f"{width}A", array=codes),
        fits.Column(name="METALLICITY_ZSUN", format="D", array=np.asarray(result.metallicities_solar, float)),
        fits.Column(name="COEFFICIENT", format="D", array=np.asarray(result.coefficients, float)),
        fits.Column(name="SURVIVING_MASS_FRACTION", format="D", array=surv),
        fits.Column(name="LIGHT_FRACTION", format="D", array=light),
        fits.Column(name="DOMINANT", format="L", array=np.arange(ns) == result.dominant_index),
    ], name="STELLAR_POPULATION")

    names = np.asarray(result.parameter_names, dtype=str)
    pu = np.full(names.size, np.nan) if result.parameter_uncertainties is None else np.asarray(result.parameter_uncertainties, float)
    nw = max(1, max(map(len, names))) if names.size else 1
    pars = fits.BinTableHDU.from_columns([
        fits.Column(name="PARAMETER", format=f"{nw}A", array=names),
        fits.Column(name="UNCERTAINTY", format="D", array=pu),
    ], name="PARAMETERS")

    gas_names = np.asarray(result.gas_names, dtype=str)
    gw = max(1, max(map(len, gas_names))) if gas_names.size else 1
    gas = fits.BinTableHDU.from_columns([
        fits.Column(name="NAME", format=f"{gw}A", array=gas_names),
        fits.Column(name="REST_WAVELENGTH", format="D", unit="Angstrom", array=np.asarray(result.gas_rest_wavelengths, float)),
        fits.Column(name="AMPLITUDE", format="D", array=np.asarray(result.gas_amplitudes, float)),
    ], name="GAS_LINES")

    hdus = [ph, spec, pop, pars, gas]
    d = result.diagnostics
    if d is not None:
        cols = []
        if d.singular_values is not None:
            cols.append(fits.Column(name="SINGULAR_VALUE", format="D", array=np.asarray(d.singular_values, float)))
        if cols:
            dh = fits.BinTableHDU.from_columns(cols, name="DIAGNOSTICS")
            if d.effective_rank is not None: dh.header["EFFRANK"] = int(d.effective_rank)
            if d.condition_number is not None: dh.header["CONDNUM"] = float(d.condition_number)
            hdus.append(dh)
        if d.correlation_matrix is not None:
            hdus.append(fits.ImageHDU(np.asarray(d.correlation_matrix, np.float32), name="SSP_CORRELATION"))
        if d.transformed_model_fluxes is not None:
            hdus.append(fits.ImageHDU(np.asarray(d.transformed_model_fluxes, np.float32), name="SSP_MODELS"))
        if d.single_ssp_chi_square is not None:
            dd = fits.BinTableHDU.from_columns([
                fits.Column(name="CHI2", format="D", array=np.asarray(d.single_ssp_chi_square, float)),
                fits.Column(name="DELTA_CHI2", format="D", array=np.asarray(d.single_ssp_delta_chi_square, float)),
                fits.Column(name="DOM_DISTANCE", format="D", array=np.asarray(d.dominant_ssp_distance, float)),
            ], name="SSP_DIAGNOSTICS")
            hdus.append(dd)

    fits.HDUList(hdus).writeto(path, overwrite=overwrite)
    return path


def load_stellar_result(path):
    """Load a result written by :func:`save_stellar_result`."""
    with fits.open(path) as hdul:
        h = hdul[0].header
        s = hdul["SPECTRUM"].data
        p = hdul["STELLAR_POPULATION"].data
        unc = np.asarray(s["FLUX_UNC"], float)
        unc = None if np.all(~np.isfinite(unc)) else unc
        surv = np.asarray(p["SURVIVING_MASS_FRACTION"], float)
        surv = None if np.all(~np.isfinite(surv)) else surv
        light = np.asarray(p["LIGHT_FRACTION"], float)
        light = None if np.all(~np.isfinite(light)) else light
        parameter_names, parameter_unc = (), None
        if "PARAMETERS" in hdul:
            pd = hdul["PARAMETERS"].data
            parameter_names = tuple(np.char.strip(np.asarray(pd["PARAMETER"]).astype(str)))
            parameter_unc = np.asarray(pd["UNCERTAINTY"], float)
        gas_names, gas_wave, gas_amp = (), np.empty(0), np.empty(0)
        if "GAS_LINES" in hdul:
            gd = hdul["GAS_LINES"].data
            gas_names = tuple(np.char.strip(np.asarray(gd["NAME"]).astype(str)))
            gas_wave = np.asarray(gd["REST_WAVELENGTH"], float)
            gas_amp = np.asarray(gd["AMPLITUDE"], float)
        diagnostics = None
        if any(name in hdul for name in ("DIAGNOSTICS", "SSP_CORRELATION", "SSP_MODELS", "SSP_DIAGNOSTICS")):
            diagnostics = StellarFitDiagnostics()
            if "DIAGNOSTICS" in hdul:
                dh = hdul["DIAGNOSTICS"]
                diagnostics.singular_values = np.asarray(dh.data["SINGULAR_VALUE"], float)
                diagnostics.effective_rank = dh.header.get("EFFRANK")
                diagnostics.condition_number = dh.header.get("CONDNUM")
            if "SSP_CORRELATION" in hdul: diagnostics.correlation_matrix = np.asarray(hdul["SSP_CORRELATION"].data, float)
            if "SSP_MODELS" in hdul: diagnostics.transformed_model_fluxes = np.asarray(hdul["SSP_MODELS"].data, float)
            if "SSP_DIAGNOSTICS" in hdul:
                sd = hdul["SSP_DIAGNOSTICS"].data
                diagnostics.single_ssp_chi_square = np.asarray(sd["CHI2"], float)
                diagnostics.single_ssp_delta_chi_square = np.asarray(sd["DELTA_CHI2"], float)
                diagnostics.dominant_ssp_distance = np.asarray(sd["DOM_DISTANCE"], float)
        return StellarFitResult(
            regime=str(h["REGIME"]).lower(), library_family=str(h.get("LIBRARY", "")),
            wave=np.asarray(s["WAVELENGTH"], float), flux=np.asarray(s["FLUX"], float), flux_unc=unc,
            mask=np.asarray(s["FIT_MASK"], bool), model=np.asarray(s["MODEL"], float),
            stellar_model=np.asarray(s["STELLAR_MODEL"], float), gas_model=np.asarray(s["GAS_MODEL"], float),
            coefficients=np.asarray(p["COEFFICIENT"], float), ages_myr=np.asarray(p["AGE_MYR"], float),
            metallicity_codes=np.char.strip(np.asarray(p["METALLICITY"]).astype(str)),
            metallicities_solar=np.asarray(p["METALLICITY_ZSUN"], float),
            ebv=float(h["EBV"]), velocity_kms=float(h["VSTAR"]), sigma_kms=float(h["SIGSTAR"]),
            gas_velocity_kms=float(h.get("VGAS", 0.0)), gas_sigma_kms=float(h.get("SIGGAS", 0.0)),
            gas_names=gas_names, gas_rest_wavelengths=gas_wave, gas_amplitudes=gas_amp,
            chi_square=float(h["CHI2"]), reduced_chi_square=float(h["RCHI2"]), degrees_of_freedom=int(h["DOF"]),
            success=bool(h.get("SUCCESS", True)), message="loaded from FITS", n_function_evaluations=int(h.get("NFEV", 0)),
            parameter_names=parameter_names, parameter_uncertainties=parameter_unc,
            coefficient_kind=str(h.get("COEFKIND", "formed_mass_Msun")), surviving_mass_fractions=surv,
            light_fractions=light, diagnostics=diagnostics, resolution_source=h.get("RESSRC"),
            metadata={k: h[k] for k in h.keys() if k not in {"SIMPLE", "BITPIX", "NAXIS", "EXTEND"}},
        )
