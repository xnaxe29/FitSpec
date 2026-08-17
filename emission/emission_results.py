"""Emission-fit result container plus FITS save/load.

Wraps the generic ``core.results.FitResult`` (reused as-is, per the
"define universal functionality once" design principle) with the
derived, line-specific quantities: per-line integrated flux/uncertainty
and per-component kinematics, in an emission-specific results table.

Note: the saved ``LINE_LIST`` table round-trips only the fields needed
to reproduce the fitted model and its derived quantities (name, ion,
rest wavelength, and any fixed-ratio tie) -- not the full atomic-term
metadata carried by ``emission.lines.EmissionLine`` (configurations,
terms, creation IP, observation reference, etc.), which lives in the
source catalog (``data/nebular_emission_line_list.csv``) rather than in
per-fit result files.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from astropy.io import fits

from core.parameters import ModelParameters
from core.results import FitResult
from core.statistics import FitStatistics

from emission.emission_model import amplitude_parameter_name
from emission.lines import EmissionLine

__all__ = ["EmissionLineMeasurement", "EmissionFitResult", "save_emission_result", "load_emission_result"]


def _ascii_safe(text: str) -> str:
    """FITS ASCII ('A') table columns cannot hold non-ASCII characters
    (e.g. the Greek letters in ion names like 'Hα', 'Lyα'); round-trip
    them losslessly via backslash-escaping instead of stripping them."""
    return text.encode("ascii", "backslashreplace").decode("ascii")


def _ascii_unsafe(text: str) -> str:
    return text.encode("ascii").decode("unicode_escape")


@dataclass
class EmissionLineMeasurement:
    """One fitted line's derived quantities, summed across every component it appears in."""

    name: str
    rest_wavelength_angstrom: float
    integrated_flux: float
    integrated_flux_uncertainty: float


@dataclass
class EmissionFitResult:
    """Result of an emission-line fit.

    Attributes
    ----------
    fit_result : FitResult
        The generic deterministic-fit result (spectrum arrays, model,
        statistics, raw ModelParameters) this was derived from.
    line_list : list[EmissionLine]
        The line list actually used for this fit (after any subsetting).
    measurements : list[EmissionLineMeasurement]
        Derived per-line integrated fluxes.
    component_velocities_kms, component_sigmas_kms : np.ndarray
        Fitted kinematics, one entry per kinematic component.
    """

    fit_result: FitResult
    line_list: "list[EmissionLine]"
    measurements: "list[EmissionLineMeasurement]"
    component_velocities_kms: np.ndarray
    component_sigmas_kms: np.ndarray

    def flux(self, name: str) -> float:
        for measurement in self.measurements:
            if measurement.name == name:
                return measurement.integrated_flux
        raise KeyError(f"No fitted measurement for line {name!r}.")


def _line_amplitude_and_uncertainty(parameters: ModelParameters, uncertainties, component_index: int, line: EmissionLine):
    """Resolve one line's integrated flux (+ propagated uncertainty) in one component."""
    if line.tied_to is None:
        key = amplitude_parameter_name(line.name)
        component = parameters.components[component_index]
        if key not in component:
            return None
        value = component[key].value
        name = f"c{component_index}_{key}"
        unc = None if uncertainties is None else uncertainties.get(name, np.nan)
        return value, (np.nan if unc is None else unc)
    else:
        tied_key = amplitude_parameter_name(line.tied_to)
        component = parameters.components[component_index]
        if tied_key not in component:
            return None
        tied_value = component[tied_key].value
        value = tied_value * line.ratio_to_tied
        name = f"c{component_index}_{tied_key}"
        tied_unc = None if uncertainties is None else uncertainties.get(name, np.nan)
        unc = np.nan if tied_unc is None else abs(tied_unc) * line.ratio_to_tied
        return value, unc


def summarize_emission_fit(fit_result: FitResult, line_list: "list[EmissionLine]") -> EmissionFitResult:
    """Build derived per-line and per-component quantities from a raw FitResult."""
    parameters = fit_result.parameters
    uncertainties = fit_result.parameter_uncertainties

    velocities = np.array([component["velocity_kms"].value for component in parameters.components], dtype=float)
    sigmas = np.array([component["sigma_kms"].value for component in parameters.components], dtype=float)

    measurements = []
    for emission_line in line_list:
        total_flux = 0.0
        variance = 0.0
        any_component = False
        for component_index in range(parameters.n_components):
            resolved = _line_amplitude_and_uncertainty(parameters, uncertainties, component_index, emission_line)
            if resolved is None:
                continue
            value, unc = resolved
            total_flux += value
            variance += 0.0 if np.isnan(unc) else unc ** 2
            any_component = True
        if not any_component:
            continue
        measurements.append(EmissionLineMeasurement(
            name=emission_line.name, rest_wavelength_angstrom=emission_line.rest_wavelength_angstrom,
            integrated_flux=float(total_flux), integrated_flux_uncertainty=float(np.sqrt(variance)),
        ))

    return EmissionFitResult(
        fit_result=fit_result, line_list=list(line_list), measurements=measurements,
        component_velocities_kms=velocities, component_sigmas_kms=sigmas,
    )


def save_emission_result(path, result: EmissionFitResult, *, overwrite=True):
    """Persist an emission fit to FITS, mirroring the stellar-result layout."""
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    fit_result = result.fit_result

    ph = fits.PrimaryHDU()
    h = ph.header
    h["METHOD"] = str(fit_result.method)[:68]
    h["NCOMP"] = int(fit_result.parameters.n_components)
    h["REDSHIFT"] = float(fit_result.redshift)
    if fit_result.resolution_source is not None:
        h["RESSRC"] = str(fit_result.resolution_source)[:68]
    metadata = fit_result.metadata or {}
    if "emission_kinematics_mode" in metadata:
        h["EMKIN"] = str(metadata["emission_kinematics_mode"])[:68]
    if "emission_flux_normalizing_factor" in metadata:
        h["FLXNORM"] = float(metadata["emission_flux_normalizing_factor"])
    if "emission_flux_reduction" in metadata:
        h["FLXRED"] = float(metadata["emission_flux_reduction"])
    if "emission_frozen_components" in metadata:
        # One '0'/'1' character per component, in order -- compact enough
        # to always fit a single FITS header card even for many
        # components, and trivially parsed back on load (see below).
        flags = metadata["emission_frozen_components"]
        h["FROZEN"] = "".join("1" if bool(flag) else "0" for flag in flags)
    stats = fit_result.statistics
    for stat_name, value in vars(stats).items():
        h[f"ST_{stat_name.upper()}"[:8]] = float(value)

    n = fit_result.wave.size
    unc = np.full(n, np.nan) if fit_result.flux_unc is None else np.asarray(fit_result.flux_unc, float)
    spec = fits.BinTableHDU.from_columns([
        fits.Column(name="WAVELENGTH", format="D", unit="Angstrom", array=np.asarray(fit_result.wave, float)),
        fits.Column(name="FLUX", format="D", array=np.asarray(fit_result.flux, float)),
        fits.Column(name="FLUX_UNC", format="D", array=unc),
        fits.Column(name="MODEL", format="D", array=np.asarray(fit_result.model, float)),
        fits.Column(name="FIT_MASK", format="L", array=np.asarray(fit_result.mask, bool)),
    ], name="SPECTRUM")

    names = np.asarray([_ascii_safe(measurement.name) for measurement in result.measurements], dtype=str)
    width = max(1, max(map(len, names))) if names.size else 1
    lines_hdu = fits.BinTableHDU.from_columns([
        fits.Column(name="NAME", format=f"{width}A", array=names),
        fits.Column(name="REST_WAVELENGTH", format="D", unit="Angstrom",
                    array=np.asarray([m.rest_wavelength_angstrom for m in result.measurements], float)),
        fits.Column(name="FLUX", format="D", array=np.asarray([m.integrated_flux for m in result.measurements], float)),
        fits.Column(name="FLUX_UNC", format="D",
                    array=np.asarray([m.integrated_flux_uncertainty for m in result.measurements], float)),
    ], name="LINES")

    kin = fits.BinTableHDU.from_columns([
        fits.Column(name="VELOCITY_KMS", format="D", array=np.asarray(result.component_velocities_kms, float)),
        fits.Column(name="SIGMA_KMS", format="D", array=np.asarray(result.component_sigmas_kms, float)),
    ], name="KINEMATICS")

    llnames = np.asarray([_ascii_safe(line.name) for line in result.line_list], dtype=str)
    llwidth = max(1, max(map(len, llnames))) if llnames.size else 1
    ions = np.asarray([_ascii_safe(line.ion) for line in result.line_list], dtype=str)
    ion_width = max(1, max(map(len, ions))) if ions.size else 1
    tied_to = np.asarray([_ascii_safe(line.tied_to) if line.tied_to else "-" for line in result.line_list], dtype=str)
    tied_width = max(1, max(map(len, tied_to))) if tied_to.size else 1
    ratio = np.asarray([line.ratio_to_tied if line.ratio_to_tied is not None else np.nan for line in result.line_list], float)
    linelist_hdu = fits.BinTableHDU.from_columns([
        fits.Column(name="NAME", format=f"{llwidth}A", array=llnames),
        fits.Column(name="ION", format=f"{ion_width}A", array=ions),
        fits.Column(name="REST_WAVELENGTH", format="D", unit="Angstrom",
                    array=np.asarray([line.rest_wavelength_angstrom for line in result.line_list], float)),
        fits.Column(name="TIED_TO", format=f"{tied_width}A", array=tied_to),
        fits.Column(name="RATIO_TO_TIED", format="D", array=ratio),
    ], name="LINE_LIST")

    params_dict = fit_result.parameters.to_dict()
    # Persist every parameter, not only the free vector.  Posterior inference
    # must be able to reconstruct fixed/tied state and proper bounds from a
    # saved deterministic result without consulting GUI widgets.
    parameter_rows = []
    unc_dict = fit_result.parameter_uncertainties or {}
    for component_index, component in enumerate(fit_result.parameters.components):
        for parameter in component.parameters:
            full_name = f"c{component_index}_{parameter.name}"
            parameter_rows.append((
                full_name, parameter.value, parameter.lower, parameter.upper,
                parameter.fixed, unc_dict.get(full_name, np.nan),
            ))
    raw_names = [row[0] for row in parameter_rows]
    pnames = np.asarray([_ascii_safe(name) for name in raw_names], dtype=str)
    pvalues = np.asarray([row[1] for row in parameter_rows], float)
    plower = np.asarray([row[2] for row in parameter_rows], float)
    pupper = np.asarray([row[3] for row in parameter_rows], float)
    pfixed = np.asarray([row[4] for row in parameter_rows], bool)
    punc = np.asarray([row[5] for row in parameter_rows], float)
    pwidth = max(1, max(map(len, pnames))) if pnames.size else 1
    params_hdu = fits.BinTableHDU.from_columns([
        fits.Column(name="PARAMETER", format=f"{pwidth}A", array=pnames),
        fits.Column(name="VALUE", format="D", array=pvalues),
        fits.Column(name="LOWER", format="D", array=plower),
        fits.Column(name="UPPER", format="D", array=pupper),
        fits.Column(name="FIXED", format="L", array=pfixed),
        fits.Column(name="UNCERTAINTY", format="D", array=punc),
    ], name="PARAMETERS")
    params_hdu.header["NCOMP"] = params_dict["n_components"]
    params_hdu.header["ACTIVEC"] = params_dict["active_component_index"]

    fits.HDUList([ph, spec, lines_hdu, kin, linelist_hdu, params_hdu]).writeto(path, overwrite=overwrite)
    return path


def load_emission_result(path) -> EmissionFitResult:
    """Load a result written by :func:`save_emission_result`.

    Reconstructs the derived (``EmissionLineMeasurement``,
    ``EmissionLine`` line list, per-component kinematics) view directly
    from the saved tables. The underlying ``fit_result.parameters``
    ``ModelParameters`` is reconstructed with every saved parameter's
    fitted value, bounds, and fixed/free state. Files written by older
    FitSpec versions without those columns remain readable; their missing
    bounds fall back to +/-inf.
    """
    from core.parameters import Component, Parameter

    with fits.open(path) as hdul:
        h = hdul[0].header
        s = hdul["SPECTRUM"].data
        unc = np.asarray(s["FLUX_UNC"], float)
        unc = None if np.all(~np.isfinite(unc)) else unc

        line_list = []
        if "LINE_LIST" in hdul:
            ld = hdul["LINE_LIST"].data
            has_ion = "ION" in ld.columns.names
            for row in ld:
                name = _ascii_unsafe(str(row["NAME"]).strip())
                wave_rest = float(row["REST_WAVELENGTH"])
                tied_to_raw = str(row["TIED_TO"]).strip() if "TIED_TO" in ld.columns.names else "-"
                tied_to_value = None if tied_to_raw == "-" else _ascii_unsafe(tied_to_raw)
                ratio_value = float(row["RATIO_TO_TIED"]) if "RATIO_TO_TIED" in ld.columns.names else np.nan
                ion = _ascii_unsafe(str(row["ION"]).strip()) if has_ion else name
                line_list.append(EmissionLine(
                    name=name, rest_wavelength_angstrom=wave_rest, ion=ion,
                    tied_to=tied_to_value,
                    ratio_to_tied=None if not np.isfinite(ratio_value) else ratio_value,
                ))

        measurements = []
        if "LINES" in hdul:
            md = hdul["LINES"].data
            for name, wave_rest, flux_value, flux_unc in zip(
                np.char.strip(np.asarray(md["NAME"]).astype(str)),
                np.asarray(md["REST_WAVELENGTH"], float),
                np.asarray(md["FLUX"], float),
                np.asarray(md["FLUX_UNC"], float),
            ):
                measurements.append(EmissionLineMeasurement(
                    name=_ascii_unsafe(str(name)), rest_wavelength_angstrom=float(wave_rest),
                    integrated_flux=float(flux_value), integrated_flux_uncertainty=float(flux_unc),
                ))

        n_components = int(h.get("NCOMP", 1))
        velocities = np.zeros(n_components)
        sigmas = np.zeros(n_components)
        if "KINEMATICS" in hdul:
            kd = hdul["KINEMATICS"].data
            velocities = np.asarray(kd["VELOCITY_KMS"], float)
            sigmas = np.asarray(kd["SIGMA_KMS"], float)

        components = [Component(parameters=[]) for _ in range(n_components)]
        parameter_uncertainties = {}
        if "PARAMETERS" in hdul:
            pd = hdul["PARAMETERS"].data
            column_names = set(pd.columns.names)
            has_bounds = "LOWER" in column_names and "UPPER" in column_names
            has_fixed = "FIXED" in column_names
            for row in pd:
                full_name = _ascii_unsafe(str(row["PARAMETER"]).strip())
                value = float(row["VALUE"])
                uncertainty = float(row["UNCERTAINTY"])
                lower = float(row["LOWER"]) if has_bounds else -np.inf
                upper = float(row["UPPER"]) if has_bounds else np.inf
                fixed = bool(row["FIXED"]) if has_fixed else False
                # names look like "c<index>_<param_name>"
                comp_str, _, param_name = full_name.partition("_")
                component_index = int(comp_str[1:])
                components[component_index].parameters.append(
                    Parameter(param_name, value, lower, upper, fixed)
                )
                if np.isfinite(uncertainty):
                    parameter_uncertainties[full_name] = uncertainty

        active_component_index = 0
        if "PARAMETERS" in hdul:
            active_component_index = int(hdul["PARAMETERS"].header.get("ACTIVEC", 0))
        model_parameters = ModelParameters(
            n_components=n_components, components=components,
            active_component_index=min(active_component_index, n_components - 1),
        )

        stat_fields = ("n_data", "n_eff", "k_params", "dof", "chi_square", "reduced_chi_square",
                       "jitter_scale", "neg2_log_likelihood", "bic", "aic", "aicc")
        stats_kwargs = {}
        for stat_name in stat_fields:
            key = f"ST_{stat_name.upper()}"[:8]
            if key in h:
                value = float(h[key])
                stats_kwargs[stat_name] = int(value) if stat_name in ("n_data", "k_params", "dof") else value
        statistics = FitStatistics(**stats_kwargs)

        fit_result = FitResult(
            parameters=model_parameters, parameter_uncertainties=parameter_uncertainties or None,
            wave=np.asarray(s["WAVELENGTH"], float), flux=np.asarray(s["FLUX"], float), flux_unc=unc,
            mask=np.asarray(s["FIT_MASK"], bool), model=np.asarray(s["MODEL"], float),
            statistics=statistics, redshift=float(h.get("REDSHIFT", 0.0)),
            resolution_source=h.get("RESSRC"), method=str(h.get("METHOD", "unknown")),
            metadata={
                # Old saved files predating this metadata key were fit under
                # whatever the code's default was at the time -- component-
                # independent, lines-within-a-component sharing kinematics,
                # which is what "tied" now correctly names (see
                # emission.emission_model's module docstring for why the
                # default changed name from the old, confusingly-labeled
                # "free").
                "emission_kinematics_mode": str(h.get("EMKIN", "tied")),
                "emission_flux_normalizing_factor": float(h.get("FLXNORM", 1.0)),
                "emission_flux_reduction": float(h.get("FLXRED", 0.0)),
                "emission_frozen_components": (
                    [char == "1" for char in str(h["FROZEN"])] if "FROZEN" in h
                    else [False] * int(h.get("NCOMP", 1))
                ),
            },
        )

        return EmissionFitResult(
            fit_result=fit_result, line_list=line_list, measurements=measurements,
            component_velocities_kms=velocities, component_sigmas_kms=sigmas,
        )
