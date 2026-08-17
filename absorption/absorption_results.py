"""Absorption-fit result container plus FITS save/load.

Wraps the generic ``core.results.FitResult`` with the derived,
component-specific quantities (column density, Doppler parameter,
velocity per kinematic component, and the shared covering fraction if
partial coverage was enabled), mirroring the stellar and emission
modules' save/load convention.

Note: the saved ``TRANSITIONS`` table round-trips only the fields
needed to reproduce the fitted model (name, ion, group, rest
wavelength, oscillator strength, damping constant) -- not each
transition's free-text ``reference`` provenance note, which lives in
the source catalog (``data/atomic_absorption_lines.csv``) rather than
in per-fit result files, matching the emission module's save/load
convention for the same reason.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from astropy.io import fits

from core.parameters import ModelParameters
from core.results import FitResult
from core.statistics import FitStatistics

from absorption.absorption_model import COVERING_FRACTION_PARAMETER
from absorption.atomic import AtomicTransition

__all__ = ["AbsorptionComponentMeasurement", "AbsorptionFitResult", "save_absorption_result", "load_absorption_result"]


@dataclass
class AbsorptionComponentMeasurement:
    """One kinematic component's fitted physical parameters."""

    logN: float
    logN_uncertainty: float
    b_kms: float
    b_kms_uncertainty: float
    velocity_kms: float
    velocity_kms_uncertainty: float
    system_label: str = ""
    # True if this component was frozen by
    # absorption.rejection.fit_joint_absorption_spectrum_with_rejection
    # as insignificant: logN should then be read as an upper limit on
    # this component's true column density, not a detection -- the
    # component is retained in the model at a value small enough that
    # it changes the total predicted absorption negligibly, rather than
    # being removed outright, so the returned fit always has exactly
    # the number of components requested and the summed column density
    # across components is unaffected.
    is_upper_limit: bool = False


@dataclass
class AbsorptionFitResult:
    """Result of an absorption-line fit.

    Attributes
    ----------
    fit_result : FitResult
        The generic deterministic-fit result this was derived from.
    transitions : list[AtomicTransition]
        The transition group actually fit.
    measurements : list[AbsorptionComponentMeasurement]
        One entry per kinematic component, in component order.
    partial_coverage : bool
        Whether the partial-covering model was used.
    covering_fraction, covering_fraction_uncertainty : float or None
        The shared fitted covering fraction (None if
        ``partial_coverage`` is False, i.e. full coverage was assumed).
    """

    fit_result: FitResult
    transitions: "list[AtomicTransition]"
    measurements: "list[AbsorptionComponentMeasurement]"
    partial_coverage: bool
    covering_fraction: "float | None" = None
    covering_fraction_uncertainty: "float | None" = None

    @property
    def total_logN(self) -> float:
        """log10 of the summed (linear) column density across every component."""
        linear_total = sum(10.0 ** measurement.logN for measurement in self.measurements)
        return float(np.log10(linear_total))


def _param_value_and_uncertainty(parameters: ModelParameters, uncertainties, component_index: int, name: str):
    component = parameters.components[component_index]
    value = component[name].value
    key = f"c{component_index}_{name}"
    uncertainty = np.nan if uncertainties is None else uncertainties.get(key, np.nan)
    return float(value), float(uncertainty)


def summarize_absorption_fit(fit_result: FitResult, transitions: "list[AtomicTransition]", *,
                              partial_coverage: bool, system_labels: "list[str] | None" = None,
                              upper_limit_flags: "list[bool] | None" = None) -> AbsorptionFitResult:
    """Build derived per-component quantities from a raw FitResult.

    system_labels : list[str], optional
        One label per component (e.g. from ``AbsorptionSystem.label``
        for a joint multi-system fit), recorded on each
        ``AbsorptionComponentMeasurement`` for traceability. Defaults to
        an empty string per component for an ordinary single-group fit.
    upper_limit_flags : list[bool], optional
        One flag per component; True marks a component frozen by
        ``absorption.rejection.fit_joint_absorption_spectrum_with_rejection``
        as insignificant (see ``AbsorptionComponentMeasurement.is_upper_limit``).
        Defaults to False for every component.
    """
    parameters = fit_result.parameters
    uncertainties = fit_result.parameter_uncertainties
    if system_labels is not None and len(system_labels) != parameters.n_components:
        raise ValueError("system_labels must have one entry per component.")
    if upper_limit_flags is not None and len(upper_limit_flags) != parameters.n_components:
        raise ValueError("upper_limit_flags must have one entry per component.")

    measurements = []
    for component_index in range(parameters.n_components):
        logN, logN_unc = _param_value_and_uncertainty(parameters, uncertainties, component_index, "logN")
        b_kms, b_unc = _param_value_and_uncertainty(parameters, uncertainties, component_index, "b_kms")
        velocity_kms, v_unc = _param_value_and_uncertainty(parameters, uncertainties, component_index, "velocity_kms")
        measurements.append(AbsorptionComponentMeasurement(
            logN=logN, logN_uncertainty=logN_unc, b_kms=b_kms, b_kms_uncertainty=b_unc,
            velocity_kms=velocity_kms, velocity_kms_uncertainty=v_unc,
            system_label=("" if system_labels is None else system_labels[component_index]),
            is_upper_limit=(False if upper_limit_flags is None else bool(upper_limit_flags[component_index])),
        ))

    covering_fraction = None
    covering_fraction_uncertainty = None
    if partial_coverage:
        covering_fraction, covering_fraction_uncertainty = _param_value_and_uncertainty(
            parameters, uncertainties, 0, COVERING_FRACTION_PARAMETER,
        )

    return AbsorptionFitResult(
        fit_result=fit_result, transitions=list(transitions), measurements=measurements,
        partial_coverage=partial_coverage, covering_fraction=covering_fraction,
        covering_fraction_uncertainty=covering_fraction_uncertainty,
    )


def save_absorption_result(path, result: AbsorptionFitResult, *, overwrite=True):
    """Persist an absorption fit to FITS, mirroring the stellar/emission-result layout."""
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    fit_result = result.fit_result

    ph = fits.PrimaryHDU()
    h = ph.header
    h["METHOD"] = str(fit_result.method)[:68]
    h["NCOMP"] = int(fit_result.parameters.n_components)
    h["REDSHIFT"] = float(fit_result.redshift)
    h["PARTCOV"] = bool(result.partial_coverage)
    metadata = fit_result.metadata or {}
    if "absorption_fit_mode" in metadata:
        h["ABSMODE"] = str(metadata["absorption_fit_mode"])[:68]
    if "absorption_subpixel" in metadata:
        h["SUBPIX"] = int(metadata["absorption_subpixel"])
    if result.covering_fraction is not None:
        h["COVFRAC"] = float(result.covering_fraction)
        h["COVFRACE"] = float(result.covering_fraction_uncertainty)
    if fit_result.resolution_source is not None:
        h["RESSRC"] = str(fit_result.resolution_source)[:68]
    for stat_name, value in vars(fit_result.statistics).items():
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

    labels = np.asarray([m.system_label for m in result.measurements], dtype=str)
    label_width = max(1, max(map(len, labels))) if labels.size else 1
    comp = fits.BinTableHDU.from_columns([
        fits.Column(name="LOGN", format="D", array=np.asarray([m.logN for m in result.measurements], float)),
        fits.Column(name="LOGN_UNC", format="D", array=np.asarray([m.logN_uncertainty for m in result.measurements], float)),
        fits.Column(name="B_KMS", format="D", array=np.asarray([m.b_kms for m in result.measurements], float)),
        fits.Column(name="B_KMS_UNC", format="D", array=np.asarray([m.b_kms_uncertainty for m in result.measurements], float)),
        fits.Column(name="VELOCITY_KMS", format="D", array=np.asarray([m.velocity_kms for m in result.measurements], float)),
        fits.Column(name="VELOCITY_KMS_UNC", format="D", array=np.asarray([m.velocity_kms_uncertainty for m in result.measurements], float)),
        fits.Column(name="SYSTEM_LABEL", format=f"{label_width}A", array=labels),
        fits.Column(name="IS_UPPER_LIMIT", format="L",
                    array=np.asarray([m.is_upper_limit for m in result.measurements], bool)),
    ], name="COMPONENTS")

    names = np.asarray([t.name for t in result.transitions], dtype=str)
    ions = np.asarray([t.ion for t in result.transitions], dtype=str)
    groups = np.asarray([t.group for t in result.transitions], dtype=str)
    width = max(1, max(map(len, names))) if names.size else 1
    ion_width = max(1, max(map(len, ions))) if ions.size else 1
    group_width = max(1, max(map(len, groups))) if groups.size else 1
    trans_hdu = fits.BinTableHDU.from_columns([
        fits.Column(name="NAME", format=f"{width}A", array=names),
        fits.Column(name="ION", format=f"{ion_width}A", array=ions),
        fits.Column(name="GROUP", format=f"{group_width}A", array=groups),
        fits.Column(name="REST_WAVELENGTH", format="D", unit="Angstrom",
                    array=np.asarray([t.rest_wavelength_angstrom for t in result.transitions], float)),
        fits.Column(name="OSCILLATOR_STRENGTH", format="D",
                    array=np.asarray([t.oscillator_strength for t in result.transitions], float)),
        fits.Column(name="DAMPING_CONSTANT", format="D", unit="s-1",
                    array=np.asarray([t.damping_constant_s for t in result.transitions], float)),
    ], name="TRANSITIONS")

    # Persist the complete parameter state. Posterior reconstruction needs
    # bounds and fixed/free status, not only the derived component table.
    parameter_rows = []
    unc_dict = fit_result.parameter_uncertainties or {}
    for component_index, component_state in enumerate(fit_result.parameters.components):
        for parameter in component_state.parameters:
            full_name = f"c{component_index}_{parameter.name}"
            parameter_rows.append((
                full_name, parameter.value, parameter.lower, parameter.upper,
                parameter.fixed, unc_dict.get(full_name, np.nan),
            ))
    pnames = np.asarray([row[0] for row in parameter_rows], dtype=str)
    pwidth = max(1, max(map(len, pnames))) if pnames.size else 1
    params_hdu = fits.BinTableHDU.from_columns([
        fits.Column(name="PARAMETER", format=f"{pwidth}A", array=pnames),
        fits.Column(name="VALUE", format="D", array=np.asarray([row[1] for row in parameter_rows], float)),
        fits.Column(name="LOWER", format="D", array=np.asarray([row[2] for row in parameter_rows], float)),
        fits.Column(name="UPPER", format="D", array=np.asarray([row[3] for row in parameter_rows], float)),
        fits.Column(name="FIXED", format="L", array=np.asarray([row[4] for row in parameter_rows], bool)),
        fits.Column(name="UNCERTAINTY", format="D", array=np.asarray([row[5] for row in parameter_rows], float)),
    ], name="PARAMETERS")
    params_hdu.header["NCOMP"] = fit_result.parameters.n_components
    params_hdu.header["ACTIVEC"] = fit_result.parameters.active_component_index

    hdus = [ph, spec, comp, trans_hdu, params_hdu]
    component_transition_names = metadata.get("absorption_component_transition_names")
    if component_transition_names:
        component_indices = []
        transition_names = []
        for component_index, names_for_component in enumerate(component_transition_names):
            for transition_name in names_for_component:
                component_indices.append(component_index)
                transition_names.append(str(transition_name))
        if transition_names:
            twidth = max(1, max(map(len, transition_names)))
            mapping_hdu = fits.BinTableHDU.from_columns([
                fits.Column(name="COMPONENT", format="J", array=np.asarray(component_indices, int)),
                fits.Column(name="TRANSITION", format=f"{twidth}A", array=np.asarray(transition_names, dtype=str)),
            ], name="COMP_TRANS")
            hdus.append(mapping_hdu)

    fits.HDUList(hdus).writeto(path, overwrite=overwrite)
    return path


def _load_component_transition_names(hdul, n_components):
    mapping = [[] for _ in range(n_components)]
    if "COMP_TRANS" not in hdul:
        return mapping
    data = hdul["COMP_TRANS"].data
    for component_index, transition_name in zip(
        np.asarray(data["COMPONENT"], int),
        np.char.strip(np.asarray(data["TRANSITION"]).astype(str)),
    ):
        if 0 <= int(component_index) < n_components:
            mapping[int(component_index)].append(str(transition_name))
    return mapping


def load_absorption_result(path) -> AbsorptionFitResult:
    """Load a result written by :func:`save_absorption_result`."""
    from core.parameters import Component, Parameter

    with fits.open(path) as hdul:
        h = hdul[0].header
        s = hdul["SPECTRUM"].data
        unc = np.asarray(s["FLUX_UNC"], float)
        unc = None if np.all(~np.isfinite(unc)) else unc

        transitions = []
        if "TRANSITIONS" in hdul:
            td = hdul["TRANSITIONS"].data
            for name, ion, group, wave_rest, f_value, gamma in zip(
                np.char.strip(np.asarray(td["NAME"]).astype(str)),
                np.char.strip(np.asarray(td["ION"]).astype(str)),
                np.char.strip(np.asarray(td["GROUP"]).astype(str)),
                np.asarray(td["REST_WAVELENGTH"], float),
                np.asarray(td["OSCILLATOR_STRENGTH"], float),
                np.asarray(td["DAMPING_CONSTANT"], float),
            ):
                transitions.append(AtomicTransition(
                    name=str(name), ion=str(ion), rest_wavelength_angstrom=float(wave_rest),
                    oscillator_strength=float(f_value), damping_constant_s=float(gamma), group=str(group),
                ))

        n_components = int(h.get("NCOMP", 1))
        partial_coverage = bool(h.get("PARTCOV", False))
        covering_fraction = float(h["COVFRAC"]) if "COVFRAC" in h else None
        covering_fraction_uncertainty = float(h["COVFRACE"]) if "COVFRACE" in h else None

        measurements = []
        if "COMPONENTS" in hdul:
            cd = hdul["COMPONENTS"].data
            has_labels = "SYSTEM_LABEL" in cd.columns.names
            labels = np.char.strip(np.asarray(cd["SYSTEM_LABEL"]).astype(str)) if has_labels else None
            has_upper_limit = "IS_UPPER_LIMIT" in cd.columns.names
            upper_limit_flags = np.asarray(cd["IS_UPPER_LIMIT"], bool) if has_upper_limit else None
            for component_index, (logN, logN_unc, b_kms, b_unc, velocity_kms, v_unc) in enumerate(zip(
                np.asarray(cd["LOGN"], float), np.asarray(cd["LOGN_UNC"], float),
                np.asarray(cd["B_KMS"], float), np.asarray(cd["B_KMS_UNC"], float),
                np.asarray(cd["VELOCITY_KMS"], float), np.asarray(cd["VELOCITY_KMS_UNC"], float),
            )):
                measurements.append(AbsorptionComponentMeasurement(
                    logN=float(logN), logN_uncertainty=float(logN_unc),
                    b_kms=float(b_kms), b_kms_uncertainty=float(b_unc),
                    velocity_kms=float(velocity_kms), velocity_kms_uncertainty=float(v_unc),
                    system_label=(str(labels[component_index]) if labels is not None else ""),
                    is_upper_limit=(bool(upper_limit_flags[component_index]) if upper_limit_flags is not None else False),
                ))

        components = [Component(parameters=[]) for _ in range(n_components)]
        parameter_uncertainties = {}
        if "PARAMETERS" in hdul:
            pd = hdul["PARAMETERS"].data
            column_names = set(pd.columns.names)
            for row in pd:
                full_name = str(row["PARAMETER"]).strip()
                comp_str, _, param_name = full_name.partition("_")
                component_index = int(comp_str[1:])
                lower = float(row["LOWER"]) if "LOWER" in column_names else -np.inf
                upper = float(row["UPPER"]) if "UPPER" in column_names else np.inf
                fixed = bool(row["FIXED"]) if "FIXED" in column_names else False
                value = float(row["VALUE"])
                components[component_index].parameters.append(Parameter(param_name, value, lower, upper, fixed))
                if "UNCERTAINTY" in column_names:
                    uncertainty = float(row["UNCERTAINTY"])
                    if np.isfinite(uncertainty):
                        parameter_uncertainties[full_name] = uncertainty
            active_component_index = int(hdul["PARAMETERS"].header.get("ACTIVEC", 0))
        else:
            # Backward compatibility with older absorption result files.
            for component_index, measurement in enumerate(measurements):
                components[component_index] = Component(parameters=[
                    Parameter("logN", measurement.logN, -np.inf, np.inf),
                    Parameter("b_kms", measurement.b_kms, -np.inf, np.inf),
                    Parameter("velocity_kms", measurement.velocity_kms, -np.inf, np.inf),
                ])
                parameter_uncertainties[f"c{component_index}_logN"] = measurement.logN_uncertainty
                parameter_uncertainties[f"c{component_index}_b_kms"] = measurement.b_kms_uncertainty
                parameter_uncertainties[f"c{component_index}_velocity_kms"] = measurement.velocity_kms_uncertainty
            if partial_coverage and covering_fraction is not None:
                components[0].parameters.append(
                    Parameter(COVERING_FRACTION_PARAMETER, covering_fraction, -np.inf, np.inf)
                )
                if covering_fraction_uncertainty is not None:
                    parameter_uncertainties[f"c0_{COVERING_FRACTION_PARAMETER}"] = covering_fraction_uncertainty
            active_component_index = 0

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
                "absorption_fit_mode": str(h.get("ABSMODE", "unknown")),
                "absorption_partial_coverage": partial_coverage,
                "absorption_subpixel": int(h.get("SUBPIX", 1)),
                "absorption_system_labels": [m.system_label for m in measurements],
                "absorption_frozen_components": [bool(m.is_upper_limit) for m in measurements],
                "absorption_component_transition_names": _load_component_transition_names(hdul, n_components),
            },
        )

        return AbsorptionFitResult(
            fit_result=fit_result, transitions=transitions, measurements=measurements,
            partial_coverage=partial_coverage, covering_fraction=covering_fraction,
            covering_fraction_uncertainty=covering_fraction_uncertainty,
        )
