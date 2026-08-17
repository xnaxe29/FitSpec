"""Atomic transition data loading for absorption-line fitting.

Reads the full atomic line list CSV (default:
``data/atomic_absorption_lines.csv``), derived from the standard VPFIT
``atom.dat`` compilation (Carswell & Webb) -- every real transition
line in that file is retained, together with its original provenance
note, so no information from the source compilation is lost. A
user-supplied atomic data file in the same CSV schema is accepted in
place of the default, per the FitSpec requirement that the absorption
module use a default atomic-data file but allow user-supplied atomic
data.

Transitions are grouped (``AtomicTransition.group``) into physical
systems/multiplets -- e.g. every H I Lyman-series line shares group
``"HI"``, both C IV lines share group ``"CIV"``, excited fine-structure
levels such as Si II* form their own separate group from ground-state
Si II -- since transitions in the same group arise from the same
absorbing population and are always fit together with one shared
column density, Doppler parameter, and velocity per kinematic component
(see ``absorption.absorption_model``). This mirrors exactly how VPFIT
itself groups transitions by species label for a fitting region.

The unidentified-line marker (species ``"??"`` in the source
compilation) is retained as an ordinary, fittable group -- it has real
oscillator-strength/damping-constant values (by convention, the same as
Lyman-alpha) and is useful for modeling an unidentified blend as a free
line without committing to an ion ID. Pure VPFIT bookkeeping markers
that are not physical transitions at all (region wavelength-shift,
in-fit continuum/zero-level adjustment, generic emission/telluric
flags) are excluded from the catalog; the equivalent nuisance-parameter
functionality is implemented directly in ``absorption.absorption_model``
(see ``region_velocity_shift``, ``continuum_adjustment``) rather than as
pseudo atomic-data rows.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

__all__ = ["AtomicTransition", "load_atomic_line_list", "select_group", "list_groups"]

DEFAULT_LINE_LIST_PATH = Path(__file__).resolve().parent.parent / "data" / "atomic_absorption_lines.csv"


@dataclass(frozen=True)
class AtomicTransition:
    """One resonance absorption transition.

    Attributes
    ----------
    name : str
        Transition identifier (e.g. ``"CIV_1548"``). Unique within one
        loaded list via an automatic disambiguating suffix in the rare
        case of an integer-wavelength collision within the same group.
    ion : str
        Ion/species label exactly as given by the source compilation
        (e.g. ``"H I"``, ``"SiII"``, ``"SiII*"`` for the excited
        fine-structure level, ``"MgIIa"`` for an alternate-source
        wavelength variant).
    rest_wavelength_angstrom : float
        Rest-frame wavelength.
    oscillator_strength : float
        Absorption oscillator strength ``f``.
    damping_constant_s : float
        Einstein A / classical damping constant :math:`\\Gamma` [s^-1],
        i.e. the natural (radiative) line-broadening rate. May be 0 for
        transitions whose radiative damping constant has not been
        measured/tabulated; the Voigt-Hjerting function reduces
        smoothly to a pure Gaussian (Doppler-core-only) profile in that
        limit (see ``absorption.profiles.voigt_hjerting``).
    group : str
        Physical system/multiplet label (species label with internal
        whitespace removed, e.g. ``"HI"``, ``"CIV"``, ``"SiII*"``);
        every transition sharing a group is fit together with one
        shared column density, Doppler parameter, and velocity per
        kinematic component.
    reference : str or None
        Free-text provenance note carried over verbatim from the source
        compilation (literature reference, calibration caveat, isotope
        note, etc.), preserved for traceability but not otherwise
        consumed by the fitting code.
    atomic_mass_amu : float or None
        Atomic/molecular mass [amu], if given in the source compilation
        (carried forward from the last explicit value for the same
        species if a given line omits it, matching VPFIT's own
        behavior). Required for thermal Doppler-parameter linking (see
        ``absorption.absorption_model.thermal_b_kms``).
    q_coefficient_cminv : float or None
        Fine-structure-constant sensitivity coefficient q [cm^-1]
        (Murphy et al. 2003 convention), if tabulated for this
        transition. Preserved for traceability; not yet consumed by a
        dedicated Delta-alpha/alpha fitting capability.
    """

    name: str
    ion: str
    rest_wavelength_angstrom: float
    oscillator_strength: float
    damping_constant_s: float
    group: str
    atomic_mass_amu: "float | None" = None
    q_coefficient_cminv: "float | None" = None
    reference: "str | None" = None

    def __post_init__(self):
        if self.rest_wavelength_angstrom <= 0:
            raise ValueError(f"Transition {self.name!r}: rest_wavelength_angstrom must be positive.")
        if self.oscillator_strength <= 0:
            raise ValueError(f"Transition {self.name!r}: oscillator_strength must be positive.")
        if self.damping_constant_s < 0:
            raise ValueError(f"Transition {self.name!r}: damping_constant_s must be non-negative.")
        if self.atomic_mass_amu is not None and self.atomic_mass_amu <= 0:
            raise ValueError(f"Transition {self.name!r}: atomic_mass_amu must be positive if given.")


def load_atomic_line_list(path=None) -> "list[AtomicTransition]":
    """Parse the atomic absorption-line list CSV.

    Parameters
    ----------
    path : str or Path, optional
        Line-list CSV to parse. Defaults to the packaged
        ``data/atomic_absorption_lines.csv``.

    Returns
    -------
    list[AtomicTransition]
    """
    path = DEFAULT_LINE_LIST_PATH if path is None else Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Atomic absorption-line list not found: {path}")

    transitions = []
    name_counts = {}
    with open(path, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"name", "ion", "rest_wavelength_angstrom", "oscillator_strength", "damping_constant_s", "group"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing required column(s) {sorted(missing)}.")
        for row_number, row in enumerate(reader, start=2):
            base_name = row["name"].strip()
            count = name_counts.get(base_name, 0)
            name_counts[base_name] = count + 1
            name = base_name if count == 0 else f"{base_name}_{chr(ord('a') + count)}"
            reference = (row.get("reference") or "").strip() or None
            mass_token = (row.get("atomic_mass_amu") or "").strip()
            q_token = (row.get("q_coefficient_cminv") or "").strip()
            transitions.append(AtomicTransition(
                name=name, ion=row["ion"].strip(),
                rest_wavelength_angstrom=float(row["rest_wavelength_angstrom"]),
                oscillator_strength=float(row["oscillator_strength"]),
                damping_constant_s=float(row["damping_constant_s"]),
                group=row["group"].strip(), reference=reference,
                atomic_mass_amu=(float(mass_token) if mass_token else None),
                q_coefficient_cminv=(float(q_token) if q_token else None),
            ))
    return transitions


def list_groups(transitions: "list[AtomicTransition]") -> "list[str]":
    """Every distinct group label present, in first-seen order."""
    seen = []
    for transition in transitions:
        if transition.group not in seen:
            seen.append(transition.group)
    return seen


def select_group(transitions: "list[AtomicTransition]", group: str) -> "list[AtomicTransition]":
    """Every transition sharing the given group label, in file order.

    This is the normal way to select the transition set for one
    absorption-line fit -- e.g. ``select_group(transitions, "CIV")``
    returns both members of the C IV doublet, which are then always fit
    together (Section on absorption fitting: shared N/b/v per component).
    """
    selected = [transition for transition in transitions if transition.group == group]
    if not selected:
        raise ValueError(f"No transitions found with group {group!r}. Available groups: {list_groups(transitions)}")
    return selected
