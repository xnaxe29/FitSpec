"""Emission line-list loading.

Parses the full nebular emission-line list CSV (default:
``data/nebular_emission_line_list.csv``). Every column from the source
table (atomic term structure, transition type, creation ionization
potential, observation reference, and the doublet-ratio metadata) is
retained verbatim on :class:`EmissionLine` -- nothing from the table is
discarded, even though the fitting code below only consumes a subset of
it. User-supplied line lists in the same schema are supported via the
``path`` argument.

Fixed-ratio amplitude ties
---------------------------
A pair of lines sharing a ``doublet_id`` is tied to a single free
amplitude parameter *only* when the table's ``physical_basis`` column
says the ratio is an atomic constant:

* ``"f-value"`` -- permitted resonance doublets from a common lower
  level (e.g. C IV 1548/1551, N V, O VI, Si IV, Al III), where the
  optically-thin flux ratio is fixed by the ratio of oscillator
  strengths (Osterbrock & Ferland 2006, Sec. 4.2).
* ``"A-value"`` -- forbidden/semiforbidden multiplets from a common
  upper level (e.g. [O III] 4959/5007, [N II] 6548/6583, [O I]
  6300/6364, [Ne III] 3868/3967), where the ratio is fixed by the
  ratio of spontaneous transition probabilities out of that level
  (Storey & Zeippen 2000).

Doublets whose ``physical_basis`` is ``"environment dependent"`` (e.g.
[O II] 3726/3729, [S II] 6716/6731, [S II] 4068/4076) are deliberately
left untied: these are collisionally-excited transitions from two
close-lying upper levels with different critical densities, so their
ratio is itself a density diagnostic (Osterbrock & Ferland 2006, Sec.
5.9) and must remain a free, independently fitted quantity rather than
a fixed constant.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field, replace
from pathlib import Path

__all__ = [
    "EmissionLine", "load_emission_line_list", "select_lines", "select_lines_in_wavelength_range",
    "apply_fixed_ratio_overrides", "DEFAULT_LINE_LIST_PATH",
]

DEFAULT_LINE_LIST_PATH = Path(__file__).resolve().parent.parent / "data" / "nebular_emission_line_list.csv"

# physical_basis values for which a fixed doublet-amplitude tie is
# scientifically justified (atomic-constant branching ratio).
_TIEABLE_PHYSICAL_BASES = {"f-value", "A-value"}


def _clean(token):
    token = (token or "").strip()
    return None if token in ("", "-") else token


def _clean_float(token):
    token = _clean(token)
    return None if token is None else float(token)


def _clean_bool(token):
    token = _clean(token)
    return None if token is None else token.strip().upper() == "TRUE"


def _clean_int(token):
    token = _clean(token)
    return None if token is None else int(float(token))


@dataclass(frozen=True)
class EmissionLine:
    """One emission-line transition, preserving every source-table column.

    Attributes
    ----------
    name : str
        Line identifier derived as ``f"{ion}{int(rest_wavelength_angstrom)}"``
        (spaces in ``ion`` replaced with underscores), matching the
        historical naming convention (e.g. ``"[O_III]5006"``,
        ``"Halpha"``-style Greek-letter names for ``Hα``/``Hβ``/etc.).
        Guaranteed unique within one loaded list.
    rest_wavelength_angstrom : float
        Rest-frame wavelength, taken directly from the table's ``#wave``
        column.
    ion : str
        Ion / species label, e.g. ``"[O III]"``, ``"He I"``, ``"Hα"``.
    lower_energy_eV, upper_energy_eV : float or None
        ``Ei (eV)`` / ``Ek (eV)``.
    lower_configuration, upper_configuration : str or None
        Split from the table's single ``Configurations`` field
        (``"lower - upper"``); None if the field was absent/unsplittable.
    lower_term, upper_term : str or None
        Split from ``Terms`` the same way.
    j_transition : str or None
        Raw ``Ji - Jk`` field (kept as a string; not all entries are
        simple numeric J values).
    multipole_type : str or None
        Transition multipole classification from the ``Type  (if not
        E1)`` column (E1, M1, E2, M1+E2, ...).
    creation_ip_eV : float or None
        ``Creation IP (eV)``.
    observation_reference : str or None
        ``Observation References``.
    line_type : str or None
        Physical classification from the table's second ``Type`` column
        (resonance, forbidden, semiforbidden, intercombination, ...).
    physical_basis : str or None
        What fixes ``intrinsic_ratio``, if anything (f-value, A-value,
        environment dependent, spin-forbidden).
    intrinsic_ratio : float or None
        The table's own per-member ratio value (not yet divided by the
        primary member -- see ``ratio_to_tied`` for the derived,
        fit-ready ratio).
    is_doublet : bool
        Whether this line was flagged as part of a doublet in the table.
    doublet_id : str or None
        Shared identifier grouping doublet members.
    doublet_member : int or None
        1-indexed position within the doublet (1 = primary).
    tied_to : str or None
        Name of another line in the same list this line's fitted
        amplitude is permanently forced to a fixed multiple of, or None
        if this line gets its own free amplitude. Derived, not a raw
        table column -- see the module docstring for the tying rule.
    ratio_to_tied : float or None
        The fixed multiplicative ratio applied to the tied-to line's
        amplitude, when ``tied_to`` is set.
    """

    name: str
    rest_wavelength_angstrom: float
    ion: str
    lower_energy_eV: "float | None" = None
    upper_energy_eV: "float | None" = None
    lower_configuration: "str | None" = None
    upper_configuration: "str | None" = None
    lower_term: "str | None" = None
    upper_term: "str | None" = None
    j_transition: "str | None" = None
    multipole_type: "str | None" = None
    creation_ip_eV: "float | None" = None
    observation_reference: "str | None" = None
    line_type: "str | None" = None
    physical_basis: "str | None" = None
    intrinsic_ratio: "float | None" = None
    is_doublet: bool = False
    doublet_id: "str | None" = None
    doublet_member: "int | None" = None
    tied_to: "str | None" = None
    ratio_to_tied: "float | None" = None

    def __post_init__(self):
        if self.rest_wavelength_angstrom <= 0:
            raise ValueError(f"Line {self.name!r}: rest_wavelength_angstrom must be positive.")
        if (self.tied_to is None) != (self.ratio_to_tied is None):
            raise ValueError(f"Line {self.name!r}: tied_to and ratio_to_tied must both be set or both be None.")
        if self.ratio_to_tied is not None and self.ratio_to_tied <= 0:
            raise ValueError(f"Line {self.name!r}: ratio_to_tied must be positive.")


def _split_pair(token, default=(None, None)):
    token = _clean(token)
    if token is None:
        return default
    parts = [part.strip() for part in token.split(" - ", 1)]
    if len(parts) != 2:
        return default
    return parts[0] or None, parts[1] or None


def _build_name(ion: str, wave: float) -> str:
    return f"{ion.replace(' ', '_')}{int(wave)}"


def load_emission_line_list(path=None) -> "list[EmissionLine]":
    """Parse the full nebular emission-line CSV into a list of EmissionLine.

    Parameters
    ----------
    path : str or Path, optional
        Line-list CSV to parse. Defaults to the packaged
        ``data/nebular_emission_line_list.csv``.

    Returns
    -------
    list[EmissionLine]
    """
    path = DEFAULT_LINE_LIST_PATH if path is None else Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Emission line list not found: {path}")

    with open(path, "r", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        header = [h.lstrip("#").strip() for h in header]
        required = {"wave", "Ion", "physical_basis", "intrinsic_ratio", "is_doublet", "doublet_id", "doublet_member"}
        missing = required - set(header)
        if missing:
            raise ValueError(f"{path}: missing required column(s) {sorted(missing)}.")
        index = {name: position for position, name in enumerate(header)}

        raw_rows = []
        name_counts = {}
        for row_number, row in enumerate(reader, start=2):
            if not row or not row[index["wave"]].strip():
                continue
            wave = _clean_float(row[index["wave"]])
            if wave is None:
                raise ValueError(f"{path}:{row_number}: missing/invalid wavelength.")
            ion = row[index["Ion"]].strip()
            base_name = _build_name(ion, wave)
            count = name_counts.get(base_name, 0)
            name_counts[base_name] = count + 1
            name = base_name if count == 0 else f"{base_name}_{chr(ord('a') + count)}"

            lower_config, upper_config = _split_pair(row[index["Configurations"]]) if "Configurations" in index else (None, None)
            lower_term, upper_term = _split_pair(row[index["Terms"]]) if "Terms" in index else (None, None)

            raw_rows.append(dict(
                row_number=row_number, name=name, wave=wave, ion=ion,
                lower_energy_eV=_clean_float(row[index["Ei (eV)"]]) if "Ei (eV)" in index else None,
                upper_energy_eV=_clean_float(row[index["Ek (eV)"]]) if "Ek (eV)" in index else None,
                lower_configuration=lower_config, upper_configuration=upper_config,
                lower_term=lower_term, upper_term=upper_term,
                j_transition=_clean(row[index["Ji - Jk"]]) if "Ji - Jk" in index else None,
                multipole_type=_clean(row[index["Type  (if not E1)"]]) if "Type  (if not E1)" in index else None,
                creation_ip_eV=_clean_float(row[index["Creation IP (eV)"]]) if "Creation IP (eV)" in index else None,
                observation_reference=_clean(row[index["Observation References"]]) if "Observation References" in index else None,
                line_type=_clean(row[index["Type"]]) if "Type" in index else None,
                physical_basis=_clean(row[index["physical_basis"]]),
                intrinsic_ratio=_clean_float(row[index["intrinsic_ratio"]]),
                is_doublet=bool(_clean_bool(row[index["is_doublet"]])),
                doublet_id=_clean(row[index["doublet_id"]]),
                doublet_member=_clean_int(row[index["doublet_member"]]),
            ))

    # Resolve fixed-ratio ties: within each doublet_id, tie every member
    # after the lowest doublet_member index to that primary member, but
    # ONLY when physical_basis marks the ratio as an atomic constant.
    by_doublet: "dict[str, list[dict]]" = {}
    for entry in raw_rows:
        if entry["is_doublet"] and entry["doublet_id"] is not None and entry["physical_basis"] in _TIEABLE_PHYSICAL_BASES:
            by_doublet.setdefault(entry["doublet_id"], []).append(entry)

    tied_to = {}
    ratio_to_tied = {}
    for doublet_id, members in by_doublet.items():
        members_with_index = [m for m in members if m["doublet_member"] is not None]
        if len(members_with_index) < 2:
            continue
        members_with_index.sort(key=lambda m: m["doublet_member"])
        primary = members_with_index[0]
        if primary["intrinsic_ratio"] in (None, 0):
            continue
        for follower in members_with_index[1:]:
            if follower["intrinsic_ratio"] is None:
                continue
            tied_to[follower["name"]] = primary["name"]
            ratio_to_tied[follower["name"]] = follower["intrinsic_ratio"] / primary["intrinsic_ratio"]

    lines = []
    for entry in raw_rows:
        lines.append(EmissionLine(
            name=entry["name"], rest_wavelength_angstrom=entry["wave"], ion=entry["ion"],
            lower_energy_eV=entry["lower_energy_eV"], upper_energy_eV=entry["upper_energy_eV"],
            lower_configuration=entry["lower_configuration"], upper_configuration=entry["upper_configuration"],
            lower_term=entry["lower_term"], upper_term=entry["upper_term"],
            j_transition=entry["j_transition"], multipole_type=entry["multipole_type"],
            creation_ip_eV=entry["creation_ip_eV"], observation_reference=entry["observation_reference"],
            line_type=entry["line_type"], physical_basis=entry["physical_basis"],
            intrinsic_ratio=entry["intrinsic_ratio"], is_doublet=entry["is_doublet"],
            doublet_id=entry["doublet_id"], doublet_member=entry["doublet_member"],
            tied_to=tied_to.get(entry["name"]), ratio_to_tied=ratio_to_tied.get(entry["name"]),
        ))
    return lines


def select_lines(line_list: "list[EmissionLine]", names=None) -> "list[EmissionLine]":
    """Restrict a parsed line list to an explicit subset of names.

    If a selected line is tied to a line NOT in the subset, the tied-to
    line is included automatically (a tie cannot reference an absent
    parameter). ``names=None`` returns the full list unchanged.
    """
    if names is None:
        return list(line_list)
    wanted = set(names)
    by_name = {emission_line.name: emission_line for emission_line in line_list}
    missing = wanted - by_name.keys()
    if missing:
        raise ValueError(f"Requested emission lines not found in line list: {sorted(missing)}")
    closure = set(wanted)
    changed = True
    while changed:
        changed = False
        for name in list(closure):
            tied_to = by_name[name].tied_to
            if tied_to is not None and tied_to not in closure:
                closure.add(tied_to)
                changed = True
    return [emission_line for emission_line in line_list if emission_line.name in closure]


def select_lines_in_wavelength_range(
    line_list: "list[EmissionLine]", wave_min, wave_max, *, redshift: float = 0.0,
) -> "list[EmissionLine]":
    """Auto-select lines whose observed-frame wavelength falls in [wave_min, wave_max].

    Reproduces the legacy behavior of generating the fitted line set
    automatically from the spectrum's wavelength coverage. Ties are
    resolved via the same closure as :func:`select_lines`.
    """
    in_range = [
        emission_line for emission_line in line_list
        if wave_min <= emission_line.rest_wavelength_angstrom * (1.0 + redshift) <= wave_max
    ]
    return select_lines(line_list, [emission_line.name for emission_line in in_range])


def apply_fixed_ratio_overrides(
    line_list: "list[EmissionLine]", species_pairs: "list[tuple[str, str]]", ratio_values: "list[float]",
) -> "list[EmissionLine]":
    """Apply user-specified fixed amplitude ratios on top of a loaded line list.

    Lets a user fix the flux ratio between any two lines at will -- e.g.
    to cross-check a published density diagnostic -- in addition to
    (and, on conflict, overriding) whatever ties the catalog itself
    already encodes (see the module docstring: only lines whose
    ``physical_basis`` is an atomic constant are tied by default; a
    pair like [SII]6716/6731 is deliberately left untied by the catalog
    since it is a density diagnostic, but a user may still choose to
    fix it here for a specific test).

    Parameters
    ----------
    line_list : list[EmissionLine]
        Typically the full catalog from :func:`load_emission_line_list`,
        so the tie is in place before any wavelength-range/explicit-list
        subsetting (:func:`select_lines`) resolves its closure.
    species_pairs : list[(str, str)]
        Each pair ``(name_a, name_b)`` names two lines already present
        in ``line_list`` (by :attr:`EmissionLine.name`). ``name_b`` is
        tied to ``name_a`` (``name_a`` becomes/remains the free,
        "primary" line for that pair).
    ratio_values : list[float]
        One positive ratio per pair, in the same order, defined as
        ``ratio_values[i] = flux(species_pairs[i][0]) / flux(species_pairs[i][1])``
        -- i.e. first-listed line's flux divided by second-listed
        line's flux. Swap a pair's order to fix the ratio the other way
        around.

    Returns
    -------
    list[EmissionLine]
        A new list (the input is left unmodified) with the requested
        ties applied. Raises ``ValueError`` if a named line isn't in
        ``line_list``, a pair ties a line to itself, or the requested
        ties (combined with the catalog's own ties) would form a cycle.
    """
    if len(species_pairs) != len(ratio_values):
        raise ValueError("species_pairs and ratio_values must be the same length.")

    by_name = {emission_line.name: emission_line for emission_line in line_list}
    tied_to_override: "dict[str, tuple[str, float]]" = {}
    for (name_a, name_b), ratio_value in zip(species_pairs, ratio_values):
        for name in (name_a, name_b):
            if name not in by_name:
                raise ValueError(f"apply_fixed_ratio_overrides: line {name!r} not found in the line list.")
        if name_a == name_b:
            raise ValueError(f"apply_fixed_ratio_overrides: cannot tie {name_a!r} to itself.")
        if ratio_value <= 0:
            raise ValueError(f"apply_fixed_ratio_overrides: ratio for ({name_a!r}, {name_b!r}) must be positive.")
        # ratio_value = flux(a)/flux(b)  =>  flux(b) = flux(a) / ratio_value
        tied_to_override[name_b] = (name_a, 1.0 / ratio_value)

    new_lines = []
    for emission_line in line_list:
        if emission_line.name in tied_to_override:
            tied_to, ratio_to_tied = tied_to_override[emission_line.name]
            new_lines.append(replace(emission_line, tied_to=tied_to, ratio_to_tied=ratio_to_tied))
        else:
            new_lines.append(emission_line)

    # Cycle check: chase tied_to pointers (post-override) from every line; a cycle
    # means the requested overrides are mutually inconsistent (e.g. A->B and B->A).
    by_name_new = {emission_line.name: emission_line for emission_line in new_lines}
    for start_name in by_name_new:
        seen = set()
        current = start_name
        while by_name_new[current].tied_to is not None:
            if current in seen:
                raise ValueError(
                    f"apply_fixed_ratio_overrides: tie cycle detected involving {start_name!r}."
                )
            seen.add(current)
            current = by_name_new[current].tied_to

    return new_lines
