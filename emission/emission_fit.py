"""Emission-line fitting entry point.

Builds an explicit-component ``ModelParameters``, wires it to the
generic ``core.fitting.fit_deterministic`` engine via
``emission.emission_model.make_emission_model_func``, and returns the
derived ``EmissionFitResult``. Also implements automatic-component-count
selection (deterministic fits for n=1..N_max, ranked by BIC plus a
configurable penalty per extra component) and automatic line-list
selection from the spectrum's wavelength coverage, both carried over
from the FitSpec legacy fitting script's ``N_max``/``penalty_factor``
and "generated automatically from the spectral range" behaviors.

``emission_flux_normalizing_factor``/``emission_flux_reduction`` are
applied exactly once, up front, by :func:`normalize_emission_spectrum` --
matching the original workflow (``orig_flux_noisy = orig_flux_clean_bin
/ flux_normalizing_factor``, applied immediately after loading, never
undone). This function does *not* re-derive or undo that scaling itself:
like continuum subtraction (see ``continuum.continuum``), normalization
is the caller's responsibility, applied once before the spectrum is
handed anywhere -- not a transient internal numerical-conditioning trick
invisible outside the optimizer, which is what made it easy to
accidentally reason about a spectrum in the wrong units elsewhere (e.g.
a "ghost" reference continuum from an unrelated, unnormalized fit).
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from core.fitting import fit_deterministic
from core.statistics import compute_fit_statistics

from emission.emission_model import (
    build_emission_parameters, make_emission_model_func,
    amplitude_parameter_name, velocity_parameter_name, sigma_parameter_name,
)
from emission.emission_results import EmissionFitResult, summarize_emission_fit
from emission.lines import (
    load_emission_line_list, select_lines, select_lines_in_wavelength_range, apply_fixed_ratio_overrides,
)
from emission.rejection import fit_with_rejection

__all__ = ["fit_emission_spectrum", "select_emission_line_list", "normalize_emission_spectrum"]


def _get(config, key, default=None):
    return config.get(key, default) if hasattr(config, "get") else default


def _raw_string(config, key, default=""):
    """Recover the original config-file text for ``key``, undoing
    ``core.config``'s generic (naive, every-comma) list-splitting if it
    already broke the value into pieces -- needed for values with their
    own internal structure (e.g. ``emission_fixed_ratio_species``'s
    nested-bracket pair-list syntax) that a flat comma-split would
    otherwise corrupt. Rejoining with "," is lossless: it exactly
    reverses the naive split regardless of what the text means.
    """
    value = _get(config, key, default)
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value)
    return str(value)


def _split_top_level(text: str) -> "list[str]":
    """Split ``text`` on commas that are not nested inside [...] brackets."""
    parts = []
    depth = 0
    current = []
    for ch in text:
        if ch == "[":
            depth += 1
            current.append(ch)
        elif ch == "]":
            depth -= 1
            if depth < 0:
                raise ValueError(f"Unbalanced ']' in {text!r}.")
            current.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
    if depth != 0:
        raise ValueError(f"Unbalanced '[' in {text!r}.")
    parts.append("".join(current))
    return parts


def _parse_species_pairs(raw: str) -> "list[tuple[str, str]]":
    """Parse ``fixed_ratio_species``'s nested-bracket pair-list syntax.

    Format: ``[name_a,name_b], [name_a,name_b], ...`` where each
    ``name`` may itself contain literal ``[`` ``]`` characters (as
    FitSpec's own generated emission-line names typically do, e.g.
    ``"[S_II]6716"``) -- handled by tracking bracket nesting depth
    rather than naive string splitting, so a pair's own delimiting
    brackets are distinguished from brackets that are part of a line
    name.
    """
    raw = raw.strip()
    if not raw:
        return []
    pairs = []
    i, n = 0, len(raw)
    while i < n:
        if raw[i].isspace() or raw[i] == ",":
            i += 1
            continue
        if raw[i] != "[":
            raise ValueError(f"emission_fixed_ratio_species: expected '[' at position {i} in {raw!r}.")
        depth = 1
        j = i + 1
        while j < n and depth > 0:
            if raw[j] == "[":
                depth += 1
            elif raw[j] == "]":
                depth -= 1
            j += 1
        if depth != 0:
            raise ValueError(f"emission_fixed_ratio_species: unbalanced brackets in {raw!r}.")
        pair_content = raw[i + 1:j - 1]
        parts = [part.strip() for part in _split_top_level(pair_content)]
        if len(parts) != 2 or not all(parts):
            raise ValueError(
                f"emission_fixed_ratio_species: expected exactly 2 line names per pair, got {parts!r} from {raw[i:j]!r}."
            )
        pairs.append((parts[0], parts[1]))
        i = j
    return pairs


def _parse_float_list(raw: str) -> "list[float]":
    """Parse ``fixed_ratio_value``'s bracket-wrapped comma list of numbers, e.g. ``[1.1, 0.9]``."""
    raw = raw.strip()
    if not raw:
        return []
    if raw.startswith("[") and raw.endswith("]"):
        raw = raw[1:-1]
    return [float(token.strip()) for token in raw.split(",") if token.strip()]


def _pair(config, key, default):
    value = _get(config, key, default)
    if isinstance(value, (list, tuple, np.ndarray)):
        vals = list(value)
    else:
        vals = [x.strip() for x in str(value).split(",")]
    if len(vals) < 2:
        return tuple(map(float, default))
    return float(vals[0]), float(vals[1])


def _as_list(config, key, cast=str):
    value = _get(config, key, None)
    if value is None or value == "" or value == []:
        return None
    if isinstance(value, (list, tuple, np.ndarray)):
        values = value
    else:
        values = [x.strip() for x in str(value).split(",") if x.strip()]
    return [cast(x) for x in values]


def select_emission_line_list(spectrum, config, line_list=None):
    """Resolve the line list to fit, per the config precedence:

    1. ``emission_lines`` (explicit subset) if set -- used exactly as given.
    2. Otherwise, every catalog line whose observed-frame wavelength falls
       within the spectrum's wavelength coverage (legacy "generated
       automatically from the spectral range" default).

    Before any of the above, ``emission_fixed_ratio_species``/
    ``emission_fixed_ratio_value`` (if set) are applied to the full
    catalog via :func:`emission.lines.apply_fixed_ratio_overrides`, so a
    user-specified tie is in place before subsetting resolves its
    closure -- letting a user fix the ratio between any two lines at
    will (e.g. to cross-check a published density-diagnostic value),
    on top of whatever ties the catalog already encodes.

    ``emission_lines`` is *this* selection's one keyword; a separate,
    optional second-stage line list (``emission_weak_lines``, resolved
    independently by ``_resolve_weak_lines``, not here) can additionally
    be fit for amplitude only, against the kinematics this selection's
    lines establish -- see ``_fit_once`` for that two-stage mechanism.
    """
    if line_list is None:
        path = _get(config, "emission_line_list_path", None)
        line_list = load_emission_line_list(None if not path else path)

    species_raw = _raw_string(config, "emission_fixed_ratio_species", "")
    value_raw = _raw_string(config, "emission_fixed_ratio_value", "")
    if species_raw.strip() or value_raw.strip():
        species_pairs = _parse_species_pairs(species_raw)
        ratio_values = _parse_float_list(value_raw)
        if len(species_pairs) != len(ratio_values):
            raise ValueError(
                f"emission_fixed_ratio_species has {len(species_pairs)} pair(s) but "
                f"emission_fixed_ratio_value has {len(ratio_values)} value(s); they must match."
            )
        if species_pairs:
            line_list = apply_fixed_ratio_overrides(line_list, species_pairs, ratio_values)

    explicit = _as_list(config, "emission_lines", str)
    if explicit is not None:
        return select_lines(line_list, explicit)

    wave = np.asarray(spectrum.wave, dtype=float)
    return select_lines_in_wavelength_range(
        line_list, float(np.nanmin(wave)), float(np.nanmax(wave)), redshift=spectrum.redshift,
    )


def normalize_emission_spectrum(spectrum, config):
    """Apply ``emission_flux_normalizing_factor``/``emission_flux_reduction``
    once, returning a *new* Spectrum: ``flux = (flux - reduction) / factor``.

    Never mutates ``spectrum`` in place -- it's typically the one Spectrum
    object shared with the stellar/absorption panels (see
    ``gui.state.SessionState``), so this returns an independent copy the
    same way ``gui.emission.EmissionGUI._continuum_subtracted_spectrum``
    does for continuum subtraction. Called once, as early as possible
    (``EmissionGUI.__init__``, before line selection, continuum
    estimation, or anything else) -- everything downstream in the
    emission session then consistently operates in these rescaled units,
    including a "ghost" reference continuum from an unrelated (and
    unnormalized) stellar fit, which callers must rescale the same way
    before comparing it (see ``gui.emission.EmissionGUI._find_stellar_continuum``).

    The normalizing factor/reduction actually used are recorded in the
    returned spectrum's ``metadata`` for provenance.
    """
    normalizing_factor = float(_get(config, "emission_flux_normalizing_factor", 1.0))
    reduction = float(_get(config, "emission_flux_reduction", 0.0))
    if normalizing_factor <= 0:
        raise ValueError("emission_flux_normalizing_factor must be strictly positive.")
    flux = (np.asarray(spectrum.flux, dtype=float) - reduction) / normalizing_factor
    flux_unc = None if spectrum.flux_unc is None else np.asarray(spectrum.flux_unc, dtype=float) / abs(normalizing_factor)
    metadata = dict(spectrum.metadata)
    metadata["emission_flux_normalizing_factor"] = normalizing_factor
    metadata["emission_flux_reduction"] = reduction
    return replace(spectrum, flux=flux, flux_unc=flux_unc, metadata=metadata)


def _resolve_weak_lines(config, main_line_list):
    """Resolve ``emission_weak_lines`` (comma-separated names) against the
    same catalog ``select_emission_line_list`` would use, disjoint from
    ``main_line_list``. Empty list if the key isn't set.
    """
    raw = _as_list(config, "emission_weak_lines", str)
    if not raw:
        return []
    path = _get(config, "emission_line_list_path", None)
    catalog = load_emission_line_list(None if not path else path)
    weak_lines = select_lines(catalog, raw)
    overlap = {line.name for line in main_line_list} & {line.name for line in weak_lines}
    if overlap:
        raise ValueError(
            f"emission_weak_lines overlaps emission_lines: {sorted(overlap)}. "
            "A line must appear in exactly one of the two lists."
        )
    return weak_lines


def _fit_once(spectrum, config, line_list, n_components, weak_lines, use_rejection):
    """Fit ``line_list`` with ``n_components`` kinematic components.

    Two optional, independently-toggled behaviors compose here:

    * Insignificant-component rejection -- freeze whole components whose
      amplitude isn't a significant detection in any of their lines,
      refitting between freezes (see ``emission.rejection.fit_with_rejection``).
      This is the same mechanism ``fit_emission_spectrum``'s automatic BIC
      component-count search reuses for every candidate N it tries, since
      a spuriously high N should get penalized by both BIC's parameter
      count *and* by however many of its components immediately freeze
      out. ``use_rejection`` (bool or None) decides this explicitly if
      given -- e.g. a live GUI checkbox -- else falls back to reading
      ``emission_reject_insignificant_components`` from ``config``. This
      never changes the returned component *count*: a component that's
      rejected is frozen in place, not deleted, and requesting N always
      returns exactly N components (with rejection off, no freezing
      happens at all -- every one of the N stays a genuinely free
      parameter for the whole fit, whatever the optimizer settles on).
    * ``emission_weak_lines`` -- a second, amplitude-only fitting stage
      for lines the main fit never sees as free parameters: after the
      (possibly rejection-aware) main fit above, every component's
      velocity_kms/sigma_kms and every *main*-list line's amplitude are
      frozen at their just-fitted values, weak lines' amplitudes are
      un-frozen (except within a component the main fit already rejected
      -- a weak line gets no independent chance to be significant in a
      kinematic system the strong lines couldn't support), and a second
      ``fit_deterministic`` pass fits only those. The two stages'
      uncertainty dicts and free-parameter counts are merged into one
      final FitResult/statistics, but every line (main and weak alike)
      ends up as an ordinary parameter of the same ``ModelParameters`` --
      saving/loading is completely unaffected by any of this.
    """
    v0 = float(_get(config, "emission_velocity_initial_kms", 0.0))
    s0 = float(_get(config, "emission_sigma_initial_kms", 50.0))
    v_bounds = _pair(config, "emission_velocity_bounds_kms", (-500.0, 500.0))
    s_bounds = _pair(config, "emission_sigma_bounds_kms", (1.0, 1000.0))
    amplitude_max = _get(config, "emission_maximum_amplitude", None)
    amplitude_bounds = (0.0, np.inf if amplitude_max in (None, "", 0) else float(amplitude_max))
    kinematics_mode = str(_get(config, "emission_kinematics_mode", "tied")).strip().lower()

    weak_names = {line.name for line in weak_lines}
    combined_line_list = list(line_list) + list(weak_lines)

    model_parameters = build_emission_parameters(
        combined_line_list, n_components,
        velocity_initial_kms=v0, velocity_bounds_kms=v_bounds,
        sigma_initial_kms=s0, sigma_bounds_kms=s_bounds,
        amplitude_bounds=amplitude_bounds, kinematics_mode=kinematics_mode,
    )
    if weak_names:
        # Stage 1 (below): weak lines contribute nothing yet, so they
        # can't influence the main fit's kinematics/amplitudes at all.
        for component in model_parameters.components:
            for name in weak_names:
                amplitude = component[amplitude_parameter_name(name)]
                amplitude.value = 0.0
                amplitude.fixed = True
                component[velocity_parameter_name(name)].fixed = True
                component[sigma_parameter_name(name)].fixed = True

    wave = np.asarray(spectrum.wave, dtype=float)
    flux = np.asarray(spectrum.flux, dtype=float)
    flux_unc = np.asarray(spectrum.flux_unc, dtype=float)
    mask = spectrum.mask
    n_valid = int(np.count_nonzero(mask)) if mask is not None else wave.size
    minimum_pixels = int(_get(config, "emission_minimum_fit_pixels", 10))
    if n_valid < minimum_pixels:
        raise ValueError(f"Too few valid pixels ({n_valid}) for emission fitting (minimum {minimum_pixels}).")
    resolution_source = None if spectrum.resolution is None else getattr(spectrum.resolution, "source", str(spectrum.resolution))
    max_function_evaluations = int(_get(config, "emission_max_function_evaluations", 20000))

    model_func = make_emission_model_func(
        combined_line_list, redshift=spectrum.redshift, resolution=spectrum.resolution, kinematics_mode=kinematics_mode,
    )

    use_rejection = str(_get(config, "emission_reject_insignificant_components", False)).strip().lower() in ("true", "1") if use_rejection is None else bool(use_rejection)
    main_free_line_names = [emission_line.name for emission_line in line_list if emission_line.tied_to is None]
    if use_rejection:
        fit_result, frozen_flags = fit_with_rejection(
            model_parameters, model_func,
            wave=wave, flux=flux, flux_unc=flux_unc, mask=mask, redshift=spectrum.redshift,
            resolution_source=resolution_source, max_function_evaluations=max_function_evaluations,
            free_line_names=main_free_line_names,
            snr_threshold=float(_get(config, "emission_reject_snr_threshold", 3.0)),
            margin_fraction=float(_get(config, "emission_reject_margin_fraction", 0.01)),
            max_passes=int(_get(config, "emission_max_rejection_passes", 3)),
        )
    else:
        fit_result = fit_deterministic(
            wave, flux, flux_unc, model_parameters, model_func,
            mask=mask, redshift=spectrum.redshift, resolution_source=resolution_source,
            max_function_evaluations=max_function_evaluations,
        )
        frozen_flags = [False] * n_components

    if weak_names:
        # Stage 2: freeze everything stage 1 just fit (main lines'
        # amplitudes, every component's kinematics), then free only the
        # weak lines' amplitudes -- except in a component rejection
        # already froze, which stays off for weak lines too.
        stage1_uncertainties = dict(fit_result.parameter_uncertainties)
        stage1_k_params = fit_result.statistics.k_params
        for index, component in enumerate(fit_result.parameters.components):
            component["velocity_kms"].fixed = True
            component["sigma_kms"].fixed = True
            for name in main_free_line_names:
                component[amplitude_parameter_name(name)].fixed = True
                component[velocity_parameter_name(name)].fixed = True
                component[sigma_parameter_name(name)].fixed = True
            if frozen_flags[index]:
                continue
            for name in weak_names:
                weak_amplitude = component[amplitude_parameter_name(name)]
                # scipy.optimize.curve_fit's trust-region-reflective method
                # gets stuck immediately if a bounded parameter starts
                # exactly on its lower bound -- confirmed directly: p0=0.0
                # with bounds=(0, upper) never moves even when the true
                # chi-square minimum is sharp and obvious, while any
                # nonzero p0 converges normally. Stage 1 deliberately left
                # this at exactly 0.0 (to silence it while frozen), so it
                # must be nudged off that boundary before being freed here
                # -- same 10%-of-upper-bound heuristic already used for
                # this in gui.emission.EmissionGUI._default_amplitude_settings.
                upper = weak_amplitude.upper
                weak_amplitude.value = 0.1 * upper if (np.isfinite(upper) and upper > 0) else 1.0
                weak_amplitude.fixed = False

        fit_result = fit_deterministic(
            wave, flux, flux_unc, fit_result.parameters, model_func,
            mask=mask, redshift=spectrum.redshift, resolution_source=resolution_source,
            max_function_evaluations=max_function_evaluations,
        )
        fit_result.parameter_uncertainties = {**stage1_uncertainties, **fit_result.parameter_uncertainties}

        # The stage-2 FitResult's own statistics only counted stage 2's
        # free parameters (weak-line amplitudes) -- recompute with the
        # true total (stage 1 + stage 2) so BIC/reduced chi-square aren't
        # misleadingly favorable.
        mask_arr = mask if mask is not None else np.ones_like(wave, dtype=bool)
        residuals = flux[mask_arr] - fit_result.model[mask_arr]
        fit_result.statistics = compute_fit_statistics(
            residuals, flux_unc[mask_arr], k_params=stage1_k_params + fit_result.statistics.k_params,
        )
        fit_result.metadata["emission_weak_line_names"] = sorted(weak_names)

    # Provenance only -- these describe units the caller already applied
    # (see normalize_emission_spectrum), not a transform performed here.
    fit_result.metadata.update({
        "emission_kinematics_mode": kinematics_mode,
        "emission_flux_normalizing_factor": float(spectrum.metadata.get("emission_flux_normalizing_factor", 1.0)),
        "emission_flux_reduction": float(spectrum.metadata.get("emission_flux_reduction", 0.0)),
        "emission_frozen_components": list(frozen_flags),
    })

    return fit_result


def fit_emission_spectrum(spectrum, config, *, line_list=None, use_rejection=None, n_components=None) -> EmissionFitResult:
    """Fit a (typically continuum-subtracted, pre-normalized) spectrum with
    N kinematic components.

    Parameters
    ----------
    spectrum : core.spectrum.Spectrum
        Must have ``flux_unc`` set (required for chi-square fitting) and,
        for a physically meaningful fit, should already have the
        continuum removed (see ``continuum.continuum``) and, if
        ``emission_flux_normalizing_factor``/``emission_flux_reduction``
        are configured, already have those applied too (see
        :func:`normalize_emission_spectrum`) -- this function does not
        apply or undo either transform itself; results come back in
        whatever units ``spectrum.flux`` was already in.
    config : object supporting ``.get(key, default)``
        See ``config/default_config_emission.dat`` for the full,
        documented set of recognized keys (line-list selection,
        kinematics, amplitude bounds, automatic component-count search,
        insignificant-component rejection, weak-line second-stage
        fitting).
    line_list : list[emission.lines.EmissionLine], optional
        Pre-loaded/pre-selected line list, bypassing all of the
        ``emission_lines``/spectral-range selection logic (and,
        consequently, ``emission_fixed_ratio_species``/
        ``emission_fixed_ratio_value``/``emission_weak_lines`` too --
        pass a list with the desired ties/weak lines already applied if
        bypassing ``select_emission_line_list`` this way).
    use_rejection : bool, optional
        Explicitly enable/disable insignificant-component rejection (see
        ``_fit_once``), overriding ``emission_reject_insignificant_components``
        from ``config`` for this call -- e.g. a live GUI checkbox (see
        ``gui.emission.EmissionGUI``'s "Help decide components"). Leave
        as ``None`` to just use whatever ``config`` says.
    n_components : int, optional
        Explicitly set the starting component count, overriding
        ``emission_n_components`` from ``config`` for this call -- e.g. a
        GUI session where the person has clicked "+"/"-" to change the
        component count interactively (``config``'s own value never
        changes when that happens, so without this the fit would
        silently revert to whatever the config file says instead of what
        the GUI currently shows). Leave as ``None`` to just use whatever
        ``config`` says.

    Returns
    -------
    EmissionFitResult
    """
    if spectrum.flux_unc is None:
        raise ValueError("Emission-line chi-square fitting requires flux_unc.")

    if line_list is None:
        line_list = select_emission_line_list(spectrum, config)
    weak_lines = _resolve_weak_lines(config, line_list)
    combined_line_list = list(line_list) + weak_lines

    n_components = int(_get(config, "emission_n_components", 1)) if n_components is None else int(n_components)
    n_max = _get(config, "emission_n_components_max", None)
    penalty_factor = float(_get(config, "emission_bic_penalty_factor", 0.0))
    if use_rejection is None:
        use_rejection = str(_get(config, "emission_reject_insignificant_components", False)).strip().lower() in ("true", "1")
    else:
        use_rejection = bool(use_rejection)

    # "Help decide components" (use_rejection) gates BOTH automatic
    # component-count search AND insignificant-component rejection
    # together, as one decision: with it off, n_components is used
    # exactly as given -- no searching upward via emission_n_components_max,
    # no freezing, full stop. n_max is only even consulted once
    # use_rejection is confirmed True.
    if not use_rejection or n_max in (None, "", 0) or int(n_max) <= n_components:
        fit_result = _fit_once(spectrum, config, line_list, n_components, weak_lines, use_rejection)
        return summarize_emission_fit(fit_result, combined_line_list)

    # Automatic component-count search: fit n=n_components..n_max, pick the
    # lowest BIC + penalty_factor * n_components (legacy N_max/penalty_factor).
    #
    # Deliberately WITHOUT rejection during the search itself: freezing a
    # padded candidate's insignificant component mid-search would quietly
    # shrink its effective parameter count for the BIC comparison while
    # the optimizer still got extra attempts at a marginally better
    # chi-square along the way -- biasing the search toward "one extra
    # component" every time, since the padding's true cost gets masked
    # from the very comparison meant to penalize it. Every candidate is
    # therefore compared on equal footing (plain fits only); rejection is
    # applied exactly once, to the winning N, as a final cleanup pass --
    # still genuinely useful there (the chosen N can still have an
    # individually-insignificant component within it), just never
    # allowed to influence which N wins.
    best_result, best_score, best_n = None, np.inf, n_components
    for candidate_n in range(n_components, int(n_max) + 1):
        candidate_fit = _fit_once(spectrum, config, line_list, candidate_n, weak_lines, use_rejection=False)
        score = candidate_fit.statistics.bic + penalty_factor * candidate_n
        if score < best_score:
            best_result, best_score, best_n = candidate_fit, score, candidate_n

    best_result = _fit_once(spectrum, config, line_list, best_n, weak_lines, use_rejection=True)
    return summarize_emission_fit(best_result, combined_line_list)
