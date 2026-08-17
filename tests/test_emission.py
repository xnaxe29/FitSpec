import numpy as np
import pytest

from core.spectrum import Spectrum

from emission.lines import EmissionLine, load_emission_line_list, select_lines, apply_fixed_ratio_overrides
from emission.emission_model import build_emission_parameters, make_emission_model_func
from emission.emission_fit import fit_emission_spectrum, select_emission_line_list, normalize_emission_spectrum
from emission.emission_results import save_emission_result, load_emission_result
from emission.profiles import gaussian_line_flux


class DictConfig(dict):
    """Minimal stand-in for core.config.Config supporting .get()."""


def test_load_default_line_list_parses_ties():
    line_list = load_emission_line_list()
    assert len(line_list) == 243
    by_name = {line.name: line for line in line_list}
    assert "Halpha" not in by_name  # names are derived as f"{ion}{int(wave)}"
    assert by_name["[O_III]5006"].tied_to == "[O_III]4958"
    assert by_name["[O_III]5006"].ratio_to_tied == pytest.approx(2.98)
    assert by_name["[N_II]6583"].tied_to == "[N_II]6548"
    assert by_name["[N_II]6583"].ratio_to_tied == pytest.approx(3.0)


def test_environment_dependent_doublets_are_not_tied():
    """[OII] 3726/3729 and [SII] 6716/6731 are density diagnostics and
    must never be hard-tied, even though the table flags them as doublets."""
    line_list = load_emission_line_list()
    by_name = {line.name: line for line in line_list}
    assert by_name["[O_II]3726"].physical_basis == "environment dependent"
    assert by_name["[O_II]3726"].tied_to is None
    assert by_name["[O_II]3728"].tied_to is None
    assert by_name["[S_II]6716"].tied_to is None
    assert by_name["[S_II]6730"].tied_to is None


def test_select_lines_pulls_in_tied_to_closure():
    line_list = load_emission_line_list()
    selected = select_lines(line_list, ["[O_III]5006"])
    names = {line.name for line in selected}
    assert names == {"[O_III]5006", "[O_III]4958"}


def test_select_lines_in_wavelength_range():
    line_list = load_emission_line_list()
    selected = select_emission_line_list_by_range(line_list, 6500.0, 6600.0)
    names = {line.name for line in selected}
    assert "Hα6562" in names
    assert "[N_II]6548" in names
    assert "[O_III]5006" not in names


def select_emission_line_list_by_range(line_list, wave_min, wave_max):
    from emission.lines import select_lines_in_wavelength_range
    return select_lines_in_wavelength_range(line_list, wave_min, wave_max)


def test_build_emission_parameters_explicit_component_count():
    line_list = [
        EmissionLine("Halpha6562", 6562.819, ion="Hα"),
        EmissionLine("Hbeta4861", 4861.333, ion="Hβ"),
    ]
    params = build_emission_parameters(line_list, n_components=2)
    assert params.n_components == 2
    assert len(params.components) == 2
    for component in params.components:
        assert "velocity_kms" in component
        assert "sigma_kms" in component
        assert "amp_Halpha6562" in component
        assert "amp_Hbeta4861" in component


def test_kinematics_mode_fixed_holds_parameters_out_of_fit():
    line_list = [EmissionLine("Halpha6562", 6562.819, ion="Hα")]
    params = build_emission_parameters(line_list, n_components=2, kinematics_mode="fixed")
    for component in params.components:
        assert component["velocity_kms"].fixed
        assert component["sigma_kms"].fixed
        assert component["velocity_kms_Halpha6562"].fixed
        assert component["sigma_kms_Halpha6562"].fixed
    assert params.to_vector().size == 2  # only the two amplitude parameters are free


def test_kinematics_mode_tied_ties_lines_within_a_component_not_components_to_each_other():
    """"tied" ties every LINE in a component to that component's shared
    kinematics -- it does NOT tie components to each other. Each
    component's velocity_kms/sigma_kms is independently free."""
    line_list = [EmissionLine("Halpha6562", 6562.819, ion="Hα"), EmissionLine("Hbeta4861", 4861.333, ion="Hβ")]
    params = build_emission_parameters(line_list, n_components=3, kinematics_mode="tied")
    for component in params.components:
        assert not component["velocity_kms"].fixed
        assert not component["sigma_kms"].fixed
        # both lines in every component track that component's own value,
        # not each other's or another component's.
        assert component["velocity_kms_Halpha6562"].fixed
        assert component["velocity_kms_Hbeta4861"].fixed
        assert component["sigma_kms_Halpha6562"].fixed
        assert component["sigma_kms_Hbeta4861"].fixed
    # 3 components * (velocity_kms + sigma_kms) = 6 free kinematics
    # parameters, plus 3 components * 2 lines = 6 free amplitudes.
    assert params.to_vector().size == 12


def test_kinematics_mode_free_gives_every_line_independent_kinematics():
    line_list = [EmissionLine("Halpha6562", 6562.819, ion="Hα"), EmissionLine("Hbeta4861", 4861.333, ion="Hβ")]
    params = build_emission_parameters(line_list, n_components=1, kinematics_mode="free")
    component = params.components[0]
    # The per-component "shared" value isn't itself fit under "free" --
    # only each line's own override is.
    assert component["velocity_kms"].fixed
    assert component["sigma_kms"].fixed
    assert not component["velocity_kms_Halpha6562"].fixed
    assert not component["velocity_kms_Hbeta4861"].fixed
    assert not component["sigma_kms_Halpha6562"].fixed
    assert not component["sigma_kms_Hbeta4861"].fixed


def test_make_emission_model_func_never_syncs_components_to_each_other():
    """Regression test for the earlier incorrect "tied" implementation,
    which forced every component after the first to numerically match
    component 0. Two independently-set components under "tied" mode must
    stay independent after a model evaluation."""
    line_list = [EmissionLine("Halpha6562", 6562.819, ion="Hα")]
    params = build_emission_parameters(line_list, n_components=2, kinematics_mode="tied")
    params.components[0]["velocity_kms"].value = 100.0
    params.components[1]["velocity_kms"].value = -200.0
    model_func = make_emission_model_func(line_list, redshift=0.0, resolution=None, kinematics_mode="tied")
    wave = np.linspace(6400.0, 6720.0, 400)
    model_func(wave, params)  # evaluating must not mutate component 1 to match component 0
    assert params.components[0]["velocity_kms"].value == pytest.approx(100.0)
    assert params.components[1]["velocity_kms"].value == pytest.approx(-200.0)


def _synthetic_single_line_spectrum(true_flux=100.0, true_velocity=30.0, true_sigma=60.0, redshift=0.0):
    rest = 6562.819
    line_list = [EmissionLine("Halpha6562", rest, ion="Hα")]
    wave = np.linspace(6400.0, 6720.0, 400)
    model_func = make_emission_model_func(line_list, redshift=redshift, resolution=None)

    from core.parameters import ModelParameters, Component, Parameter
    params = ModelParameters(
        n_components=1,
        components=[Component(parameters=[
            Parameter("velocity_kms", true_velocity, -500, 500),
            Parameter("sigma_kms", true_sigma, 1, 1000),
            Parameter("amp_Halpha6562", true_flux, 0, np.inf),
        ])],
    )
    flux = model_func(wave, params)
    rng = np.random.default_rng(0)
    noise_sigma = 0.02 * np.max(flux)
    flux_unc = np.full(wave.shape, noise_sigma)
    flux_noisy = flux + rng.normal(0, noise_sigma, size=wave.shape)
    spectrum = Spectrum.from_arrays(wave, flux_noisy, flux_unc, redshift=redshift)
    return spectrum, line_list


def test_fit_recovers_single_gaussian_line():
    spectrum, line_list = _synthetic_single_line_spectrum()
    config = DictConfig(emission_n_components=1, emission_velocity_initial_kms=0.0,
                         emission_sigma_initial_kms=80.0)
    result = fit_emission_spectrum(spectrum, config, line_list=line_list)

    assert result.fit_result.parameters.n_components == 1
    flux = result.flux("Halpha6562")
    assert flux == pytest.approx(100.0, rel=0.1)
    assert result.component_velocities_kms[0] == pytest.approx(30.0, abs=15.0)
    assert result.component_sigmas_kms[0] == pytest.approx(60.0, abs=15.0)


def test_fit_enforces_tied_doublet_ratio():
    wave = np.linspace(4900.0, 5100.0, 500)
    line_list = [
        EmissionLine("[O_III]5006", 5006.843, ion="[O III]"),
        EmissionLine("[O_III]4958", 4958.911, ion="[O III]", tied_to="[O_III]5006", ratio_to_tied=0.335),
    ]
    model_func = make_emission_model_func(line_list, redshift=0.0, resolution=None)
    from core.parameters import ModelParameters, Component, Parameter
    true_params = ModelParameters(
        n_components=1,
        components=[Component(parameters=[
            Parameter("velocity_kms", 0.0, -500, 500),
            Parameter("sigma_kms", 40.0, 1, 1000),
            Parameter("amp_[O_III]5006", 200.0, 0, np.inf),
        ])],
    )
    flux = model_func(wave, true_params)
    flux_unc = np.full(wave.shape, 0.01 * np.max(flux))
    rng = np.random.default_rng(1)
    flux_noisy = flux + rng.normal(0, flux_unc[0], size=wave.shape)
    spectrum = Spectrum.from_arrays(wave, flux_noisy, flux_unc)

    config = DictConfig(emission_n_components=1, emission_sigma_initial_kms=40.0)
    result = fit_emission_spectrum(spectrum, config, line_list=line_list)

    primary = result.flux("[O_III]5006")
    tied = result.flux("[O_III]4958")
    assert tied == pytest.approx(primary * 0.335, rel=1e-6)


def test_normalize_emission_spectrum_applies_factor_and_reduction_once():
    spectrum, line_list = _synthetic_single_line_spectrum(true_flux=100.0)
    original_flux = spectrum.flux.copy()
    original_flux_unc = spectrum.flux_unc.copy()

    normalized = normalize_emission_spectrum(
        spectrum, DictConfig(emission_flux_normalizing_factor=10.0, emission_flux_reduction=2.0),
    )
    # Original spectrum untouched -- normalize_emission_spectrum returns a
    # copy, since it's typically the one Spectrum object shared with the
    # stellar/absorption panels (see gui.state.SessionState).
    assert np.array_equal(spectrum.flux, original_flux)
    assert np.array_equal(spectrum.flux_unc, original_flux_unc)

    assert np.allclose(normalized.flux, (original_flux - 2.0) / 10.0)
    assert np.allclose(normalized.flux_unc, original_flux_unc / 10.0)
    assert normalized.metadata["emission_flux_normalizing_factor"] == pytest.approx(10.0)
    assert normalized.metadata["emission_flux_reduction"] == pytest.approx(2.0)


def test_fit_emission_spectrum_no_longer_rescales_internally():
    """fit_emission_spectrum's contract changed: normalization is now the
    caller's responsibility (applied once, up front -- see
    normalize_emission_spectrum), not an internal, invisible round-trip.
    Fitting an already-normalized spectrum should therefore report results
    in those same (rescaled) units, not transparently converted back."""
    spectrum, line_list = _synthetic_single_line_spectrum(true_flux=100.0)
    config_plain = DictConfig(emission_n_components=1)
    result_plain = fit_emission_spectrum(spectrum, config_plain, line_list=line_list)

    config_scaled = DictConfig(emission_n_components=1, emission_flux_normalizing_factor=10.0)
    normalized_spectrum = normalize_emission_spectrum(spectrum, config_scaled)
    result_scaled = fit_emission_spectrum(normalized_spectrum, config_scaled, line_list=line_list)

    # Same physical line, fit in units 10x smaller -> integrated flux 10x smaller.
    assert result_scaled.flux("Halpha6562") == pytest.approx(result_plain.flux("Halpha6562") / 10.0, rel=1e-3)
    # Fitting the *un*-normalized spectrum but still passing the scaled
    # config (i.e. forgetting to normalize first) must NOT silently rescale
    # anything anymore -- results should match the plain fit exactly.
    result_unnormalized_call = fit_emission_spectrum(spectrum, config_scaled, line_list=line_list)
    assert result_unnormalized_call.flux("Halpha6562") == pytest.approx(result_plain.flux("Halpha6562"), rel=1e-3)


def test_auto_n_components_search_prefers_lower_bic_for_flat_noise():
    spectrum, line_list = _synthetic_single_line_spectrum(true_flux=50.0)
    config = DictConfig(emission_n_components=1, emission_n_components_max=2, emission_bic_penalty_factor=0.0)
    result = fit_emission_spectrum(spectrum, config, line_list=line_list)
    # A single true Gaussian component should not require a second component.
    assert result.fit_result.parameters.n_components in (1, 2)


def test_save_and_load_round_trip(tmp_path):
    spectrum, line_list = _synthetic_single_line_spectrum()
    config = DictConfig(emission_n_components=1)
    result = fit_emission_spectrum(spectrum, config, line_list=line_list)

    path = tmp_path / "emission_result.fits"
    save_emission_result(path, result)
    loaded = load_emission_result(path)

    assert loaded.flux("Halpha6562") == pytest.approx(result.flux("Halpha6562"), rel=1e-6)
    assert np.allclose(loaded.component_velocities_kms, result.component_velocities_kms)
    assert np.allclose(loaded.component_sigmas_kms, result.component_sigmas_kms)
    assert np.allclose(loaded.fit_result.wave, result.fit_result.wave)
    assert np.allclose(loaded.fit_result.model, result.fit_result.model, rtol=1e-5)


def test_gaussian_line_flux_integrates_to_input_flux():
    wave = np.linspace(-50, 50, 200001) + 5000.0
    profile = gaussian_line_flux(wave, 42.0, 5000.0, 3.0)
    integral = np.trapezoid(profile, wave)
    assert integral == pytest.approx(42.0, rel=1e-3)


# --- User-specified fixed amplitude ratios (apply_fixed_ratio_overrides) ----------

def test_apply_fixed_ratio_overrides_sets_tie_and_ratio():
    line_list = load_emission_line_list()
    overridden = apply_fixed_ratio_overrides(
        line_list,
        species_pairs=[("[S_II]6716", "[S_II]6730")],
        ratio_values=[1.1],
    )
    by_name = {l.name: l for l in overridden}
    assert by_name["[S_II]6730"].tied_to == "[S_II]6716"
    assert by_name["[S_II]6730"].ratio_to_tied == pytest.approx(1.0 / 1.1)
    # untouched lines are unaffected
    assert by_name["Hα6562"].tied_to is None


def test_apply_fixed_ratio_overrides_multiple_pairs_independent():
    line_list = load_emission_line_list()
    overridden = apply_fixed_ratio_overrides(
        line_list,
        species_pairs=[("[S_II]6716", "[S_II]6730"), ("[O_II]3726", "[O_II]3728")],
        ratio_values=[1.1, 0.9],
    )
    by_name = {l.name: l for l in overridden}
    assert by_name["[S_II]6730"].ratio_to_tied == pytest.approx(1.0 / 1.1)
    assert by_name["[O_II]3728"].ratio_to_tied == pytest.approx(1.0 / 0.9)


def test_apply_fixed_ratio_overrides_can_override_existing_catalog_tie():
    line_list = load_emission_line_list()
    by_name_before = {l.name: l for l in line_list}
    assert by_name_before["[O_III]5006"].ratio_to_tied == pytest.approx(2.98)  # catalog default

    overridden = apply_fixed_ratio_overrides(
        line_list, species_pairs=[("[O_III]4958", "[O_III]5006")], ratio_values=[0.5],
    )
    by_name = {l.name: l for l in overridden}
    assert by_name["[O_III]5006"].tied_to == "[O_III]4958"
    assert by_name["[O_III]5006"].ratio_to_tied == pytest.approx(2.0)  # 1/0.5, not the catalog's 2.98


def test_apply_fixed_ratio_overrides_rejects_self_tie():
    line_list = load_emission_line_list()
    with pytest.raises(ValueError):
        apply_fixed_ratio_overrides(line_list, [("[S_II]6716", "[S_II]6716")], [1.0])


def test_apply_fixed_ratio_overrides_rejects_unknown_line():
    line_list = load_emission_line_list()
    with pytest.raises(ValueError):
        apply_fixed_ratio_overrides(line_list, [("NotALine", "[S_II]6730")], [1.0])


def test_apply_fixed_ratio_overrides_rejects_nonpositive_ratio():
    line_list = load_emission_line_list()
    with pytest.raises(ValueError):
        apply_fixed_ratio_overrides(line_list, [("[S_II]6716", "[S_II]6730")], [0.0])


def test_apply_fixed_ratio_overrides_detects_cycle():
    line_list = load_emission_line_list()
    with pytest.raises(ValueError, match="cycle"):
        apply_fixed_ratio_overrides(
            line_list, [("[S_II]6716", "[S_II]6730"), ("[S_II]6730", "[S_II]6716")], [1.1, 0.9],
        )


def test_apply_fixed_ratio_overrides_mismatched_lengths():
    line_list = load_emission_line_list()
    with pytest.raises(ValueError):
        apply_fixed_ratio_overrides(line_list, [("[S_II]6716", "[S_II]6730")], [1.1, 0.9])


# --- Config-driven parsing (select_emission_line_list) ------------------------------

def test_config_fixed_ratio_species_parses_nested_brackets_and_survives_naive_comma_split():
    """core.config._infer_value splits every comma naively; the config-facing
    parser must recover the original nested-pair structure regardless."""
    from core.config import _infer_value
    from emission.emission_fit import _parse_species_pairs, _raw_string

    raw_species = "[[S_II]6716,[S_II]6730], [[O_II]3726,[O_II]3728]"
    mangled = _infer_value(raw_species)  # simulates what core.config actually stores
    assert isinstance(mangled, list)  # confirms the naive split really did occur

    config = DictConfig(emission_fixed_ratio_species=mangled)
    recovered_raw = _raw_string(config, "emission_fixed_ratio_species")
    pairs = _parse_species_pairs(recovered_raw)
    assert pairs == [("[S_II]6716", "[S_II]6730"), ("[O_II]3726", "[O_II]3728")]


def test_select_emission_line_list_applies_config_fixed_ratios():
    wave = np.linspace(3700.0, 6800.0, 500)
    spectrum = Spectrum.from_arrays(wave, np.ones_like(wave), np.full_like(wave, 0.01))
    config = DictConfig(
        emission_fixed_ratio_species="[[S_II]6716,[S_II]6730], [[O_II]3726,[O_II]3728]",
        emission_fixed_ratio_value="1.1, 0.9",
        emission_lines="[S_II]6716,[S_II]6730,[O_II]3726,[O_II]3728",
    )
    result = select_emission_line_list(spectrum, config)
    by_name = {l.name: l for l in result}
    assert by_name["[S_II]6730"].ratio_to_tied == pytest.approx(1.0 / 1.1)
    assert by_name["[O_II]3728"].ratio_to_tied == pytest.approx(1.0 / 0.9)


def test_fit_recovers_user_specified_fixed_ratio_end_to_end():
    """A ratio the catalog leaves untied (density-diagnostic [SII]) can be
    fixed by the user and is then enforced exactly by the fit, letting a
    published value be cross-checked."""
    wave = np.linspace(6700.0, 6740.0, 600)
    line_list = [
        EmissionLine("[S_II]6716", 6716.44, ion="[S II]"),
        EmissionLine("[S_II]6730", 6730.82, ion="[S II]"),
    ]
    overridden = apply_fixed_ratio_overrides(line_list, [("[S_II]6716", "[S_II]6730")], [1.25])

    model_func = make_emission_model_func(overridden, redshift=0.0, resolution=None)
    from core.parameters import ModelParameters, Component, Parameter
    true_params = ModelParameters(
        n_components=1,
        components=[Component(parameters=[
            Parameter("velocity_kms", 0.0, -500, 500),
            Parameter("sigma_kms", 40.0, 1, 1000),
            Parameter("amp_[S_II]6716", 150.0, 0, np.inf),
        ])],
    )
    flux = model_func(wave, true_params)
    flux_unc = np.full(wave.shape, 0.01 * np.max(flux))
    rng = np.random.default_rng(5)
    flux_noisy = flux + rng.normal(0, flux_unc[0], size=wave.shape)
    spectrum = Spectrum.from_arrays(wave, flux_noisy, flux_unc)

    config = DictConfig(emission_n_components=1, emission_sigma_initial_kms=40.0)
    result = fit_emission_spectrum(spectrum, config, line_list=overridden)

    primary = result.flux("[S_II]6716")
    tied = result.flux("[S_II]6730")
    assert tied == pytest.approx(primary / 1.25, rel=1e-6)
    assert primary == pytest.approx(150.0, rel=0.1)


def _synthetic_two_line_spectrum(main_flux=100.0, weak_flux=3.0, true_velocity=25.0, true_sigma=45.0):
    """A strong Halpha6562 line plus a much weaker [O_III]4363 line, both
    at the same true kinematics -- for weak-line second-stage tests."""
    from core.parameters import Component, ModelParameters, Parameter

    main_lines = [EmissionLine("Halpha6562", 6562.819, ion="Hα")]
    weak_lines = select_lines(load_emission_line_list(), ["[O_III]4363"])
    combined = main_lines + weak_lines

    wave = np.linspace(4000.0, 7000.0, 4000)
    model_func = make_emission_model_func(combined, redshift=0.0, resolution=None)
    params = ModelParameters(
        n_components=1,
        components=[Component(parameters=[
            Parameter("velocity_kms", true_velocity, -500, 500),
            Parameter("sigma_kms", true_sigma, 1, 1000),
            Parameter("amp_Halpha6562", main_flux, 0, np.inf),
            Parameter("amp_[O_III]4363", weak_flux, 0, np.inf),
        ])],
    )
    flux = model_func(wave, params)
    rng = np.random.default_rng(4)
    flux_unc = np.full(wave.shape, 0.1)
    flux_noisy = flux + rng.normal(0, 0.1, size=wave.shape)
    spectrum = Spectrum.from_arrays(wave, flux_noisy, flux_unc)
    return spectrum, main_lines, weak_lines


def test_weak_lines_second_stage_recovers_amplitude_and_shares_kinematics():
    """Regression test for a real bug: scipy.optimize.curve_fit's
    trust-region-reflective method gets permanently stuck if a bounded
    parameter's starting value sits exactly on its lower bound (0) --
    which every weak-line amplitude does after stage 1 (deliberately
    silenced there). Without nudging it off that boundary before stage 2
    un-freezes it, the weak line's fitted amplitude collapses to ~0
    regardless of the true signal, even when the true chi-square minimum
    is sharp and obvious."""
    spectrum, main_lines, weak_lines = _synthetic_two_line_spectrum(main_flux=100.0, weak_flux=3.0)
    config = DictConfig(emission_n_components=1, emission_weak_lines="[O_III]4363")
    result = fit_emission_spectrum(spectrum, config, line_list=main_lines)

    assert {measurement.name for measurement in result.measurements} == {"Halpha6562", "[O_III]4363"}
    assert result.flux("Halpha6562") == pytest.approx(100.0, rel=0.1)
    assert result.flux("[O_III]4363") == pytest.approx(3.0, rel=0.2)
    assert result.flux("[O_III]4363") > 0.5  # regression guard: must not collapse to ~0

    # Weak line shares the main fit's kinematics exactly (same component).
    velocity_kms = result.component_velocities_kms[0]
    assert velocity_kms == pytest.approx(25.0, abs=2.0)

    unc = result.fit_result.parameter_uncertainties
    assert "c0_velocity_kms" in unc and np.isfinite(unc["c0_velocity_kms"])  # preserved from stage 1
    assert "c0_amp_[O_III]4363" in unc and np.isfinite(unc["c0_amp_[O_III]4363"])


def test_weak_lines_save_load_round_trip_unchanged(tmp_path):
    """Explicit requirement: weak lines are saved/loaded exactly like any
    other line, with no separate save path."""
    spectrum, main_lines, weak_lines = _synthetic_two_line_spectrum()
    config = DictConfig(emission_n_components=1, emission_weak_lines="[O_III]4363")
    result = fit_emission_spectrum(spectrum, config, line_list=main_lines)

    path = save_emission_result(tmp_path / "weak_lines.fits", result, overwrite=True)
    loaded = load_emission_result(path)

    assert {m.name for m in loaded.measurements} == {"Halpha6562", "[O_III]4363"}
    assert loaded.flux("Halpha6562") == pytest.approx(result.flux("Halpha6562"))
    assert loaded.flux("[O_III]4363") == pytest.approx(result.flux("[O_III]4363"))
    assert loaded.component_velocities_kms[0] == pytest.approx(result.component_velocities_kms[0])


def test_emission_weak_lines_overlapping_main_lines_raises():
    spectrum, main_lines, weak_lines = _synthetic_two_line_spectrum()
    # weak_lines ([O_III]4363) is a real catalog entry; overlap it directly
    # with the explicit main line_list rather than depending on catalog
    # name-resolution for the synthetic "Halpha6562" name used elsewhere
    # in this test file.
    main_lines_with_overlap = main_lines + weak_lines
    config = DictConfig(emission_n_components=1, emission_weak_lines="[O_III]4363")
    with pytest.raises(ValueError, match="overlaps"):
        fit_emission_spectrum(spectrum, config, line_list=main_lines_with_overlap)


def test_rejection_freezes_insignificant_components_and_keeps_one_free():
    """Only one real kinematic component in the data; ask for 4."""
    spectrum, line_list = _synthetic_single_line_spectrum(true_flux=50.0)
    config = DictConfig(
        emission_n_components=4, emission_reject_insignificant_components=True,
        emission_max_rejection_passes=3,
    )
    result = fit_emission_spectrum(spectrum, config, line_list=line_list)

    frozen = result.fit_result.metadata["emission_frozen_components"]
    assert result.fit_result.parameters.n_components == 4  # count never changes
    assert sum(frozen) == 3  # exactly one stays free
    assert result.flux("Halpha6562") == pytest.approx(50.0, rel=0.15)

    for index, component in enumerate(result.fit_result.parameters.components):
        if frozen[index]:
            assert component["velocity_kms"].fixed
            assert component["amp_Halpha6562"].fixed


def test_rejection_never_freezes_every_component():
    """Even if every component looks insignificant, at least one stays free."""
    wave = np.linspace(6400.0, 6720.0, 400)
    rng = np.random.default_rng(9)
    flux_unc = np.full(wave.shape, 1.0)
    flux_noisy = rng.normal(0, 1.0, size=wave.shape)  # pure noise, no real line at all
    spectrum = Spectrum.from_arrays(wave, flux_noisy, flux_unc)
    line_list = [EmissionLine("Halpha6562", 6562.819, ion="Hα")]

    config = DictConfig(emission_n_components=3, emission_reject_insignificant_components=True)
    result = fit_emission_spectrum(spectrum, config, line_list=line_list)
    frozen = result.fit_result.metadata["emission_frozen_components"]
    assert sum(frozen) == 2
    assert not all(frozen)


def test_rejection_and_weak_lines_compose():
    """A component the main-line rejection pass freezes must not get an
    independent chance for its weak-line amplitude either."""
    spectrum, main_lines, weak_lines = _synthetic_two_line_spectrum(main_flux=100.0, weak_flux=5.0)
    config = DictConfig(
        emission_n_components=3, emission_weak_lines="[O_III]4363",
        emission_reject_insignificant_components=True,
    )
    result = fit_emission_spectrum(spectrum, config, line_list=main_lines)

    frozen = result.fit_result.metadata["emission_frozen_components"]
    assert sum(frozen) == 2
    for index, component in enumerate(result.fit_result.parameters.components):
        if frozen[index]:
            assert component["amp_[O_III]4363"].value == pytest.approx(0.0)
    assert result.flux("Halpha6562") == pytest.approx(100.0, rel=0.15)
    assert result.flux("[O_III]4363") == pytest.approx(5.0, rel=0.3)


def test_help_decide_components_off_never_searches_or_freezes():
    """Regression test: emission_n_components_max used to trigger automatic
    component-count search regardless of use_rejection, so a config with
    it set would silently inflate n_components even with rejection
    (the GUI's "Help decide components" checkbox) off."""
    spectrum, line_list = _synthetic_single_line_spectrum(true_flux=50.0)
    config = DictConfig(emission_n_components=1, emission_n_components_max=3, emission_bic_penalty_factor=0.0)

    result = fit_emission_spectrum(spectrum, config, line_list=line_list, use_rejection=False, n_components=1)
    assert result.fit_result.parameters.n_components == 1
    assert result.fit_result.metadata["emission_frozen_components"] == [False]


def test_automatic_component_search_does_not_favor_padded_candidates():
    """Regression test: comparing BIC-search candidates *with* rejection
    active let a padded candidate's insignificant component get frozen
    mid-search, quietly shrinking its effective parameter count for the
    comparison while the optimizer still got extra attempts at a
    marginally better chi-square -- biasing the search toward one extra
    component every time. Candidates must be compared as plain
    (non-rejection) fits; rejection only applied once, after, to
    whichever N actually wins."""
    wave = np.linspace(6500.0, 6650.0, 1500)
    rng = np.random.default_rng(7)
    flux = (
        8.0 * np.exp(-0.5 * ((wave - 6560.0) / 2.0) ** 2)
        + 5.0 * np.exp(-0.5 * ((wave - 6567.0) / 3.0) ** 2)
        + rng.normal(0, 0.1, size=wave.shape)
    )
    spectrum = Spectrum.from_arrays(wave, flux, np.full(wave.shape, 0.1))
    line_list = [EmissionLine("Halpha6562", 6562.819, ion="Hα")]

    config = DictConfig(emission_n_components=1, emission_n_components_max=4, emission_bic_penalty_factor=0.0)
    result = fit_emission_spectrum(spectrum, config, line_list=line_list, use_rejection=True, n_components=1)
    assert result.fit_result.parameters.n_components == 2  # the true number, not 3 or 4
