"""Legacy-parity plotting for unified FitSpec stellar results.

The deterministic plotting layouts in this module are direct adaptations of
``stellar_plotting_function_v8.py`` from the original standalone stellar code.
Only the data-access layer has been changed to consume FitSpec's unified
``StellarFitResult`` object.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from stellar.stellar_results import load_stellar_result

__all__ = [
    "plot_stellar_fit",
    "plot_stellar_diagnostics",
    "plot_stellar_observational_summary",
    "write_stellar_diagnostics_text",
    "save_stellar_plot_products",
]


def _load(x):
    return load_stellar_result(x) if isinstance(x, (str, bytes, Path)) else x


def _positive_coefficients(r):
    c = np.asarray(r.coefficients, float)
    return np.where(np.isfinite(c) & (c > 0.0), c, 0.0)


def _weighted_population_summary(r):
    coeff = _positive_coefficients(r)
    total = float(np.sum(coeff))
    frac = coeff / total if total > 0 else np.zeros_like(coeff)
    positive = coeff > 0
    if np.any(positive):
        age = float(np.average(np.asarray(r.ages_myr, float)[positive], weights=coeff[positive]))
        z = float(np.average(np.asarray(r.metallicities_solar, float)[positive], weights=coeff[positive]))
    else:
        age = np.nan
        z = np.nan
    return coeff, total, frac, age, z


def plot_stellar_fit(result_or_path, output_path=None):
    """Reproduce the original saved-fit figure for a FitSpec result."""
    r = _load(result_or_path)
    wave = np.asarray(r.wave, float)
    flux = np.asarray(r.flux, float)
    unc = np.asarray(r.flux_unc, float) if r.flux_unc is not None else np.full_like(flux, np.nan)
    model = np.asarray(r.model, float)
    gas_model = np.asarray(r.gas_model, float)
    fit_mask = np.asarray(r.mask, bool)

    valid = np.isfinite(wave) & np.isfinite(flux) & np.isfinite(model)
    if r.flux_unc is not None:
        valid &= np.isfinite(unc) & (unc > 0)
    used = valid & fit_mask
    masked = valid & ~fit_mask

    residual_sigma = np.full(wave.size, np.nan, dtype=float)
    if r.flux_unc is not None:
        residual_sigma[valid] = (flux[valid] - model[valid]) / unc[valid]
    else:
        residual_sigma[valid] = flux[valid] - model[valid]

    coeff, total_coeff, coeff_fraction, average_age, average_z = _weighted_population_summary(r)
    ages = np.asarray(r.ages_myr, float)
    metallicity_codes = np.asarray(r.metallicity_codes).astype(str)
    metallicities_zsun = np.asarray(r.metallicities_solar, float)
    dominant_index = int(r.dominant_index)

    plot_figure = plt.figure(figsize=(9, 7))
    grid = plot_figure.add_gridspec(
        3, 2,
        height_ratios=[5.0, 2.0, 5.0],
        width_ratios=[1.35, 1.0],
        hspace=0.2,
        wspace=0.2,
    )
    spectrum_axis = plot_figure.add_subplot(grid[0, :])
    residual_axis = plot_figure.add_subplot(grid[1, :], sharex=spectrum_axis)
    residual_position = residual_axis.get_position()
    residual_axis.set_position([
        residual_position.x0,
        residual_position.y0 + 0.03,
        residual_position.width,
        residual_position.height,
    ])
    population_axis = plot_figure.add_subplot(grid[2, 0])
    table_axis = plot_figure.add_subplot(grid[2, 1])

    if r.flux_unc is not None:
        spectrum_axis.errorbar(
            wave[masked], flux[masked], yerr=unc[masked], fmt='.',
            markersize=3, linewidth=0.5, alpha=0.2,
            label='Masked', zorder=1,
        )
        spectrum_axis.errorbar(
            wave[used], flux[used], yerr=unc[used], fmt='.',
            markersize=3, linewidth=0.5, alpha=0.8,
            label='Used in fit', zorder=2,
        )
    else:
        spectrum_axis.plot(wave[masked], flux[masked], '.', ms=3, alpha=0.2, label='Masked')
        spectrum_axis.plot(wave[used], flux[used], '.', ms=3, alpha=0.8, label='Used in fit')
    spectrum_axis.plot(wave, model, color='red', linewidth=2.0, label='Best-fit composite', zorder=3)
    if np.any(np.isfinite(gas_model) & (np.abs(gas_model) > 0)):
        spectrum_axis.plot(wave, gas_model, linewidth=1.0, linestyle=':', label='Gas emission')
    spectrum_axis.set_ylabel('Flux')
    cluster = str((getattr(r, 'metadata', {}) or {}).get('cluster', (getattr(r, 'metadata', {}) or {}).get('cluster_id', 'Stellar population fit')))
    spectrum_axis.set_title(cluster)
    spectrum_axis.legend(loc='best')
    spectrum_axis.tick_params(labelbottom=False)

    residual_axis.scatter(wave[masked], residual_sigma[masked], s=4, alpha=0.2)
    residual_axis.scatter(wave[used], residual_sigma[used], s=4, alpha=0.8)
    residual_axis.axhline(0.0, linewidth=1.0)
    if r.flux_unc is not None:
        residual_axis.axhline(3.0, linestyle='--', linewidth=1.0)
        residual_axis.axhline(-3.0, linestyle='--', linewidth=1.0)
        residual_axis.set_ylim(-3.5, 3.5)
        residual_axis.set_ylabel('Residual\n($\\sigma$)')
    else:
        residual_axis.set_ylabel('Residual')
    residual_axis.set_xlabel('Wavelength')

    if str(r.regime).lower() == 'uv':
        unique_ages = np.unique(ages)
        unique_metallicities = np.unique(metallicity_codes)
        bottom = np.zeros(unique_ages.size, dtype=float)
        for metallicity in unique_metallicities:
            contribution = np.zeros(unique_ages.size, dtype=float)
            for index, age in enumerate(unique_ages):
                selection = np.isclose(ages, age) & (metallicity_codes == metallicity)
                contribution[index] = np.sum(coeff_fraction[selection]) * 100.0
            zvals = metallicities_zsun[metallicity_codes == metallicity]
            zmed = float(np.nanmedian(zvals)) if zvals.size else np.nan
            population_axis.bar(unique_ages, contribution, bottom=bottom, label=f'{zmed:.2f} $Z_{{\\odot}}$')
            bottom += contribution
        population_axis.set_xlabel('Stellar age (Myr)')
        population_axis.set_ylabel('Mass contribution (%)')
        population_axis.set_title('Fitted SSP contributions')
        handles, labels = population_axis.get_legend_handles_labels()
        if handles:
            population_axis.legend(handles, labels, title='Metallicity', ncol=2)
    else:
        positive = coeff > 0
        if np.any(positive):
            sizes = 40.0 + 1200.0 * coeff_fraction[positive]
            sc = population_axis.scatter(
                np.log10(np.clip(ages[positive], 1e-30, None)),
                metallicities_zsun[positive],
                s=sizes,
                c=coeff_fraction[positive],
            )
            plot_figure.colorbar(sc, ax=population_axis, label='formed-mass-scale fraction')
        population_axis.set_xlabel('log10 age [Myr]')
        population_axis.set_ylabel(r'$Z/Z_\odot$')
        population_axis.set_title('Optical SSP solution')

    table_axis.axis('off')
    summary_rows = [
        ['Mass-weighted age', f'{average_age:.3g} Myr' if np.isfinite(average_age) else '—'],
        ['Mass-weighted metallicity', f'{average_z:.2f} $Z_{{\\odot}}$' if np.isfinite(average_z) else '—'],
        ['Total stellar mass', f'{total_coeff:.4g} M$_\\odot$'],
        ['Dominant SSP mass', f'{coeff[dominant_index]:.4g} M$_\\odot$'],
        ['Dominant age', f'{ages[dominant_index]:.4g} Myr'],
        ['Dominant metallicity', f'{metallicities_zsun[dominant_index]:.2f} $Z_{{\\odot}}$'],
        ['E(B-V)', f'{r.ebv:.4g}'],
        ['Velocity', f'{r.velocity_kms:.4g} km s$^{{-1}}$'],
        ['Velocity dispersion', f'{r.sigma_kms:.4g} km s$^{{-1}}$'],
        ['Reduced $\\chi^2$', f'{r.reduced_chi_square:.4g}'],
        ['Degrees of freedom', f'{r.degrees_of_freedom:.0f}'],
    ]
    result_table = table_axis.table(
        cellText=summary_rows,
        colLabels=['Parameter', 'Value'],
        loc='center',
        cellLoc='left',
        colLoc='left',
    )
    result_table.auto_set_font_size(False)
    result_table.set_fontsize(8)
    plot_figure.align_ylabels([spectrum_axis, residual_axis, population_axis])
    try:
        plot_figure.canvas.manager.set_window_title('Saved Stellar Fit')
    except Exception:
        pass
    if output_path is not None:
        plot_figure.savefig(output_path, dpi=150)
        print(f'{output_path} saved')
    return plot_figure


def _diagnostic_basis_indices(r, n_basis):
    """Map stored diagnostic rows to full FitSpec population indices."""
    n_full = np.asarray(r.coefficients).size
    if n_basis == n_full:
        return np.arange(n_basis, dtype=int)
    raw = (getattr(r, 'metadata', {}) or {}).get('candidate_basis_indices')
    if raw is not None:
        try:
            idx = np.asarray([int(x) for x in str(raw).split(',') if str(x).strip()], dtype=int)
            if idx.size == n_basis and np.all((idx >= 0) & (idx < n_full)):
                return idx
        except Exception:
            pass
    positive = np.flatnonzero(np.asarray(r.coefficients, float) > 0)
    if positive.size == n_basis:
        return positive
    return np.arange(n_basis, dtype=int)


def _diagnostic_vector_on_basis(values, basis_idx, n_basis, n_full):
    arr = np.asarray(values, float)
    if arr.size == n_basis:
        return arr
    if arr.size == n_full:
        return arr[basis_idx]
    raise ValueError(f'Diagnostic vector has length {arr.size}; expected {n_basis} or {n_full}.')


def plot_stellar_diagnostics(result_or_path, output_path=None, text_output_path=None):
    """Reproduce the original six-panel SSP-degeneracy diagnostics."""
    r = _load(result_or_path)
    d = r.diagnostics
    if d is None or d.correlation_matrix is None or d.transformed_model_fluxes is None:
        raise ValueError('This result does not contain SSP diagnostics.')

    wave = np.asarray(r.wave, float)
    fit_mask = np.asarray(r.mask, bool)
    coeff_full = _positive_coefficients(r)
    ages_full = np.asarray(r.ages_myr, float)
    codes_full = np.asarray(r.metallicity_codes).astype(str)
    z_full = np.asarray(r.metallicities_solar, float)
    n_full = coeff_full.size

    correlation_matrix = np.asarray(d.correlation_matrix, float)
    n_models = correlation_matrix.shape[0]
    if correlation_matrix.shape != (n_models, n_models):
        raise ValueError('Stored SSP correlation matrix is not square.')
    basis_idx = _diagnostic_basis_indices(r, n_models)
    if basis_idx.size != n_models:
        raise ValueError('Could not map SSP diagnostics to the saved population table.')

    ages = ages_full[basis_idx]
    metallicities = codes_full[basis_idx]
    metallicities_z_sun = z_full[basis_idx]
    masses = coeff_full[basis_idx]
    total_mass = np.sum(masses)
    mass_fraction = masses / total_mass if total_mass > 0 else np.zeros_like(masses)
    model_fluxes = np.asarray(d.transformed_model_fluxes, float)
    if model_fluxes.shape != (n_models, wave.size):
        raise ValueError('Stored transformed SSP models do not match diagnostics/wavelength dimensions.')

    single_ssp_chi_square = _diagnostic_vector_on_basis(d.single_ssp_chi_square, basis_idx, n_models, n_full)
    single_ssp_delta_chi_square = _diagnostic_vector_on_basis(d.single_ssp_delta_chi_square, basis_idx, n_models, n_full)
    dominant_ssp_distance = _diagnostic_vector_on_basis(d.dominant_ssp_distance, basis_idx, n_models, n_full)
    singular_values = np.asarray(d.singular_values if d.singular_values is not None else [], float)
    effective_rank = int(d.effective_rank) if d.effective_rank is not None else 0
    condition_number = float(d.condition_number) if d.condition_number is not None else np.nan

    dominant_full = int(r.dominant_index)
    hit = np.flatnonzero(basis_idx == dominant_full)
    dominant_index = int(hit[0]) if hit.size else int(np.nanargmax(masses))

    unique_ages = np.unique(ages)
    unique_metallicities = np.unique(metallicities)
    unique_metallicities_z_sun = np.asarray([
        np.nanmedian(metallicities_z_sun[metallicities == metallicity])
        for metallicity in unique_metallicities
    ], dtype=float)
    dominant_correlations = correlation_matrix[dominant_index].copy()
    alternative_indices = np.arange(n_models)
    alternative_indices = alternative_indices[alternative_indices != dominant_index]
    alternative_indices = alternative_indices[np.argsort(dominant_correlations[alternative_indices])[::-1]]
    top_degenerate_indices = alternative_indices[:5]

    plot_figure2 = plt.figure(figsize=(21, 13.5))
    grid = plot_figure2.add_gridspec(2, 3, hspace=0.52, wspace=0.48)
    correlation_axis = plot_figure2.add_subplot(grid[0, 0])
    chi_square_axis = plot_figure2.add_subplot(grid[0, 1])
    mass_axis = plot_figure2.add_subplot(grid[0, 2])
    dominant_correlation_axis = plot_figure2.add_subplot(grid[1, 0])
    spectral_axis = plot_figure2.add_subplot(grid[1, 1])
    summary_axis = plot_figure2.add_subplot(grid[1, 2])

    correlation_image = correlation_axis.imshow(
        correlation_matrix, origin='lower', aspect='auto', interpolation='nearest',
        vmin=-1.0, vmax=1.0, cmap='coolwarm',
    )
    n_corr_ticks = min(10, n_models)
    corr_tick_indices = np.unique(np.linspace(0, n_models - 1, n_corr_ticks, dtype=int))
    corr_tick_labels = [
        f'{ages[i]:g} Myr\nZ={metallicities_z_sun[i]:.2f} Zsun'
        for i in corr_tick_indices
    ]
    correlation_axis.set_xticks(corr_tick_indices)
    correlation_axis.set_xticklabels(corr_tick_labels, rotation=45, ha='right', fontsize=7)
    correlation_axis.set_yticks(corr_tick_indices)
    correlation_axis.set_yticklabels(corr_tick_labels, fontsize=7)
    correlation_axis.set_xlabel('SSP age / metallicity')
    correlation_axis.set_ylabel('SSP age / metallicity')
    correlation_axis.set_title('1. Full SSP correlation matrix')
    correlation_axis.axvline(dominant_index, linestyle='--', linewidth=1.2, alpha=0.9, zorder=4)
    correlation_axis.axhline(dominant_index, linestyle='--', linewidth=1.2, alpha=0.9, zorder=4)
    correlation_axis.scatter(
        dominant_index, dominant_index, marker='*', s=420,
        facecolor='white', edgecolor='black', linewidth=1.8,
        zorder=10, clip_on=False, label='Dominant SSP',
    )
    correlation_axis.annotate(
        f'Dominant: {ages[dominant_index]:g} Myr, Z={metallicities_z_sun[dominant_index]:.2f} Zsun',
        xy=(dominant_index, dominant_index), xytext=(8, 10), textcoords='offset points',
        fontsize=7, bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.85), zorder=11,
    )
    correlation_axis.legend(loc='lower right', fontsize=8)
    correlation_colorbar = plot_figure2.colorbar(correlation_image, ax=correlation_axis, fraction=0.046, pad=0.04)
    correlation_colorbar.set_label('Pearson correlation')
    for metallicity in unique_metallicities:
        metallicity_indices = np.where(metallicities == metallicity)[0]
        if metallicity_indices.size:
            last_index = metallicity_indices.max()
            if last_index < n_models - 1:
                boundary = last_index + 0.5
                correlation_axis.axvline(boundary, linewidth=0.6, alpha=0.5)
                correlation_axis.axhline(boundary, linewidth=0.6, alpha=0.5)

    chi_square_grid = np.full((unique_ages.size, unique_metallicities.size), np.nan, dtype=float)
    for age_index, age in enumerate(unique_ages):
        for metallicity_index, metallicity in enumerate(unique_metallicities):
            selection = np.isclose(ages, age) & (metallicities == metallicity)
            matching_indices = np.where(selection)[0]
            if matching_indices.size:
                matching_values = single_ssp_delta_chi_square[matching_indices]
                if np.any(np.isfinite(matching_values)):
                    chi_square_grid[age_index, metallicity_index] = np.nanmin(matching_values)
    plotted_chi_square_grid = np.log10(chi_square_grid + 1.0)
    chi_square_image = chi_square_axis.imshow(
        plotted_chi_square_grid, origin='lower', aspect='auto', interpolation='nearest', cmap='viridis'
    )
    chi_square_axis.set_xticks(np.arange(unique_metallicities.size))
    chi_square_axis.set_xticklabels([f'{m:.2f}' for m in unique_metallicities_z_sun], rotation=45, ha='right')
    chi_square_axis.set_yticks(np.arange(unique_ages.size))
    chi_square_axis.set_yticklabels([f'{age:g}' for age in unique_ages])
    chi_square_axis.set_xlabel('Metallicity ($Z/Z_{\\odot}$)')
    chi_square_axis.set_ylabel('Age (Myr)')
    chi_square_axis.set_title('2. Single-SSP $\\chi^2$ landscape')
    if np.any(np.isfinite(single_ssp_chi_square)):
        best_single_index = int(np.nanargmin(single_ssp_chi_square))
        best_age_index = int(np.argmin(np.abs(unique_ages - ages[best_single_index])))
        matching_metallicity = np.where(unique_metallicities == metallicities[best_single_index])[0]
        if matching_metallicity.size:
            chi_square_axis.scatter(int(matching_metallicity[0]), best_age_index, marker='*', s=190, edgecolor='black', linewidth=0.8)
    chi_square_colorbar = plot_figure2.colorbar(chi_square_image, ax=chi_square_axis, fraction=0.046, pad=0.04)
    chi_square_colorbar.set_label('$\\log_{10}(\\Delta\\chi^2 + 1)$')

    model_positions = np.arange(n_models)
    mass_axis.bar(model_positions, mass_fraction * 100.0)
    mass_axis.scatter(dominant_index, mass_fraction[dominant_index] * 100.0, marker='*', s=180, edgecolor='black', linewidth=0.8, zorder=5)
    n_mass_ticks = min(10, n_models)
    mass_tick_positions = np.unique(np.linspace(0, n_models - 1, n_mass_ticks, dtype=int))
    mass_axis.set_xticks(mass_tick_positions)
    mass_axis.set_xticklabels([
        f'{ages[i]:g} Myr\nZ={metallicities_z_sun[i]:.2f} Zsun' for i in mass_tick_positions
    ], rotation=45, ha='right', fontsize=7)
    mass_axis.set_xlabel('SSP age / metallicity')
    mass_axis.set_ylabel('Mass contribution (%)')
    mass_axis.set_title('3. Fitted SSP mass distribution')
    mass_axis.set_xlim(-0.75, n_models - 0.25)

    dominant_correlation_axis.plot(model_positions, dominant_correlations, marker='o', markersize=4, linewidth=1.2)
    dominant_correlation_axis.axhline(0.95, linestyle='--', linewidth=1.0, label='$r = 0.95$')
    dominant_correlation_axis.scatter(dominant_index, dominant_correlations[dominant_index], marker='*', s=190, edgecolor='black', linewidth=0.8, zorder=5, label='Dominant SSP')
    for model_index in top_degenerate_indices:
        dominant_correlation_axis.annotate(
            f'{ages[model_index]:g} Myr\nZ={metallicities_z_sun[model_index]:.2f} Zsun',
            (model_index, dominant_correlations[model_index]), xytext=(0, 6),
            textcoords='offset points', ha='center', fontsize=7,
        )
    dominant_correlation_axis.set_xticks(mass_tick_positions)
    dominant_correlation_axis.set_xticklabels([
        f'{ages[i]:g} Myr\nZ={metallicities_z_sun[i]:.2f} Zsun' for i in mass_tick_positions
    ], rotation=45, ha='right', fontsize=7)
    dominant_correlation_axis.set_xlabel('SSP age / metallicity')
    dominant_correlation_axis.set_ylabel('Correlation with dominant SSP')
    finite_corr = dominant_correlations[np.isfinite(dominant_correlations)]
    ymin = min(-1.0, float(np.nanmin(finite_corr)) - 0.05) if finite_corr.size else -1.0
    dominant_correlation_axis.set_ylim(ymin, 1.02)
    dominant_correlation_axis.set_title('4. Degeneracy with dominant SSP')
    dominant_correlation_axis.legend(loc='best', fontsize=8)

    comparison_valid = fit_mask & np.isfinite(wave) & np.all(np.isfinite(model_fluxes), axis=0)
    dominant_model = model_fluxes[dominant_index].copy()
    dominant_reference = np.nanmedian(np.abs(dominant_model[comparison_valid])) if np.any(comparison_valid) else np.nan
    if not np.isfinite(dominant_reference) or dominant_reference <= 0.0:
        dominant_reference = 1.0
    dominant_normalized = dominant_model / dominant_reference
    spectral_axis.plot(
        wave[comparison_valid], dominant_normalized[comparison_valid], linewidth=2.2,
        label=f'Dominant: {ages[dominant_index]:g} Myr, Z={metallicities_z_sun[dominant_index]:.2f} Zsun, mass={masses[dominant_index]:.3g} Msun',
    )
    for model_index in top_degenerate_indices:
        comparison_model = model_fluxes[model_index].copy()
        denominator = np.sum(comparison_model[comparison_valid] ** 2)
        if denominator <= 0.0:
            continue
        scale_to_dominant = np.sum(dominant_model[comparison_valid] * comparison_model[comparison_valid]) / denominator
        comparison_normalized = scale_to_dominant * comparison_model / dominant_reference
        spectral_axis.plot(
            wave[comparison_valid], comparison_normalized[comparison_valid], linewidth=1.0, alpha=0.75,
            label=f'{ages[model_index]:g} Myr, Z={metallicities_z_sun[model_index]:.2f} Zsun, mass={masses[model_index]:.3g} Msun, r={dominant_correlations[model_index]:.3f}',
        )
    spectral_axis.set_xlabel('Wavelength')
    spectral_axis.set_ylabel('Normalized model flux')
    spectral_axis.set_title('5. Most degenerate transformed SSP spectra')
    spectral_axis.legend(loc='best', fontsize=7)

    summary_axis.axis('off')
    active_ssps = int(np.count_nonzero(masses > 0.0))
    high_correlation_indices = np.where((dominant_correlations >= 0.95) & (model_positions != dominant_index))[0]
    summary_lines = [
        'SSP degeneracy summary', '', 'Dominant SSP', '------------',
        f'Age:          {ages[dominant_index]:g} Myr',
        f'Metallicity:  {metallicities_z_sun[dominant_index]:.2f} Z_sun',
        f'Mass:         {masses[dominant_index]:.5g} M_sun', '',
        'Library diagnostics', '-------------------',
        f'SSP models:        {n_models}',
        f'Active SSPs:        {active_ssps}',
        f'Effective rank:     {effective_rank}',
        f'Condition number:   {condition_number:.4g}',
        f'Used pixels:        {int(np.count_nonzero(fit_mask))}',
        f'SSPs with r>=0.95:  {high_correlation_indices.size}',
        f'Reduced chi-square: {r.reduced_chi_square:.4g}', '',
        'Top degeneracies', '----------------',
    ]
    for model_index in top_degenerate_indices:
        summary_lines.append(
            f'{ages[model_index]:g} Myr, Z={metallicities_z_sun[model_index]:.2f} Z_sun, '
            f'mass={masses[model_index]:.5g} M_sun, r={dominant_correlations[model_index]:.4f}, '
            f'D={dominant_ssp_distance[model_index]:.4f}'
        )
    if text_output_path is not None:
        np.savetxt(text_output_path, np.asarray(summary_lines, dtype=str), fmt='%s')
        print(f'{text_output_path} saved')
    summary_axis.text(0.02, 0.98, '\n'.join(summary_lines), transform=summary_axis.transAxes, va='top', ha='left', fontsize=10, family='monospace')
    summary_axis.set_title('6. Numerical summary')
    cluster_name = str((getattr(r, 'metadata', {}) or {}).get('cluster', (getattr(r, 'metadata', {}) or {}).get('cluster_id', 'Stellar population fit')))
    plot_figure2.suptitle(f'{cluster_name}: SSP degeneracy diagnostics', fontsize=15, y=0.99)
    try:
        plot_figure2.canvas.manager.set_window_title('Saved Stellar Fit Diagnostics')
    except Exception:
        pass
    if output_path is not None:
        plot_figure2.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f'{output_path} saved')
    return plot_figure2


OPTICAL_STELLAR_FEATURE_WINDOWS = (
    ('Mg', 5150.0, 5195.0),
    ('Na D', 5876.875, 5909.375),
    ('Fe 6189', 6145.0, 6201.0),
    ('Ca II 8542', 8522.0, 8562.0),
    ('Ca II 8662', 8642.0, 8682.0),
)
UV_STELLAR_WIND_WINDOWS = (
    ('N V P-Cygni 1238,1242', 1225.0, 1255.0),
    ('Si IV P-Cygni 1394,1403', 1378.0, 1418.0),
    ('C IV P-Cygni 1548,1550', 1532.0, 1568.0),
)
UV_EMISSION_WINDOWS = (
    ('He II 1640', 1628.0, 1652.0),
    ('O III] 1661,1666', 1648.0, 1678.0),
    ('C III] 1907,1909', 1894.0, 1920.0),
)


def _shade_masked_pixels(ax, wave, fit_mask):
    wave = np.asarray(wave, float)
    fit_mask = np.asarray(fit_mask, bool)
    bad = ~fit_mask
    if not np.any(bad):
        return
    idx = np.flatnonzero(bad)
    starts = [idx[0]]
    ends = []
    for a, b in zip(idx[:-1], idx[1:]):
        if b != a + 1:
            ends.append(a)
            starts.append(b)
    ends.append(idx[-1])
    for i0, i1 in zip(starts, ends):
        ax.axvspan(wave[max(0, i0)], wave[min(wave.size - 1, i1)], alpha=0.12)


def _plot_spectral_stamp(ax, wave, flux, unc, model, stellar, gas, fit_mask, lo, hi, title):
    sel = np.isfinite(wave) & (wave >= lo) & (wave <= hi)
    if np.count_nonzero(sel) < 2:
        ax.axis('off')
        ax.text(0.5, 0.5, f'{title}\nnot covered', ha='center', va='center')
        return
    w, f, m, s, g, mk = wave[sel], flux[sel], model[sel], stellar[sel], gas[sel], fit_mask[sel]
    e = unc[sel] if unc is not None else None
    ax.plot(w, f, lw=0.8, label='Data')
    if e is not None:
        finite_unc = np.isfinite(e) & (e > 0)
        if np.any(finite_unc):
            ax.fill_between(w, f-e, f+e, where=finite_unc, alpha=0.10, linewidth=0)
    ax.plot(w, m, lw=1.2, label='Total model')
    ax.plot(w, s, lw=1.0, ls='--', label='Stellar')
    if np.any(np.isfinite(g) & (np.abs(g) > 0)):
        ax.plot(w, g, lw=0.9, ls=':', label='Gas')
    _shade_masked_pixels(ax, w, mk)
    ax.set_xlim(lo, hi)
    ax.set_title(title, fontsize=9)
    ax.tick_params(direction='in', top=True, right=True, labelsize=8)


def plot_stellar_observational_summary(result_or_path, output_path=None):
    """Port of the original UV/optical observational-summary figure."""
    r = _load(result_or_path)
    wave = np.asarray(r.wave, float)
    flux = np.asarray(r.flux, float)
    unc = None if r.flux_unc is None else np.asarray(r.flux_unc, float)
    model = np.asarray(r.model, float)
    stellar = np.asarray(r.stellar_model, float)
    gas = np.asarray(r.gas_model, float)
    fit_mask = np.asarray(r.mask, bool)

    if str(r.regime).lower() == 'uv':
        fig = plt.figure(figsize=(18, 11))
        gs = fig.add_gridspec(3, 6, height_ratios=[1.0, 1.0, 1.25], hspace=0.42, wspace=0.42)
        for i, (name, lo, hi) in enumerate(UV_STELLAR_WIND_WINDOWS):
            ax = fig.add_subplot(gs[0, 2*i:2*i+2])
            _plot_spectral_stamp(ax, wave, flux, unc, model, stellar, gas, fit_mask, lo, hi, name)
            if i == 0:
                ax.set_ylabel(r'$L_\lambda$')
        for i, (name, lo, hi) in enumerate(UV_EMISSION_WINDOWS):
            ax = fig.add_subplot(gs[1, 2*i:2*i+2])
            _plot_spectral_stamp(ax, wave, flux, unc, model, stellar, gas, fit_mask, lo, hi, name)
            if i == 0:
                ax.set_ylabel(r'$L_\lambda$')
        ages = np.asarray(r.ages_myr, float)
        mass = _positive_coefficients(r)
        mass_fraction = mass / mass.sum() if mass.sum() > 0 else np.zeros_like(mass)
        light = np.asarray(r.light_fractions, float) if r.light_fractions is not None else mass_fraction.copy()
        light = np.where(np.isfinite(light) & (light >= 0), light, 0.0)
        light = light / light.sum() if light.sum() > 0 else np.zeros_like(light)
        unique_age = np.unique(ages)
        mass_by_age = np.asarray([np.sum(mass_fraction[np.isclose(ages, a)]) for a in unique_age], float)
        light_by_age = np.asarray([np.sum(light[np.isclose(ages, a)]) for a in unique_age], float)
        axw = fig.add_subplot(gs[2, :])
        axw.plot(unique_age, mass_by_age, marker='o', lw=1.4, label='Relative fitted stellar mass')
        axw.plot(unique_age, light_by_age, marker='s', lw=1.4, ls='--', label='Relative stellar luminosity')
        axw.set_xlabel('SSP age [Myr]')
        axw.set_ylabel('Relative contribution')
        axw.set_title('Relative stellar mass and luminosity versus SSP age')
        axw.tick_params(direction='in', top=True, right=True)
        axw.legend(loc='best', fontsize=9)
    else:
        fig = plt.figure(figsize=(18, 11))
        gs = fig.add_gridspec(3, 10, height_ratios=[1.0, 1.0, 1.25], hspace=0.38, wspace=0.45)
        for i, (name, lo, hi) in enumerate(OPTICAL_STELLAR_FEATURE_WINDOWS):
            ax = fig.add_subplot(gs[0, 2*i:2*i+2])
            _plot_spectral_stamp(ax, wave, flux, unc, model, stellar, gas, fit_mask, lo, hi, name)
            if i == 0:
                ax.set_ylabel(r'$L_\lambda$')
        # In FitSpec the gas nuisance result stores line wavelengths/amplitudes.
        names = list(r.gas_names)
        rests = np.asarray(r.gas_rest_wavelengths, float)
        amps = np.asarray(r.gas_amplitudes, float)
        selected = []
        if rests.size:
            good = np.isfinite(rests) & (rests >= np.nanmin(wave)) & (rests <= np.nanmax(wave))
            idx = np.flatnonzero(good)
            if idx.size:
                idx = idx[np.argsort(np.abs(amps[idx]))[::-1]][:5]
                selected = [(names[j] if j < len(names) else f'line {j}', rests[j]) for j in idx]
        for i in range(5):
            ax = fig.add_subplot(gs[1, 2*i:2*i+2])
            if i >= len(selected):
                ax.axis('off'); ax.text(0.5, 0.5, 'No fitted line', ha='center', va='center'); continue
            name, center = selected[i]
            half_width = max(12.0, center*600.0/299792.458)
            _plot_spectral_stamp(ax, wave, flux, unc, model, stellar, gas, fit_mask, center-half_width, center+half_width, f'{name}  {center:.1f} Å')
            if i == 0:
                ax.set_ylabel(r'$L_\lambda$')
        ages = np.asarray(r.ages_myr, float)
        current = np.asarray(r.current_mass_coefficients, float)
        current = np.where(np.isfinite(current) & (current > 0), current, 0.0)
        mfrac = current/current.sum() if current.sum()>0 else current
        lfrac = np.asarray(r.light_fractions, float) if r.light_fractions is not None else mfrac.copy()
        lfrac = np.where(np.isfinite(lfrac)&(lfrac>=0),lfrac,0.0); lfrac=lfrac/lfrac.sum() if lfrac.sum()>0 else lfrac
        unique_age=np.unique(ages)
        mass_by_age=np.asarray([np.sum(mfrac[np.isclose(ages,a)]) for a in unique_age],float)
        light_by_age=np.asarray([np.sum(lfrac[np.isclose(ages,a)]) for a in unique_age],float)
        axw=fig.add_subplot(gs[2,:]); axw.plot(unique_age/1000.0,mass_by_age,marker='o',lw=1.4,label='Relative current stellar mass'); axw.plot(unique_age/1000.0,light_by_age,marker='s',lw=1.4,ls='--',label='Relative stellar luminosity'); axw.set_xscale('log'); axw.set_xlabel('SSP age [Gyr]'); axw.set_ylabel('Relative contribution'); axw.set_title('Relative stellar mass and luminosity versus SSP age'); axw.tick_params(direction='in',top=True,right=True); axw.legend(loc='best',fontsize=9)

    cluster = str((getattr(r, 'metadata', {}) or {}).get('cluster', (getattr(r, 'metadata', {}) or {}).get('cluster_id', '')))
    fig.suptitle(f'{cluster} — {str(r.regime).upper()} stellar + gas fitting summary'.strip(' —'), fontsize=14, y=0.995)
    handles, labels = [], []
    for ax in fig.axes[:5 if str(r.regime).lower() == 'optical' else 3]:
        h, l = ax.get_legend_handles_labels()
        if h:
            handles, labels = h, l
            break
    if handles:
        fig.legend(handles, labels, loc='upper center', ncol=len(handles), bbox_to_anchor=(0.5, 0.965), frameon=False)
    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f'{output_path} saved')
    return fig


def write_stellar_diagnostics_text(result_or_path, output_path):
    """Write the same numerical summary used in the six-panel diagnostic plot."""
    r = _load(result_or_path)
    d = r.diagnostics
    output_path = Path(output_path).expanduser()
    if d is None or d.correlation_matrix is None:
        lines = ['SSP degeneracy summary', '', 'No SSP diagnostics stored.']
        np.savetxt(output_path, np.asarray(lines, dtype=str), fmt='%s')
        return output_path
    corr = np.asarray(d.correlation_matrix, float)
    n = corr.shape[0]
    basis_idx = _diagnostic_basis_indices(r, n)
    masses = _positive_coefficients(r)[basis_idx]
    ages = np.asarray(r.ages_myr, float)[basis_idx]
    z = np.asarray(r.metallicities_solar, float)[basis_idx]
    dom_full = int(r.dominant_index)
    hit = np.flatnonzero(basis_idx == dom_full)
    dom = int(hit[0]) if hit.size else int(np.nanargmax(masses))
    dc = corr[dom]
    alternatives = np.arange(n); alternatives = alternatives[alternatives != dom]; alternatives = alternatives[np.argsort(dc[alternatives])[::-1]][:5]
    dist = _diagnostic_vector_on_basis(d.dominant_ssp_distance, basis_idx, n, np.asarray(r.coefficients).size)
    lines = [
        'SSP degeneracy summary', '', 'Dominant SSP', '------------',
        f'Age:          {ages[dom]:g} Myr',
        f'Metallicity:  {z[dom]:.2f} Z_sun',
        f'Mass:         {masses[dom]:.5g} M_sun', '',
        'Library diagnostics', '-------------------',
        f'SSP models:        {n}',
        f'Active SSPs:        {int(np.count_nonzero(masses > 0))}',
        f'Effective rank:     {int(d.effective_rank) if d.effective_rank is not None else 0}',
        f'Condition number:   {float(d.condition_number) if d.condition_number is not None else np.nan:.4g}',
        f'Used pixels:        {int(np.count_nonzero(r.mask))}',
        f'SSPs with r>=0.95:  {int(np.count_nonzero((dc >= 0.95) & (np.arange(n) != dom)))}',
        f'Reduced chi-square: {r.reduced_chi_square:.4g}', '',
        'Top degeneracies', '----------------',
    ]
    for i in alternatives:
        lines.append(f'{ages[i]:g} Myr, Z={z[i]:.2f} Z_sun, mass={masses[i]:.5g} M_sun, r={dc[i]:.4f}, D={dist[i]:.4f}')
    np.savetxt(output_path, np.asarray(lines, dtype=str), fmt='%s')
    print(f'{output_path} saved')
    return output_path


def _product_stem(result_path):
    """Version-neutral FitSpec product stem."""
    path = Path(result_path).expanduser()
    return path.with_suffix('')


def save_stellar_plot_products(result_or_path, result_path):
    """Save all deterministic stellar products using version-neutral names.

    For ``stellar_fit.fits`` this writes::

        stellar_fit.pdf
        stellar_fit_summary.pdf
        stellar_fit_diag.pdf
        stellar_fit_diag.txt
    """
    base = _product_stem(result_path)
    main_pdf = base.with_suffix('.pdf')
    summary_pdf = base.with_name(base.name + '_summary.pdf')
    diag_pdf = base.with_name(base.name + '_diag.pdf')
    diag_txt = base.with_name(base.name + '_diag.txt')

    fig_main = plot_stellar_fit(result_or_path, output_path=main_pdf)
    fig_diag = plot_stellar_diagnostics(result_or_path, output_path=diag_pdf, text_output_path=diag_txt)
    fig_summary = plot_stellar_observational_summary(result_or_path, output_path=summary_pdf)

    for fig in (fig_main, fig_diag, fig_summary):
        plt.close(fig)
    return {
        'main_pdf': main_pdf,
        'summary_pdf': summary_pdf,
        'diag_pdf': diag_pdf,
        'diag_txt': diag_txt,
    }
