"""Per-line "stamp" grids and the stacked common-velocity-scale plot.

The stacked plot reproduces RDGEN's signature velocity-plot capability
(its cursor-mode ``v``/``u`` commands): several transitions/lines,
often from different ions, plotted on one shared velocity axis relative
to a common reference redshift, stacked with a vertical offset so
common kinematic structure across species is immediately visible (see
the VPFIT/RDGEN documentation review). Both functions here are
duck-typed over "line-like" objects -- anything with a ``.name`` and a
``.rest_wavelength_angstrom`` -- so they work identically for
``emission.lines.EmissionLine`` and ``absorption.atomic.AtomicTransition``
without this module importing either science module directly.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from astropy.constants import c as _c

__all__ = ["wavelength_to_velocity_kms", "plot_emission_stamps", "plot_line_stamps", "plot_velocity_stack"]

C_KMS = _c.to("km/s").value


def wavelength_to_velocity_kms(wave, rest_wavelength_angstrom: float, redshift: float = 0.0):
    """Invert the relativistic Doppler formula: velocity offset from a
    rest wavelength at a given systemic redshift, for each point in
    ``wave`` (observed frame). Consistent with the forward transform
    used throughout the emission/absorption line-profile code.
    """
    wave = np.asarray(wave, dtype=float)
    ratio = wave / (float(rest_wavelength_angstrom) * (1.0 + float(redshift)))
    ratio_squared = ratio ** 2
    return C_KMS * (ratio_squared - 1.0) / (ratio_squared + 1.0)


def plot_emission_stamps(
    wave, flux, flux_unc, continuum, total_line_model, component_line_models, lines, *,
    redshift: float = 0.0, velocity_half_width_kms: float = 300.0, x_axis: str = "velocity",
    ncols: int = 4, panel_size=(3.4, 3.2), fontsize: int = 9, sigma_lines=(-3.0, 3.0),
    component_colors=None, frozen_components=None,
):
    """Grid of per-line stamps in velocity/flux space, each paired with a
    residual sub-panel directly below it.

    Continuum handling matches the original ``bic_emission_fitting.py``/
    ``dynesty_mcmc_module.py`` stamp plot exactly: subtraction, not
    normalization. The data (`flux`/`flux_unc`) is plotted as-is; the
    continuum is drawn as its own dashed curve; every model curve (total
    and per-component) is `line-only model + continuum`, so it's
    directly comparable to the raw data on the same additive scale.
    Dividing by the continuum (an earlier version of this function) is
    deliberately not done here. For a continuum-subtracted, zero-baseline
    view instead, pass `continuum` as all-zeros and `flux` already
    subtracted (see ``gui.emission.EmissionGUI._plot_stamps``, which
    does that swap based on ``emission_fig_normalized``).

    Pure plotting: every array must already be on a shared `wave` grid
    (see ``gui.emission.EmissionGUI._plot_stamps``, which evaluates the
    line-only models -- science stays out of this module, per the
    "plotting is separate from fitting" design principle). Works equally
    well from a live preview (before ever clicking Fit) or a completed
    fit result, since it only needs arrays, not a ``FitResult``/config
    object -- the point being to help pick good initial values by eye.

    Parameters
    ----------
    wave : array-like
        Observed-frame wavelengths, shared by every array below.
    flux, flux_unc : array-like
        The data and its uncertainty, in the spectrum's own flux units
        (whatever ``emission_flux_normalizing_factor`` already rescaled
        them to -- this function does no further rescaling).
    continuum : array-like
        The continuum, on the same `wave` grid and units as `flux`.
    total_line_model : array-like
        The full (every component, every line) *line-only* model (no
        continuum baked in) -- this function adds `continuum` to it.
    component_line_models : list[array-like]
        One *line-only* curve per kinematic component (continuum not
        included -- added here), plotted in separate colors so each
        component's contribution to a blended line is visually
        distinguishable.
    lines : list
        Objects with ``.name`` and ``.rest_wavelength_angstrom`` (e.g.
        ``emission.lines.EmissionLine``) -- one stamp per entry.
    frozen_components : list[bool], optional
        One flag per entry in `component_line_models` (see
        ``emission.rejection``/``emission_reject_insignificant_components``)
        -- a frozen component's curve is drawn dotted, thinner, and
        lower-alpha instead of the usual dashed line, and its legend
        entry reads "C{i} (frozen)", so a component the fit judged
        insignificant is visually distinguishable from one that's
        actually detected. Omit (or pass all-False) if the fit didn't
        use rejection at all.
    x_axis : {"velocity", "rest_wavelength", "observed_wavelength"}
        What each stamp's x-axis actually shows. The *window* covered is
        always defined by `velocity_half_width_kms` regardless of this
        choice (converted to a wavelength half-width per line when
        plotting by wavelength), so every stamp still covers the same
        physical velocity range -- only the axis labeling/coordinate
        changes. The vertical reference line marks the systemic redshift
        (v=0) position either way: 0 km/s, the line's rest wavelength,
        or the line's redshifted-but-not-Doppler-shifted wavelength,
        respectively.
    sigma_lines : (float, float)
        Residual-panel horizontal guide lines, in units of sigma.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if x_axis not in ("velocity", "rest_wavelength", "observed_wavelength"):
        raise ValueError("x_axis must be 'velocity', 'rest_wavelength', or 'observed_wavelength'.")

    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)
    flux_unc = np.asarray(flux_unc, dtype=float)
    continuum = np.asarray(continuum, dtype=float)
    total_curve = np.asarray(total_line_model, dtype=float) + continuum
    component_curves = [np.asarray(m, dtype=float) + continuum for m in component_line_models]
    frozen_components = (
        [False] * len(component_curves) if frozen_components is None else list(frozen_components)
    )
    if len(frozen_components) != len(component_curves):
        raise ValueError("frozen_components must have one entry per component_line_models entry.")

    n = len(lines)
    if n == 0:
        raise ValueError("lines must be non-empty.")
    ncols = max(1, min(ncols, n))
    nrows = int(np.ceil(n / ncols))
    # Matches the original stamp plot's own component palette exactly.
    palette = component_colors or ["magenta", "cyan", "orange", "purple"]

    fig, axes = plt.subplots(
        nrows * 2, ncols, figsize=(panel_size[0] * ncols, panel_size[1] * nrows),
        squeeze=False, gridspec_kw={"height_ratios": [3, 1] * nrows, "hspace": 0.08},
        constrained_layout=True,
    )

    # One shared legend for the whole figure, built from whichever panel
    # is the first to actually render -- not hardcoded to index 0, since
    # that specific line can fall outside velocity_half_width_kms (a
    # "not covered" panel skips plotting anything at all, so labeling
    # only ever index==0 could leave the entire figure without a legend
    # whenever the very first line happens to be the uncovered one).
    labeled_yet = False

    for index, line in enumerate(lines):
        row_pair, col = divmod(index, ncols)
        ax_main = axes[row_pair * 2][col]
        ax_resid = axes[row_pair * 2 + 1][col]

        velocity = wavelength_to_velocity_kms(wave, line.rest_wavelength_angstrom, redshift)
        in_window = np.abs(velocity) <= velocity_half_width_kms
        if not np.any(in_window):
            ax_main.text(0.5, 0.5, f"{line.name}\nnot covered", ha="center", va="center",
                         transform=ax_main.transAxes, fontsize=fontsize)
            ax_main.set_xticks([]); ax_main.set_yticks([])
            ax_resid.axis("off")
            continue
        should_label = not labeled_yet
        labeled_yet = True

        # The window is always velocity-defined; only the plotted
        # coordinate (and reference-line position) changes with x_axis.
        if x_axis == "velocity":
            x = velocity[in_window]
            x_reference = 0.0
            x_half_width = velocity_half_width_kms
        elif x_axis == "rest_wavelength":
            x = wave[in_window] / (1.0 + redshift)
            x_reference = line.rest_wavelength_angstrom
            edge_velocity = np.array([-velocity_half_width_kms, velocity_half_width_kms])
            beta = edge_velocity / C_KMS
            edge_wave = line.rest_wavelength_angstrom * np.sqrt((1 + beta) / (1 - beta))
            x_half_width = float(np.max(np.abs(edge_wave - line.rest_wavelength_angstrom)))
        else:  # observed_wavelength
            x = wave[in_window]
            x_reference = line.rest_wavelength_angstrom * (1.0 + redshift)
            edge_velocity = np.array([-velocity_half_width_kms, velocity_half_width_kms])
            beta = edge_velocity / C_KMS
            edge_wave = x_reference * np.sqrt((1 + beta) / (1 - beta))
            x_half_width = float(np.max(np.abs(edge_wave - x_reference)))

        # drawstyle="steps-mid" connects consecutive data points with a
        # step-line (matching the original's ds="steps-mid"), rather than
        # leaving isolated markers.
        ax_main.errorbar(
            x, flux[in_window], yerr=flux_unc[in_window],
            drawstyle="steps-mid", color="tab:blue", ecolor="tab:blue", lw=1.0, elinewidth=0.6,
            zorder=2, label="data" if should_label else None,
        )
        ax_main.plot(x, continuum[in_window], ls="--", color="green", lw=1.0, zorder=2,
                     label="continuum" if should_label else None)
        for component_index, component_curve in enumerate(component_curves):
            is_frozen = frozen_components[component_index]
            label = f"C{component_index} (frozen)" if is_frozen else f"C{component_index}"
            ax_main.plot(
                x, component_curve[in_window],
                lw=(0.7 if is_frozen else 1.0), ls=(":" if is_frozen else "--"), alpha=(0.35 if is_frozen else 0.6),
                color=palette[component_index % len(palette)], zorder=3,
                label=label if should_label else None,
            )
        ax_main.plot(x, total_curve[in_window], lw=2.0, color="tab:red", zorder=4,
                     label="total" if should_label else None)
        ax_main.axvline(x_reference, lw=0.8, color="gray", ls="-", zorder=1)
        ax_main.set_title(line.name, fontsize=fontsize + 1)
        ax_main.tick_params(labelsize=fontsize - 1, labelbottom=False)
        ax_main.set_xlim(x_reference - x_half_width, x_reference + x_half_width)
        if col == 0:
            ax_main.set_ylabel("Flux", fontsize=fontsize)

        good = np.isfinite(flux_unc[in_window]) & (flux_unc[in_window] > 0)
        residual = np.full(x.shape, np.nan)
        residual[good] = (flux[in_window] - total_curve[in_window])[good] / flux_unc[in_window][good]
        ax_resid.scatter(x, residual, s=5, color="tab:blue")
        ax_resid.axhline(0.0, lw=0.8, color="k")
        for level in sigma_lines:
            ax_resid.axhline(level, lw=0.6, ls="--", color="0.6")
        ax_resid.axvline(x_reference, lw=0.8, color="gray", ls="-", zorder=1)
        y_limit = max(4.0, max(abs(s) for s in sigma_lines) + 1.0)
        ax_resid.set_ylim(-y_limit, y_limit)
        ax_resid.set_xlim(x_reference - x_half_width, x_reference + x_half_width)
        ax_resid.tick_params(labelsize=fontsize - 1)
        if col == 0:
            ax_resid.set_ylabel(r"Resid. ($\sigma$)", fontsize=fontsize - 1)

    for index in range(n, nrows * ncols):
        row_pair, col = divmod(index, ncols)
        axes[row_pair * 2][col].axis("off")
        axes[row_pair * 2 + 1][col].axis("off")

    handles, labels = [], []
    for ax in (axes[row_pair * 2][col] for row_pair in range(nrows) for col in range(ncols)):
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            break
    if handles:
        # "outside upper center" (not bbox_to_anchor) is the
        # constrained-layout-aware way to place a figure-level legend --
        # with constrained_layout=True (set above), a legend positioned
        # via bbox_to_anchor sits outside the area constrained_layout
        # reserves space for, so it exists on the Figure object but can
        # end up clipped/invisible in the actual rendered output
        # (on-screen or saved). "outside upper center" tells the layout
        # engine to reserve room for it up front instead.
        #
        # ncol is capped, not set to len(handles): forcing every entry
        # (data/continuum/total/one per component) onto a single row
        # made the legend wider than the figure itself for anything but
        # a wide multi-column grid -- e.g. a single-line, single-column
        # plot is only `panel_size[0]` inches wide, nowhere near enough
        # for a 6-entry one-row legend, so it silently overflowed off
        # the canvas. Capping lets it wrap across a couple of rows instead.
        fig.legend(handles, labels, loc="outside upper center", ncol=min(len(handles), 3),
                   fontsize=fontsize - 1, frameon=False)
    x_axis_label = {
        "velocity": "Velocity (km/s)",
        "rest_wavelength": r"Rest wavelength ($\AA$)",
        "observed_wavelength": r"Observed wavelength ($\AA$)",
    }[x_axis]
    fig.supxlabel(x_axis_label, fontsize=fontsize + 1)
    return fig


def plot_line_stamps(
    wave, flux, lines, *, flux_unc=None, model=None, redshift: float = 0.0,
    velocity_half_width_kms: float = 300.0, ncols: int = 4,
    panel_size=(3.2, 2.4), fontsize: int = 9, sharey: bool = False,
):
    """Grid of small, per-line panels, each zoomed to +/- ``velocity_half_width_kms``
    around one line, in velocity space -- useful for visually comparing
    line shapes/widths/detections across many transitions at a glance.

    Parameters
    ----------
    lines : list
        Objects with ``.name`` and ``.rest_wavelength_angstrom`` (e.g.
        ``emission.lines.EmissionLine`` or ``absorption.atomic.AtomicTransition``).

    Returns
    -------
    matplotlib.figure.Figure
    """
    from plotting.spectrum import plot_spectrum_only

    wave = np.asarray(wave, dtype=float)
    n = len(lines)
    if n == 0:
        raise ValueError("lines must be non-empty.")
    ncols = max(1, min(ncols, n))
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(panel_size[0] * ncols, panel_size[1] * nrows),
        sharey=sharey, squeeze=False,
    )

    for index, line in enumerate(lines):
        ax = axes[index // ncols][index % ncols]
        velocity = wavelength_to_velocity_kms(wave, line.rest_wavelength_angstrom, redshift)
        in_window = np.abs(velocity) <= velocity_half_width_kms
        if not np.any(in_window):
            ax.text(0.5, 0.5, f"{line.name}\nnot covered", ha="center", va="center", transform=ax.transAxes,
                     fontsize=fontsize)
            ax.set_xticks([]); ax.set_yticks([])
            continue
        plot_spectrum_only(
            velocity[in_window], flux[in_window],
            flux_unc=(None if flux_unc is None else np.asarray(flux_unc)[in_window]),
            model=(None if model is None else np.asarray(model)[in_window]),
            ax=ax, title=line.name, xlabel="", ylabel="", fontsize=fontsize, show_legend=False,
        )
        ax.axvline(0.0, lw=0.6, ls="--", color="0.5")

    for index in range(n, nrows * ncols):
        axes[index // ncols][index % ncols].axis("off")

    fig.supxlabel("Velocity (km/s)", fontsize=fontsize + 1)
    fig.tight_layout()
    return fig


def plot_velocity_stack(
    wave, flux, lines, *, flux_unc=None, model=None, reference_redshift: float = 0.0,
    velocity_range_kms=(-500.0, 500.0), component_velocities_kms=None,
    normalize: bool = True, y_boost: float = 1.0, figsize=None, fontsize: int = 10,
    title=None,
):
    """Stack several lines on one shared velocity axis (RDGEN-style).

    Each line occupies one vertical slot, offset from the next by
    ``y_boost`` (in normalized-flux units if ``normalize`` is True), so
    common velocity structure across species -- e.g. "is the same
    kinematic system present in both H I and C IV?" -- is visible at a
    glance, exactly the diagnostic RDGEN's stacked velocity plots are
    built around.

    Parameters
    ----------
    lines : list
        Objects with ``.name`` and ``.rest_wavelength_angstrom``, plotted
        top-to-bottom in the given order.
    component_velocities_kms : list[float], optional
        If given, a vertical marker is drawn at each of these
        velocities on every panel -- e.g. an absorption/emission fit's
        per-component fitted velocities, so the reader can see exactly
        where the model places each kinematic component relative to
        every line at once.
    normalize : bool, default True
        Divide each line's flux window by its own local median before
        stacking, so lines of very different absolute flux/depth are
        equally visible (does not affect fitted values -- display only).

    Returns
    -------
    matplotlib.figure.Figure
    """
    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)
    n = len(lines)
    if n == 0:
        raise ValueError("lines must be non-empty.")
    v_lo, v_hi = velocity_range_kms
    figsize = figsize or (7.0, max(3.0, 0.9 * n))

    fig, ax = plt.subplots(figsize=figsize)
    for index, line in enumerate(lines):
        velocity = wavelength_to_velocity_kms(wave, line.rest_wavelength_angstrom, reference_redshift)
        in_window = (velocity >= v_lo) & (velocity <= v_hi) & np.isfinite(flux)
        if not np.any(in_window):
            continue
        y = flux[in_window].astype(float)
        if normalize:
            scale = np.nanmedian(np.abs(y)) or 1.0
            y = y / scale
        offset = index * y_boost
        ax.plot(velocity[in_window], y + offset, lw=1.0, color=f"C{index % 10}")
        if model is not None:
            model_y = np.asarray(model, dtype=float)[in_window]
            if normalize:
                model_y = model_y / scale
            ax.plot(velocity[in_window], model_y + offset, lw=1.2, ls="--", color="k", alpha=0.7)
        ax.text(v_lo + 0.02 * (v_hi - v_lo), offset + (1.05 if normalize else 0.0), line.name,
                fontsize=fontsize - 1, va="bottom")

    ax.axvline(0.0, lw=0.8, color="0.3")
    if component_velocities_kms:
        for velocity_kms in component_velocities_kms:
            ax.axvline(velocity_kms, lw=0.8, ls=":", color="C3")

    ax.set_xlim(v_lo, v_hi)
    ax.set_xlabel("Velocity (km/s)", fontsize=fontsize)
    ax.set_ylabel(("Normalized flux" if normalize else "Flux") + " (stacked)", fontsize=fontsize)
    ax.tick_params(labelsize=fontsize - 1)
    if title:
        ax.set_title(title, fontsize=fontsize + 1)
    fig.tight_layout()
    return fig
