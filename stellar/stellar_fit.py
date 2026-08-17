"""Unified UV/optical stellar-population fitting for FitSpec.

This is the FitSpec port of the architecture developed in
``stellar_fitting_function_v8.py``.  There is deliberately ONE deterministic
stellar fitting engine.  UV/optical classification changes defaults and the
selected HDF5 stellar library; it does not dispatch to FICUS, pPXF, SESSAMME,
or any other independent backend.

By default the full SSP library is screened once with a vectorized single-SSP
chi-square calculation.  Only a small candidate basis is then transformed and
solved at every nonlinear trial.  Users can explicitly request the historical
full-library variable-projection solve when desired.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings
import numpy as np
from scipy.optimize import least_squares, lsq_linear
from astropy.constants import c, L_sun
from astropy.cosmology import Planck18
import astropy.units as u

from stellar.stellar_models import (
    DEFAULT_H5_PATHS, StellarLibrary, classify_spectral_regime,
    load_stellar_library, shift_broaden_resample_library,
    physical_light_fractions,
)
from stellar.stellar_results import StellarFitDiagnostics, StellarFitResult

C_KMS = c.to("km/s").value
L_SUN_ERG_S = L_sun.cgs.value

__all__ = [
    "DEFAULT_UV_EMISSION_LINES", "DEFAULT_OPTICAL_EMISSION_LINES",
    "prepare_stellar_library_from_config", "build_emission_line_templates",
    "fit_stellar_spectrum", "evaluate_stellar_result_on_grid",
    "calculate_stellar_diagnostics",
]

DEFAULT_UV_EMISSION_LINES = (
    ("Lyalpha",1215.6701),("NV1239",1238.821),("NV1243",1242.804),
    ("NIV]1483",1483.32),("NIV]1486",1486.50),("CIV1548",1548.204),
    ("CIV1551",1550.781),("HeII1640",1640.42),("OIII]1661",1660.809),
    ("OIII]1666",1666.150),("NIII]1750",1750.0),("SiIII]1883",1882.71),
    ("SiIII]1892",1892.03),("CIII]1907",1906.68),("CIII]1909",1908.73),
    ("CII]2326",2326.0),
)
DEFAULT_OPTICAL_EMISSION_LINES = (
    ("[OII]3726",3726.032),("[OII]3729",3728.815),("Hdelta",4101.742),
    ("Hgamma",4340.471),("[FeIII]4658",4658.05),("HeII4686",4685.71),
    ("Hbeta",4861.333),("[OIII]4959",4958.911),("[OIII]5007",5006.843),
    ("[FeIII]5084",5084.77),("HeI5876",5875.61),("[OI]6300",6300.304),
    ("[SIII]6312",6312.06),("[OI]6364",6363.776),("[NII]6548",6548.05),
    ("Halpha",6562.819),("[NII]6583",6583.45),("[SII]6716",6716.44),
    ("[SII]6731",6730.82),("[OII]7319",7319.99),("[OII]7330",7330.73),
    ("[FeII]8617",8616.95),("[SIII]9069",9068.6),
)


def _get(config, key, default=None):
    return config.get(key, default) if hasattr(config, "get") else default


def _as_list(config, key, cast=str):
    value = _get(config, key, None)
    if value is None or value == "" or value == []:
        return None
    if isinstance(value, (list, tuple, np.ndarray)):
        values = value
    else:
        values = [x.strip() for x in str(value).split(",") if x.strip()]
    return [cast(x) for x in values]


def _pair(config, key, default):
    value = _get(config, key, default)
    if isinstance(value, (list, tuple, np.ndarray)):
        vals = list(value)
    else:
        vals = [x.strip() for x in str(value).split(",")]
    if len(vals) < 2:
        return tuple(map(float, default))
    return float(vals[0]), float(vals[1])


def _optional_float(config, key, default=None):
    value = _get(config, key, default)
    if value is None or value == "":
        return None
    return float(value)


def _bool(config, key, default=False):
    value = _get(config, key, default)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true","1","yes","y","on"}


def _rest_wave(spectrum):
    return np.asarray(spectrum.wave, float) / (1.0 + float(spectrum.redshift))


def _flux_to_lsun_factor(redshift, config=None):
    """Return multiplier converting observed F_lambda to rest Lsun/Angstrom.

    The historical stellar fitter used 10 kpc for z=0.  FitSpec preserves that
    reproducible default but allows ``stellar_distance_kpc`` to override it.
    """
    z = float(redshift)
    distance_kpc = _optional_float(config, "stellar_distance_kpc", None)
    if distance_kpc is not None:
        distance_cm = (distance_kpc * u.kpc).to_value(u.cm)
    elif z == 0.0:
        distance_cm = (10.0 * u.kpc).to_value(u.cm)
    else:
        distance_cm = Planck18.luminosity_distance(z).to_value(u.cm)
    return 4.0 * np.pi * distance_cm**2 * (1.0 + z) / L_SUN_ERG_S


def prepare_stellar_library_from_config(spectrum, config):
    """Select and lazily load the HDF5 SSP subset needed for one spectrum.

    ``stellar_library=auto`` chooses pySB99 for UV and E-MILES for optical.
    Any of the four built-in HDF5 families can be explicitly selected in either
    regime when its wavelength coverage supports the observation.
    """
    wave = _rest_wave(spectrum)
    boundary = float(_get(config, "uv_optical_boundary_angstrom", 3000.0))
    regime = classify_spectral_regime(wave, boundary)
    family = str(_get(config, "stellar_library", "auto")).strip().lower()
    aliases = {"pysb99":"sb99","starburst99":"sb99","miles":"emiles","e-miles":"emiles"}
    family = aliases.get(family, family)
    if family in {"", "auto"}:
        family = "sb99" if regime == "uv" else "emiles"
    if family not in {"bpass", "sb99", "slug", "emiles"}:
        raise ValueError("stellar_library must be auto, sb99, bpass, slug, or emiles.")
    default_path = DEFAULT_H5_PATHS[family]
    path = Path(str(_get(config, f"{family}_h5_path", default_path))).expanduser()
    common_wave = (float(np.nanmin(wave)), float(np.nanmax(wave)))
    ages = _as_list(config, "stellar_ages_myr", float)
    age_range = _pair(config, "stellar_age_range_myr", (0.0, 20000.0))

    if family == "sb99":
        library = load_stellar_library(
            family, path, wave_range=common_wave,
            track=str(_get(config,"sb99_track","GENEC")),
            rotation=str(_get(config,"sb99_rotation","nonrot")),
            spectra_library=str(_get(config,"sb99_spectra_library","PoWR")),
            metallicity_labels=_as_list(config,"sb99_metallicities",str),
            ages_myr=ages, age_range_myr=age_range,
            intrinsic_model_resolution_R=_optional_float(config,"sb99_model_resolution_R",None),
        )
    elif family == "bpass":
        library = load_stellar_library(
            family, path, wave_range=common_wave,
            imf=str(_get(config,"bpass_imf","imf135_300")),
            population=str(_get(config,"bpass_population","binary")),
            metallicity_codes=_as_list(config,"bpass_metallicities",str),
            ages_myr=ages, age_range_myr=age_range,
            reference_mass_msun=float(_get(config,"bpass_reference_mass_msun",1.0e6)),
            intrinsic_model_resolution_R=_optional_float(config,"bpass_model_resolution_R",None),
        )
    elif family == "slug":
        library = load_stellar_library(
            family, path, wave_range=common_wave,
            slug_family=str(_get(config,"slug_family","auto")),
            subfamily=str(_get(config,"slug_subfamily","auto")),
            specsyn=str(_get(config,"slug_specsyn","auto")),
            metallicity_labels=_as_list(config,"slug_metallicities",str),
            ages_myr=ages, age_range_myr=age_range,
            intrinsic_model_resolution_R=_optional_float(config,"slug_model_resolution_R",None),
        )
    else:
        age_gyr = (age_range[0]/1000.0, age_range[1]/1000.0)
        library = load_stellar_library(
            family, path, wave_range=common_wave,
            isochrone=str(_get(config,"emiles_isochrone","Padova00")),
            imf=str(_get(config,"emiles_imf","ku")),
            imf_slope=_optional_float(config,"emiles_imf_slope",None),
            age_range_gyr=age_gyr,
            metal_range_dex=_pair(config,"emiles_metal_range_dex",(-5.0,2.0)),
            intrinsic_model_resolution_R=_optional_float(config,"emiles_model_resolution_R",None),
            intrinsic_fwhm_angstrom=_optional_float(config,"emiles_fwhm_angstrom",2.51),
        )
    return regime, library


def _doppler_factor(velocity_kms):
    beta = float(velocity_kms)/C_KMS
    if abs(beta) >= 1:
        raise ValueError("|velocity| must be smaller than c.")
    return np.sqrt((1+beta)/(1-beta))


def build_emission_line_templates(wave_rest, velocity_kms=0.0, sigma_kms=50.0,
                                  resolution=None, line_list=None,
                                  full_wave=None, fit_mask=None):
    """Peak-normalized Gaussian gas templates used only as nuisance terms.

    This preserves v8's optional simultaneous gas treatment so emission lines
    do not force the stellar continuum solution.  It does not replace the
    dedicated FitSpec emission-line science module.
    """
    wave = np.asarray(wave_rest,float)
    lines = tuple(line_list or ())
    full_wave = wave if full_wave is None else np.asarray(full_wave,float)
    fit_mask = np.ones(full_wave.size,bool) if fit_mask is None else np.asarray(fit_mask,bool)
    names, restwaves, columns = [], [], []
    for name, rest in lines:
        center = float(rest)*_doppler_factor(velocity_kms)
        if not (wave[0] <= center <= wave[-1]):
            continue
        nearest = int(np.nanargmin(np.abs(full_wave-center)))
        if not fit_mask[nearest]:
            continue
        sigma_intrinsic = center*max(0.0,float(sigma_kms))/C_KMS
        sigma_inst = 0.0 if resolution is None else float(np.asarray(resolution.sigma_angstrom(np.array([center])))[0])
        sigma_A = float(np.hypot(sigma_intrinsic,sigma_inst))
        pixel_sigma = max(np.nanmedian(np.diff(wave))/2.355, 1e-8)
        sigma_A = max(sigma_A,pixel_sigma)
        profile = np.exp(-0.5*((wave-center)/sigma_A)**2)
        if np.nanmax(profile) <= 0:
            continue
        columns.append(profile); names.append(str(name)); restwaves.append(float(rest))
    if not columns:
        return np.empty((wave.size,0),float), tuple(), np.empty(0,float)
    return np.column_stack(columns), tuple(names), np.asarray(restwaves,float)


def _fit_arrays(spectrum, config):
    wave = _rest_wave(spectrum)
    flux = np.asarray(spectrum.flux,float)
    if spectrum.flux_unc is None:
        raise ValueError("Stellar chi-square fitting requires flux_unc.")
    unc = np.asarray(spectrum.flux_unc,float)
    mask = np.ones(wave.size,bool) if spectrum.mask is None else np.asarray(spectrum.mask,bool).copy()
    mask &= np.isfinite(wave)&np.isfinite(flux)&np.isfinite(unc)&(unc>0)
    if np.count_nonzero(mask) < int(_get(config,"stellar_minimum_fit_pixels",20)):
        raise ValueError("Too few valid stellar-fitting pixels remain after masking.")
    factor = _flux_to_lsun_factor(spectrum.redshift, config)
    return wave, flux*factor, unc*factor, mask, factor


def _solve_linear(design, flux, unc, mask, *, n_stellar, gas_peak_max_factor=2.0,
                  method="bvls", tolerance=1e-8, max_iter=3000):
    A = np.asarray(design[mask,:],float); b=np.asarray(flux[mask],float); e=np.asarray(unc[mask],float)
    Aw=A/e[:,None]; bw=b/e
    scales=np.linalg.norm(Aw,axis=0)
    if np.any(~np.isfinite(scales)|(scales<=0)):
        bad=np.flatnonzero(~np.isfinite(scales)|(scales<=0))
        raise ValueError(f"Zero/invalid linear design columns at indices {bad.tolist()}.")
    As=Aw/scales[None,:]
    lower=np.zeros(A.shape[1],float); upper=np.full(A.shape[1],np.inf,float)
    if A.shape[1] > n_stellar:
        peak=max(float(np.nanmax(np.abs(b))),1e-30)*float(gas_peak_max_factor)
        upper[n_stellar:]=peak*scales[n_stellar:]
    opt=lsq_linear(As,bw,bounds=(lower,upper),method=method,tol=float(tolerance),max_iter=int(max_iter))
    coeff=np.maximum(opt.x/scales,0.0)
    model=np.asarray(design@coeff,float)
    return coeff, model, opt


def _parameter_covariance(opt, reduced_chi_square):
    jac=np.asarray(getattr(opt,"jac",np.empty((0,0))),float)
    n=jac.shape[1] if jac.ndim==2 else 0
    if n==0 or jac.shape[0] <= n or not np.all(np.isfinite(jac)):
        return None,np.full(n,np.nan)
    try:
        cov=np.linalg.pinv(jac.T@jac)
        if np.isfinite(reduced_chi_square) and reduced_chi_square>0: cov*=reduced_chi_square
        err=np.sqrt(np.clip(np.diag(cov),0,None))
        return cov,err
    except np.linalg.LinAlgError:
        return None,np.full(n,np.nan)



def _subset_library(library: StellarLibrary, indices) -> StellarLibrary:
    """Return a lightweight view containing only selected SSP rows."""
    idx=np.asarray(indices,dtype=int)
    surviving=None if library.surviving_mass_fractions is None else np.asarray(library.surviving_mass_fractions)[idx]
    mdex=None if library.metallicities_dex is None else np.asarray(library.metallicities_dex)[idx]
    return StellarLibrary(
        wave=library.wave, flux_per_mass=np.asarray(library.flux_per_mass)[idx],
        flux_reference=np.asarray(library.flux_reference)[idx], ages_myr=np.asarray(library.ages_myr)[idx],
        metallicity_codes=np.asarray(library.metallicity_codes)[idx],
        metallicities_solar=np.asarray(library.metallicities_solar)[idx],
        reference_masses=np.asarray(library.reference_masses)[idx],
        labels=tuple(library.labels[i] for i in idx),
        source_paths=tuple(library.source_paths[i] for i in idx), family=library.family,
        model_resolution_R=library.model_resolution_R, model_fwhm_angstrom=library.model_fwhm_angstrom,
        surviving_mass_fractions=surviving, metallicities_dex=mdex, normalization=library.normalization,
        metadata=dict(library.metadata),
    )


def _single_ssp_chi_square_fast(models, flux, unc, mask):
    """Vectorized best-scale chi-square for every SSP, with scale constrained >=0."""
    M=np.asarray(models,float)
    y=np.asarray(flux,float); e=np.asarray(unc,float); m=np.asarray(mask,bool)
    valid=m&np.isfinite(y)&np.isfinite(e)&(e>0)&np.all(np.isfinite(M),axis=0)
    if np.count_nonzero(valid)<2:
        return np.full(M.shape[0],np.inf)
    Mw=M[:,valid]/e[valid][None,:]
    yw=y[valid]/e[valid]
    dot=Mw@yw
    norm=np.einsum("ij,ij->i",Mw,Mw)
    scale=np.zeros(M.shape[0],float)
    good=norm>0
    scale[good]=np.maximum(0.0,dot[good]/norm[good])
    y2=float(np.dot(yw,yw))
    chi=y2-2.0*scale*dot+scale*scale*norm
    chi[~np.isfinite(chi)]=np.inf
    return np.maximum(chi,0.0)


def _select_deterministic_basis(library, wave, flux, unc, mask, config, *, resolution, ebv, velocity_kms, sigma_kms, progress=True):
    mode=str(_get(config,"stellar_template_selection","candidate")).strip().lower()
    if mode not in {"candidate","full"}:
        raise ValueError("stellar_template_selection must be 'candidate' or 'full'.")
    if mode=="full" or library.n_models<=1:
        indices=np.arange(library.n_models,dtype=int)
        return indices, None, None
    maximum=max(1,int(_get(config,"stellar_candidate_max",30)))
    maximum=min(maximum,library.n_models)
    if progress:
        print(f"[FitSpec:stellar] screening {library.n_models} SSPs for candidate basis",flush=True)
    full_models=shift_broaden_resample_library(
        library,wave,ebv=ebv,velocity_kms=velocity_kms,sigma_kms=sigma_kms,
        attenuation_law=str(_get(config,"stellar_attenuation_law","calzetti00")),resolution=resolution)
    chi=_single_ssp_chi_square_fast(full_models,flux,unc,mask)
    order=np.argsort(chi,kind="stable")
    finite=order[np.isfinite(chi[order])]
    if finite.size==0:
        raise ValueError("Stellar candidate screening found no finite SSP chi-square values.")
    indices=np.sort(finite[:maximum])
    if progress:
        best=int(finite[0]); print(
            f"[FitSpec:stellar] selected {indices.size} candidate SSPs from {library.n_models}; "
            f"best single-SSP index={best}, chi2={chi[best]:.7g}",flush=True)
    return indices, chi, full_models

def calculate_stellar_diagnostics(transformed_models, observed_flux, observed_uncertainty,
                                  fit_mask, coefficients, effective_rank_tolerance=0.001):
    """Port of the v8 SSP degeneracy diagnostics to the common result schema."""
    models=np.asarray(transformed_models,float)
    flux=np.asarray(observed_flux,float); unc=np.asarray(observed_uncertainty,float); mask=np.asarray(fit_mask,bool)
    valid=mask&np.isfinite(flux)&np.isfinite(unc)&(unc>0)&np.all(np.isfinite(models),axis=0)
    if np.count_nonzero(valid)<2: return StellarFitDiagnostics()
    M=models[:,valid]; Mw=M/unc[valid][None,:]
    norms=np.linalg.norm(Mw,axis=1); safe=np.where(norms>0,norms,1.0); Mn=Mw/safe[:,None]
    corr=np.clip(Mn@Mn.T,-1,1)
    try: s=np.linalg.svd(Mn.T,compute_uv=False)
    except np.linalg.LinAlgError: s=np.empty(0)
    if s.size:
        threshold=float(effective_rank_tolerance)*s[0]; rank=int(np.count_nonzero(s>threshold)); cond=float(s[0]/s[-1]) if s[-1]>0 else np.inf
    else: rank=0; cond=np.inf
    y=flux[valid]; e=unc[valid]; chi=np.full(models.shape[0],np.nan)
    for i,row in enumerate(M):
        den=np.sum((row/e)**2)
        if den>0:
            scale=max(0.0,np.sum((row/e)*(y/e))/den)
            chi[i]=np.sum(((y-scale*row)/e)**2)
    delta=chi-np.nanmin(chi) if np.any(np.isfinite(chi)) else chi.copy()
    coeff=np.asarray(coefficients,float); dom=int(np.nanargmax(np.where(np.isfinite(coeff),coeff,-np.inf)))
    distance=np.sqrt(np.maximum(0.0,2.0*(1.0-corr[dom])))
    return StellarFitDiagnostics(
        correlation_matrix=corr,singular_values=s,effective_rank=rank,condition_number=cond,
        single_ssp_chi_square=chi,single_ssp_delta_chi_square=delta,
        dominant_ssp_distance=distance,transformed_model_fluxes=models,
    )


def fit_stellar_spectrum(spectrum, config, *, library: StellarLibrary | None=None,
                         regime: str | None=None, line_list=None):
    """Run the single universal FitSpec stellar fitter on UV or optical data."""
    if library is None:
        selected_regime, library = prepare_stellar_library_from_config(spectrum,config)
        regime = selected_regime if regime is None else str(regime).lower()
    else:
        regime = classify_spectral_regime(_rest_wave(spectrum),float(_get(config,"uv_optical_boundary_angstrom",3000.0))) if regime is None else str(regime).lower()

    wave,flux,unc,mask,flux_factor=_fit_arrays(spectrum,config)
    resolution=getattr(spectrum,"resolution",None)
    include_gas=_bool(config,"stellar_include_gas",False)
    if line_list is None:
        line_list=DEFAULT_UV_EMISSION_LINES if regime=="uv" else DEFAULT_OPTICAL_EMISSION_LINES

    ebv0=float(_get(config,"stellar_ebv_initial",0.05)); v0=float(_get(config,"stellar_velocity_initial_kms",0.0)); s0=float(_get(config,"stellar_sigma_initial_kms",50.0))
    gv0=float(_get(config,"stellar_gas_velocity_initial_kms",0.0)); gs0=float(_get(config,"stellar_gas_sigma_initial_kms",50.0))
    ebv_bounds=_pair(config,"stellar_ebv_bounds",(0,2)); v_bounds=_pair(config,"stellar_velocity_bounds_kms",(-500,500)); s_bounds=_pair(config,"stellar_sigma_bounds_kms",(0,1000))
    gv_bounds=_pair(config,"stellar_gas_velocity_bounds_kms",(-500,500)); gs_bounds=_pair(config,"stellar_gas_sigma_bounds_kms",(1,1000))

    # Lightweight default: stellar kinematics are fixed at their configured
    # initial values unless the user explicitly enables either parameter.
    fit_velocity=_bool(config,"stellar_fit_velocity",False)
    fit_sigma=_bool(config,"stellar_fit_sigma",False)
    names=["ebv"]
    initial=[ebv0]; lower=[ebv_bounds[0]]; upper=[ebv_bounds[1]]
    if fit_velocity:
        names.append("velocity_kms"); initial.append(v0); lower.append(v_bounds[0]); upper.append(v_bounds[1])
    if fit_sigma:
        names.append("sigma_kms"); initial.append(s0); lower.append(s_bounds[0]); upper.append(s_bounds[1])
    if include_gas:
        names.extend(["gas_velocity_kms","gas_sigma_kms"])
        initial.extend([gv0,gs0]); lower.extend([gv_bounds[0],gs_bounds[0]]); upper.extend([gv_bounds[1],gs_bounds[1]])
    names=tuple(names); x0=np.asarray(initial,float); lo=np.asarray(lower,float); hi=np.asarray(upper,float)
    if np.any(x0<lo)|np.any(x0>hi): raise ValueError("Initial stellar parameters lie outside configured bounds.")

    def unpack_nonlinear(theta):
        values={"ebv":ebv0,"velocity_kms":v0,"sigma_kms":s0,
                "gas_velocity_kms":gv0,"gas_sigma_kms":gs0}
        values.update({name:float(value) for name,value in zip(names,theta)})
        return values

    progress=_bool(config,"stellar_progress",True)
    basis_indices, screening_chi2, screening_models = _select_deterministic_basis(
        library,wave,flux,unc,mask,config,resolution=resolution,ebv=ebv0,velocity_kms=v0,sigma_kms=s0,progress=progress)
    fit_library=_subset_library(library,basis_indices)
    solver=str(_get(config,"stellar_mass_solver","bvls")).lower(); tol=float(_get(config,"stellar_mass_tolerance",1e-8)); maxiter=int(_get(config,"stellar_mass_max_iterations",3000))
    gas_peak=float(_get(config,"stellar_gas_peak_max_factor",2.0))
    cache={}
    def evaluate(theta):
        pars=unpack_nonlinear(theta)
        stellar=shift_broaden_resample_library(
            fit_library,wave,ebv=pars["ebv"],velocity_kms=pars["velocity_kms"],sigma_kms=pars["sigma_kms"],
            attenuation_law=str(_get(config,"stellar_attenuation_law","calzetti00")),resolution=resolution)
        G=np.empty((wave.size,0)); gas_names=(); gas_rest=np.empty(0)
        if include_gas:
            G,gas_names,gas_rest=build_emission_line_templates(
                wave,pars["gas_velocity_kms"],pars["gas_sigma_kms"],resolution,line_list,full_wave=wave,fit_mask=mask)
        design=np.column_stack([stellar.T,G]) if G.shape[1] else stellar.T
        coeff,model,lin=_solve_linear(design,flux,unc,mask,n_stellar=fit_library.n_models,gas_peak_max_factor=gas_peak,method=solver,tolerance=tol,max_iter=maxiter)
        cache["last"]=(stellar,G,gas_names,gas_rest,coeff,model,lin)
        return stellar,G,gas_names,gas_rest,coeff,model,lin
    progress_every=max(1,int(_get(config,"stellar_progress_every",5)))
    objective_calls={"n":0,"best_chi2":np.inf}
    def objective(theta):
        *_,model,_lin=evaluate(theta)
        resid=(flux[mask]-model[mask])/unc[mask]
        objective_calls["n"]+=1
        chi2=float(np.dot(resid,resid))
        improved=chi2 < objective_calls["best_chi2"]*(1.0-1e-10)
        if improved:
            objective_calls["best_chi2"]=chi2
        if progress and (objective_calls["n"]==1 or improved or objective_calls["n"]%progress_every==0):
            values=", ".join(f"{name}={value:.6g}" for name,value in zip(names,theta))
            print(f"[FitSpec:stellar] eval {objective_calls['n']:4d}: chi2={chi2:.7g} | {values}",flush=True)
        return resid

    jacobian=str(_get(config,"stellar_nonlinear_jacobian","2-point")).strip().lower()
    if jacobian not in {"2-point","3-point","cs"}:
        raise ValueError("stellar_nonlinear_jacobian must be '2-point', '3-point', or 'cs'.")
    nonlinear_tol=float(_get(config,"stellar_nonlinear_tolerance",1e-5))
    max_nfev=int(_get(config,"stellar_max_nfev",100))
    verbose=int(_get(config,"stellar_verbose",2))
    if progress:
        print(
            f"[FitSpec:stellar] optimizer start: nonlinear={len(names)}, SSPs={fit_library.n_models}/{library.n_models}, "
            f"gas={'on' if include_gas else 'off'}, jac={jacobian}, max_nfev={max_nfev}",
            flush=True,
        )
    opt=least_squares(objective,x0,bounds=(lo,hi),method="trf",jac=jacobian,loss="linear",
        ftol=nonlinear_tol,xtol=nonlinear_tol,gtol=nonlinear_tol,
        max_nfev=max_nfev,verbose=verbose)
    if progress:
        print(f"[FitSpec:stellar] optimizer stop: {opt.message} (nfev={opt.nfev}, cost={opt.cost:.7g})",flush=True)
    stellar,G,gas_names,gas_rest,coeff,model,lin=evaluate(opt.x)
    candidate_coeff=coeff[:fit_library.n_models]; gas_amp=coeff[fit_library.n_models:]
    stellar_coeff=np.zeros(library.n_models,float); stellar_coeff[basis_indices]=candidate_coeff
    stellar_model=stellar.T@candidate_coeff
    gas_model=G@gas_amp if G.shape[1] else np.zeros_like(stellar_model)
    residual=(flux-model); chi=float(np.sum((residual[mask]/unc[mask])**2))
    active=int(np.count_nonzero(stellar_coeff>max(1e-12,1e-8*np.nanmax(stellar_coeff) if np.any(stellar_coeff>0) else 1e-12)))
    active_gas=int(np.count_nonzero(gas_amp>0)); dof=int(np.count_nonzero(mask)-active-active_gas-len(names)); red=chi/dof if dof>0 else np.nan
    cov,perr=_parameter_covariance(opt,red)
    fitted=unpack_nonlinear(opt.x)
    light=physical_light_fractions(library,stellar_coeff,wave_range=(wave[mask].min(),wave[mask].max()))
    # Expensive correlation/rank diagnostics are evaluated only on the fitted
    # candidate basis.  Full-library single-SSP chi-square values are retained
    # for later posterior candidate selection without constructing an NxN matrix.
    diagnostics=calculate_stellar_diagnostics(stellar,flux,unc,mask,candidate_coeff,float(_get(config,"stellar_effective_rank_tolerance",0.001)))
    if screening_chi2 is None or fit_velocity or fit_sigma or abs(float(fitted["ebv"])-ebv0)>1e-12:
        final_full=shift_broaden_resample_library(
            library,wave,ebv=fitted["ebv"],velocity_kms=fitted["velocity_kms"],sigma_kms=fitted["sigma_kms"],
            attenuation_law=str(_get(config,"stellar_attenuation_law","calzetti00")),resolution=resolution)
        full_chi=_single_ssp_chi_square_fast(final_full,flux,unc,mask)
    else:
        full_chi=np.asarray(screening_chi2,float)
    diagnostics.single_ssp_chi_square=full_chi
    diagnostics.single_ssp_delta_chi_square=full_chi-np.nanmin(full_chi[np.isfinite(full_chi)]) if np.any(np.isfinite(full_chi)) else full_chi.copy()
    diagnostics.transformed_model_fluxes=stellar
    inv_factor=1.0/flux_factor
    resolution_source=None if resolution is None else getattr(resolution,"source",str(resolution))
    metadata=dict(library.metadata); metadata.update({
        "library_normalization":library.normalization,"input_redshift":float(spectrum.redshift),
        "simultaneous_gas":bool(include_gas),"fit_velocity":bool(fit_velocity),"fit_sigma":bool(fit_sigma),
        "template_selection":str(_get(config,"stellar_template_selection","candidate")).strip().lower(),
        "candidate_basis_size":int(fit_library.n_models),
        "candidate_basis_indices":",".join(map(str,basis_indices.tolist())),
        "flux_to_lsun_factor":float(flux_factor),"attenuation_law":str(_get(config,"stellar_attenuation_law","calzetti00"))})
    return StellarFitResult(
        regime=regime,library_family=library.family,wave=wave,flux=flux*inv_factor,flux_unc=unc*inv_factor,
        mask=mask,model=model*inv_factor,stellar_model=stellar_model*inv_factor,gas_model=gas_model*inv_factor,
        coefficients=stellar_coeff,ages_myr=library.ages_myr,metallicity_codes=library.metallicity_codes,metallicities_solar=library.metallicities_solar,
        ebv=fitted["ebv"],velocity_kms=fitted["velocity_kms"],sigma_kms=fitted["sigma_kms"],
        gas_velocity_kms=fitted["gas_velocity_kms"] if include_gas else 0.0,
        gas_sigma_kms=fitted["gas_sigma_kms"] if include_gas else 0.0,
        gas_names=gas_names,gas_rest_wavelengths=gas_rest,gas_amplitudes=gas_amp*inv_factor,
        chi_square=chi,reduced_chi_square=float(red),degrees_of_freedom=dof,success=bool(opt.success),message=str(opt.message),n_function_evaluations=int(opt.nfev),
        parameter_names=names,parameter_uncertainties=perr,nonlinear_covariance=cov,
        coefficient_kind="formed_mass_Msun",surviving_mass_fractions=library.surviving_mass_fractions,light_fractions=light,
        diagnostics=diagnostics,resolution_source=resolution_source,metadata=metadata,optimizer_result=opt,
    )


def evaluate_stellar_result_on_grid(result: StellarFitResult, library: StellarLibrary, target_wave,
                                    resolution=None, attenuation_law="calzetti00"):
    """Reconstruct a saved solution continuously on an arbitrary rest-frame grid."""
    wave=np.asarray(target_wave,float)
    stellar=shift_broaden_resample_library(library,wave,ebv=result.ebv,velocity_kms=result.velocity_kms,sigma_kms=result.sigma_kms,attenuation_law=attenuation_law,resolution=resolution)
    stellar_model=stellar.T@np.asarray(result.coefficients,float)
    line_list=list(zip(result.gas_names,result.gas_rest_wavelengths))
    G,_,_=build_emission_line_templates(wave,result.gas_velocity_kms,result.gas_sigma_kms,resolution,line_list,full_wave=wave,fit_mask=np.ones(wave.size,bool))
    gas_model=G@np.asarray(result.gas_amplitudes,float) if G.shape[1] else np.zeros_like(stellar_model)
    return stellar_model,gas_model,stellar_model+gas_model
