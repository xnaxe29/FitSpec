"""Unified FitSpec stellar-population library interface and template transforms.

This module is the only stellar-library interface used by FitSpec.  Native
BPASS, pySB99, SLUG and E-MILES products are expected to have been translated
into the project HDF5 schemas before fitting.  The fitting code therefore never
needs to understand the original distribution files.

Instrumental resolution is observation-side state.  HDF5 spectra are kept in
native form and are convolved only when a model is evaluated against a
Spectrum.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import warnings

import h5py
import numpy as np
from astropy.constants import L_sun, c

from core.resolution import ResolutionModel, convolve_variable_gaussian

__all__ = [
    "Z_SUN", "UV_OPTICAL_BOUNDARY_ANGSTROM", "DEFAULT_H5_PATHS",
    "StellarLibrary", "classify_spectral_regime", "load_stellar_library",
    "load_bpass_library", "load_sb99_library", "load_slug_library",
    "load_emiles_library", "attenuation_transmission",
    "shift_broaden_resample_library", "physical_light_fractions",
]

Z_SUN = 0.014
C_KMS = c.to("km/s").value
L_SUN_ERG_S = L_sun.cgs.value
UV_OPTICAL_BOUNDARY_ANGSTROM = 3000.0

# These reproduce the user's current local library locations but every path is
# overridable from default_config_stellar.dat/config.dat/CLI.
DEFAULT_H5_PATHS = {
    "bpass": Path("/Users/aranjan/Documents/software/stellar_libraries/BPASS/BPASS_v2.2.1_population_library.h5"),
    "emiles": Path("/Users/aranjan/Documents/software/stellar_libraries/MILES/EMILES_population_library.h5"),
    "sb99": Path("/Users/aranjan/Documents/software/stellar_libraries/pySB99/SB99_population_library.h5"),
    "slug": Path("/Users/aranjan/Documents/software/stellar_libraries/SLUG/SLUG_population_library.h5"),
}


@dataclass(frozen=True)
class StellarLibrary:
    """Common in-memory view of only the SSPs needed for the current fit."""
    wave: np.ndarray
    flux_per_mass: np.ndarray
    flux_reference: np.ndarray
    ages_myr: np.ndarray
    metallicity_codes: np.ndarray
    metallicities_solar: np.ndarray
    reference_masses: np.ndarray
    labels: tuple[str, ...]
    source_paths: tuple[str, ...]
    family: str
    model_resolution_R: float | None = None
    model_fwhm_angstrom: float | None = None
    surviving_mass_fractions: np.ndarray | None = None
    metallicities_dex: np.ndarray | None = None
    normalization: str = "formed_mass_Msun"
    metadata: dict = field(default_factory=dict)

    @property
    def n_models(self) -> int:
        return int(self.flux_per_mass.shape[0])

    @property
    def n_wave(self) -> int:
        return int(self.wave.size)

    @property
    def library_metadata(self) -> dict:
        """Compatibility alias for branch/source metadata across libraries.

        FitSpec historically exposed stellar-library branch information as
        ``library_metadata``.  The unified ``StellarLibrary`` stores the same
        information in ``metadata``; this property provides one consistent
        public interface for BPASS, SB99, SLUG, and E-MILES without duplicating
        state.
        """
        return self.metadata


def classify_spectral_regime(wave_rest, boundary_angstrom=UV_OPTICAL_BOUNDARY_ANGSTROM) -> str:
    """Choose UV or optical from rest-frame coverage; no mixed mode yet."""
    wave = np.asarray(wave_rest, dtype=float)
    wave = wave[np.isfinite(wave)]
    if wave.size < 2:
        raise ValueError("At least two finite rest-frame wavelengths are required.")
    lo, hi = float(wave.min()), float(wave.max())
    boundary = float(boundary_angstrom)
    if hi <= boundary:
        return "uv"
    if lo >= boundary:
        return "optical"
    uv_span = boundary - lo
    optical_span = hi - boundary
    return "uv" if uv_span >= optical_span else "optical"


def _decode(value):
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode(errors="replace")
    return value.item() if isinstance(value, np.ndarray) and value.shape == () else value


def _freeze(*arrays):
    for arr in arrays:
        if isinstance(arr, np.ndarray):
            arr.setflags(write=False)


def _safe_log10(values):
    values = np.asarray(values, dtype=float)
    out = np.full(values.shape, np.nan)
    good = np.isfinite(values) & (values > 0)
    out[good] = np.log10(values[good])
    return out


def _wave_slice(wave, wave_range=None, padding_angstrom=100.0):
    wave = np.asarray(wave, dtype=float)
    if wave_range is None:
        return slice(None), wave.copy()
    lo, hi = map(float, wave_range)
    keep = np.flatnonzero((wave >= lo-padding_angstrom) & (wave <= hi+padding_angstrom))
    if keep.size < 2:
        raise ValueError(
            f"No model wavelengths overlap {wave_range}; library covers "
            f"{wave.min():.1f}-{wave.max():.1f} A."
        )
    sl = slice(int(keep[0]), int(keep[-1])+1)
    return sl, wave[sl].copy()


def _nearest_age_indices(available, requested=None, age_range_myr=None):
    available = np.asarray(available, dtype=float)
    allowed = np.ones(available.size, dtype=bool)
    if age_range_myr is not None:
        lo, hi = map(float, age_range_myr)
        allowed &= (available >= lo) & (available <= hi)
    indices = np.flatnonzero(allowed)
    if indices.size == 0:
        raise ValueError("Age selection leaves no SSPs.")
    if requested is None or len(requested) == 0:
        return indices
    picked = [int(indices[np.argmin(np.abs(available[indices]-float(age)))]) for age in requested]
    return np.asarray(sorted(set(picked)), dtype=int)


def _metal_code_to_absolute_z(code):
    text = str(code).strip().lower()
    if text in {"zem5": "1e-5", "zem4": "1e-4"}:
        text = {"zem5": "1e-5", "zem4": "1e-4"}[text]
    if text.startswith("z"):
        text = text[1:]
    try:
        return float(int(text))/1000.0 if text.isdigit() else float(text)
    except ValueError:
        return np.nan


def _match_codes(available, requested):
    available = [str(x) for x in available]
    if requested is None or len(requested) == 0:
        return available
    out = []
    for request in map(str, requested):
        request = request.strip()
        if request in available:
            out.append(request); continue
        target = _metal_code_to_absolute_z(request)
        candidates = np.asarray([_metal_code_to_absolute_z(x) for x in available])
        hit = np.flatnonzero(np.isfinite(candidates) & np.isclose(candidates, target, rtol=0, atol=max(1e-8, abs(target)*1e-6)))
        if not hit.size:
            raise ValueError(f"Requested metallicity {request!r} unavailable; available={available}")
        out.append(available[int(hit[0])])
    return list(dict.fromkeys(out))


def _finish_library(*, wave, flux_reference, ages, codes, zsolar, refs, labels,
                    source_paths, family, metadata, normalization="formed_mass_Msun",
                    flux_per_mass=None, surviving=None, mdex=None,
                    model_resolution_R=None, model_fwhm_angstrom=None):
    wave = np.asarray(wave, float); flux_reference = np.asarray(flux_reference, float)
    ages = np.asarray(ages, float); codes = np.asarray(codes, str)
    zsolar = np.asarray(zsolar, float); refs = np.asarray(refs, float)
    if flux_per_mass is None:
        flux_per_mass = flux_reference / refs[:, None]
    flux_per_mass = np.asarray(flux_per_mass, float)
    surviving = None if surviving is None else np.asarray(surviving, float)
    mdex = _safe_log10(zsolar) if mdex is None else np.asarray(mdex, float)
    if flux_reference.shape != flux_per_mass.shape or flux_per_mass.shape != (ages.size, wave.size):
        raise ValueError("Stellar-library arrays have inconsistent dimensions.")
    _freeze(wave, flux_reference, flux_per_mass, ages, codes, zsolar, refs, surviving, mdex)
    return StellarLibrary(
        wave, flux_per_mass, flux_reference, ages, codes, zsolar, refs,
        tuple(labels), tuple(source_paths), family,
        model_resolution_R=model_resolution_R,
        model_fwhm_angstrom=model_fwhm_angstrom,
        surviving_mass_fractions=surviving, metallicities_dex=mdex,
        normalization=normalization, metadata=dict(metadata),
    )


def load_bpass_library(filename, *, imf="imf135_300", population="binary",
                       metallicity_codes=None, ages_myr=None, age_range_myr=None,
                       wave_range=None, reference_mass_msun=1e6,
                       intrinsic_model_resolution_R=None):
    path = Path(filename).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"BPASS HDF5 library not found: {path}")
    population = {"bin":"binary", "binary":"binary", "sin":"single", "single":"single"}.get(str(population).lower())
    if population is None:
        raise ValueError("BPASS population must be binary or single.")
    with h5py.File(path, "r") as h5:
        if not all(x in h5 for x in ("axes/wavelength_A", "axes/age_Myr", "models")):
            raise ValueError("File does not match the FitSpec BPASS HDF5 schema.")
        imfs = list(h5["models"].keys())
        if imf not in imfs:
            fallback = "imf135_300" if "imf135_300" in imfs else imfs[0]
            warnings.warn(f"BPASS IMF {imf!r} unavailable; using {fallback!r}.")
            imf = fallback
        root_name = f"models/{imf}/{population}"
        root = h5[root_name]
        zsel = _match_codes(list(root.keys()), metallicity_codes)
        all_age = np.asarray(h5["axes/age_Myr"][:], float)
        aidx = _nearest_age_indices(all_age, ages_myr, age_range_myr)
        wsl, wave = _wave_slice(np.asarray(h5["axes/wavelength_A"][:], float), wave_range)
        rows=[]; ages=[]; codes=[]; zsolar=[]; refs=[]; labels=[]; paths=[]
        for code in zsel:
            group = root[code]
            spectra = np.asarray(group["flux"][aidx, wsl], float)
            zabs = float(group.attrs.get("metallicity_Z", _metal_code_to_absolute_z(code)))
            for k, ai in enumerate(aidx):
                rows.append(spectra[k]); ages.append(all_age[ai]); codes.append(code)
                zsolar.append(zabs/Z_SUN); refs.append(float(reference_mass_msun))
                labels.append(f"BPASS/{imf}/{population}/{code}/age={all_age[ai]:g}Myr")
                paths.append(f"{path}::/{root_name}/{code}")
    return _finish_library(
        wave=wave, flux_reference=rows, ages=ages, codes=codes, zsolar=zsolar,
        refs=refs, labels=labels, source_paths=paths, family="bpass",
        metadata={"imf":imf, "population":population, "source":str(path)},
        model_resolution_R=intrinsic_model_resolution_R,
    )


def load_sb99_library(filename, *, track="GENEC", rotation="nonrot", spectra_library="PoWR",
                      metallicity_labels=None, ages_myr=None, age_range_myr=None,
                      wave_range=None, intrinsic_model_resolution_R=None):
    path = Path(filename).expanduser().resolve()
    if not path.is_file(): raise FileNotFoundError(f"pySB99 HDF5 library not found: {path}")
    root_name = f"models/{track}/{rotation}/{spectra_library}"
    with h5py.File(path, "r") as h5:
        if root_name not in h5: raise ValueError(f"pySB99 branch /{root_name} not found.")
        root = h5[root_name]; available = list(root.keys())
        selected = available if not metallicity_labels else [str(x) for x in metallicity_labels]
        missing = [x for x in selected if x not in available]
        if missing: raise ValueError(f"pySB99 metallicities unavailable: {missing}; available={available}")
        rows=[]; ages=[]; codes=[]; zsolar=[]; refs=[]; labels=[]; paths=[]; wave_ref=None
        for met in selected:
            grp=root[met]; wsl,wave=_wave_slice(np.asarray(grp["wavelength_A"][:],float),wave_range)
            if wave_ref is None: wave_ref=wave
            elif wave.shape != wave_ref.shape or not np.allclose(wave,wave_ref,rtol=1e-10,atol=1e-8):
                raise ValueError("Selected pySB99 groups do not share a common wavelength grid.")
            av=np.asarray(grp["age_Myr"][:],float); idx=_nearest_age_indices(av,ages_myr,age_range_myr)
            spec=np.asarray(grp["flux"][idx,wsl],float)
            ref=float(grp.attrs.get("reference_mass_Msun",grp["flux"].attrs.get("reference_mass_Msun",1e6)))
            zabs=float(grp.attrs.get("metallicity_Z",np.nan))
            for k,ai in enumerate(idx):
                rows.append(spec[k]); ages.append(av[ai]); codes.append(met); zsolar.append(zabs/Z_SUN if np.isfinite(zabs) else np.nan); refs.append(ref)
                labels.append(f"pySB99/{track}/{rotation}/{spectra_library}/{met}/age={av[ai]:g}Myr")
                paths.append(f"{path}::/{root_name}/{met}")
    flux_native=np.asarray(rows,float); refs_arr=np.asarray(refs,float)

    # pySB99 high-resolution population spectra are not written directly in
    # physical erg s^-1 A^-1.  In specsyn_hires(), the summed population flux
    # carries an explicit /1e20 scaling, while the saved logarithmic product is
    # log10(population_flux + tiny) + 20.  The FitSpec HDF5 builder inverts the
    # logarithm back to the native *linear* pySB99 array, so an additional
    # factor of 1e20 is required to recover the physical luminosity density.
    #
    # Detect this from the provenance metadata written by the builder rather
    # than exposing a user-facing unit switch.  Future HDF5 builders may write
    # an explicit physical_flux_scale_to_erg_s_A attribute; prefer it when
    # present.
    physical_scales=[]
    representations=[]
    with h5py.File(path, "r") as h5:
        root=h5[root_name]
        for met in selected:
            grp=root[met]
            dflux=grp["flux"]
            explicit = dflux.attrs.get(
                "physical_flux_scale_to_erg_s_A",
                grp.attrs.get("physical_flux_scale_to_erg_s_A", None),
            )
            representation=str(grp.attrs.get("flux_representation", "")).strip()
            saved_transform=str(grp.attrs.get("source_saved_transform", "")).strip()
            representations.append(representation)
            if explicit is not None:
                scale=float(explicit)
            elif (
                "pysb99" in representation.lower()
                and "+ 20" in saved_transform.replace("+20", "+ 20")
            ):
                scale=1.0e20
            else:
                raise ValueError(
                    "Cannot determine the physical normalization of the pySB99 "
                    f"HDF5 group {grp.name!r}. The library must contain either "
                    "physical_flux_scale_to_erg_s_A metadata or the FitSpec "
                    "pySB99 builder provenance attributes flux_representation and "
                    "source_saved_transform. Rebuild the SB99 HDF5 library with "
                    "the current builder rather than supplying a guessed unit."
                )
            if not np.isfinite(scale) or scale <= 0.0:
                raise ValueError(
                    f"Invalid pySB99 physical flux scale {scale!r} in {grp.name}."
                )
            physical_scales.append(scale)

    physical_scales=np.asarray(physical_scales,float)
    if not np.allclose(physical_scales, physical_scales[0], rtol=0.0, atol=0.0):
        raise ValueError(
            "Selected pySB99 metallicity groups use inconsistent physical flux "
            f"scales: {physical_scales.tolist()}"
        )
    physical_scale=float(physical_scales[0])
    flux_ref=flux_native * physical_scale
    flux_pm=flux_ref/(refs_arr[:,None]*L_SUN_ERG_S)

    print(
        "[FitSpec:stellar] pySB99 normalization: "
        f"native linear flux x {physical_scale:.1e} -> erg/s/A; "
        "then divided by reference mass and L_sun"
    )
    return _finish_library(
        wave=wave_ref, flux_reference=flux_ref, flux_per_mass=flux_pm, ages=ages,
        codes=codes, zsolar=zsolar, refs=refs_arr, labels=labels, source_paths=paths,
        family="sb99", metadata={
            "track":track,"rotation":rotation,"spectra_library":spectra_library,
            "source":str(path),"native_flux_representation":"pySB99 linear high-resolution population flux",
            "physical_flux_scale_to_erg_s_A":physical_scale,
            "physical_flux_unit":"erg/s/A",
        },
        model_resolution_R=intrinsic_model_resolution_R,
    )


def _slug_leaves(root):
    leaves=[]
    def visitor(_name,obj):
        if isinstance(obj,h5py.Group) and all(x in obj for x in ("flux","age_Myr","wavelength_A")):
            leaves.append(obj.name)
    root.visititems(visitor)
    return leaves


def _slug_deterministic_branches(h5):
    """Return available (family, subfamily, specsyn) deterministic SLUG branches."""
    base="models/deterministic"
    if base not in h5:
        return []
    leaves=_slug_leaves(h5[base])
    branches=set()
    base_parts=Path('/'+base).parts
    for leaf in leaves:
        parts=Path(leaf).parts
        # h5py object names are absolute, e.g.
        # /models/deterministic/MIST/foo/bar/Z020
        try:
            i=parts.index('deterministic')
        except ValueError:
            continue
        rel=parts[i+1:]
        if len(rel) >= 3:
            branches.add(tuple(rel[:3]))
    return sorted(branches)


def _resolve_slug_branch(h5, family, subfamily, specsyn):
    available=_slug_deterministic_branches(h5)
    if not available:
        raise ValueError("No deterministic SLUG branches containing flux, age_Myr, and wavelength_A were found under /models/deterministic.")

    requested=(str(family).strip(), str(subfamily).strip(), str(specsyn).strip())
    auto=lambda x: x.lower() in {'', 'auto', 'default', '*'}
    matches=[]
    for branch in available:
        if ((auto(requested[0]) or branch[0] == requested[0]) and
            (auto(requested[1]) or branch[1] == requested[1]) and
            (auto(requested[2]) or branch[2] == requested[2])):
            matches.append(branch)

    if not matches:
        formatted='\n  '.join('/models/deterministic/'+'/'.join(x) for x in available)
        raise ValueError(
            "Requested SLUG deterministic branch does not exist: "
            f"/models/deterministic/{requested[0]}/{requested[1]}/{requested[2]}\n"
            "Available deterministic branches are:\n  "+formatted
        )

    chosen=matches[0]
    if any(auto(x) for x in requested):
        print('[FitSpec:stellar] SLUG auto-selected branch: '
              '/models/deterministic/' + '/'.join(chosen))
        if len(matches) > 1:
            print(f'[FitSpec:stellar] SLUG auto-selection matched {len(matches)} branches; using the first sorted match. Set slug_family/slug_subfamily/slug_specsyn explicitly to choose another.')
    return chosen


def load_slug_library(filename, *, family="auto", subfamily="auto", specsyn="auto",
                      metallicity_labels=None, ages_myr=None, age_range_myr=None,
                      wave_range=None, intrinsic_model_resolution_R=None):
    path=Path(filename).expanduser().resolve()
    if not path.is_file(): raise FileNotFoundError(f"SLUG HDF5 library not found: {path}")
    with h5py.File(path,"r") as h5:
        family, subfamily, specsyn = _resolve_slug_branch(h5, family, subfamily, specsyn)
        prefix=f"models/deterministic/{family}/{subfamily}/{specsyn}"
        leaves=_slug_leaves(h5[prefix])
        if metallicity_labels:
            wanted=set(map(str,metallicity_labels)); leaves=[p for p in leaves if Path(p).name in wanted]
        if not leaves: raise ValueError(f"No deterministic SLUG spectra selected under /{prefix}.")
        rows=[]; ages=[]; codes=[]; zsolar=[]; refs=[]; labels=[]; paths=[]; wave_ref=None
        for leaf in leaves:
            grp=h5[leaf]; wsl,wave=_wave_slice(np.asarray(grp["wavelength_A"][:],float),wave_range)
            if wave_ref is None: wave_ref=wave
            elif wave.shape != wave_ref.shape or not np.allclose(wave,wave_ref,rtol=1e-10,atol=1e-8):
                raise ValueError("Selected SLUG groups do not share a common wavelength grid.")
            av=np.asarray(grp["age_Myr"][:],float); idx=_nearest_age_indices(av,ages_myr,age_range_myr)
            spec=np.asarray(grp["flux"][idx,wsl],float); ref=float(grp.attrs.get("cluster_mass_Msun",1.0))
            code=Path(leaf).name
            if "metallicity_Zsun" in grp.attrs: zsun=float(grp.attrs["metallicity_Zsun"])
            elif "absolute_Z" in grp.attrs: zsun=float(grp.attrs["absolute_Z"])/Z_SUN
            elif "FeH" in grp.attrs: zsun=10**float(grp.attrs["FeH"])
            else: zsun=np.nan
            for k,ai in enumerate(idx):
                rows.append(spec[k]); ages.append(av[ai]); codes.append(code); zsolar.append(zsun); refs.append(ref)
                labels.append(f"SLUG/{family}/{subfamily}/{specsyn}/{code}/age={av[ai]:g}Myr"); paths.append(f"{path}::/{leaf}")
    flux_ref=np.asarray(rows,float); refs_arr=np.asarray(refs,float)
    flux_pm=flux_ref/(refs_arr[:,None]*L_SUN_ERG_S)
    return _finish_library(
        wave=wave_ref, flux_reference=flux_ref, flux_per_mass=flux_pm, ages=ages,
        codes=codes,zsolar=zsolar,refs=refs_arr,labels=labels,source_paths=paths,
        family="slug",metadata={"population_mode":"deterministic","family":family,"subfamily":subfamily,"specsyn":specsyn,"source":str(path)},
        model_resolution_R=intrinsic_model_resolution_R,
    )


def load_emiles_library(filename, *, isochrone="Padova00", imf="ku", imf_slope=None,
                        age_range_gyr=None, metal_range_dex=None, wave_range=None,
                        intrinsic_model_resolution_R=None, intrinsic_fwhm_angstrom=2.51):
    path=Path(filename).expanduser().resolve()
    if not path.is_file(): raise FileNotFoundError(f"E-MILES HDF5 library not found: {path}")
    with h5py.File(path,"r") as h5:
        if "wavelength_A" not in h5 or "models" not in h5: raise ValueError("File does not match the FitSpec E-MILES HDF5 schema.")
        if isochrone not in h5["models"]: raise ValueError(f"E-MILES isochrone {isochrone!r} unavailable.")
        iso=h5["models"][isochrone]
        if imf not in iso: raise ValueError(f"E-MILES IMF {imf!r} unavailable; available={list(iso.keys())}")
        branch=iso[imf]
        slopes=[k for k in branch.keys() if isinstance(branch[k],h5py.Group) and str(k).startswith("slope_")]
        if slopes:
            if imf_slope is None and len(slopes)!=1: raise ValueError(f"E-MILES IMF {imf} has multiple slopes {slopes}; specify imf_slope.")
            target=slopes[0] if imf_slope is None else min(slopes,key=lambda k:abs(float(branch[k].attrs.get("imf_slope",np.inf))-float(imf_slope)))
            branch=branch[target]
        wsl,wave=_wave_slice(np.asarray(h5["wavelength_A"][:],float),wave_range)
        groups=[]
        for name,obj in branch.items():
            if isinstance(obj,h5py.Group) and "flux" in obj and "age_Gyr" in obj:
                met=float(obj.attrs.get("metallicity_dex",np.nan))
                if metal_range_dex is None or float(metal_range_dex[0]) <= met <= float(metal_range_dex[1]): groups.append((met,name,obj))
        groups.sort(key=lambda x:x[0])
        if not groups: raise ValueError("E-MILES metallicity selection leaves no SSPs.")
        rows=[]; ages=[]; codes=[]; zsolar=[]; refs=[]; surviving=[]; mdex=[]; labels=[]; paths=[]
        for met,_name,grp in groups:
            av=np.asarray(grp["age_Gyr"][:],float)
            age_range_myr=None if age_range_gyr is None else (1000*float(age_range_gyr[0]),1000*float(age_range_gyr[1]))
            idx=_nearest_age_indices(av*1000,None,age_range_myr)
            spec=np.asarray(grp["flux"][idx,wsl],float)
            if "Mass_star_remn" in grp: sf=np.asarray(grp["Mass_star_remn"][idx],float)
            elif "Mass_total" in grp: sf=np.asarray(grp["Mass_total"][idx],float)
            else: sf=np.ones(idx.size)
            for k,ai in enumerate(idx):
                rows.append(spec[k]); ages.append(av[ai]*1000); codes.append(f"{met:+.4f}"); zsolar.append(10**met); refs.append(1.0); surviving.append(sf[k]); mdex.append(met)
                labels.append(f"EMILES/{isochrone}/{imf}/[M/H]={met:+.4f}/age={av[ai]:g}Gyr"); paths.append(f"{path}::/{grp.name}")
    return _finish_library(
        wave=wave,flux_reference=rows,flux_per_mass=np.asarray(rows,float),ages=ages,codes=codes,zsolar=zsolar,refs=refs,
        labels=labels,source_paths=paths,family="emiles",surviving=surviving,mdex=mdex,
        normalization="native_initial_ssp", metadata={"isochrone":isochrone,"imf":imf,"source":str(path)},
        model_resolution_R=intrinsic_model_resolution_R,
        model_fwhm_angstrom=(None if intrinsic_model_resolution_R is not None else intrinsic_fwhm_angstrom),
    )


def load_stellar_library(family, filename=None, *, wave_range=None, **kwargs):
    """Unified lazy HDF5 entry point."""
    aliases={"miles":"emiles","e-miles":"emiles","pysb99":"sb99","starburst99":"sb99"}
    family=aliases.get(str(family).strip().lower(),str(family).strip().lower())
    if family not in DEFAULT_H5_PATHS: raise ValueError("family must be bpass, sb99, slug, or emiles.")
    path=DEFAULT_H5_PATHS[family] if filename is None else Path(filename)
    func={"bpass":load_bpass_library,"sb99":load_sb99_library,"slug":load_slug_library,"emiles":load_emiles_library}[family]
    # ``load_slug_library`` itself has a keyword named ``family`` describing
    # the SLUG evolutionary-track family (e.g. MIST).  The unified wrapper
    # already uses ``family`` for the top-level library selector, so expose
    # the nested SLUG selector as ``slug_family`` to avoid passing the same
    # Python argument name twice.
    if family == "slug" and "slug_family" in kwargs:
        kwargs = dict(kwargs)
        kwargs["family"] = kwargs.pop("slug_family")
    return func(path,wave_range=wave_range,**kwargs)


def attenuation_transmission(wave_angstrom, ebv, law="calzetti00", rv=4.05):
    """Internal stellar attenuation; Calzetti is dependency-free default."""
    wave=np.asarray(wave_angstrom,float); ebv=float(ebv)
    if ebv < 0 or not np.isfinite(ebv): raise ValueError("ebv must be finite and non-negative.")
    if ebv == 0: return np.ones_like(wave)
    name=str(law).strip().lower()
    if name not in {"calzetti00","calzetti"}:
        # Optional dust_extinction dependency for the non-default laws.
        import astropy.units as u
        from dust_extinction.parameter_averages import CCM89, F99, G23
        from dust_extinction.averages import G03_LMCAvg, G03_SMCBar
        mapping={"cardelli89":CCM89(Rv=3.1),"fitzpatrick99":F99(Rv=3.1),"gordon23_mw":G23(Rv=3.1),"gordon03_lmc":G03_LMCAvg(),"gordon03_smc":G03_SMCBar()}
        if name not in mapping: raise ValueError(f"Unsupported attenuation law {law!r}.")
        return np.asarray(mapping[name].extinguish(wave*u.AA,Ebv=ebv),float)
    micron=np.clip(wave/1e4,0.12,2.2); inv=1/micron; k=np.empty_like(micron)
    uv=micron<0.63
    k[uv]=2.659*(-2.156+1.509*inv[uv]-0.198*inv[uv]**2+0.011*inv[uv]**3)+rv
    k[~uv]=2.659*(-1.857+1.04*inv[~uv])+rv
    return 10**(-0.4*ebv*k)


def _doppler_factor(velocity_kms):
    beta=float(velocity_kms)/C_KMS
    if abs(beta)>=1: raise ValueError("velocity magnitude must be below c.")
    return np.sqrt((1+beta)/(1-beta))


def shift_broaden_resample_library(library: StellarLibrary, target_wave, *, ebv=0.0,
                                   velocity_kms=0.0, sigma_kms=0.0,
                                   attenuation_law="calzetti00",
                                   resolution: ResolutionModel | None=None):
    """Transform all SSPs onto the observed/rest-frame target wavelength grid.

    Instrument resolution and stellar velocity dispersion are treated as
    independent Gaussian broadenings. The native template LSF is removed in
    quadrature only when explicit library metadata supplies it. Wavelength
    sampling is never treated as an intrinsic spectral resolution.
    """
    target=np.asarray(target_wave,float)
    if target.ndim != 1 or np.any(np.diff(target)<=0): raise ValueError("target_wave must be increasing 1-D.")
    shifted_wave=np.asarray(library.wave,float)*_doppler_factor(velocity_kms)
    transmission=attenuation_transmission(shifted_wave,ebv,attenuation_law)
    flux=np.asarray(library.flux_per_mass,float)*transmission[None,:]

    # Stellar sigma in Angstrom at each target point.
    sigma_star_A=target*max(0.0,float(sigma_kms))/C_KMS
    if resolution is None:
        sigma_target=np.asarray(sigma_star_A,float)
    else:
        sigma_target=np.hypot(sigma_star_A,resolution.sigma_angstrom(target))

    # Remove an explicitly known intrinsic template Gaussian width in quadrature.
    # Never infer an LSF from wavelength sampling.
    if library.model_fwhm_angstrom is not None:
        sigma_native=np.full(target.shape,float(library.model_fwhm_angstrom)/2.354820045)
    elif library.model_resolution_R is not None and np.isfinite(library.model_resolution_R) and library.model_resolution_R>0:
        sigma_native=(target/float(library.model_resolution_R))/2.354820045
    else:
        sigma_native=np.zeros_like(target)
    unresolved = sigma_native > sigma_target + 1e-12
    if np.any(unresolved):
        warnings.warn(
            "The requested/data resolution is finer than the intrinsic stellar-template "
            "resolution over part of the fitted wavelength range. FitSpec cannot deconvolve "
            "or sharpen the templates; the additional convolution is set to zero there.",
            RuntimeWarning, stacklevel=2,
        )
    kernel_sigma=np.sqrt(np.clip(sigma_target**2-sigma_native**2,0,None))

    transformed=np.empty((library.n_models,target.size),float)
    for i,row in enumerate(flux):
        transformed[i]=convolve_variable_gaussian(shifted_wave,row,target,kernel_sigma)
    return transformed


def physical_light_fractions(library: StellarLibrary, coefficients, *, wave_range=None):
    coeff=np.clip(np.asarray(coefficients,float),0,None)
    if coeff.shape != (library.n_models,): raise ValueError("coefficient length mismatch.")
    use=np.isfinite(library.wave)
    if wave_range is not None: use &= (library.wave>=wave_range[0])&(library.wave<=wave_range[1])
    if np.count_nonzero(use)<2: return np.zeros_like(coeff)
    lum=np.trapezoid(np.clip(library.flux_per_mass[:,use],0,None),library.wave[use],axis=1)*coeff
    total=np.nansum(lum)
    return lum/total if np.isfinite(total) and total>0 else np.zeros_like(coeff)
