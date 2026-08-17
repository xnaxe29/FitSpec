"""Unified FitSpec stellar GUI for the v8-derived stellar engine.

The GUI never chooses a separate UV/optical fitting backend.  It previews the
selected SSP, edits the universal FitSpec mask, and calls one
``fit_stellar_spectrum`` routine.  Display smoothing is visualization-only.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, CheckButtons, Slider

from core.rebinning import compute_display_smoothing
from gui.mask_controller import MaskController
from stellar.stellar_fit_v3 import prepare_stellar_library_from_config, fit_stellar_spectrum
from stellar.stellar_models_v3 import StellarLibrary, shift_broaden_resample_library
from stellar.stellar_results_v3 import save_stellar_result, load_stellar_result

__all__=["StellarGUI"]


def _get(config,key,default=None):
    return config.get(key,default) if hasattr(config,"get") else default


def _pair(config,key,default):
    v=_get(config,key,default)
    vals=list(v) if isinstance(v,(list,tuple,np.ndarray)) else [x.strip() for x in str(v).split(",")]
    return (float(vals[0]),float(vals[1])) if len(vals)>=2 else tuple(map(float,default))


class StellarGUI:
    def __init__(self,spectrum,config,*,result_path="stellar_fit.fits",mask_path=None):
        self.spectrum=spectrum; self.config=config; self.result_path=Path(result_path)
        self.regime,self.library=prepare_stellar_library_from_config(spectrum,config)
        self.wave_rest=np.asarray(spectrum.wave,float)/(1+float(spectrum.redshift)); self.result=None

        self.fig,self.ax=plt.subplots(figsize=(13,8)); plt.subplots_adjust(left=.10,right=.78,bottom=.38)
        self.data_line,=self.ax.plot([],[],drawstyle="steps-mid",alpha=.45,label="data")
        self.masked_line,=self.ax.plot([],[],".",ms=3,alpha=.18,label="masked")
        self.preview_line,=self.ax.plot([],[],lw=1.3,label="selected SSP")
        self.best_line,=self.ax.plot([],[],lw=2,ls="--",label="best fit")
        self.gas_line,=self.ax.plot([],[],lw=1,ls=":",label="gas nuisance model")
        self.ax.set_xlabel(r"Rest-frame wavelength ($\AA$)"); self.ax.set_ylabel("Flux")
        self.ax.set_title(f"FitSpec stellar fitting — unified engine [{self.regime.upper()} / {self.library.family}]")
        self.ax.legend(fontsize=8)

        ages=np.unique(self.library.ages_myr)
        self.age_slider=Slider(plt.axes([.13,.27,.28,.025]),"Age [Myr]",ages.min(),ages.max(),valinit=ages[0],valstep=ages)
        if self.library.metallicities_dex is not None and np.any(np.isfinite(self.library.metallicities_dex)):
            mets=np.unique(self.library.metallicities_dex[np.isfinite(self.library.metallicities_dex)]); self.met_values=mets; self.met_is_numeric=True
            self.met_slider=Slider(plt.axes([.13,.23,.28,.025]),"[M/H]",mets.min(),mets.max(),valinit=mets[0],valstep=mets)
        else:
            self.met_values=np.unique(self.library.metallicity_codes.astype(str)); self.met_is_numeric=False
            self.met_slider=Slider(plt.axes([.13,.23,.28,.025]),"Metallicity",0,len(self.met_values)-1,valinit=0,valstep=1)

        eb=_pair(config,"stellar_ebv_bounds",(0,2)); vb=_pair(config,"stellar_velocity_bounds_kms",(-500,500)); sb=_pair(config,"stellar_sigma_bounds_kms",(0,1000))
        self.ebv_slider=Slider(plt.axes([.13,.19,.28,.025]),"E(B-V)",eb[0],eb[1],valinit=float(_get(config,"stellar_ebv_initial",.05)),valstep=.01)
        self.vel_slider=Slider(plt.axes([.48,.27,.23,.025]),r"$v_\star$ [km/s]",vb[0],vb[1],valinit=float(_get(config,"stellar_velocity_initial_kms",0)),valstep=5)
        self.sig_slider=Slider(plt.axes([.48,.23,.23,.025]),r"$\sigma_\star$ [km/s]",sb[0],sb[1],valinit=float(_get(config,"stellar_sigma_initial_kms",50)),valstep=5)
        self.smooth_slider=Slider(plt.axes([.48,.19,.23,.025]),"Display bin",1,100,valinit=1,valstep=1)
        self.options=CheckButtons(plt.axes([.80,.60,.17,.08]),["Include gas nuisance"],[str(_get(config,"stellar_include_gas",True)).lower() not in {"false","0","no"}])

        self.fit_button=Button(plt.axes([.80,.09,.055,.04]),"Fit"); self.fit_button.on_clicked(self._fit)
        self.load_button=Button(plt.axes([.86,.09,.055,.04]),"Load Fit"); self.load_button.on_clicked(self._load_fit)
        self.save_button=Button(plt.axes([.92,.09,.055,.04]),"Save Fit"); self.save_button.on_clicked(self._save_fit)

        self.mask_controller=MaskController(
            spectrum,fit_mode="stellar",included_intervals=_get(config,"included_intervals",None),
            excluded_intervals=_get(config,"excluded_intervals",None),mask_path=mask_path,
            on_change=lambda _mask,_reason:self._refresh_data(),
        )
        self.mask_controller.connect_matplotlib(
            self.ax,selector_check_axes=plt.axes([.80,.51,.17,.055]),mode_axes=plt.axes([.80,.39,.17,.10]),
            save_button_axes=plt.axes([.80,.31,.075,.04]),load_button_axes=plt.axes([.89,.31,.075,.04]),
            reset_button_axes=plt.axes([.80,.25,.165,.04]),status_axes=plt.axes([.80,.18,.17,.05]),
        )
        for widget in (self.age_slider,self.met_slider,self.ebv_slider,self.vel_slider,self.sig_slider): widget.on_changed(self._preview)
        self.smooth_slider.on_changed(self._refresh_data); self.options.on_clicked(self._toggle_gas)
        self._refresh_data(); self._preview()

    def _toggle_gas(self,*_):
        # Runtime GUI choice overrides the config value for the current fit.
        self.config.values["stellar_include_gas"]=bool(self.options.get_status()[0]) if hasattr(self.config,"values") else bool(self.options.get_status()[0])

    def _selected_index(self):
        age=float(self.age_slider.val); am=np.isclose(self.library.ages_myr,age,atol=1e-8)
        if self.met_is_numeric:
            hit=np.flatnonzero(am&np.isclose(self.library.metallicities_dex,float(self.met_slider.val),atol=1e-8))
        else:
            code=str(self.met_values[int(round(self.met_slider.val))]); hit=np.flatnonzero(am&(self.library.metallicity_codes.astype(str)==code))
        candidates=np.flatnonzero(am)
        return int(hit[0]) if hit.size else int(candidates[0])

    def _refresh_data(self,*_):
        bins=max(1,int(round(self.smooth_slider.val))) if hasattr(self,"smooth_slider") else 1
        if bins>1 and self.spectrum.flux_unc is not None:
            w,f,_=compute_display_smoothing(self.spectrum,bins,min_coverage=.5); w=w/(1+float(self.spectrum.redshift))
        else:
            w=self.wave_rest; f=np.asarray(self.spectrum.flux,float)
        self.data_line.set_data(w,f)
        mask=np.ones(self.wave_rest.size,bool) if self.spectrum.mask is None else np.asarray(self.spectrum.mask,bool)
        self.masked_line.set_data(self.wave_rest[~mask],np.asarray(self.spectrum.flux,float)[~mask])
        self.ax.relim(); self.ax.autoscale_view(); self.fig.canvas.draw_idle()

    def _preview(self,*_):
        i=self._selected_index()
        lib=StellarLibrary(
            wave=self.library.wave,flux_per_mass=self.library.flux_per_mass[i:i+1],flux_reference=self.library.flux_reference[i:i+1],
            ages_myr=self.library.ages_myr[i:i+1],metallicity_codes=self.library.metallicity_codes[i:i+1],metallicities_solar=self.library.metallicities_solar[i:i+1],
            reference_masses=self.library.reference_masses[i:i+1],labels=(self.library.labels[i],),source_paths=(self.library.source_paths[i],),family=self.library.family,
            model_resolution_R=self.library.model_resolution_R,model_fwhm_angstrom=self.library.model_fwhm_angstrom,
            surviving_mass_fractions=None if self.library.surviving_mass_fractions is None else self.library.surviving_mass_fractions[i:i+1],
            metallicities_dex=None if self.library.metallicities_dex is None else self.library.metallicities_dex[i:i+1],normalization=self.library.normalization,metadata=self.library.metadata,
        )
        transformed=shift_broaden_resample_library(lib,self.wave_rest,ebv=float(self.ebv_slider.val),velocity_kms=float(self.vel_slider.val),sigma_kms=float(self.sig_slider.val),attenuation_law=str(_get(self.config,"stellar_attenuation_law","calzetti00")),resolution=getattr(self.spectrum,"resolution",None))[0]
        data=np.asarray(self.spectrum.flux,float); good=np.isfinite(data)&np.isfinite(transformed)&(np.abs(transformed)>0)
        scale=np.nanmedian(data[good])/np.nanmedian(transformed[good]) if np.any(good) else 1.0
        self.preview_line.set_data(self.wave_rest,transformed*scale); self.fig.canvas.draw_idle()

    def _fit(self,*_):
        self._toggle_gas(); self.result=fit_stellar_spectrum(self.spectrum,self.config,library=self.library,regime=self.regime)
        self.best_line.set_data(self.result.wave,self.result.model); self.gas_line.set_data(self.result.wave,self.result.gas_model)
        self.fig.canvas.draw_idle(); print(f"stellar fit: chi2nu={self.result.reduced_chi_square:.5g}, E(B-V)={self.result.ebv:.4g}, v={self.result.velocity_kms:.3f}, sigma={self.result.sigma_kms:.3f}")

    def _save_fit(self,*_):
        if self.result is None: self._fit()
        path=save_stellar_result(self.result_path,self.result,overwrite=True); print(f"Saved {path}")

    def _load_fit(self,*_):
        self.result=load_stellar_result(self.result_path); self.best_line.set_data(self.result.wave,self.result.model); self.gas_line.set_data(self.result.wave,self.result.gas_model); self.fig.canvas.draw_idle(); print(f"Loaded {self.result_path}")

    def show(self):
        plt.show()
