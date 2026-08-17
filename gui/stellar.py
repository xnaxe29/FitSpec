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
from gui.axis_limits import compute_sensible_limits
from gui.mask_controller import MaskController
from stellar.stellar_fit import prepare_stellar_library_from_config, fit_stellar_spectrum
from stellar.stellar_models import StellarLibrary, shift_broaden_resample_library
from stellar.stellar_results import save_stellar_result, load_stellar_result
from stellar.inference import run_stellar_inference, stellar_population_samples
from core.results import PosteriorResult
from plotting.diagnostics import plot_posterior_corner
from plotting.stellar import plot_stellar_fit, save_stellar_plot_products

__all__=["StellarGUI"]


def _get(config,key,default=None):
    return config.get(key,default) if hasattr(config,"get") else default


def _pair(config,key,default):
    v=_get(config,key,default)
    vals=list(v) if isinstance(v,(list,tuple,np.ndarray)) else [x.strip() for x in str(v).split(",")]
    return (float(vals[0]),float(vals[1])) if len(vals)>=2 else tuple(map(float,default))


class StellarGUI:
    def _log(self, message):
        print(f"[FitSpec:stellar] {message}", flush=True)

    def __init__(self,spectrum,config,*,result_path="stellar_fit.fits",mask_path=None,state=None):
        self.spectrum=spectrum; self.config=config; self.result_path=Path(result_path); self.state=state
        self._log("opening stellar panel")
        self.regime,self.library=prepare_stellar_library_from_config(spectrum,config)
        self._log(f"loaded {self.library.family} library with {self.library.n_models} SSP templates [{self.regime.upper()}]")
        self.wave_rest=np.asarray(spectrum.wave,float)/(1+float(spectrum.redshift)); self.result=None
        self.posterior=None
        self.posterior_path=Path(_get(config,"stellar_inference_output_path","stellar_posterior.npz"))
        if self.state is not None: self.state.register_panel("stellar",self)

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
        self.age_slider=Slider(plt.axes([.14,.27,.17,.025]),"Age [Myr]",ages.min(),ages.max(),valinit=ages[0],valstep=ages)
        if self.library.metallicities_dex is not None and np.any(np.isfinite(self.library.metallicities_dex)):
            mets=np.unique(self.library.metallicities_dex[np.isfinite(self.library.metallicities_dex)]); self.met_values=mets; self.met_is_numeric=True
            self.met_slider=Slider(plt.axes([.14,.23,.17,.025]),"[M/H]",mets.min(),mets.max(),valinit=mets[0],valstep=mets)
        else:
            self.met_values=np.unique(self.library.metallicity_codes.astype(str)); self.met_is_numeric=False
            self.met_slider=Slider(plt.axes([.14,.23,.17,.025]),"Metallicity",0,len(self.met_values)-1,valinit=0,valstep=1)

        eb=_pair(config,"stellar_ebv_bounds",(0,2)); vb=_pair(config,"stellar_velocity_bounds_kms",(-500,500)); sb=_pair(config,"stellar_sigma_bounds_kms",(0,1000))
        self.ebv_slider=Slider(plt.axes([.14,.19,.17,.025]),"E(B-V)",eb[0],eb[1],valinit=float(_get(config,"stellar_ebv_initial",.05)),valstep=.01)
        self.vel_slider=Slider(plt.axes([.54,.27,.13,.025]),r"$v_\star$ [km/s]",vb[0],vb[1],valinit=float(_get(config,"stellar_velocity_initial_kms",0)),valstep=5)
        self.sig_slider=Slider(plt.axes([.54,.23,.13,.025]),r"$\sigma_\star$ [km/s]",sb[0],sb[1],valinit=float(_get(config,"stellar_sigma_initial_kms",50)),valstep=5)
        self.smooth_slider=Slider(plt.axes([.54,.19,.13,.025]),"Display bin",1,100,valinit=1,valstep=1)
        self.options=CheckButtons(
            plt.axes([.80,.57,.17,.13]),
            ["Fit stellar velocity","Fit stellar sigma","Include gas nuisance"],
            [
                str(_get(config,"stellar_fit_velocity",False)).lower() not in {"false","0","no"},
                str(_get(config,"stellar_fit_sigma",False)).lower() not in {"false","0","no"},
                str(_get(config,"stellar_include_gas",False)).lower() not in {"false","0","no"},
            ],
        )

        self.view_button=Button(plt.axes([.80,.14,.165,.04]),"Reset View"); self.view_button.on_clicked(self._reset_view)
        self.fit_button=Button(plt.axes([.80,.09,.04,.04]),"Fit"); self.fit_button.on_clicked(self._fit)
        self.plot_button=Button(plt.axes([.845,.09,.04,.04]),"Plot"); self.plot_button.on_clicked(self._plot_fit)
        self.load_button=Button(plt.axes([.89,.09,.04,.04]),"Load"); self.load_button.on_clicked(self._load_fit)
        self.save_button=Button(plt.axes([.935,.09,.04,.04]),"Save"); self.save_button.on_clicked(self._save_fit)
        self.posterior_button=Button(plt.axes([.80,.03,.04,.04]),"Post"); self.posterior_button.on_clicked(self._run_posterior)
        self.posterior_load_button=Button(plt.axes([.845,.03,.04,.04]),"Load P"); self.posterior_load_button.on_clicked(self._load_posterior)
        self.posterior_save_button=Button(plt.axes([.89,.03,.04,.04]),"Save P"); self.posterior_save_button.on_clicked(self._save_posterior)
        self.posterior_plot_button=Button(plt.axes([.935,.03,.04,.04]),"Plot P"); self.posterior_plot_button.on_clicked(self._plot_posterior)

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
        self.smooth_slider.on_changed(self._refresh_data); self.options.on_clicked(self._toggle_fit_options)
        self._refresh_data(rescale=True); self._preview()
        self._log(f"display range: {self.wave_rest[np.isfinite(self.wave_rest)].min():.3f}--{self.wave_rest[np.isfinite(self.wave_rest)].max():.3f} Angstrom (rest frame)")

    def _set_runtime_config(self,key,value):
        values=getattr(self.config,"values",None)
        if isinstance(values,dict):
            values[key]=value
        elif hasattr(self.config,"__setitem__"):
            self.config[key]=value

    def _toggle_fit_options(self,*_):
        fit_velocity,fit_sigma,include_gas=map(bool,self.options.get_status())
        self._set_runtime_config("stellar_fit_velocity",fit_velocity)
        self._set_runtime_config("stellar_fit_sigma",fit_sigma)
        self._set_runtime_config("stellar_include_gas",include_gas)
        self._log(
            f"fit options: v_star={'fit' if fit_velocity else 'fixed'}, "
            f"sigma_star={'fit' if fit_sigma else 'fixed'}, gas nuisance={'on' if include_gas else 'off'}"
        )

    def _selected_index(self):
        age=float(self.age_slider.val); am=np.isclose(self.library.ages_myr,age,atol=1e-8)
        if self.met_is_numeric:
            hit=np.flatnonzero(am&np.isclose(self.library.metallicities_dex,float(self.met_slider.val),atol=1e-8))
        else:
            code=str(self.met_values[int(round(self.met_slider.val))]); hit=np.flatnonzero(am&(self.library.metallicity_codes.astype(str)==code))
        candidates=np.flatnonzero(am)
        return int(hit[0]) if hit.size else int(candidates[0])

    def _refresh_data(self,*_,rescale=False):
        bins=max(1,int(round(self.smooth_slider.val))) if hasattr(self,"smooth_slider") else 1
        if bins>1 and self.spectrum.flux_unc is not None:
            w,f,_=compute_display_smoothing(self.spectrum,bins,min_coverage=.5); w=w/(1+float(self.spectrum.redshift))
        else:
            w=self.wave_rest; f=np.asarray(self.spectrum.flux,float)
        self.data_line.set_data(w,f)
        mask=np.ones(self.wave_rest.size,bool) if self.spectrum.mask is None else np.asarray(self.spectrum.mask,bool)
        self.masked_line.set_data(self.wave_rest[~mask],np.asarray(self.spectrum.flux,float)[~mask])
        if rescale:
            self._set_sensible_limits()
        self.fig.canvas.draw_idle()

    def _set_sensible_limits(self):
        # Shared with gui.emission via gui.axis_limits -- see that module
        # for why this replaced ax.relim()/autoscale_view() (Matplotlib's
        # silent (0, 1) fallback) in the first place. True min/max by
        # default, not percentile-clipped, so the full spectrum is always
        # visible.
        wave = np.asarray(self.wave_rest, float)
        flux = np.asarray(self.spectrum.flux, float)
        mask = None if self.spectrum.mask is None else np.asarray(self.spectrum.mask, bool)
        limits = compute_sensible_limits(wave, flux, mask)
        if limits is None:
            return
        xlim, ylim = limits
        self.ax.set_xlim(*xlim)
        if ylim is not None:
            self.ax.set_ylim(*ylim)

    def _reset_view(self,*_):
        self._set_sensible_limits()
        self.fig.canvas.draw_idle()
        xlim=self.ax.get_xlim(); ylim=self.ax.get_ylim()
        self._log(f"view reset: x={xlim[0]:.3f}..{xlim[1]:.3f} Angstrom, y={ylim[0]:.4g}..{ylim[1]:.4g}")

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
        self._toggle_fit_options()
        fit_velocity,fit_sigma,include_gas=map(bool,self.options.get_status())
        self._log("starting deterministic stellar fit")
        self._log(
            f"settings: SSPs={self.library.n_models}, v_star={'fit' if fit_velocity else 'fixed'}, "
            f"sigma_star={'fit' if fit_sigma else 'fixed'}, gas={'on' if include_gas else 'off'}, "
            f"Jacobian={_get(self.config,'stellar_nonlinear_jacobian','2-point')}, "
            f"max_nfev={int(_get(self.config,'stellar_max_nfev',100))}"
        )
        try:
            self.result=fit_stellar_spectrum(self.spectrum,self.config,library=self.library,regime=self.regime)
        except Exception as exc:
            self._log(f"FIT FAILED: {type(exc).__name__}: {exc}")
            raise
        if self.state is not None: self.state.set_result("stellar",self.result)
        self.best_line.set_data(self.result.wave,self.result.model); self.gas_line.set_data(self.result.wave,self.result.gas_model)
        self.fig.canvas.draw_idle()
        self._log(
            f"fit complete: success={self.result.success}, nfev={self.result.n_function_evaluations}, "
            f"chi2nu={self.result.reduced_chi_square:.5g}, E(B-V)={self.result.ebv:.4g}, "
            f"v={self.result.velocity_kms:.3f} km/s, sigma={self.result.sigma_kms:.3f} km/s"
        )

    def _plot_fit(self,*_):
        if self.result is None:
            self._log("no deterministic stellar fit is available; run Fit or Load first")
            return None
        self._log("saving deterministic stellar plot products")
        products=save_stellar_plot_products(self.result,self.result_path)
        self._log(f"saved main fit plot: {products['main_pdf']}")
        self._log(f"saved observational summary: {products['summary_pdf']}")
        self._log(f"saved SSP diagnostics plot: {products['diag_pdf']}")
        self._log(f"saved SSP diagnostics text: {products['diag_txt']}")
        return products

    def _save_fit(self,*_):
        if self.result is None:
            self._fit()
        if self.result is None:
            return None
        path=save_stellar_result(self.result_path,self.result,overwrite=True)
        self._log(f"saved fit: {path}")
        products=save_stellar_plot_products(self.result,self.result_path)
        self._log(f"saved plot: {products['main_pdf']}")
        self._log(f"saved summary: {products['summary_pdf']}")
        self._log(f"saved diagnostics plot: {products['diag_pdf']}")
        self._log(f"saved diagnostics text: {products['diag_txt']}")
        return path

    def _load_fit(self,*_):
        self.result=load_stellar_result(self.result_path); self.state.set_result("stellar",self.result) if self.state is not None else None; self.best_line.set_data(self.result.wave,self.result.model); self.gas_line.set_data(self.result.wave,self.result.gas_model); self.fig.canvas.draw_idle(); self._log(f"loaded fit: {self.result_path}")

    def _run_posterior(self,*_):
        if self.result is None:
            self._fit()
        self.posterior=run_stellar_inference(
            self.result,self.spectrum,self.config,library=self.library,regime=self.regime,
        )
        if self.posterior is not None and self.state is not None: self.state.set_posterior("stellar",self.posterior)
        if self.posterior is None:
            self._log("posterior sampling disabled; set stellar_inference_method = emcee or dynesty")
            return
        engine=self.posterior.metadata.get("engine","posterior")
        kind=self.posterior.metadata.get("posterior_kind","unknown")
        self._log(f"{engine}: {self.posterior.samples.shape[0]} stored samples [{kind}]")
        summary=self.posterior.summary()
        for name in self.posterior.parameter_names[:10]:
            q=summary[name]; print(f"  {name}: {q['50']:.6g} (-{q['50']-q['16']:.3g}, +{q['84']-q['50']:.3g})")
        if len(self.posterior.parameter_names)>10:
            print(f"  ... {len(self.posterior.parameter_names)-10} additional sampled parameters")
        if kind=="conditional_full":
            try:
                pop=stellar_population_samples(self.posterior,self.result)
                q=np.percentile(pop["total_formed_mass_msun"],[16,50,84])
                print(f"  total formed mass [Msun]: {q[1]:.6g} (-{q[1]-q[0]:.3g}, +{q[2]-q[1]:.3g})")
            except ValueError:
                pass

    def _load_posterior(self,*_):
        self.posterior=PosteriorResult.load_npz(self.posterior_path)
        if self.state is not None: self.state.set_posterior("stellar",self.posterior)
        self._log(f"loaded posterior: {self.posterior_path} ({self.posterior.samples.shape[0]} samples)")

    def _save_posterior(self,*_):
        if self.posterior is None:
            self._run_posterior()
        if self.posterior is None:
            return
        path=self.posterior.save_npz(self.posterior_path); self._log(f"saved posterior: {path}")

    def _plot_posterior(self,*_):
        if self.posterior is None:
            self._run_posterior()
        if self.posterior is None:
            return
        max_parameters=int(_get(self.config,"stellar_inference_corner_max_parameters",10))
        figure=plot_posterior_corner(self.posterior,max_parameters=max_parameters)
        figure.suptitle("FitSpec stellar posterior",fontsize=12)
        figure.show()

    def show(self):
        plt.show()
