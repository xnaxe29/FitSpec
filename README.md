# FitSpec

**FitSpec** is a universal Python spectral-fitting application for stellar continua, nebular emission lines, and absorption lines within one shared GUI and fitting framework. The three science modules use the same spectrum object, masking infrastructure, instrumental-resolution handling, statistics/results layer, plotting primitives, and optional Bayesian inference backends.

## Current capabilities

FitSpec currently includes:

- a universal `Spectrum` container and text/FITS spectrum loading;
- three-tier masking with a common working/saved mask state;
- flux-conservative, gap-aware rebinning and explicit instrumental-resolution handling;
- continuum estimation;
- unified UV/optical stellar-population fitting with non-negative SSP coefficients;
- multi-component emission-line fitting, kinematic ties, fixed line ratios, and BIC-based component search;
- Voigt/optical-depth absorption fitting, partial covering, cross-ion joint fitting, thermal/turbulent linking, abundance ties, upper limits, and freeze-based component rejection;
- shared posterior inference with `emcee` and `dynesty` adapters for stellar, emission, and absorption fitting;
- a common application shell and shared session state for all three fitting modes;
- deterministic-result and posterior save/load support plus science-specific plotting and posterior diagnostics.

The detailed architecture and scientific conventions are documented in the accompanying FitSpec Overleaf manual. That documentation should be treated as the full technical reference; this README is only the repository entry point.

## Directory layout

```text
FitSpec/
├── fitspec.py                 # top-level application launcher
├── test_run.py                # lightweight end-to-end smoke test
├── config/                    # default configuration files
├── core/                      # spectrum, I/O, masks, rebinning, resolution, fitting, results
├── continuum/                 # continuum estimation
├── stellar/                   # unified stellar fitting and stellar inference adapter
├── emission/                  # emission-line fitting and inference adapter
├── absorption/                # absorption-line fitting and inference adapter
├── inference/                 # sampler-independent posterior framework + backends
├── plotting/                  # shared and science-specific plotting
├── gui/                       # shared GUI shell, state, controllers, and panels
├── data/                      # bundled line lists / resolution data / test data
└── tests/                     # pytest test suite
```

## Python dependencies

The current source tree uses the following core scientific packages:

```text
numpy
scipy
matplotlib
astropy
h5py
scikit-learn
statsmodels
```

For Bayesian posterior sampling, install the desired optional backend(s):

```text
emcee
dynesty
```

For development/testing, install:

```text
pytest
```

A typical environment can therefore be prepared with:

```bash
python -m pip install numpy scipy matplotlib astropy h5py scikit-learn statsmodels pytest emcee dynesty
```

## Running FitSpec

From the top-level `FitSpec/` directory, the normal application entry point is:

```bash
python fitspec.py spectrum.dat
```

Common examples include:

```bash
python fitspec.py spectrum.dat --run-dir ./my_run
python fitspec.py spectrum.dat --mode stellar
python fitspec.py spectrum.dat --mode emission
python fitspec.py spectrum.dat --mode absorption
python fitspec.py spectrum.dat --redshift 0.003
python fitspec.py --session fitspec_session.npz
```

Use:

```bash
python fitspec.py --help
```

for the authoritative command-line options in the installed version.

## Input spectra

FitSpec accepts delimited text spectra and FITS binary tables through `core.io`.

For headerless text files, columns are interpreted in the following order:

```text
wavelength  flux  flux_unc  continuum  model
```

Only wavelength and flux are mandatory for loading. Fitting methods that use chi-square statistics require a valid uncertainty array.

Headered files may use common aliases such as `wavelength`, `wave`, `flux`, `flam`, `error`, `flux_unc`, `continuum`, and `model`. Input wavelengths are structurally cleaned, sorted, and deduplicated by the universal `Spectrum` loader.

## Configuration

FitSpec ships mode-specific defaults under `config/`. A run directory may contain a single user `config.dat` overriding defaults.

The unified application permits that one `config.dat` to contain stellar, emission, and absorption keywords together. Each science mode applies the keys belonging to its schema, while truly unknown configuration keywords remain errors rather than being silently ignored.

The default behavior for posterior inference remains deterministic. Enable `emcee` or `dynesty` explicitly using the appropriate stellar/emission/absorption inference configuration keywords.

## Stellar libraries

The stellar fitter uses the unified HDF5 stellar-library interface in `stellar/stellar_models.py`. The fitting architecture can operate in both UV and optical regimes through the appropriate wavelength coverage and library selection; it does not use separate external UV and optical fitting branches.

Library paths are configurable. The source code contains developer-machine defaults for convenience, but portable installations should override those paths in configuration.

## Bayesian inference

The shared `inference/` package provides priors, normalized Gaussian likelihoods, posterior-problem construction, `emcee`, `dynesty`, model-selection helpers, posterior results, and diagnostics.

Emission and absorption inference sample the finalized deterministic physical model. Stellar inference supports an explicit conditional posterior over the selected SSP basis and an explicitly labelled profile mode; the latter is not presented as a marginalized stellar-population posterior.

Posterior sampling is optional and is not required for ordinary deterministic FitSpec operation.

## Tests

Run the complete unit/integration suite from the top-level directory with:

```bash
pytest -q
```

The suite covers core masking, spectrum/model machinery, I/O, rebinning, resolution, stellar/emission/absorption fitting, plotting, GUI controllers, application state, and all three inference adapters.

A faster application-shell smoke test is also provided:

```bash
python test_run.py
```

`test_run.py` deliberately does **not** start an expensive science fit or posterior sampler. It verifies spectrum loading, unified configuration loading, shared session state, all three application modes, and session persistence without requiring external stellar-library files.

## Development principle

FitSpec is organized around one common spectral-fitting architecture. Shared numerical operations belong in `core/`, sampler-independent posterior machinery belongs in `inference/`, and only genuinely science-specific behavior belongs in `stellar/`, `emission/`, or `absorption/`. New functionality should extend these shared interfaces rather than create parallel fitting pipelines.

## Status

The major application architecture is implemented: core infrastructure, continuum handling, stellar fitting, emission fitting, absorption fitting, posterior inference, plotting, the shared GUI shell, and the top-level launcher are present. Development should now emphasize end-to-end validation on real spectra, regression testing, numerical edge cases, and documentation synchronization.
