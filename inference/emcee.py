"""emcee posterior-sampling backend (optional dependency)."""
from __future__ import annotations

import numpy as np

from core.results import PosteriorResult
from inference.problem import PosteriorProblem

__all__ = ["run_emcee"]


def _import_emcee():
    try:
        import emcee as package
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            "The emcee backend requires the optional 'emcee' package. Install emcee>=3 and retry."
        ) from exc
    return package


def _initial_walkers(problem, nwalkers, rng, scale):
    if nwalkers < 2 * problem.ndim:
        raise ValueError("emcee requires at least 2 * ndim walkers for the default stretch move.")
    center = problem.initial_position
    if center is None or not np.isfinite(problem.log_prior(center)):
        return problem.priors.sample(rng, nwalkers)

    # Parameter-scale-aware cloud around the deterministic solution.  Invalid
    # points are replaced with independent prior draws rather than clipped,
    # which avoids parking walkers exactly on hard prior boundaries.
    prior_draws = problem.priors.sample(rng, nwalkers)
    spread = np.std(prior_draws, axis=0)
    spread = np.where(spread > 0, spread, np.maximum(np.abs(center), 1.0))
    walkers = center + rng.normal(size=(nwalkers, problem.ndim)) * spread * float(scale)
    for i in range(nwalkers):
        if not np.isfinite(problem.log_prior(walkers[i])):
            walkers[i] = prior_draws[i]
    return walkers


def run_emcee(
    problem: PosteriorProblem, *, nwalkers=None, nsteps=2000, burn=500, thin=1,
    initial_scale=1e-3, random_seed=None, progress=False, pool=None,
    deterministic_result=None,
) -> PosteriorResult:
    """Sample ``problem`` with emcee's ensemble sampler.

    Returned samples are flattened after ``burn`` and ``thin``.  Acceptance
    fractions and (when estimable) integrated autocorrelation times are stored
    in ``PosteriorResult.metadata`` for convergence assessment.
    """
    package = _import_emcee()
    ndim = problem.ndim
    nwalkers = max(2 * ndim, 32) if nwalkers is None else int(nwalkers)
    nsteps, burn, thin = int(nsteps), int(burn), int(thin)
    if nsteps <= 0 or burn < 0 or burn >= nsteps or thin <= 0:
        raise ValueError("Require nsteps > 0, 0 <= burn < nsteps, and thin > 0.")

    rng = np.random.default_rng(random_seed)
    walkers = _initial_walkers(problem, nwalkers, rng, initial_scale)
    sampler = package.EnsembleSampler(nwalkers, ndim, problem.log_probability, pool=pool)
    sampler.run_mcmc(walkers, nsteps, progress=progress)
    samples = np.asarray(sampler.get_chain(discard=burn, thin=thin, flat=True), dtype=float)
    log_prob = np.asarray(sampler.get_log_prob(discard=burn, thin=thin, flat=True), dtype=float)

    try:
        tau = np.asarray(sampler.get_autocorr_time(discard=burn, thin=thin, quiet=True), dtype=float).tolist()
    except Exception:
        tau = None
    metadata = dict(problem.metadata)
    metadata.update({
        "engine": "emcee", "nwalkers": nwalkers, "nsteps": nsteps,
        "burn": burn, "thin": thin, "random_seed": random_seed,
        "mean_acceptance_fraction": float(np.mean(sampler.acceptance_fraction)),
        "acceptance_fraction": np.asarray(sampler.acceptance_fraction, dtype=float).tolist(),
        "autocorrelation_time": tau,
    })
    return PosteriorResult(samples=samples, log_probability=log_prob,
                           parameter_names=problem.parameter_names,
                           deterministic_result=deterministic_result, metadata=metadata)
