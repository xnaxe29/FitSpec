"""dynesty nested-sampling backend (optional dependency)."""
from __future__ import annotations

import numpy as np

from core.results import PosteriorResult
from inference.problem import PosteriorProblem

__all__ = ["run_dynesty"]


def _import_dynesty():
    try:
        import dynesty as package
        from dynesty import utils as dyfunc
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            "The dynesty backend requires the optional 'dynesty' package. Install dynesty>=2 and retry."
        ) from exc
    return package, dyfunc


def run_dynesty(
    problem: PosteriorProblem, *, dynamic=True, nlive=500, dlogz=0.1,
    sample="auto", bound="multi", random_seed=None, progress=False, pool=None,
    queue_size=None, deterministic_result=None, **run_kwargs,
) -> PosteriorResult:
    """Run static or dynamic nested sampling and return equal-weight samples.

    The native weighted samples and evidence information are summarized in
    metadata; equal-weight posterior samples are returned in the common
    ``PosteriorResult.samples`` array for downstream diagnostics.
    """
    package, dyfunc = _import_dynesty()
    nlive = int(nlive)
    if nlive <= 1:
        raise ValueError("nlive must be > 1.")
    rng = np.random.default_rng(random_seed)
    common = dict(bound=bound, sample=sample, rstate=rng)
    if pool is not None:
        common["pool"] = pool
        common["queue_size"] = int(queue_size or 1)

    if dynamic:
        sampler = package.DynamicNestedSampler(problem.log_likelihood, problem.prior_transform,
                                                problem.ndim, **common)
        kwargs = dict(run_kwargs)
        kwargs.setdefault("nlive_init", nlive)
        kwargs.setdefault("dlogz_init", float(dlogz))
        kwargs.setdefault("print_progress", bool(progress))
        sampler.run_nested(**kwargs)
    else:
        sampler = package.NestedSampler(problem.log_likelihood, problem.prior_transform,
                                        problem.ndim, nlive=nlive, **common)
        kwargs = dict(run_kwargs)
        kwargs.setdefault("dlogz", float(dlogz))
        kwargs.setdefault("print_progress", bool(progress))
        sampler.run_nested(**kwargs)

    results = sampler.results
    weights = np.exp(np.asarray(results.logwt) - float(results.logz[-1]))
    samples = np.asarray(dyfunc.resample_equal(np.asarray(results.samples), weights, rstate=rng), dtype=float)
    log_prob = np.asarray([problem.log_probability(row) for row in samples], dtype=float)
    metadata = dict(problem.metadata)
    metadata.update({
        "engine": "dynesty", "dynamic": bool(dynamic), "nlive": nlive,
        "dlogz": float(dlogz), "sample": sample, "bound": bound,
        "random_seed": random_seed, "log_evidence": float(results.logz[-1]),
        "log_evidence_error": float(results.logzerr[-1]),
        "n_likelihood_calls": int(np.sum(results.ncall)),
    })
    return PosteriorResult(samples=samples, log_probability=log_prob,
                           parameter_names=problem.parameter_names,
                           deterministic_result=deterministic_result, metadata=metadata)
