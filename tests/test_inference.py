"""Tests for the shared inference layer; external emcee/dynesty are optional."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from core.parameters import Component, ModelParameters, Parameter
from core.results import PosteriorResult
from inference.priors import UniformPrior, GaussianPrior, LogUniformPrior, TruncatedGaussianPrior, PriorSet
from inference.problem import PosteriorProblem, gaussian_log_likelihood, model_parameters_problem
from inference.model_selection import compare_models, select_best_model


class TestPriors(unittest.TestCase):
    def test_uniform_transform_and_support(self):
        p = UniformPrior(-2.0, 6.0)
        self.assertAlmostEqual(p.transform(0.25), 0.0)
        self.assertTrue(np.isfinite(p.log_probability(1.0)))
        self.assertEqual(p.log_probability(7.0), -np.inf)

    def test_gaussian_transform_median(self):
        p = GaussianPrior(3.0, 2.0)
        self.assertAlmostEqual(p.transform(0.5), 3.0, places=12)

    def test_loguniform(self):
        p = LogUniformPrior(1.0, 100.0)
        self.assertAlmostEqual(p.transform(0.5), 10.0, places=12)

    def test_truncated_gaussian_transform_is_bounded(self):
        p = TruncatedGaussianPrior(0.0, 1.0, -1.0, 2.0)
        vals = [p.transform(u) for u in (0.0, 0.3, 1.0)]
        self.assertTrue(all(-1.0 <= v <= 2.0 for v in vals))

    def test_from_model_parameters_requires_proper_bounds(self):
        pars = ModelParameters(1, [Component([Parameter("x", 0.0, -np.inf, np.inf)])])
        with self.assertRaises(ValueError):
            PriorSet.from_model_parameters(pars)


class TestProblem(unittest.TestCase):
    def test_diagonal_gaussian_likelihood(self):
        y = np.array([1.0, 2.0])
        m = np.array([1.0, 2.0])
        s = np.array([0.5, 0.5])
        expected = -0.5 * np.sum(np.log(2*np.pi*s*s))
        self.assertAlmostEqual(gaussian_log_likelihood(y, m, s), expected)

    def test_covariance_gaussian_likelihood(self):
        y = np.array([0.0, 0.0])
        cov = np.array([[2.0, 0.3], [0.3, 1.0]])
        ll = gaussian_log_likelihood(y, y, None, covariance=cov)
        sign, logdet = np.linalg.slogdet(cov)
        self.assertEqual(sign, 1)
        self.assertAlmostEqual(ll, -0.5*(logdet + 2*np.log(2*np.pi)))

    def test_model_parameters_adapter_recovers_best_region(self):
        wave = np.linspace(-1, 1, 21)
        true = 2.0
        flux = true * wave
        unc = np.full_like(wave, 0.1)
        pars = ModelParameters(1, [Component([Parameter("slope", 1.5, 0.0, 4.0)])])

        def model(w, mp):
            return mp.components[0]["slope"].value * w

        problem = model_parameters_problem(wave, flux, unc, pars, model)
        self.assertGreater(problem.log_probability([2.0]), problem.log_probability([1.0]))
        # Adapter operates on a copy, not the caller's state.
        problem.log_probability([3.0])
        self.assertEqual(pars.components[0]["slope"].value, 1.5)

    def test_posterior_problem_prior_transform(self):
        priors = PriorSet([UniformPrior(0, 10)], ["x"])
        problem = PosteriorProblem(["x"], lambda x: -0.5*x[0]**2, priors, np.array([1.0]))
        self.assertAlmostEqual(problem.prior_transform([0.2])[0], 2.0)
        self.assertEqual(problem.log_probability([-1.0]), -np.inf)


class _Stats:
    def __init__(self, bic):
        self.bic = bic


class _Result:
    def __init__(self, bic):
        self.statistics = _Stats(bic)


class TestModelSelection(unittest.TestCase):
    def test_compare_and_driver(self):
        ranked = compare_models([_Result(5), _Result(2), _Result(9)], labels=["a", "b", "c"])
        self.assertEqual([x.label for x in ranked], ["b", "a", "c"])
        best, ranked2 = select_best_model([1, 2, 3], lambda n: _Result((n-2)**2), criterion="bic")
        self.assertEqual(best.statistics.bic, 0)
        self.assertEqual(ranked2[0].label, "2")


class TestPosteriorResult(unittest.TestCase):
    def test_summary_map_and_roundtrip(self):
        samples = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]])
        result = PosteriorResult(samples, np.array([-2.0, -1.0, -3.0]), ["a", "b"], metadata={"engine":"test"})
        self.assertTrue(np.allclose(result.maximum_posterior_sample, [1.0, 2.0]))
        self.assertIn("50", result.summary()["a"])
        with tempfile.TemporaryDirectory() as td:
            path = result.save_npz(Path(td)/"posterior.npz")
            loaded = PosteriorResult.load_npz(path)
        self.assertTrue(np.allclose(loaded.samples, samples))
        self.assertEqual(loaded.metadata["engine"], "test")


class TestOptionalBackends(unittest.TestCase):
    def test_missing_backends_raise_clear_import_error(self):
        priors = PriorSet([UniformPrior(-1, 1)], ["x"])
        problem = PosteriorProblem(["x"], lambda x: -0.5*x[0]**2, priors, np.array([0.0]))
        try:
            import emcee  # noqa: F401
        except ImportError:
            from inference.emcee import run_emcee
            with self.assertRaisesRegex(ImportError, "optional 'emcee'"):
                run_emcee(problem, nwalkers=4, nsteps=4, burn=1)
        try:
            import dynesty  # noqa: F401
        except ImportError:
            from inference.dynesty import run_dynesty
            with self.assertRaisesRegex(ImportError, "optional 'dynesty'"):
                run_dynesty(problem, nlive=10)


if __name__ == "__main__":
    unittest.main()
