"""Generic model-selection helpers and explicit candidate search."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from core.statistics import (
    bayesian_information_criterion,
    akaike_information_criterion,
)

__all__ = ["ModelSelectionEntry", "compare_models", "select_best_model"]


@dataclass(frozen=True)
class ModelSelectionEntry:
    label: str
    score: float
    result: object


def _statistic(result, criterion: str) -> float:
    criterion = criterion.lower()
    stats = getattr(result, "statistics", None)
    if stats is None and hasattr(result, "fit_result"):
        stats = getattr(result.fit_result, "statistics", None)
    if stats is None:
        raise TypeError("Each result must expose .statistics or .fit_result.statistics.")
    aliases = {"chi2": "chi_square", "reduced_chi2": "reduced_chi_square"}
    attr = aliases.get(criterion, criterion)
    if not hasattr(stats, attr):
        raise ValueError(f"Unknown model-selection criterion {criterion!r}.")
    return float(getattr(stats, attr))


def compare_models(results, *, criterion="bic", labels=None) -> list[ModelSelectionEntry]:
    """Rank already-fitted candidate models from smallest to largest score."""
    results = list(results)
    if not results:
        raise ValueError("results must be non-empty.")
    labels = [str(i) for i in range(len(results))] if labels is None else list(labels)
    if len(labels) != len(results):
        raise ValueError("labels must have one entry per result.")
    entries = [ModelSelectionEntry(str(label), _statistic(result, criterion), result)
               for label, result in zip(labels, results)]
    return sorted(entries, key=lambda entry: (not np.isfinite(entry.score), entry.score))


def select_best_model(candidate_values, fit_callable: Callable, *, criterion="bic"):
    """Fit explicit candidates and return ``(best_result, ranked_entries)``.

    ``fit_callable(candidate)`` is intentionally generic.  A science module
    decides what a candidate means (typically an explicit component count).
    """
    candidates = list(candidate_values)
    if not candidates:
        raise ValueError("candidate_values must be non-empty.")
    results = [fit_callable(candidate) for candidate in candidates]
    ranked = compare_models(results, criterion=criterion, labels=[str(c) for c in candidates])
    return ranked[0].result, ranked
