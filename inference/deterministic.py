"""Deterministic-inference compatibility layer.

The canonical implementation remains :func:`core.fitting.fit_deterministic`;
this module exists so users can access deterministic and posterior engines from
one inference namespace without duplicating optimizer code.
"""
from core.fitting import fit_deterministic

__all__ = ["fit_deterministic"]
