"""Explicit, component-based model parameters.

Directly implements the "explicit state beats implicit inference" design
principle: the number of model components is always an explicit integer
that something (config, or the GUI) sets on purpose, never inferred from
whether a value happens to be scalar or list-valued. This was a real bug
in the legacy emission-line GUI, where whether ``curvefit_v_init`` etc.
were floats or lists silently determined the number of fitted components.

Also implements the GUI behavior described in the FitSpec fundamentals:
a "number of components" control and an independent "active component"
control, where switching the active component changes which one the
amplitude/velocity/velocity_dispersion widgets edit, while every other
component's values are retained rather than reset.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


__all__ = ["Parameter", "Component", "ModelParameters", "ParameterTie", "apply_ties"]


@dataclass
class Parameter:
    """One named scalar model parameter.

    Attributes
    ----------
    name : str
        Parameter name (e.g. "amplitude", "velocity_kms").
    value : float
        Current value.
    lower, upper : float
        Bounds, used by both deterministic optimizers and as the default
        (uniform) prior range for posterior sampling. Use -inf/+inf for
        an unbounded parameter.
    fixed : bool, default False
        If True, excluded from the free-parameter vector entirely (held
        at ``value``, not optimized/sampled).
    """

    name: str
    value: float
    lower: float = -np.inf
    upper: float = np.inf
    fixed: bool = False

    def __post_init__(self):
        if self.lower > self.upper:
            raise ValueError(f"Parameter {self.name!r}: lower bound exceeds upper bound.")
        if not (self.lower <= self.value <= self.upper):
            raise ValueError(
                f"Parameter {self.name!r}: value {self.value} is outside bounds "
                f"[{self.lower}, {self.upper}]."
            )


@dataclass
class Component:
    """One physical model component: an ordered set of named Parameters.

    E.g. one emission-line kinematic component (amplitude, velocity_kms,
    velocity_dispersion_kms), or one absorption Voigt component
    (column_density, b_kms, velocity_kms).
    """

    parameters: "list[Parameter]" = field(default_factory=list)

    def __getitem__(self, name: str) -> Parameter:
        for parameter in self.parameters:
            if parameter.name == name:
                return parameter
        raise KeyError(f"No parameter named {name!r} in this component.")

    def __contains__(self, name: str) -> bool:
        return any(parameter.name == name for parameter in self.parameters)

    def names(self) -> "list[str]":
        return [parameter.name for parameter in self.parameters]

    def copy(self) -> "Component":
        return Component(parameters=[
            Parameter(p.name, p.value, p.lower, p.upper, p.fixed) for p in self.parameters
        ])


@dataclass
class ModelParameters:
    """A full model: an explicit number of Components, plus which one is active.

    Attributes
    ----------
    n_components : int
        The number of components. Always set explicitly (by config or by
        the GUI's "number of components" control) -- never inferred from
        ``len(components)`` alone being treated as optional/implicit
        elsewhere; ``len(components)`` is required to equal
        ``n_components`` at all times (enforced in ``__post_init__`` and
        by every mutating method).
    components : list[Component]
        One entry per component, length always ``n_components``.
    active_component_index : int
        Index into ``components`` of the currently "active" component
        (e.g. the one GUI amplitude/velocity/velocity_dispersion widgets
        currently edit). Changing this does not modify any component's
        values -- every component retains its own last-set values
        independently, matching the fundamentals doc's requirement that
        switching the active component keeps other components' settings
        in memory.
    """

    n_components: int
    components: "list[Component]"
    active_component_index: int = 0

    def __post_init__(self):
        if self.n_components < 1:
            raise ValueError("n_components must be >= 1.")
        if len(self.components) != self.n_components:
            raise ValueError(
                f"len(components) ({len(self.components)}) must equal "
                f"n_components ({self.n_components})."
            )
        if not (0 <= self.active_component_index < self.n_components):
            raise ValueError("active_component_index out of range.")

    @property
    def active_component(self) -> Component:
        return self.components[self.active_component_index]

    def set_active(self, index: int) -> None:
        """Change which component is active. Does not alter any values."""
        if not (0 <= index < self.n_components):
            raise ValueError(f"active component index {index} out of range [0, {self.n_components}).")
        self.active_component_index = index

    def add_component(self, component: Component, *, make_active: bool = True) -> None:
        """Append a new component, incrementing n_components explicitly."""
        self.components.append(component)
        self.n_components += 1
        if make_active:
            self.active_component_index = self.n_components - 1

    def remove_component(self, index: int) -> None:
        """Remove a component by index, decrementing n_components explicitly."""
        if self.n_components <= 1:
            raise ValueError("Cannot remove the last remaining component.")
        if not (0 <= index < self.n_components):
            raise ValueError(f"component index {index} out of range [0, {self.n_components}).")
        del self.components[index]
        self.n_components -= 1
        if self.active_component_index >= self.n_components:
            self.active_component_index = self.n_components - 1

    def free_parameters(self):
        """Yield (component_index, Parameter) for every non-fixed parameter, in order."""
        for component_index, component in enumerate(self.components):
            for parameter in component.parameters:
                if not parameter.fixed:
                    yield component_index, parameter

    def to_vector(self) -> np.ndarray:
        """Flatten every free parameter's current value into one array, in order."""
        return np.array([parameter.value for _, parameter in self.free_parameters()], dtype=float)

    def from_vector(self, vector) -> None:
        """Write values from a flat array back into the free parameters, in order."""
        vector = np.asarray(vector, dtype=float)
        free = list(self.free_parameters())
        if vector.size != len(free):
            raise ValueError(f"Expected {len(free)} free-parameter values, got {vector.size}.")
        for value, (_, parameter) in zip(vector, free):
            if not (parameter.lower <= value <= parameter.upper):
                raise ValueError(
                    f"Value {value} for parameter {parameter.name!r} is outside bounds "
                    f"[{parameter.lower}, {parameter.upper}]."
                )
            parameter.value = value

    def bounds(self):
        """(lower_array, upper_array) for every free parameter, in the same order as to_vector()."""
        lower = np.array([p.lower for _, p in self.free_parameters()], dtype=float)
        upper = np.array([p.upper for _, p in self.free_parameters()], dtype=float)
        return lower, upper

    def parameter_names(self) -> "list[str]":
        """Names of every free parameter, in the same order as to_vector(), prefixed by component index."""
        return [f"c{component_index}_{parameter.name}" for component_index, parameter in self.free_parameters()]

    def copy(self) -> "ModelParameters":
        return ModelParameters(
            n_components=self.n_components,
            components=[component.copy() for component in self.components],
            active_component_index=self.active_component_index,
        )

    def to_dict(self) -> dict:
        """Serialize to a plain, JSON-able dict (for the save/load GUI button)."""
        return {
            "n_components": self.n_components,
            "active_component_index": self.active_component_index,
            "components": [
                [
                    {"name": p.name, "value": p.value, "lower": p.lower, "upper": p.upper, "fixed": p.fixed}
                    for p in component.parameters
                ]
                for component in self.components
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ModelParameters":
        components = [
            Component(parameters=[Parameter(**parameter_data) for parameter_data in component_data])
            for component_data in data["components"]
        ]
        return cls(
            n_components=data["n_components"], components=components,
            active_component_index=data.get("active_component_index", 0),
        )

@dataclass
class ParameterTie:
    """Tie one follower parameter's value to a (possibly transformed) leader parameter's value.

    Generalizes the ad hoc cross-component tying used by the emission
    module's kinematics and the absorption module's covering fraction
    into a single reusable primitive: a science module builds its
    ``ModelParameters`` with every follower's ``Parameter.fixed`` set to
    True (so the optimizer never treats it as free), lists the ties it
    wants, and calls :func:`apply_ties` once per model evaluation --
    typically as the first thing a ``model_func`` closure does -- to
    synchronize every follower's ``.value`` from its leader before the
    model itself is built.

    Ties may be chained: a parameter that is itself a follower of one
    tie may act as the leader of another, as long as ``apply_ties``
    processes them in dependency order (the order of the input list).

    Attributes
    ----------
    leader : (int, str)
        ``(component_index, parameter_name)`` of the value this tie
        reads from.
    follower : (int, str)
        ``(component_index, parameter_name)`` of the value this tie
        writes to. Must already be ``fixed=True``.
    transform : callable, default identity
        ``transform(leader_value) -> follower_value``. The default
        ``lambda value: value`` gives a simple equal-value tie (e.g. two
        ions sharing one redshift). Pass a closure for anything else --
        e.g. a mass-scaled thermal Doppler-parameter link, or a fixed
        multiplicative column-density-ratio link.
    """

    leader: "tuple[int, str]"
    follower: "tuple[int, str]"
    transform: "Callable[[float], float]" = staticmethod(lambda value: value)


def apply_ties(model_parameters: ModelParameters, ties: "list[ParameterTie] | None") -> None:
    """Synchronize every follower parameter's value from its leader, in list order.

    No-op if ``ties`` is None or empty. Safe to call unconditionally at
    the top of a ``model_func`` closure.
    """
    if not ties:
        return
    for tie in ties:
        leader_index, leader_name = tie.leader
        follower_index, follower_name = tie.follower
        leader_value = model_parameters.components[leader_index][leader_name].value
        model_parameters.components[follower_index][follower_name].value = tie.transform(leader_value)
