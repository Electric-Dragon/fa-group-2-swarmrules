from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntFlag
from typing import Callable

import numpy as np


class Cell(IntFlag):
    """Bit flags used in Observation.vision."""

    EMPTY = 0
    WALL = 1
    TARGET = 2
    CARGO = 4


@dataclass(frozen=True)
class Rules:
    """Teacher-defined rules for one scenario.

    Coordinates are measured in grid-cell units. ``grid_shape`` is
    ``(height, width)``.
    """

    grid_shape: tuple[int, int]
    n_agents: int

    day_turns: int
    night_turns: int
    day_vision_radius: int
    night_vision_radius: int

    battery_capacity: float
    initial_energy: float
    night_recharge: float
    rest_recharge_factor: float

    move_cost: float
    push_cost: float
    pheromone_cost: float

    memory_size: int
    max_agent_types: int
    max_pheromones: int

    cargo_size: float
    cargo_mobility: float


@dataclass(frozen=True)
class Pheromone:
    """One student-defined pheromone channel.

    ``decay`` is the fraction lost per turn.  For example ``0.02`` means
    that 2% of the existing amount disappears each turn.
    """

    decay: float


@dataclass(frozen=True)
class Config:
    """Immutable configuration returned once by ``setup(rules)``."""

    pheromones: tuple[Pheromone, ...] = ()

    # Counts for types 1, 2, ...; all remaining agents become type 0.
    agent_type_counts: tuple[int, ...] = ()

    # One initial memory vector for each used type, starting with type 0.
    # If omitted, all memories start at zero.
    initial_memory: tuple[tuple[float, ...], ...] = ()

    # Optional immutable constants prepared in setup(), indexed by type.
    parameters: tuple[tuple[float, ...], ...] = ()


@dataclass(frozen=True)
class Observation:
    """Information available to one agent for one turn.

    ``vision`` is a square array of :class:`Cell` bit flags centered on the
    agent's current grid cell.  Its size depends on the current day/night
    phase.  ``pheromones`` always has shape ``(n_channels, 3, 3)``.

    ``cell_position`` gives the fractional x/y position inside the current
    grid cell; absolute world coordinates are deliberately not exposed.
    """

    energy: float
    cell_position: tuple[float, float]
    vision: np.ndarray
    pheromones: np.ndarray


@dataclass(frozen=True)
class Action:
    """Action requested by an agent for one turn.

    ``move`` is a continuous x/y displacement request.  Moving into the cargo
    generates a push automatically.  ``pheromones`` contains one non-negative
    emission amount per configured pheromone channel.
    """

    move: tuple[float, float] = (0.0, 0.0)
    pheromones: tuple[float, ...] = ()


@dataclass(frozen=True)
class Scenario:
    """Teacher-defined scenario.

    Students normally obtain scenarios via ``load_scenario()`` and do not
    construct or modify this object themselves.
    """

    name: str
    rules: Rules
    walls: np.ndarray
    agent_positions: np.ndarray
    cargo_position: tuple[float, float]
    target_mask: np.ndarray


@dataclass(frozen=True)
class Result:
    success: bool
    turns: int


SetupFunction = Callable[[Rules], Config]
ActFunction = Callable[[Observation, np.ndarray, int, Config], tuple[Action, np.ndarray]]
