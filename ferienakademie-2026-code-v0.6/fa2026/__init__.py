"""Ferienakademie 2026 shared-project framework."""

from .api import Action, Cell, Config, Observation, Pheromone, Result, Rules, Scenario
from .engine import execute
from .scenarios import load_scenario

__all__ = [
    "Action",
    "Cell",
    "Config",
    "Observation",
    "Pheromone",
    "Result",
    "Rules",
    "Scenario",
    "execute",
    "load_scenario",
]

__version__ = "0.6.0"
