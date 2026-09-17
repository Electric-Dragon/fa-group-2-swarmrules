import numpy as np
from .base import BaseObject


class BaseAgent(BaseObject):
    def __init__(
        self,
        agent_id: int,
        position: np.ndarray,
        theta: float = 0.0,
        radius: float = 0.4,
        diff: float = 0.5,
        rotdiff: float = 0.5
    ):
        super().__init__(obj_id=agent_id, position=position, radius=radius, is_static=False)
        self.diff = float(diff)
        self.rotdiff = float(rotdiff)
        self.theta = float(theta)

    def compute_self_propulsion(self) -> np.ndarray:
        """Hook for active agent motion, steering, or internal force."""
        return np.zeros(2, dtype=float)

    def update_internal_state(self, dt: float, env: "BaseObject | None" = None) -> None:
        """Hook for updating agent internal states (e.g., energy, state machine)."""
        pass


class PheromoneAgent(BaseAgent):
    def __init__(
        self,
        agent_id: int,
        position: np.ndarray,
        deposition_rate: float = 1.0,
        **kwargs,
    ):
        super().__init__(agent_id=agent_id, position=position, **kwargs)
        self.deposition_rate = float(deposition_rate)

    def update_internal_state(self, dt: float, env=None) -> None:
        if env is not None:
            env.deposit_pheromone(self.position, self.deposition_rate * dt)
