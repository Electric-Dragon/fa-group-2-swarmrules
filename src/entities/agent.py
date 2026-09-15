import numpy as np
from .base import BaseObject


class BaseAgent(BaseObject):
    def __init__(
        self,
        agent_id: int,
        position: np.ndarray,
        radius: float = 0.4,
        mobility: float = 1.0,
        diffusion_coeff: float = 0.5,
    ):
        super().__init__(obj_id=agent_id, position=position, radius=radius, is_static=False)
        self.mobility = float(mobility)  # \mu = 1 / \gamma
        self.diffusion_coeff = float(diffusion_coeff)  # D

    def compute_self_propulsion(self) -> np.ndarray:
        """Hook for active agent motion, steering, or internal force."""
        return np.zeros(2, dtype=float)

    def update_internal_state(self, dt: float) -> None:
        """Hook for updating agent internal states (e.g., energy, state machine)."""
        pass
