import numpy as np
from .base import BaseObject


class Attractor(BaseObject):
    def __init__(self, obj_id: int, position: np.ndarray, strength: float = 5.0, cutoff: float = 10.0, radius: float = 0.6):
        super().__init__(obj_id=obj_id, position=position, radius=radius, is_static=True)
        self.strength = strength
        self.cutoff = cutoff

    def force_on(self, other: BaseObject) -> np.ndarray:
        delta = self.position - other.position
        dist = np.linalg.norm(delta)
        if dist < 1e-5 or dist > self.cutoff:
            return np.zeros(2, dtype=float)
        direction = delta / dist
        magnitude = self.strength / (dist**2 + 0.1)
        return magnitude * direction


class Repeller(BaseObject):
    def __init__(self, obj_id: int, position: np.ndarray, strength: float = 10.0, cutoff: float = 5.0, radius: float = 0.6):
        super().__init__(obj_id=obj_id, position=position, radius=radius, is_static=True)
        self.strength = strength
        self.cutoff = cutoff

    def force_on(self, other: BaseObject) -> np.ndarray:
        delta = other.position - self.position
        dist = np.linalg.norm(delta)
        if dist < 1e-5 or dist > self.cutoff:
            return np.zeros(2, dtype=float)
        direction = delta / dist
        magnitude = self.strength / (dist**2 + 0.1)
        return magnitude * direction
