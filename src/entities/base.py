import numpy as np


class BaseObject:
    def __init__(self, obj_id: int, position: np.ndarray, radius: float = 0.5, is_static: bool = True):
        self.id = obj_id
        self.position = np.array(position, dtype=float)
        self.radius = float(radius)
        self.is_static = is_static

    def force_on(self, other: "BaseObject") -> np.ndarray:
        """Compute the force exerted by this object on another entity.
        Returns a 2D numpy array representing the force vector [Fx, Fy].
        Override this method in subclasses to provide custom interactions.
        """
        return np.zeros(2, dtype=float)
