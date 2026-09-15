import numpy as np


class GridEnvironment:
    def __init__(self, width: int, height: int, cell_size: float = 1.0):
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.wall_mask = np.zeros((height, width), dtype=bool)

    @property
    def bounds(self) -> tuple[float, float]:
        return self.width * self.cell_size, self.height * self.cell_size

    def generate_random_walls(self, wall_density: float = 0.2, seed: int | None = None) -> None:
        rng = np.random.default_rng(seed)
        self.wall_mask = rng.random((self.height, self.width)) < wall_density

    def is_cell_wall(self, ix: int, iy: int) -> bool:
        if 0 <= ix < self.width and 0 <= iy < self.height:
            return bool(self.wall_mask[iy, ix])
        return True  # Outer boundaries treated as solid

    def is_position_in_wall(self, pos: np.ndarray, radius: float = 0.0) -> bool:
        min_x = int(np.floor((pos[0] - radius) / self.cell_size))
        max_x = int(np.floor((pos[0] + radius) / self.cell_size))
        min_y = int(np.floor((pos[1] - radius) / self.cell_size))
        max_y = int(np.floor((pos[1] + radius) / self.cell_size))

        for iy in range(min_y, max_y + 1):
            for ix in range(min_x, max_x + 1):
                if self.is_cell_wall(ix, iy):
                    return True
        return False

    def resolve_collision(self, old_pos: np.ndarray, new_pos: np.ndarray, radius: float) -> np.ndarray:
        max_x, max_y = self.bounds
        # Enforce external continuous boundaries
        clamped_x = np.clip(new_pos[0], radius, max_x - radius)
        clamped_y = np.clip(new_pos[1], radius, max_y - radius)
        target = np.array([clamped_x, clamped_y], dtype=float)

        # Separate axis resolution for sliding against grid walls
        test_x = np.array([target[0], old_pos[1]])
        if not self.is_position_in_wall(test_x, radius):
            resolved_x = target[0]
        else:
            resolved_x = old_pos[0]

        test_y = np.array([resolved_x, target[1]])
        if not self.is_position_in_wall(test_y, radius):
            resolved_y = target[1]
        else:
            resolved_y = old_pos[1]

        return np.array([resolved_x, resolved_y], dtype=float)
