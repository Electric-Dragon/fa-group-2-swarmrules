import numpy as np


class GridEnvironment:
    def __init__(self, width: int, height: int, cell_size: float = 1.0, decay_rate: float = 0.05):
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.decay_rate = decay_rate
        self.wall_mask = np.zeros((height, width), dtype=bool)
        self.pheromone_grid = np.zeros((height, width), dtype=float)

    def deposit_pheromone(self, pos: np.ndarray, amount: float) -> None:
        ix = int(pos[0] / self.cell_size)
        iy = int(pos[1] / self.cell_size)
        if 0 <= ix < self.width and 0 <= iy < self.height and not self.wall_mask[iy, ix]:
            self.pheromone_grid[iy, ix] += amount

    def update_pheromones(self, dt: float) -> None:
        self.pheromone_grid *= np.maximum(0.0, 1.0 - self.decay_rate * dt)
        self.pheromone_grid[self.wall_mask] = 0.0

    @property
    def bounds(self) -> tuple[float, float]:
        return self.width * self.cell_size, self.height * self.cell_size

    def generate_random_walls(self, wall_density: float = 0.2, seed: int | None = None) -> None:
        rng = np.random.default_rng(seed)
        self.wall_mask = rng.random((self.height, self.width)) < wall_density

    def generate_labyrinth(self, seed: int | None = None) -> None:
        rng = np.random.default_rng(seed)

        # Start with all cells as walls
        self.wall_mask = np.ones((self.height, self.width), dtype=bool)

        start_x = 1 if self.width > 2 else 0
        start_y = 1 if self.height > 2 else 0

        self.wall_mask[start_y, start_x] = False
        stack = [(start_x, start_y)]

        directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]

        while stack:
            cx, cy = stack[-1]

            neighbors = []
            for dx, dy in directions:
                nx, ny = cx + dx, cy + dy
                if 0 < nx < self.width - 1 and 0 < ny < self.height - 1 and self.wall_mask[ny, nx]:
                    neighbors.append((nx, ny, dx // 2, dy // 2))

            if neighbors:
                idx = rng.integers(len(neighbors))
                nx, ny, wall_dx, wall_dy = neighbors[idx]

                self.wall_mask[cy + wall_dy, cx + wall_dx] = False
                self.wall_mask[ny, nx] = False
                stack.append((nx, ny))
            else:
                stack.pop()

    def generate_hybrid_environment(
        self, maze_threshold: float = 0.5, field_obstacle_density: float = 0.15, seed: int | None = None
    ) -> None:
        """Generates an organic hybrid environment using smooth low-frequency noise."""
        rng = np.random.default_rng(seed)
        self.generate_labyrinth(seed=seed)
        maze_walls = self.wall_mask.copy()

        # Generate smooth low-frequency noise using randomized Fourier modes
        y, x = np.mgrid[0 : self.height, 0 : self.width]
        intensity = np.zeros((self.height, self.width))
        for _ in range(3):
            fx, fy = rng.uniform(0.5, 2.0, size=2)
            phase = rng.uniform(0, 2 * np.pi)
            intensity += np.sin(2 * np.pi * (fx * x / self.width + fy * y / self.height) + phase)

        # Normalize intensity to [0, 1]
        intensity = (intensity - intensity.min()) / (intensity.max() - intensity.min() + 1e-8)

        # In maze areas keep labyrinth walls; in open areas place sparse obstacles
        open_mask = intensity < maze_threshold
        self.wall_mask = np.where(open_mask, rng.random((self.height, self.width)) < field_obstacle_density, maze_walls)

        # Preserve boundary walls
        self.wall_mask[[0, -1], :] = True
        self.wall_mask[:, [0, -1]] = True

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
