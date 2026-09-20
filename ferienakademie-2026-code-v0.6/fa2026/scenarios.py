from __future__ import annotations

import numpy as np

from .api import Rules, Scenario


def _boundary_walls(height: int, width: int) -> np.ndarray:
    walls = np.zeros((height, width), dtype=bool)
    walls[0, :] = True
    walls[-1, :] = True
    walls[:, 0] = True
    walls[:, -1] = True
    return walls


def _freeze(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


def _random_agent_positions(
    rules: Rules,
    walls: np.ndarray,
    cargo: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    """Reproducibly distribute agents over accessible free space."""
    height, width = rules.grid_shape
    rng = np.random.default_rng(seed)
    positions: list[tuple[float, float]] = []
    half = rules.cargo_size / 2.0

    while len(positions) < rules.n_agents:
        pos = np.array(
            [rng.uniform(1.2, width - 1.2), rng.uniform(1.2, height - 1.2)],
            dtype=float,
        )
        if walls[int(pos[1]), int(pos[0])]:
            continue
        if abs(pos[0] - cargo[0]) <= half and abs(pos[1] - cargo[1]) <= half:
            continue
        positions.append((float(pos[0]), float(pos[1])))

    return np.asarray(positions, dtype=float)


def _playground() -> Scenario:
    """Forgiving fixed scenario used by ``getting_started.ipynb``."""

    # Large enough that the cargo and target are not usually visible together.
    # The demo controller therefore has to explore before transport becomes directed.
    height, width = 36, 48
    walls = _boundary_walls(height, width)

    # The introductory playground is intentionally open. The challenge here is
    # local exploration and collective transport, not maze navigation.
    target = np.zeros((height, width), dtype=bool)
    target[11:26, 38:47] = True

    rules = Rules(
        grid_shape=(height, width),
        n_agents=80,
        vision_radius=8,
        battery_capacity=30.0,
        initial_energy=20.0,
        recharge=0.5,
        move_cost=0.20,
        push_cost=1.20,
        pheromone_cost=0.02,
        memory_size=4,
        max_agent_types=4,
        max_pheromones=4,
        cargo_size=2.0,
        cargo_mobility=0.065,
    )

    cargo = np.array([24.0, 18.0], dtype=float)
    positions = _random_agent_positions(rules, walls, cargo, seed=2026)

    return Scenario(
        name="playground",
        rules=rules,
        walls=_freeze(walls),
        agent_positions=_freeze(positions),
        cargo_position=(float(cargo[0]), float(cargo[1])),
        target_mask=_freeze(target),
    )


def _showcase() -> Scenario:
    """Feature-rich scenario used by ``feature_demo.ipynb``.

    This is a discussion/demo map rather than a benchmark. It deliberately adds
    internal walls and tighter sight range so several controller mechanisms are
    visible at once.
    """

    height, width = 36, 48
    walls = _boundary_walls(height, width)

    # A few simple barriers with generous gaps. The geometry is deliberately
    # easy to inspect rather than designed as a difficult maze.
    walls[4:14, 15] = True
    walls[19:33, 15] = True
    walls[5:25, 30] = True
    walls[29:34, 30] = True
    walls[10, 16:25] = True
    walls[10, 27:30] = True
    walls[27, 4:12] = True
    walls[27, 13:15] = True

    target = np.zeros((height, width), dtype=bool)
    target[14:24, 39:47] = True

    rules = Rules(
        grid_shape=(height, width),
        n_agents=120,
        vision_radius=6,
        battery_capacity=26.0,
        initial_energy=16.0,
        recharge=0.40,
        move_cost=0.22,
        push_cost=1.35,
        pheromone_cost=0.02,
        memory_size=6,
        max_agent_types=4,
        max_pheromones=4,
        cargo_size=2.0,
        cargo_mobility=0.055,
    )

    cargo = np.array([9.0, 18.0], dtype=float)
    positions = _random_agent_positions(rules, walls, cargo, seed=2027)

    return Scenario(
        name="showcase",
        rules=rules,
        walls=_freeze(walls),
        agent_positions=_freeze(positions),
        cargo_position=(float(cargo[0]), float(cargo[1])),
        target_mask=_freeze(target),
    )


def load_scenario(name: str) -> Scenario:
    """Load a fixed teacher-defined scenario by name."""
    normalized = name.strip().lower()
    if normalized == "playground":
        return _playground()
    if normalized == "showcase":
        return _showcase()
    raise KeyError(
        f"unknown scenario {name!r}; available scenarios: 'playground', 'showcase'"
    )
