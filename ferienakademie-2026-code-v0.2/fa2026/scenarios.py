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
    """Return an array that students can inspect but not modify in-place."""
    array.setflags(write=False)
    return array


def _playground() -> Scenario:
    """Easy fixed scenario used by the getting-started notebook.

    The playground is intentionally forgiving: agents have ample energy, the
    cargo is easy to move, daytime visibility is generous, and the target is
    large.  It is meant to exercise the complete API, not to be the final
    optimization challenge.
    """

    height, width = 18, 24
    walls = _boundary_walls(height, width)

    # Two harmless wall segments make local wall sensing visible without
    # obstructing the direct cargo-to-target route.
    walls[4, 15:19] = True
    walls[13, 15:19] = True

    target = np.zeros((height, width), dtype=bool)
    target[7:12, 18:23] = True

    rules = Rules(
        grid_shape=(height, width),
        n_agents=80,
        day_turns=15,
        night_turns=5,
        day_vision_radius=7,
        night_vision_radius=1,
        battery_capacity=25.0,
        initial_energy=25.0,
        night_recharge=3.0,
        rest_recharge_factor=2.0,
        move_cost=0.30,
        push_cost=0.60,
        pheromone_cost=0.06,
        memory_size=4,
        max_agent_types=4,
        max_pheromones=4,
        cargo_size=2.0,
        cargo_mobility=0.12,
    )

    # Fixed initial positions.  Agents start throughout the left half instead
    # of being artificially clustered around the cargo.
    rng = np.random.default_rng(2026)
    positions: list[tuple[float, float]] = []
    while len(positions) < rules.n_agents:
        pos = np.array(
            [rng.uniform(1.5, 10.0), rng.uniform(1.5, height - 1.5)],
            dtype=float,
        )
        if walls[int(pos[1]), int(pos[0])]:
            continue
        positions.append((float(pos[0]), float(pos[1])))

    return Scenario(
        name="playground",
        rules=rules,
        walls=_freeze(walls),
        agent_positions=_freeze(np.asarray(positions, dtype=float)),
        cargo_position=(12.0, 9.0),
        target_mask=_freeze(target),
    )


def load_scenario(name: str) -> Scenario:
    """Load a fixed teacher-defined scenario by name."""
    normalized = name.strip().lower()
    if normalized == "playground":
        return _playground()
    raise KeyError(f"unknown scenario {name!r}; available scenarios: 'playground'")
