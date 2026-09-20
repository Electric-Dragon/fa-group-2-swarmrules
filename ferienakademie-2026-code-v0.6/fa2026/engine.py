from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .api import (
    Action,
    ActFunction,
    Cell,
    Config,
    Observation,
    Pheromone,
    Result,
    Scenario,
    SetupFunction,
)

_EPS = 1e-12
_GEOM_EPS = 1e-9
_COLLISION_STEP = 0.05  # cargo only; agent motion uses exact grid traversal


@dataclass
class _AgentState:
    position: np.ndarray
    energy: float
    memory: np.ndarray
    agent_type: int


@dataclass
class _PreparedAction:
    move: np.ndarray
    push: np.ndarray
    pheromones: np.ndarray
    energy_cost: float


class Engine:
    """Internal simulation engine. Student code normally uses :func:`execute`."""

    def __init__(self, scenario: Scenario, setup: SetupFunction, act: ActFunction):
        self.scenario = scenario
        self.rules = scenario.rules
        self.act_fn = act

        self._validate_scenario()
        self.config = setup(self.rules)
        if not isinstance(self.config, Config):
            raise TypeError("setup(rules) must return fa2026.Config")
        self._validate_config(self.config)

        self.walls = np.asarray(scenario.walls, dtype=bool).copy()
        self.target_mask = np.asarray(scenario.target_mask, dtype=bool).copy()
        self.cargo_position = np.asarray(scenario.cargo_position, dtype=float).copy()

        self._static_cells = np.zeros(self.rules.grid_shape, dtype=np.uint8)
        self._static_cells[self.walls] |= int(Cell.WALL)
        self._static_cells[self.target_mask] |= int(Cell.TARGET)

        self._pheromone_decay = np.array(
            [p.decay for p in self.config.pheromones], dtype=float
        )
        if len(self.config.pheromones):
            self._pheromone_unit_cost = self.rules.pheromone_cost / self._pheromone_decay
        else:
            self._pheromone_unit_cost = np.zeros(0, dtype=float)
        self.pheromone_fields = np.zeros(
            (len(self.config.pheromones), *self.rules.grid_shape), dtype=float
        )

        agent_types = self._build_agent_types()
        memories = self._build_initial_memories(agent_types)
        self.agents = [
            _AgentState(
                position=np.asarray(pos, dtype=float).copy(),
                energy=float(self.rules.initial_energy),
                memory=memories[i],
                agent_type=int(agent_types[i]),
            )
            for i, pos in enumerate(np.asarray(scenario.agent_positions, dtype=float))
        ]
        self.turn = 0

    # ------------------------------------------------------------------
    # Validation and configuration

    def _validate_scenario(self) -> None:
        r = self.rules
        h, w = r.grid_shape
        if h <= 0 or w <= 0:
            raise ValueError("grid_shape must contain positive dimensions")
        if r.n_agents <= 0:
            raise ValueError("n_agents must be positive")
        if r.vision_radius < 0:
            raise ValueError("vision_radius must be non-negative")
        if r.battery_capacity <= 0:
            raise ValueError("battery_capacity must be positive")
        if not (0 <= r.initial_energy <= r.battery_capacity):
            raise ValueError("initial_energy must be within the battery capacity")
        if r.recharge < 0:
            raise ValueError("recharge must be non-negative")
        if r.move_cost < 0 or r.push_cost < 0 or r.pheromone_cost < 0:
            raise ValueError("energy costs must be non-negative")
        if r.memory_size < 0:
            raise ValueError("memory_size must be non-negative")
        if r.max_agent_types < 1 or r.max_pheromones < 0:
            raise ValueError("invalid type/pheromone limits")
        if r.cargo_size <= 0 or r.cargo_mobility < 0:
            raise ValueError("invalid cargo parameters")

        walls = np.asarray(self.scenario.walls, dtype=bool)
        target = np.asarray(self.scenario.target_mask, dtype=bool)
        positions = np.asarray(self.scenario.agent_positions, dtype=float)
        if walls.shape != (h, w):
            raise ValueError(f"walls must have shape {(h, w)}")
        if target.shape != (h, w):
            raise ValueError(f"target_mask must have shape {(h, w)}")
        if not np.any(target):
            raise ValueError("target_mask must contain at least one target cell")
        if np.any(walls & target):
            raise ValueError("target cells may not also be walls")
        if positions.shape != (r.n_agents, 2):
            raise ValueError(f"agent_positions must have shape {(r.n_agents, 2)}")
        if not np.all(np.isfinite(positions)):
            raise ValueError("agent_positions must be finite")

        cargo = np.asarray(self.scenario.cargo_position, dtype=float)
        if cargo.shape != (2,) or not np.all(np.isfinite(cargo)):
            raise ValueError("cargo_position must contain two finite coordinates")
        if self._cargo_overlaps_wall(cargo, walls=walls):
            raise ValueError("initial cargo overlaps a wall")
        for pos in positions:
            if self._point_in_wall(pos, walls=walls):
                raise ValueError("an initial agent position lies inside a wall")
        # Agents deliberately may overlap cargo and one another.

    def _validate_config(self, config: Config) -> None:
        r = self.rules
        if len(config.pheromones) > r.max_pheromones:
            raise ValueError("too many pheromone channels")
        for p in config.pheromones:
            if not isinstance(p, Pheromone):
                raise TypeError("Config.pheromones must contain Pheromone objects")
            if not (0.0 < p.decay <= 1.0):
                raise ValueError("pheromone decay must be in (0, 1]")
            if p.color is not None and not isinstance(p.color, str):
                raise TypeError("pheromone color must be a color string or None")

        if len(config.agent_type_counts) > r.max_agent_types - 1:
            raise ValueError("too many non-zero agent types")
        if any(int(c) != c or c < 0 for c in config.agent_type_counts):
            raise ValueError("agent_type_counts must contain non-negative integers")
        if sum(config.agent_type_counts) > r.n_agents:
            raise ValueError("agent_type_counts exceed the available agents")

        n_types = 1 + len(config.agent_type_counts)
        if config.initial_memory and len(config.initial_memory) != n_types:
            raise ValueError("initial_memory must contain one vector per used agent type")
        for memory in config.initial_memory:
            if len(memory) != r.memory_size:
                raise ValueError("each initial memory vector must have rules.memory_size entries")
            if not np.all(np.isfinite(np.asarray(memory, dtype=float))):
                raise ValueError("initial memory values must be finite")

        if config.parameters and len(config.parameters) != n_types:
            raise ValueError("parameters must contain one tuple per used agent type")
        for parameters in config.parameters:
            if not np.all(np.isfinite(np.asarray(parameters, dtype=float))):
                raise ValueError("Config.parameters must contain finite numbers")

    def _build_agent_types(self) -> np.ndarray:
        types = np.zeros(self.rules.n_agents, dtype=int)
        cursor = 0
        for agent_type, count in enumerate(self.config.agent_type_counts, start=1):
            count = int(count)
            types[cursor : cursor + count] = agent_type
            cursor += count
        return types

    def _build_initial_memories(self, agent_types: np.ndarray) -> list[np.ndarray]:
        if self.config.initial_memory:
            templates = [np.asarray(m, dtype=float) for m in self.config.initial_memory]
        else:
            templates = [np.zeros(self.rules.memory_size, dtype=float)] * (
                1 + len(self.config.agent_type_counts)
            )
        return [templates[t].copy() for t in agent_types]

    # ------------------------------------------------------------------
    # State used by visualization/tests

    @property
    def agent_positions(self) -> np.ndarray:
        return np.array([a.position for a in self.agents], dtype=float)

    @property
    def agent_energies(self) -> np.ndarray:
        return np.array([a.energy for a in self.agents], dtype=float)

    # ------------------------------------------------------------------
    # Geometry

    def _point_in_bounds(self, pos: np.ndarray) -> bool:
        h, w = self.rules.grid_shape
        return 0.0 <= pos[0] < w and 0.0 <= pos[1] < h

    def _cell_is_wall(self, ix: int, iy: int, walls: np.ndarray | None = None) -> bool:
        if walls is None:
            walls = self.walls
        h, w = walls.shape
        return ix < 0 or iy < 0 or ix >= w or iy >= h or bool(walls[iy, ix])

    def _point_in_wall(self, pos: np.ndarray, walls: np.ndarray | None = None) -> bool:
        if walls is None:
            walls = self.walls
        x, y = float(pos[0]), float(pos[1])
        h, w = walls.shape
        if x < 0.0 or y < 0.0 or x >= w or y >= h:
            return True
        return bool(walls[int(math.floor(y)), int(math.floor(x))])

    def _cargo_bounds(
        self, cargo_position: np.ndarray | None = None
    ) -> tuple[float, float, float, float]:
        if cargo_position is None:
            cargo_position = self.cargo_position
        half = self.rules.cargo_size / 2.0
        x, y = float(cargo_position[0]), float(cargo_position[1])
        return x - half, x + half, y - half, y + half

    def _point_in_cargo(
        self, pos: np.ndarray, cargo_position: np.ndarray | None = None
    ) -> bool:
        """Return True for points inside or on the cargo square."""
        xmin, xmax, ymin, ymax = self._cargo_bounds(cargo_position)
        x, y = float(pos[0]), float(pos[1])
        return (
            xmin - _GEOM_EPS <= x <= xmax + _GEOM_EPS
            and ymin - _GEOM_EPS <= y <= ymax + _GEOM_EPS
        )

    def _cargo_overlaps_wall(
        self, cargo_position: np.ndarray, walls: np.ndarray | None = None
    ) -> bool:
        if walls is None:
            walls = self.walls
        xmin, xmax, ymin, ymax = self._cargo_bounds(cargo_position)
        h, w = self.rules.grid_shape
        if xmin < 0 or ymin < 0 or xmax > w or ymax > h:
            return True

        ix0 = max(0, int(math.floor(xmin + _GEOM_EPS)))
        ix1 = min(w - 1, int(math.floor(xmax - _GEOM_EPS)))
        iy0 = max(0, int(math.floor(ymin + _GEOM_EPS)))
        iy1 = min(h - 1, int(math.floor(ymax - _GEOM_EPS)))
        return bool(np.any(walls[iy0 : iy1 + 1, ix0 : ix1 + 1]))

    def _move_agent_until_wall(self, start: np.ndarray, displacement: np.ndarray) -> np.ndarray:
        """Move a point along a segment, stopping at the first crossed wall cell.

        Grid traversal makes this exact with respect to the wall grid and avoids
        small integration steps, so even very long requested moves cannot tunnel
        through a one-cell wall.
        """
        dx, dy = float(displacement[0]), float(displacement[1])
        if abs(dx) <= _EPS and abs(dy) <= _EPS:
            return start.copy()

        x, y = float(start[0]), float(start[1])
        ix, iy = int(math.floor(x)), int(math.floor(y))
        step_x = 1 if dx > 0 else (-1 if dx < 0 else 0)
        step_y = 1 if dy > 0 else (-1 if dy < 0 else 0)

        if step_x:
            boundary_x = ix + 1 if step_x > 0 else ix
            t_max_x = (boundary_x - x) / dx
            t_delta_x = 1.0 / abs(dx)
        else:
            t_max_x = math.inf
            t_delta_x = math.inf

        if step_y:
            boundary_y = iy + 1 if step_y > 0 else iy
            t_max_y = (boundary_y - y) / dy
            t_delta_y = 1.0 / abs(dy)
        else:
            t_max_y = math.inf
            t_delta_y = math.inf

        def stop_at(t_hit: float) -> np.ndarray:
            # Stay infinitesimally on the free side of the crossed boundary.
            t_safe = max(0.0, float(np.nextafter(t_hit, -math.inf)))
            return start + t_safe * displacement

        while True:
            t_next = min(t_max_x, t_max_y)
            if t_next > 1.0:
                return start + displacement

            # Crossing an exact grid corner: conservatively disallow corner
            # cutting if either side cell or the diagonal cell is a wall.
            if abs(t_max_x - t_max_y) <= 1e-14:
                candidates: list[tuple[int, int]] = []
                if step_x:
                    candidates.append((ix + step_x, iy))
                if step_y:
                    candidates.append((ix, iy + step_y))
                if step_x and step_y:
                    candidates.append((ix + step_x, iy + step_y))
                if any(self._cell_is_wall(cx, cy) for cx, cy in candidates):
                    return stop_at(t_next)
                ix += step_x
                iy += step_y
                t_max_x += t_delta_x
                t_max_y += t_delta_y
            elif t_max_x < t_max_y:
                next_ix = ix + step_x
                if self._cell_is_wall(next_ix, iy):
                    return stop_at(t_max_x)
                ix = next_ix
                t_max_x += t_delta_x
            else:
                next_iy = iy + step_y
                if self._cell_is_wall(ix, next_iy):
                    return stop_at(t_max_y)
                iy = next_iy
                t_max_y += t_delta_y

    def _move_cargo_until_wall(self, displacement: np.ndarray) -> np.ndarray:
        """Move cargo along a straight segment, stopping at the first wall."""
        length = float(np.linalg.norm(displacement))
        if length <= _EPS:
            return self.cargo_position.copy()

        # Only one cargo exists, so a small robust sweep is cheap compared with
        # per-agent control. The step is far below a grid-cell width, preventing
        # tunnelling even for large requested displacements.
        n_steps = max(1, int(math.ceil(length / _COLLISION_STEP)))
        previous_t = 0.0
        for j in range(1, n_steps + 1):
            t = j / n_steps
            pos = self.cargo_position + t * displacement
            if not self._cargo_overlaps_wall(pos):
                previous_t = t
                continue
            lo, hi = previous_t, t
            for _ in range(30):
                mid = 0.5 * (lo + hi)
                mid_pos = self.cargo_position + mid * displacement
                if self._cargo_overlaps_wall(mid_pos):
                    hi = mid
                else:
                    lo = mid
            return self.cargo_position + lo * displacement
        return self.cargo_position + displacement

    # ------------------------------------------------------------------
    # Observation

    def _cargo_overlaps_cell(self, ix: int, iy: int) -> bool:
        xmin, xmax, ymin, ymax = self._cargo_bounds()
        return not (
            xmax <= ix + _GEOM_EPS
            or xmin >= ix + 1 - _GEOM_EPS
            or ymax <= iy + _GEOM_EPS
            or ymin >= iy + 1 - _GEOM_EPS
        )

    def _visible_cell_map(self) -> np.ndarray:
        cells = self._static_cells.copy()
        xmin, xmax, ymin, ymax = self._cargo_bounds()
        h, w = self.rules.grid_shape
        ix0 = max(0, int(math.floor(xmin + _GEOM_EPS)))
        ix1 = min(w - 1, int(math.floor(xmax - _GEOM_EPS)))
        iy0 = max(0, int(math.floor(ymin + _GEOM_EPS)))
        iy1 = min(h - 1, int(math.floor(ymax - _GEOM_EPS)))
        for iy in range(iy0, iy1 + 1):
            for ix in range(ix0, ix1 + 1):
                if self._cargo_overlaps_cell(ix, iy):
                    cells[iy, ix] |= int(Cell.CARGO)
        return cells

    def _vision_for(
        self, position: np.ndarray, padded_cells: np.ndarray | None = None
    ) -> np.ndarray:
        radius = self.rules.vision_radius
        cx = int(math.floor(float(position[0])))
        cy = int(math.floor(float(position[1])))
        if padded_cells is None:
            padded_cells = np.pad(
                self._visible_cell_map(),
                radius,
                mode="constant",
                constant_values=int(Cell.WALL),
            )
        size = 2 * radius + 1
        vision = padded_cells[cy : cy + size, cx : cx + size].copy()
        vision.setflags(write=False)
        return vision

    def _pheromones_for(
        self, position: np.ndarray, padded_pheromones: np.ndarray | None = None
    ) -> np.ndarray:
        n_channels = self.pheromone_fields.shape[0]
        if n_channels == 0:
            patch = np.zeros((0, 3, 3), dtype=float)
            patch.setflags(write=False)
            return patch
        cx = int(math.floor(float(position[0])))
        cy = int(math.floor(float(position[1])))
        if padded_pheromones is None:
            padded_pheromones = np.pad(
                self.pheromone_fields,
                ((0, 0), (1, 1), (1, 1)),
                mode="constant",
            )
        patch = padded_pheromones[:, cy : cy + 3, cx : cx + 3].copy()
        patch.setflags(write=False)
        return patch

    def _observation_for(
        self,
        agent: _AgentState,
        padded_cells: np.ndarray | None = None,
        padded_pheromones: np.ndarray | None = None,
    ) -> Observation:
        frac = agent.position - np.floor(agent.position)
        return Observation(
            energy=float(agent.energy),
            cell_position=(float(frac[0]), float(frac[1])),
            vision=self._vision_for(agent.position, padded_cells),
            pheromones=self._pheromones_for(agent.position, padded_pheromones),
        )

    # ------------------------------------------------------------------
    # Turn update

    def _normalize_action(self, action: Action) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not isinstance(action, Action):
            raise TypeError("act(...) must return an Action as its first result")

        move = np.asarray(action.move, dtype=float)
        push = np.asarray(action.push, dtype=float)
        if move.shape != (2,) or not np.all(np.isfinite(move)):
            raise ValueError("Action.move must contain two finite numbers")
        if push.shape != (2,) or not np.all(np.isfinite(push)):
            raise ValueError("Action.push must contain two finite numbers")

        emissions = np.asarray(action.pheromones, dtype=float)
        n_pheromones = len(self.config.pheromones)
        if emissions.size == 0 and n_pheromones > 0:
            emissions = np.zeros(n_pheromones, dtype=float)
        if emissions.shape != (n_pheromones,):
            raise ValueError(f"Action.pheromones must contain {n_pheromones} values")
        if not np.all(np.isfinite(emissions)) or np.any(emissions < 0):
            raise ValueError("pheromone emission amounts must be finite and non-negative")
        return move, push, emissions

    def _prepare_action(self, action: Action, energy: float) -> _PreparedAction:
        """Validate an action and scale it uniformly if the battery is too low.

        Move and push costs are quadratic in vector magnitude. Pheromone cost is
        linear in emitted amount. If the full simultaneous request is too
        expensive, all three action amplitudes are multiplied by the same factor
        ``s``. The engine solves ``M*s^2 + P*s = energy`` exactly, where ``M``
        is the requested mechanical cost and ``P`` the requested pheromone cost.
        """
        move, push, emissions = self._normalize_action(action)
        mechanical_cost = (
            self.rules.move_cost * float(np.dot(move, move))
            + self.rules.push_cost * float(np.dot(push, push))
        )
        pheromone_cost = float(np.dot(emissions, self._pheromone_unit_cost))
        requested_cost = mechanical_cost + pheromone_cost
        if not math.isfinite(requested_cost):
            raise ValueError("requested action is too large")

        if requested_cost <= energy + _EPS:
            return _PreparedAction(move, push, emissions, max(0.0, requested_cost))

        if energy <= _EPS or requested_cost <= _EPS:
            return _PreparedAction(
                np.zeros(2, dtype=float),
                np.zeros(2, dtype=float),
                np.zeros_like(emissions),
                0.0,
            )

        # M*s^2 + P*s = E, choosing the positive root in [0, 1].
        if mechanical_cost <= _EPS:
            scale = energy / pheromone_cost
        elif pheromone_cost <= _EPS:
            scale = math.sqrt(energy / mechanical_cost)
        else:
            discriminant = pheromone_cost * pheromone_cost + 4.0 * mechanical_cost * energy
            scale = (
                2.0 * energy
                / (pheromone_cost + math.sqrt(discriminant))
            )
        scale = min(1.0, max(0.0, scale))

        scaled_move = move * scale
        scaled_push = push * scale
        scaled_emissions = emissions * scale
        actual_cost = mechanical_cost * scale * scale + pheromone_cost * scale
        return _PreparedAction(
            scaled_move,
            scaled_push,
            scaled_emissions,
            min(energy, max(0.0, actual_cost)),
        )

    def _deposit_at(self, position: np.ndarray, amounts: np.ndarray, grid: np.ndarray) -> None:
        if amounts.size == 0 or not np.any(amounts > 0.0):
            return
        ix = int(math.floor(float(position[0])))
        iy = int(math.floor(float(position[1])))
        h, w = self.rules.grid_shape
        if 0 <= ix < w and 0 <= iy < h and not self.walls[iy, ix]:
            grid[:, iy, ix] += amounts

    def _cargo_inside_target(self) -> bool:
        """Return True when every grid cell overlapped by the cargo is target."""
        xmin, xmax, ymin, ymax = self._cargo_bounds()
        h, w = self.rules.grid_shape
        if xmin < 0 or ymin < 0 or xmax > w or ymax > h:
            return False
        ix0 = max(0, int(math.floor(xmin + _GEOM_EPS)))
        ix1 = min(w - 1, int(math.floor(xmax - _GEOM_EPS)))
        iy0 = max(0, int(math.floor(ymin + _GEOM_EPS)))
        iy1 = min(h - 1, int(math.floor(ymax - _GEOM_EPS)))
        return bool(np.all(self.target_mask[iy0 : iy1 + 1, ix0 : ix1 + 1]))

    def step(self) -> bool:
        """Execute one turn. Return True if the task is solved afterwards."""

        # 1) Recharge, then sense the same beginning-of-turn world for everyone.
        recharge = self.rules.recharge
        capacity = self.rules.battery_capacity
        for agent in self.agents:
            agent.energy = min(capacity, agent.energy + recharge)

        radius = self.rules.vision_radius
        padded_cells = np.pad(
            self._visible_cell_map(),
            radius,
            mode="constant",
            constant_values=int(Cell.WALL),
        )
        padded_pheromones = (
            np.pad(
                self.pheromone_fields,
                ((0, 0), (1, 1), (1, 1)),
                mode="constant",
            )
            if self.pheromone_fields.shape[0] > 0
            else None
        )

        prepared: list[_PreparedAction] = []
        new_memories: list[np.ndarray] = []

        # 2) Evaluate all controllers from the same snapshot.
        for agent in self.agents:
            observation = self._observation_for(agent, padded_cells, padded_pheromones)
            memory_view = agent.memory.copy()
            memory_view.setflags(write=False)
            returned = self.act_fn(observation, memory_view, agent.agent_type, self.config)
            if not isinstance(returned, tuple) or len(returned) != 2:
                raise TypeError("act(...) must return (Action, new_memory)")
            action, new_memory = returned
            new_memory = np.asarray(new_memory, dtype=float)
            if new_memory.shape != (self.rules.memory_size,):
                raise ValueError(
                    f"act(...) must return memory with shape {(self.rules.memory_size,)}"
                )
            if not np.all(np.isfinite(new_memory)):
                raise ValueError("memory values must be finite")
            prepared.append(self._prepare_action(action, agent.energy))
            new_memories.append(new_memory.copy())

        # 3) Spend requested energy whether or not geometry later blocks the action.
        for agent, action in zip(self.agents, prepared):
            agent.energy = max(0.0, agent.energy - action.energy_cost)

        # 4) Agents move. They may overlap one another and cargo; only walls block.
        for agent, action in zip(self.agents, prepared):
            agent.position = self._move_agent_until_wall(agent.position, action.move)

        # 5) Pushes from agents overlapping the cargo after movement add vectorially.
        total_push = np.zeros(2, dtype=float)
        for agent, action in zip(self.agents, prepared):
            if self._point_in_cargo(agent.position):
                total_push += action.push
        self.cargo_position = self._move_cargo_until_wall(
            self.rules.cargo_mobility * total_push
        )

        # 6) Persist memory returned by the controller.
        for agent, memory in zip(self.agents, new_memories):
            agent.memory = memory

        # 7) Emit pheromones at final agent positions. Existing field decays first;
        #    new deposits therefore become visible on the next turn without being
        #    immediately decayed: P(t+1) = (1-d) P(t) + Q(t).
        if self.pheromone_fields.shape[0] > 0:
            self.pheromone_fields *= (1.0 - self._pheromone_decay)[:, None, None]
            self.pheromone_fields[:, self.walls] = 0.0
            deposit_grid = np.zeros_like(self.pheromone_fields)
            for agent, action in zip(self.agents, prepared):
                self._deposit_at(agent.position, action.pheromones, deposit_grid)
            self.pheromone_fields += deposit_grid

        self.turn += 1
        return self._cargo_inside_target()


def execute(
    scenario: Scenario,
    setup: SetupFunction,
    act: ActFunction,
    *,
    visualize: bool = True,
    delay: float = 0.02,
    draw_every: int = 1,
    max_turns: int = 100_000,
) -> Result:
    """Run a scenario until solved or until the technical safety limit.

    ``visualize``, ``delay`` and ``draw_every`` affect display only.
    ``max_turns`` is a technical safety limit, not part of the game rules.
    """
    if draw_every < 1:
        raise ValueError("draw_every must be at least 1")
    if max_turns < 1:
        raise ValueError("max_turns must be positive")
    if delay < 0:
        raise ValueError("delay must be non-negative")

    engine = Engine(scenario, setup, act)
    visualizer = None
    if visualize:
        from .visualization import Visualizer

        visualizer = Visualizer(engine)
        visualizer.draw(delay=delay)

    if engine._cargo_inside_target():
        if visualizer is not None:
            visualizer.finish()
        return Result(success=True, turns=0)

    success = False
    while engine.turn < max_turns:
        success = engine.step()
        if visualizer is not None and (engine.turn % draw_every == 0 or success):
            visualizer.draw(delay=delay)
        if success:
            break

    if visualizer is not None:
        visualizer.finish()
    return Result(success=success, turns=engine.turn)
