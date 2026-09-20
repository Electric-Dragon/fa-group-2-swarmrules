from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import numpy as np

from .api import (
    Action,
    ActFunction,
    Cell,
    Config,
    Observation,
    Pheromone,
    Result,
    Rules,
    Scenario,
    SetupFunction,
)

_EPS = 1e-9
_GEOM_EPS = 1e-6
_COLLISION_STEP = 0.05


@dataclass
class _AgentState:
    position: np.ndarray
    energy: float
    memory: np.ndarray
    agent_type: int


@dataclass
class _PreparedAction:
    requested_pheromones: np.ndarray
    rest_intent: bool
    energy_after_move: float
    contact_position: np.ndarray
    push_vector: np.ndarray
    follow_vector: np.ndarray


class Engine:
    """Internal simulation engine.

    The public entry point is :func:`execute`; students normally do not need
    to instantiate this class.
    """

    def __init__(self, scenario: Scenario, setup: SetupFunction, act: ActFunction):
        self.scenario = scenario
        self.rules = scenario.rules
        self.setup_fn = setup
        self.act_fn = act

        self._validate_scenario()
        self.config = setup(self.rules)
        if not isinstance(self.config, Config):
            raise TypeError("setup(rules) must return fa2026.Config")
        self._validate_config(self.config)

        self.walls = np.asarray(scenario.walls, dtype=bool).copy()
        self.target_mask = np.asarray(scenario.target_mask, dtype=bool).copy()
        self.cargo_position = np.asarray(scenario.cargo_position, dtype=float).copy()

        # Static part of the visible cell map.  Cargo is added once per turn.
        self._static_cells = np.zeros(self.rules.grid_shape, dtype=np.uint8)
        self._static_cells[self.walls] |= int(Cell.WALL)
        self._static_cells[self.target_mask] |= int(Cell.TARGET)

        self.pheromone_fields = np.zeros(
            (len(self.config.pheromones), *self.rules.grid_shape), dtype=float
        )

        agent_types = self._build_agent_types()
        initial_memories = self._build_initial_memories(agent_types)
        self.agents = [
            _AgentState(
                position=np.asarray(pos, dtype=float).copy(),
                energy=float(self.rules.initial_energy),
                memory=initial_memories[i],
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
        if r.day_turns <= 0 or r.night_turns <= 0:
            raise ValueError("day_turns and night_turns must be positive")
        if r.day_vision_radius < 0 or r.night_vision_radius < 0:
            raise ValueError("vision radii must be non-negative")
        if r.battery_capacity <= 0:
            raise ValueError("battery_capacity must be positive")
        if not (0 <= r.initial_energy <= r.battery_capacity):
            raise ValueError("initial_energy must be within the battery capacity")
        if r.night_recharge < 0 or r.rest_recharge_factor < 1:
            raise ValueError("invalid recharge parameters")
        if r.move_cost < 0 or r.push_cost < 0 or r.pheromone_cost < 0:
            raise ValueError("energy costs must be non-negative")
        if r.memory_size < 0:
            raise ValueError("memory_size must be non-negative")
        if r.max_agent_types < 1 or r.max_pheromones < 0:
            raise ValueError("invalid type/pheromone limits")
        if r.cargo_size <= 0 or r.cargo_mobility < 0:
            raise ValueError("invalid cargo parameters")

        walls = np.asarray(self.scenario.walls)
        target = np.asarray(self.scenario.target_mask)
        positions = np.asarray(self.scenario.agent_positions)
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
        if cargo.shape != (2,):
            raise ValueError("cargo_position must contain exactly two coordinates")
        if not np.all(np.isfinite(cargo)):
            raise ValueError("cargo_position must be finite")
        if self._cargo_overlaps_wall(cargo, walls=walls):
            raise ValueError("initial cargo overlaps a wall")
        for pos in positions:
            if self._point_in_wall(pos, walls=walls):
                raise ValueError("an initial agent position lies inside a wall")
            if self._point_in_cargo(pos, cargo_position=cargo):
                raise ValueError("an initial agent position lies inside the cargo")

    def _validate_config(self, config: Config) -> None:
        r = self.rules
        if len(config.pheromones) > r.max_pheromones:
            raise ValueError("too many pheromone channels")
        for p in config.pheromones:
            if not isinstance(p, Pheromone):
                raise TypeError("Config.pheromones must contain Pheromone objects")
            if not (0.0 < p.decay <= 1.0):
                raise ValueError("pheromone decay must be in (0, 1]")

        if len(config.agent_type_counts) > r.max_agent_types - 1:
            raise ValueError("too many non-zero agent types")
        if any(int(c) != c or c < 0 for c in config.agent_type_counts):
            raise ValueError("agent_type_counts must contain non-negative integers")
        if sum(config.agent_type_counts) > r.n_agents:
            raise ValueError("agent_type_counts exceed the available agents")

        n_used_types = 1 + len(config.agent_type_counts)
        if config.initial_memory and len(config.initial_memory) != n_used_types:
            raise ValueError("initial_memory must contain one vector per used agent type")
        for mem in config.initial_memory:
            if len(mem) != r.memory_size:
                raise ValueError("each initial memory vector must have rules.memory_size entries")
            if not np.all(np.isfinite(np.asarray(mem, dtype=float))):
                raise ValueError("initial memory values must be finite")

        if config.parameters and len(config.parameters) != n_used_types:
            raise ValueError("parameters must contain one tuple per used agent type")
        for params in config.parameters:
            if not np.all(np.isfinite(np.asarray(params, dtype=float))):
                raise ValueError("Config.parameters must contain finite numbers")

    def _build_agent_types(self) -> np.ndarray:
        n = self.rules.n_agents
        types = np.zeros(n, dtype=int)
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
    # Public-ish state helpers used by visualization/tests

    @property
    def is_day(self) -> bool:
        cycle = self.rules.day_turns + self.rules.night_turns
        phase = self.turn % cycle
        return phase < self.rules.day_turns

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

    def _point_in_wall(self, pos: np.ndarray, walls: Optional[np.ndarray] = None) -> bool:
        if walls is None:
            walls = self.walls
        if not self._point_in_bounds(np.asarray(pos, dtype=float)):
            return True
        ix = int(math.floor(float(pos[0])))
        iy = int(math.floor(float(pos[1])))
        return bool(walls[iy, ix])

    def _cargo_bounds(self, cargo_position: Optional[np.ndarray] = None) -> tuple[float, float, float, float]:
        if cargo_position is None:
            cargo_position = self.cargo_position
        half = self.rules.cargo_size / 2.0
        x, y = float(cargo_position[0]), float(cargo_position[1])
        return x - half, x + half, y - half, y + half

    def _point_in_cargo(self, pos: np.ndarray, cargo_position: Optional[np.ndarray] = None) -> bool:
        """Return True only when a point penetrates the cargo interior.

        A point exactly on the cargo boundary is a valid contact position, not an
        overlap.  Keeping this distinction is important for pushers: they should
        be able to remain in contact with a cargo face without being repeatedly
        classified as lying inside the cargo.
        """
        xmin, xmax, ymin, ymax = self._cargo_bounds(cargo_position)
        x, y = float(pos[0]), float(pos[1])
        return (xmin + _GEOM_EPS) < x < (xmax - _GEOM_EPS) and (ymin + _GEOM_EPS) < y < (ymax - _GEOM_EPS)

    def _cargo_overlaps_wall(self, cargo_position: np.ndarray, walls: Optional[np.ndarray] = None) -> bool:
        if walls is None:
            walls = self.walls
        xmin, xmax, ymin, ymax = self._cargo_bounds(cargo_position)
        h, w = self.rules.grid_shape
        if xmin < 0 or ymin < 0 or xmax > w or ymax > h:
            return True

        # Cells touched by the square. Touching a cell boundary alone is not an overlap.
        ix0 = int(math.floor(xmin + _GEOM_EPS))
        ix1 = int(math.floor(xmax - _GEOM_EPS))
        iy0 = int(math.floor(ymin + _GEOM_EPS))
        iy1 = int(math.floor(ymax - _GEOM_EPS))
        ix0 = max(ix0, 0)
        iy0 = max(iy0, 0)
        ix1 = min(ix1, w - 1)
        iy1 = min(iy1, h - 1)
        return bool(np.any(walls[iy0 : iy1 + 1, ix0 : ix1 + 1]))

    def _segment_first_hit(
        self, start: np.ndarray, displacement: np.ndarray, *, include_cargo: bool = True
    ) -> tuple[np.ndarray, Optional[str], float]:
        """Move a point along a straight segment until first wall/cargo contact.

        Returns ``(last_valid_position, hit_kind, hit_fraction)``.  ``hit_fraction``
        is approximately the fraction of the requested displacement completed when
        contact is reached.
        """
        length = float(np.linalg.norm(displacement))
        if length <= _EPS:
            return start.copy(), None, 1.0

        n_steps = max(1, int(math.ceil(length / _COLLISION_STEP)))
        previous_t = 0.0
        previous_pos = start.copy()

        def kind_at(pos: np.ndarray) -> Optional[str]:
            if self._point_in_wall(pos):
                return "wall"
            if include_cargo and self._point_in_cargo(pos):
                return "cargo"
            return None

        for j in range(1, n_steps + 1):
            t = j / n_steps
            pos = start + t * displacement
            kind = kind_at(pos)
            if kind is None:
                previous_t = t
                previous_pos = pos
                continue

            # Refine contact between the last valid and first invalid positions.
            lo, hi = previous_t, t
            hit_kind = kind
            for _ in range(30):
                mid = 0.5 * (lo + hi)
                mid_pos = start + mid * displacement
                mid_kind = kind_at(mid_pos)
                if mid_kind is None:
                    lo = mid
                else:
                    hi = mid
                    hit_kind = mid_kind
            contact = start + lo * displacement
            return contact, hit_kind, lo

        return start + displacement, None, 1.0

    def _move_cargo_until_wall(self, displacement: np.ndarray) -> np.ndarray:
        length = float(np.linalg.norm(displacement))
        if length <= _EPS:
            return self.cargo_position.copy()

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
                pos_mid = self.cargo_position + mid * displacement
                if self._cargo_overlaps_wall(pos_mid):
                    hi = mid
                else:
                    lo = mid
            return self.cargo_position + lo * displacement
        return self.cargo_position + displacement

    def _depenetrate_agent_from_cargo(self, pos: np.ndarray) -> np.ndarray:
        if not self._point_in_cargo(pos):
            return pos
        xmin, xmax, ymin, ymax = self._cargo_bounds()
        x, y = float(pos[0]), float(pos[1])
        candidates = [
            np.array([xmin, y]),
            np.array([xmax, y]),
            np.array([x, ymin]),
            np.array([x, ymax]),
        ]
        candidates.sort(key=lambda c: float(np.linalg.norm(c - pos)))
        for candidate in candidates:
            if not self._point_in_wall(candidate) and not self._point_in_cargo(candidate):
                return candidate
        # Exceptional geometry (e.g. cargo flush against walls): keep the old point.
        return pos

    # ------------------------------------------------------------------
    # Observation

    def _cargo_overlaps_cell(self, ix: int, iy: int) -> bool:
        xmin, xmax, ymin, ymax = self._cargo_bounds()
        cell_xmin, cell_xmax = ix, ix + 1
        cell_ymin, cell_ymax = iy, iy + 1
        return not (
            xmax <= cell_xmin + _GEOM_EPS
            or xmin >= cell_xmax - _GEOM_EPS
            or ymax <= cell_ymin + _GEOM_EPS
            or ymin >= cell_ymax - _GEOM_EPS
        )

    def _visible_cell_map(self) -> np.ndarray:
        """Grid of Cell bit flags for the current world state."""
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
        self, position: np.ndarray, padded_cells: Optional[np.ndarray] = None
    ) -> np.ndarray:
        radius = self.rules.day_vision_radius if self.is_day else self.rules.night_vision_radius
        cx = int(math.floor(float(position[0])))
        cy = int(math.floor(float(position[1])))

        if padded_cells is None:
            padded_cells = np.pad(
                self._visible_cell_map(),
                radius,
                mode="constant",
                constant_values=int(Cell.WALL),
            )

        # Coordinates shift by +radius in the padded array; the slice starts
        # at the original (cx, cy), yielding exactly the centered local patch.
        size = 2 * radius + 1
        vision = padded_cells[cy : cy + size, cx : cx + size].copy()

        vision.setflags(write=False)
        return vision

    def _pheromones_for(
        self, position: np.ndarray, padded_pheromones: Optional[np.ndarray] = None
    ) -> np.ndarray:
        n = self.pheromone_fields.shape[0]
        if n == 0:
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
        padded_cells: Optional[np.ndarray] = None,
        padded_pheromones: Optional[np.ndarray] = None,
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

    def _normalize_action(self, action: Action) -> tuple[np.ndarray, np.ndarray, bool]:
        if not isinstance(action, Action):
            raise TypeError("act(...) must return an Action as its first result")
        move = np.asarray(action.move, dtype=float)
        if move.shape != (2,) or not np.all(np.isfinite(move)):
            raise ValueError("Action.move must contain two finite numbers")

        q = np.asarray(action.pheromones, dtype=float)
        n_phero = len(self.config.pheromones)
        if q.size == 0 and n_phero > 0:
            q = np.zeros(n_phero, dtype=float)
        if q.shape != (n_phero,):
            raise ValueError(f"Action.pheromones must contain {n_phero} values")
        if not np.all(np.isfinite(q)) or np.any(q < 0):
            raise ValueError("pheromone emission amounts must be finite and non-negative")

        rest_intent = float(np.linalg.norm(move)) <= _EPS and bool(np.all(q <= _EPS))
        return move, q, rest_intent

    def _affordable_base_move(self, move: np.ndarray, energy: float) -> tuple[np.ndarray, float]:
        length = float(np.linalg.norm(move))
        if length <= _EPS or self.rules.move_cost <= 0:
            return move.copy(), 0.0
        cost = self.rules.move_cost * length * length
        if cost <= energy + _EPS:
            return move.copy(), cost
        affordable_length = math.sqrt(max(energy, 0.0) / self.rules.move_cost)
        if affordable_length <= _EPS:
            return np.zeros(2), 0.0
        return move * (affordable_length / length), energy

    def _prepare_action(self, agent: _AgentState, action: Action) -> _PreparedAction:
        move_req, q_req, rest_intent = self._normalize_action(action)
        move, move_energy = self._affordable_base_move(move_req, agent.energy)
        energy_left = max(0.0, agent.energy - move_energy)

        contact, hit_kind, fraction = self._segment_first_hit(agent.position, move, include_cargo=True)
        push = np.zeros(2, dtype=float)
        follow = np.zeros(2, dtype=float)

        if hit_kind == "cargo":
            blocked = (1.0 - fraction) * move
            blocked_len = float(np.linalg.norm(blocked))
            if blocked_len > _EPS and self.rules.push_cost > 0 and energy_left > _EPS:
                full_push_cost = self.rules.push_cost * blocked_len * blocked_len
                if full_push_cost <= energy_left + _EPS:
                    push = blocked.copy()
                    follow = blocked.copy()
                    energy_left -= full_push_cost
                else:
                    affordable_len = math.sqrt(energy_left / self.rules.push_cost)
                    push = blocked * (affordable_len / blocked_len)
                    follow = push.copy()
                    energy_left = 0.0
            elif blocked_len > _EPS and self.rules.push_cost <= 0:
                push = blocked.copy()
                follow = blocked.copy()

        return _PreparedAction(
            requested_pheromones=q_req,
            rest_intent=rest_intent,
            energy_after_move=energy_left,
            contact_position=contact,
            push_vector=push,
            follow_vector=follow,
        )

    def _emit_pheromones(self, agent: _AgentState, requested: np.ndarray) -> np.ndarray:
        if requested.size == 0 or np.all(requested <= _EPS):
            return np.zeros_like(requested)
        if self.rules.pheromone_cost <= 0:
            actual = requested.copy()
        else:
            decays = np.array([p.decay for p in self.config.pheromones], dtype=float)
            unit_cost = self.rules.pheromone_cost / decays
            total_cost = float(np.dot(requested, unit_cost))
            if total_cost <= agent.energy + _EPS:
                actual = requested.copy()
                agent.energy = max(0.0, agent.energy - total_cost)
            elif total_cost > _EPS and agent.energy > _EPS:
                scale = agent.energy / total_cost
                actual = requested * scale
                agent.energy = 0.0
            else:
                actual = np.zeros_like(requested)

        if self.rules.pheromone_cost <= 0:
            # No energy change in this branch.
            pass
        return actual

    def _deposit_at(self, position: np.ndarray, amounts: np.ndarray, deposit_grid: np.ndarray) -> None:
        if amounts.size == 0:
            return
        ix = int(math.floor(float(position[0])))
        iy = int(math.floor(float(position[1])))
        h, w = self.rules.grid_shape
        if 0 <= ix < w and 0 <= iy < h and not self.walls[iy, ix]:
            deposit_grid[:, iy, ix] += amounts

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
        """Execute one turn. Returns True if the task is solved afterwards."""

        # Base night recharge happens before sensing/action.
        if not self.is_day:
            for agent in self.agents:
                agent.energy = min(
                    self.rules.battery_capacity,
                    agent.energy + self.rules.night_recharge,
                )

        vision_radius = (
            self.rules.day_vision_radius if self.is_day else self.rules.night_vision_radius
        )
        padded_cells = np.pad(
            self._visible_cell_map(),
            vision_radius,
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
        observations = [
            self._observation_for(a, padded_cells, padded_pheromones)
            for a in self.agents
        ]
        raw_actions: list[Action] = []
        new_memories: list[np.ndarray] = []

        # All controllers see the same beginning-of-turn world state.
        for agent, obs in zip(self.agents, observations):
            memory_view = agent.memory.copy()
            memory_view.setflags(write=False)
            action, new_memory = self.act_fn(
                obs, memory_view, agent.agent_type, self.config
            )
            new_memory = np.asarray(new_memory, dtype=float)
            if new_memory.shape != (self.rules.memory_size,):
                raise ValueError(
                    f"act(...) must return memory with shape {(self.rules.memory_size,)}"
                )
            if not np.all(np.isfinite(new_memory)):
                raise ValueError("memory values must be finite")
            raw_actions.append(action)
            new_memories.append(new_memory.copy())

        prepared = [self._prepare_action(a, act) for a, act in zip(self.agents, raw_actions)]

        # Apply movement energy and first movement phase.
        for agent, pa in zip(self.agents, prepared):
            agent.energy = min(pa.energy_after_move, self.rules.battery_capacity)
            agent.position = pa.contact_position.copy()

        # Sum all push contributions and move cargo once (overdamped cargo).
        total_push = np.sum([pa.push_vector for pa in prepared], axis=0) if prepared else np.zeros(2)
        cargo_displacement = self.rules.cargo_mobility * total_push
        self.cargo_position = self._move_cargo_until_wall(cargo_displacement)

        # Cargo does not receive impulses from agents it happens to move over.
        for agent in self.agents:
            if self._point_in_cargo(agent.position):
                agent.position = self._depenetrate_agent_from_cargo(agent.position)

        # Pushers continue along the affordable remainder of their original move.
        for agent, pa in zip(self.agents, prepared):
            if float(np.linalg.norm(pa.follow_vector)) <= _EPS:
                continue
            start = agent.position.copy()
            end, _, _ = self._segment_first_hit(start, pa.follow_vector, include_cargo=True)
            agent.position = end

        # New memory becomes persistent only after all physical actions.
        for agent, mem in zip(self.agents, new_memories):
            agent.memory = mem

        # Pheromone emission has second priority after movement/pushing.
        deposit_grid = np.zeros_like(self.pheromone_fields)
        for agent, pa in zip(self.agents, prepared):
            actual = self._emit_pheromones(agent, pa.requested_pheromones)
            self._deposit_at(agent.position, actual, deposit_grid)

        # Old pheromone decays; new pheromone is then deposited and is sensed next turn.
        for k, p in enumerate(self.config.pheromones):
            self.pheromone_fields[k] *= (1.0 - p.decay)
        self.pheromone_fields[:, self.walls] = 0.0
        self.pheromone_fields += deposit_grid

        # Extra night recharge for agents that intended to spend no energy at all.
        if not self.is_day and self.rules.rest_recharge_factor > 1.0:
            extra = self.rules.night_recharge * (self.rules.rest_recharge_factor - 1.0)
            for agent, pa in zip(self.agents, prepared):
                if pa.rest_intent:
                    agent.energy = min(self.rules.battery_capacity, agent.energy + extra)

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
    pheromone_channel: int = 0,
) -> Result:
    """Run one scenario until solved or until the technical safety limit.

    ``visualize``, ``delay`` and ``draw_every`` affect display only; they never
    alter the game mechanics.
    """
    if draw_every < 1:
        raise ValueError("draw_every must be at least 1")
    if max_turns < 1:
        raise ValueError("max_turns must be positive")

    engine = Engine(scenario, setup, act)
    visualizer = None
    if visualize:
        from .visualization import Visualizer

        visualizer = Visualizer(engine, pheromone_channel=pheromone_channel)
        visualizer.draw(delay=delay)

    if engine._cargo_inside_target():
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
