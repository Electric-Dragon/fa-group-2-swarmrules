from dataclasses import replace

import numpy as np
import pytest

from fa2026 import Action, Config, Pheromone, Scenario, execute, load_scenario
from fa2026.engine import Engine


def stationary_setup(rules):
    return Config(initial_memory=((0.0,) * rules.memory_size,))


def stationary_act(observation, memory, agent_type, config):
    return Action(), memory


def _custom_scenario(base, *, positions=None, cargo=None, walls=None, rules=None, target=None):
    return Scenario(
        name="test",
        rules=rules or base.rules,
        walls=base.walls if walls is None else walls,
        agent_positions=base.agent_positions if positions is None else positions,
        cargo_position=base.cargo_position if cargo is None else cargo,
        target_mask=base.target_mask if target is None else target,
    )


def _one_special_agent_setup(rules):
    zero = (0.0,) * rules.memory_size
    return Config(agent_type_counts=(1,), initial_memory=(zero, zero))


def test_playground_loads_and_runs():
    scenario = load_scenario("playground")
    result = execute(
        scenario,
        stationary_setup,
        stationary_act,
        visualize=False,
        max_turns=3,
    )
    assert result.success is False
    assert result.turns == 3


def test_energy_recharges_every_turn():
    scenario = load_scenario("playground")
    engine = Engine(scenario, stationary_setup, stationary_act)
    engine.agents[0].energy = 0.0
    engine.step()
    assert np.isclose(engine.agents[0].energy, scenario.rules.recharge)


def test_agents_can_move_through_cargo():
    scenario = load_scenario("playground")
    positions = scenario.agent_positions.copy()
    positions[:] = np.array([2.0, 2.0])
    cargo = np.asarray(scenario.cargo_position, dtype=float)
    positions[0] = cargo + np.array([-3.0, 0.0])
    rules = replace(scenario.rules, move_cost=0.0)
    custom = _custom_scenario(scenario, positions=positions, rules=rules)

    def act(observation, memory, agent_type, config):
        if agent_type == 1:
            return Action(move=(4.0, 0.0)), memory
        return Action(), memory

    engine = Engine(custom, _one_special_agent_setup, act)
    engine.step()
    assert engine.agents[0].position[0] > cargo[0]


def test_large_agent_move_cannot_tunnel_through_wall():
    scenario = load_scenario("playground")
    walls = scenario.walls.copy()
    walls[1:-1, 10] = True
    positions = scenario.agent_positions.copy()
    positions[:] = np.array([2.5, 2.5])
    positions[0] = np.array([2.5, 18.5])
    rules = replace(scenario.rules, move_cost=0.0)
    custom = _custom_scenario(scenario, positions=positions, walls=walls, rules=rules)

    def act(observation, memory, agent_type, config):
        return (Action(move=(100.0, 0.0)), memory) if agent_type == 1 else (Action(), memory)

    engine = Engine(custom, _one_special_agent_setup, act)
    engine.step()
    x, y = engine.agents[0].position
    assert x < 10.0
    assert not engine._point_in_wall(np.array([x, y]))


def test_push_requires_overlap_after_movement_but_costs_energy_anyway():
    scenario = load_scenario("playground")
    positions = scenario.agent_positions.copy()
    positions[:] = np.array([2.0, 2.0])
    custom = _custom_scenario(scenario, positions=positions)

    def act(observation, memory, agent_type, config):
        if agent_type == 1:
            return Action(push=(1.0, 0.0)), memory
        return Action(), memory

    engine = Engine(custom, _one_special_agent_setup, act)
    before_cargo = engine.cargo_position.copy()
    before_energy = engine.agents[0].energy
    engine.step()
    assert np.allclose(engine.cargo_position, before_cargo)
    available = min(scenario.rules.battery_capacity, before_energy + scenario.rules.recharge)
    assert np.isclose(engine.agents[0].energy, available - scenario.rules.push_cost)


def test_agent_can_move_into_cargo_and_push_in_same_turn():
    scenario = load_scenario("playground")
    positions = scenario.agent_positions.copy()
    positions[:] = np.array([2.0, 2.0])
    cargo = np.asarray(scenario.cargo_position, dtype=float)
    positions[0] = cargo + np.array([-1.5, 0.0])
    custom = _custom_scenario(scenario, positions=positions)

    def act(observation, memory, agent_type, config):
        if agent_type == 1:
            return Action(move=(0.75, 0.0), push=(1.0, 0.0)), memory
        return Action(), memory

    engine = Engine(custom, _one_special_agent_setup, act)
    before = engine.cargo_position.copy()
    engine.step()
    assert engine._point_in_cargo(engine.agents[0].position, cargo_position=before)
    assert engine.cargo_position[0] > before[0]


def test_opposing_pushes_cancel_but_both_cost_energy():
    scenario = load_scenario("playground")
    positions = scenario.agent_positions.copy()
    positions[:] = np.array([2.0, 2.0])
    positions[0] = np.array(scenario.cargo_position)
    positions[1] = np.array(scenario.cargo_position)
    custom = _custom_scenario(scenario, positions=positions)

    def setup(rules):
        zero = (0.0,) * rules.memory_size
        return Config(agent_type_counts=(1, 1), initial_memory=(zero, zero, zero))

    def act(observation, memory, agent_type, config):
        if agent_type == 1:
            return Action(push=(1.0, 0.0)), memory
        if agent_type == 2:
            return Action(push=(-1.0, 0.0)), memory
        return Action(), memory

    engine = Engine(custom, setup, act)
    before = engine.cargo_position.copy()
    engine.step()
    assert np.allclose(engine.cargo_position, before)
    assert engine.agents[0].energy < scenario.rules.initial_energy + scenario.rules.recharge
    assert engine.agents[1].energy < scenario.rules.initial_energy + scenario.rules.recharge


def test_large_cargo_push_cannot_tunnel_through_wall():
    scenario = load_scenario("playground")
    walls = scenario.walls.copy()
    walls[16:21, 30] = True
    positions = scenario.agent_positions.copy()
    positions[:] = np.array([2.0, 2.0])
    positions[0] = np.array(scenario.cargo_position)
    rules = replace(
        scenario.rules,
        battery_capacity=1_000_000.0,
        initial_energy=1_000_000.0,
        recharge=0.0,
        push_cost=0.0,
        cargo_mobility=1.0,
    )
    custom = _custom_scenario(scenario, positions=positions, walls=walls, rules=rules)

    def act(observation, memory, agent_type, config):
        return (Action(push=(100.0, 0.0)), memory) if agent_type == 1 else (Action(), memory)

    engine = Engine(custom, _one_special_agent_setup, act)
    engine.step()
    # cargo half-size is 1; wall starts at x=30, so center cannot pass x=29
    assert engine.cargo_position[0] <= 29.0 + 1e-6
    assert not engine._cargo_overlaps_wall(engine.cargo_position)


def test_full_action_is_scaled_uniformly_when_energy_is_insufficient():
    scenario = load_scenario("playground")

    def setup(rules):
        return Config(
            pheromones=(Pheromone(decay=0.5),),
            initial_memory=((0.0,) * rules.memory_size,),
        )

    engine = Engine(scenario, setup, stationary_act)
    requested = Action(move=(10.0, 0.0), push=(5.0, 0.0), pheromones=(100.0,))
    prepared = engine._prepare_action(requested, energy=1.0)
    scales = [
        prepared.move[0] / 10.0,
        prepared.push[0] / 5.0,
        prepared.pheromones[0] / 100.0,
    ]
    assert np.allclose(scales, scales[0])
    assert 0.0 < scales[0] < 1.0
    assert np.isclose(prepared.energy_cost, 1.0)


def test_pheromone_update_is_decay_then_new_deposit():
    scenario = load_scenario("playground")

    def setup(rules):
        return Config(
            pheromones=(Pheromone(decay=0.25, color="blue"), Pheromone(decay=0.50)),
            initial_memory=((0.0,) * rules.memory_size,),
        )

    def act(observation, memory, agent_type, config):
        m = memory.copy()
        if m[0] < 0.5:
            m[0] = 1.0
            return Action(pheromones=(1.0, 2.0)), m
        return Action(), m

    engine = Engine(scenario, setup, act)
    pos = engine.agents[0].position.copy()
    ix, iy = int(pos[0]), int(pos[1])
    engine.step()
    assert np.allclose(engine.pheromone_fields[:, iy, ix], [1.0, 2.0])
    engine.step()
    assert np.allclose(engine.pheromone_fields[:, iy, ix], [0.75, 1.0])


def test_observation_arrays_are_read_only():
    scenario = load_scenario("playground")

    def setup(rules):
        return Config(
            pheromones=(Pheromone(decay=0.2),),
            initial_memory=((0.0,) * rules.memory_size,),
        )

    engine = Engine(scenario, setup, stationary_act)
    obs = engine._observation_for(engine.agents[0])
    assert obs.vision.flags.writeable is False
    assert obs.pheromones.flags.writeable is False


def test_scenario_arrays_are_read_only():
    scenario = load_scenario("playground")
    assert scenario.walls.flags.writeable is False
    assert scenario.agent_positions.flags.writeable is False
    assert scenario.target_mask.flags.writeable is False


def test_success_requires_full_cargo_containment():
    scenario = load_scenario("playground")
    engine = Engine(scenario, stationary_setup, stationary_act)
    ys, xs = np.nonzero(scenario.target_mask)
    target_center = np.array([float(np.mean(xs) + 0.5), float(np.mean(ys) + 0.5)])
    engine.cargo_position = np.array([float(np.min(xs) + 0.1), target_center[1]])
    assert engine._cargo_inside_target() is False
    engine.cargo_position = target_center
    assert engine._cargo_inside_target() is True


def test_invalid_action_values_are_rejected():
    scenario = load_scenario("playground")

    def bad_act(observation, memory, agent_type, config):
        return Action(move=(np.nan, 0.0)), memory

    engine = Engine(scenario, stationary_setup, bad_act)
    with pytest.raises(ValueError, match="finite"):
        engine.step()


def test_invalid_memory_shape_is_rejected():
    scenario = load_scenario("playground")

    def bad_act(observation, memory, agent_type, config):
        return Action(), np.zeros(1)

    engine = Engine(scenario, stationary_setup, bad_act)
    with pytest.raises(ValueError, match="memory"):
        engine.step()


def test_invalid_pheromone_configuration_is_rejected():
    scenario = load_scenario("playground")

    def setup(rules):
        return Config(pheromones=(Pheromone(decay=0.0),))

    with pytest.raises(ValueError, match="decay"):
        Engine(scenario, setup, stationary_act)


def test_repeated_runs_are_deterministic():
    scenario = load_scenario("playground")

    def act(observation, memory, agent_type, config):
        return Action(move=(0.17, -0.11)), memory

    a = Engine(scenario, stationary_setup, act)
    b = Engine(scenario, stationary_setup, act)
    for _ in range(25):
        a.step()
        b.step()
    assert np.array_equal(a.agent_positions, b.agent_positions)
    assert np.array_equal(a.agent_energies, b.agent_energies)
    assert np.array_equal(a.cargo_position, b.cargo_position)


def test_execute_rejects_invalid_display_options():
    scenario = load_scenario("playground")
    with pytest.raises(ValueError):
        execute(scenario, stationary_setup, stationary_act, visualize=False, delay=-1)
    with pytest.raises(ValueError):
        execute(scenario, stationary_setup, stationary_act, visualize=False, draw_every=0)


def test_showcase_loads_and_runs():
    scenario = load_scenario("showcase")
    result = execute(
        scenario,
        stationary_setup,
        stationary_act,
        visualize=False,
        max_turns=2,
    )
    assert result.success is False
    assert result.turns == 2
    assert np.any(scenario.walls[1:-1, 1:-1])


def test_agent_type_counts_assign_permanent_types():
    scenario = load_scenario("showcase")

    def setup(rules):
        zero = (0.0,) * rules.memory_size
        return Config(
            agent_type_counts=(3, 2),
            initial_memory=(zero, zero, zero),
        )

    engine = Engine(scenario, setup, stationary_act)
    types = np.array([agent.agent_type for agent in engine.agents])
    assert np.array_equal(types[:5], [1, 1, 1, 2, 2])
    assert np.all(types[5:] == 0)
