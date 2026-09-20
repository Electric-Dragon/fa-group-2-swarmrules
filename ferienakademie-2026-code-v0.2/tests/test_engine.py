import numpy as np

from fa2026 import Action, Config, Pheromone, execute, load_scenario
from fa2026.engine import Engine


def stationary_setup(rules):
    return Config(initial_memory=((0.0,) * rules.memory_size,))


def stationary_act(observation, memory, agent_type, config):
    return Action(), memory


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


def test_day_and_night_change_vision_shape_and_recharge():
    scenario = load_scenario("playground")
    engine = Engine(scenario, stationary_setup, stationary_act)
    obs_day = engine._observation_for(engine.agents[0])
    assert obs_day.vision.shape == (2 * scenario.rules.day_vision_radius + 1,) * 2

    # Advance to the first night turn.
    for _ in range(scenario.rules.day_turns):
        engine.step()
    assert engine.is_day is False
    before = engine.agents[0].energy
    obs_night = engine._observation_for(engine.agents[0])
    assert obs_night.vision.shape == (2 * scenario.rules.night_vision_radius + 1,) * 2
    engine.step()
    assert engine.agents[0].energy >= before


def test_pheromone_decay_and_next_turn_visibility():
    scenario = load_scenario("playground")

    def setup(rules):
        return Config(
            pheromones=(Pheromone(decay=0.25),),
            initial_memory=((0.0,) * rules.memory_size,),
        )

    def act(observation, memory, agent_type, config):
        return Action(pheromones=(1.0,)), memory

    engine = Engine(scenario, setup, act)
    start_pos = engine.agents[0].position.copy()
    ix, iy = int(start_pos[0]), int(start_pos[1])
    assert engine.pheromone_fields[0, iy, ix] == 0.0
    engine.step()
    assert engine.pheromone_fields[0, iy, ix] > 0.0
    obs = engine._observation_for(engine.agents[0])
    assert np.max(obs.pheromones) > 0.0


def test_opposing_pushes_cancel():
    scenario = load_scenario("playground")
    rules = scenario.rules
    # Create a tiny deterministic custom placement around the cargo.
    positions = scenario.agent_positions.copy()
    positions[:] = np.array([2.0, 2.0])
    positions[0] = np.array([10.8, 9.0])
    positions[1] = np.array([13.2, 9.0])
    scenario = type(scenario)(
        name="push_test",
        rules=rules,
        walls=scenario.walls,
        agent_positions=positions,
        cargo_position=scenario.cargo_position,
        target_mask=scenario.target_mask,
    )

    def setup(rules):
        # Two types so the two test agents can push in opposite directions.
        return Config(
            agent_type_counts=(1, 1),
            initial_memory=((0.0,) * rules.memory_size,) * 3,
        )

    def act(observation, memory, agent_type, config):
        if agent_type == 1:
            return Action(move=(1.0, 0.0)), memory
        if agent_type == 2:
            return Action(move=(-1.0, 0.0)), memory
        return Action(), memory

    engine = Engine(scenario, setup, act)
    before = engine.cargo_position.copy()
    engine.step()
    assert np.linalg.norm(engine.cargo_position - before) < 1e-5


def test_agents_never_finish_inside_cargo_in_demo_run():
    """Cargo motion may overtake point agents, but overlap must be resolved."""
    from examples.weak_controller import setup, act

    scenario = load_scenario("playground")
    engine = Engine(scenario, setup, act)
    for _ in range(100):
        engine.step()
        assert not any(engine._point_in_cargo(a.position) for a in engine.agents)


def test_cargo_boundary_is_contact_not_overlap():
    scenario = load_scenario("playground")
    engine = Engine(scenario, stationary_setup, stationary_act)
    xmin, xmax, ymin, ymax = engine._cargo_bounds()
    assert not engine._point_in_cargo(np.array([xmin, 0.5 * (ymin + ymax)]))
    assert not engine._point_in_cargo(np.array([xmax, 0.5 * (ymin + ymax)]))
    assert engine._point_in_cargo(np.array([0.5 * (xmin + xmax), 0.5 * (ymin + ymax)]))


def test_playground_reference_controller_solves_easy_scenario():
    from examples.playground_controller import act, setup

    scenario = load_scenario("playground")
    result = execute(
        scenario,
        setup,
        act,
        visualize=False,
        max_turns=500,
    )
    assert result.success is True
    assert result.turns < 100


def test_scenario_arrays_are_read_only():
    scenario = load_scenario("playground")
    assert scenario.walls.flags.writeable is False
    assert scenario.agent_positions.flags.writeable is False
    assert scenario.target_mask.flags.writeable is False


def test_success_requires_all_overlapped_cargo_cells_to_be_target():
    scenario = load_scenario("playground")
    engine = Engine(scenario, stationary_setup, stationary_act)

    # Center of the cargo is in the target, but the square still sticks out.
    engine.cargo_position = np.array([18.1, 9.0])
    assert engine._cargo_inside_target() is False

    # Fully inside the generous target rectangle.
    engine.cargo_position = np.array([20.0, 9.0])
    assert engine._cargo_inside_target() is True


def test_cargo_has_no_momentum_between_turns():
    scenario = load_scenario("playground")
    positions = scenario.agent_positions.copy()
    positions[:] = np.array([2.0, 2.0])
    positions[0] = np.array([10.8, 9.0])
    custom = type(scenario)(
        name="no_momentum_test",
        rules=scenario.rules,
        walls=scenario.walls,
        agent_positions=positions,
        cargo_position=scenario.cargo_position,
        target_mask=scenario.target_mask,
    )

    def setup(rules):
        return Config(
            agent_type_counts=(1,),
            initial_memory=((0.0,) * rules.memory_size,) * 2,
        )

    def act(observation, memory, agent_type, config):
        new_memory = memory.copy()
        if agent_type == 1 and memory[0] < 0.5:
            new_memory[0] = 1.0
            return Action(move=(1.0, 0.0)), new_memory
        return Action(), new_memory

    engine = Engine(custom, setup, act)
    before = engine.cargo_position.copy()
    engine.step()
    after_push = engine.cargo_position.copy()
    engine.step()
    after_rest = engine.cargo_position.copy()

    assert after_push[0] > before[0]
    assert np.allclose(after_rest, after_push)
