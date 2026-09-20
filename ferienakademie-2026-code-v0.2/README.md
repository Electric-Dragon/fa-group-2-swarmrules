# Ferienakademie 2026 – shared project framework (v0.2)

This repository contains the shared-project framework for **Learning Rules for
Life-Like Emergent Behavior**.

The student-facing model is deliberately small:

```python
config = setup(rules)
action, new_memory = act(observation, memory, agent_type, config)
```

The world is turn-based. Agents have continuous positions on a grid-based
playing field, local vision, finite energy and limited memory. They communicate
through optional pheromone fields. Moving into the cargo produces a collective
push; the cargo is overdamped and has no momentum between turns.

## Recommended starting point

Open `getting_started.ipynb`. It contains a complete but simple controller that
solves the forgiving `playground` scenario and demonstrates the main API:

- local vision and limited memory
- day/night and battery recharge
- pheromone emission and sensing
- cargo detection and cooperative pushing
- visual and headless execution

The playground is deliberately easy. The same controller interface can later be
used with more difficult teacher-defined scenarios and parameter sets.

## Scripted run

A command-line equivalent is available in `examples/demo_run.py`:

```bash
python -m examples.demo_run
```

For a headless run:

```python
from fa2026 import execute, load_scenario
from examples.playground_controller import setup, act

scenario = load_scenario("playground")
result = execute(scenario, setup, act, visualize=False, max_turns=500)
print(result)
```

`visualize`, `delay`, and `draw_every` affect only display, never the game.

## Current mechanics

- deterministic, turn-based simulation
- continuous point agents, grid-based world
- fixed day/night cycle and local vision
- battery, quadratic movement cost and night recharge
- one square cargo and one target area
- collective pushing with no cargo momentum
- student-defined pheromone channels with relative decay
- limited memory and permanent agent types
- Matplotlib visualization
- fixed teacher-defined scenarios

The numerical parameters and final challenge scenarios are intentionally not
final yet. The next step is to test/tune the mechanics and then build harder
optimization scenarios.
