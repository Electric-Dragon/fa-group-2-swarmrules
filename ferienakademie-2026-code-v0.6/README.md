# Ferienakademie 2026 – shared project framework

This is a compact working version of the shared-project framework for
**Learning Rules for Life-Like Emergent Behavior**. It is intended for testing
and discussion before the final student version is frozen.

The controller interface is deliberately small:

```python
config = setup(rules)
action, new_memory = act(observation, memory, agent_type, config)
```

Agents have continuous positions on a grid-based playing field, local vision,
finite energy and limited memory. They may move, push one cargo object and
communicate through optional pheromone fields. Agents can overlap one another
and the cargo; walls block agents and cargo.

## Current mechanics

- Energy recharges by a fixed amount at the beginning of every turn.
- Movement and pushing have quadratic energy costs.
- Pheromone emission has a linear cost proportional to `amount / decay`, so
  long-lived signals are more expensive.
- Movement, pushing and pheromone emission may all be requested in the same
  turn. Their costs add.
- If the battery cannot afford the full simultaneous action, move, push and
  pheromone amplitudes are all reduced by the same factor while preserving the
  requested proportions.
- Energy is spent for the requested action even if a wall blocks movement or a
  push has no effect.
- A push affects cargo only if the agent overlaps the cargo **after agent
  movement** in that turn.
- Push vectors from all overlapping agents add. Cargo has no momentum.
- Pheromones are emitted at final agent positions after movement/pushing and are
  first sensed on the next turn.
- Existing pheromone follows `P(t+1) = (1-d) P(t) + Q(t)`.
- The score is the number of turns required to place the cargo completely
  inside the target.

All agents receive observations from the same beginning-of-turn world state.
They have no absolute position, turn number or agent ID.

## Getting started

Install dependencies:

```bash
pip install -r requirements.txt
```

Open `getting_started.ipynb`. It contains a deliberately simple and inefficient
baseline for the fixed `playground` scenario, plus a visual run and a timed
headless run for quick experiments.

For collaborator discussion, `feature_demo.ipynb` uses the separate `showcase`
scenario to display more mechanisms at once: internal walls, three agent types,
persistent memory and three visible pheromone channels. Its controller is
intentionally crude and is not meant as a reference solution.

For a quick framework check after changing code:

```bash
python -m pytest -q
```

### Pheromones

Several pheromone channels are supported up to `rules.max_pheromones`. A
channel is hidden by default; give it a color to show it in the visualization:

```python
Pheromone(decay=0.08, color="blue")
```

If several colored channels overlap, their display colors are blended. Each
channel is normalized independently for display only. Color never changes the
simulation or the values agents sense.

When several permanent agent types are configured, the visualizer colors the
types differently for inspection; this is display-only as well.

Display options such as `visualize`, `delay` and `draw_every` never change game
mechanics.

## Repository structure

```text
fa2026/
    __init__.py
    api.py
    engine.py
    scenarios.py
    visualization.py
getting_started.ipynb
feature_demo.ipynb
tests/test_engine.py
requirements.txt
```

The code is intentionally kept small so mechanics can still be changed easily
after discussion.
