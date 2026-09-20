"""Simple controller that solves the easy playground scenario.

The strategy is intentionally transparent rather than optimal.  It uses one
pheromone channel, local vision, local memory, and the day/night cycle.  It is
meant as a working example of the complete student API.
"""

from __future__ import annotations

import math

import numpy as np

from fa2026 import Action, Cell, Config, Pheromone


def setup(rules):
    # One moderately persistent recruitment pheromone.
    # parameters = (step length, pheromone amount, night vision size, stand-off)
    return Config(
        pheromones=(Pheromone(decay=0.10),),
        initial_memory=((0.0, 0.0, 0.0, 0.0),),
        parameters=((
            0.42,
            0.18,
            2 * rules.night_vision_radius + 1,
            0.5 * rules.cargo_size + 0.15,
        ),),
    )


def _visible_vector(observation, flag):
    """Vector from the agent to the mean center of visible cells with `flag`."""
    mask = (observation.vision & int(flag)) != 0
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None

    center = observation.vision.shape[0] // 2
    dx = xs - center + 0.5 - observation.cell_position[0]
    dy = ys - center + 0.5 - observation.cell_position[1]
    return np.array([float(np.mean(dx)), float(np.mean(dy))])


def _unit(vector):
    length = float(np.linalg.norm(vector))
    if length == 0.0:
        return np.zeros(2)
    return vector / length


def act(observation, memory, agent_type, config):
    m = memory.copy()
    step, emission, night_size, stand_off = config.parameters[agent_type]

    cargo = _visible_vector(observation, Cell.CARGO)
    target = _visible_vector(observation, Cell.TARGET)

    # Night is recognizable from the smaller vision window.  This simple
    # strategy uses it only for resting/recharging.
    if observation.vision.shape[0] == int(night_size):
        return Action(), m

    # If cargo and target are both visible, first get behind the cargo and then
    # push it toward the target.
    if cargo is not None and target is not None:
        push_direction = _unit(target - cargo)
        behind_cargo = cargo - stand_off * push_direction

        along = float(np.dot(cargo, push_direction))
        lateral = cargo - along * push_direction
        ready_to_push = (
            np.linalg.norm(behind_cargo) < 0.65
            or (along > 0.0 and np.linalg.norm(lateral) < 0.45)
        )

        if ready_to_push:
            move = step * push_direction
        else:
            move = step * _unit(behind_cargo)

        return Action(move=tuple(move), pheromones=(emission,)), m

    # If only the cargo is visible, approach it and leave a recruitment trail.
    if cargo is not None:
        return Action(
            move=tuple(step * _unit(cargo)),
            pheromones=(emission,),
        ), m

    # Follow a neighboring pheromone maximum if one is available.
    pheromone = observation.pheromones[0]
    if float(np.max(pheromone)) > 1e-8:
        py, px = np.unravel_index(np.argmax(pheromone), pheromone.shape)
        direction = np.array([px - 1, py - 1], dtype=float)
        if np.linalg.norm(direction) > 0.0:
            return Action(move=tuple(step * _unit(direction))), m

    # Otherwise explore deterministically.  Memory stores heading, a small
    # countdown, and an initialization flag.
    if m[2] < 0.5:
        phase = (
            observation.cell_position[0]
            + 0.61803398875 * observation.cell_position[1]
        ) % 1.0
        m[0] = 2.0 * math.pi * phase
        m[1] = 8.0
        m[2] = 1.0

    angle = float(m[0])
    countdown = float(m[1]) - 1.0
    if countdown <= 0.0:
        angle = (angle + 2.399963229728653) % (2.0 * math.pi)
        countdown = 8.0

    move = step * np.array([math.cos(angle), math.sin(angle)])

    # Immediate wall avoidance.
    center = observation.vision.shape[0] // 2
    sx = 1 if move[0] > 0 else (-1 if move[0] < 0 else 0)
    sy = 1 if move[1] > 0 else (-1 if move[1] < 0 else 0)
    if sx and int(observation.vision[center, center + sx]) & int(Cell.WALL):
        move[0] *= -1
    if sy and int(observation.vision[center + sy, center]) & int(Cell.WALL):
        move[1] *= -1

    m[0] = math.atan2(move[1], move[0])
    m[1] = countdown
    return Action(move=tuple(move)), m
