"""Deliberately weak example strategy.

The controller is deterministic but produces different wandering directions from
local initial conditions.  It demonstrates movement, memory, energy and cargo
pushing without implementing a useful collective strategy.
"""

import math
import numpy as np

from fa2026 import Action, Cell, Config


def setup(rules):
    # One agent type, no pheromones.  Memory = angle, countdown, initialized, spare.
    return Config(initial_memory=((0.0, 0.0, 0.0, 0.0),))


def act(observation, memory, agent_type, config):
    m = memory.copy()

    # Give initially identical agents different deterministic directions using
    # only their local sub-cell position (no global coordinates or agent IDs).
    if m[2] < 0.5:
        phase = (observation.cell_position[0] + 0.61803398875 * observation.cell_position[1]) % 1.0
        m[0] = 2.0 * math.pi * phase
        m[1] = 6.0
        m[2] = 1.0

    # When energy is nearly exhausted, do nothing and wait for a night recharge.
    if observation.energy < 0.35:
        return Action(), m

    angle = float(m[0])
    countdown = float(m[1]) - 1.0
    if countdown <= 0.0:
        # A deterministic "random-looking" turn.  This is intentionally not a
        # good navigation strategy.
        angle = (angle + 2.399963229728653) % (2.0 * math.pi)
        countdown = 6.0

    step = 0.38
    direction = np.array([math.cos(angle), math.sin(angle)]) * step

    # Crude wall avoidance using only immediate visible neighbours.
    vision = observation.vision
    center = vision.shape[0] // 2
    sx = 1 if direction[0] > 0 else (-1 if direction[0] < 0 else 0)
    sy = 1 if direction[1] > 0 else (-1 if direction[1] < 0 else 0)
    if sx and int(vision[center, center + sx]) & int(Cell.WALL):
        direction[0] *= -1
        angle = math.atan2(direction[1], direction[0])
    if sy and int(vision[center + sy, center]) & int(Cell.WALL):
        direction[1] *= -1
        angle = math.atan2(direction[1], direction[0])

    m[0] = angle
    m[1] = countdown
    return Action(move=(float(direction[0]), float(direction[1]))), m
