import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import numpy as np
from .simulation import Simulation


class SimulationVisualizer:
    def __init__(self, sim: Simulation):
        self.sim = sim
        self.fig, self.ax = plt.subplots(figsize=(7, 7))

    def _draw_walls(self) -> None:
        mask = self.sim.env.wall_mask
        cell = self.sim.env.cell_size
        for iy in range(mask.shape[0]):
            for ix in range(mask.shape[1]):
                if mask[iy, ix]:
                    rect = patches.Rectangle(
                        (ix * cell, iy * cell), cell, cell, facecolor="black", edgecolor="none"
                    )
                    self.ax.add_patch(rect)

    def run(self, steps: int = 500, interval: int = 30) -> None:
        max_x, max_y = self.sim.env.bounds
        self.ax.set_xlim(0, max_x)
        self.ax.set_ylim(0, max_y)
        self.ax.set_aspect("equal")
        self.ax.set_title("Overdamped Brownian Dynamics")

        self._draw_walls()

        # Render objects
        for obj in self.sim.objects:
            circle = patches.Circle(obj.position, obj.radius, color="green", alpha=0.7)
            self.ax.add_patch(circle)

        # Agent scatter plot
        agent_scatter = self.ax.scatter([], [], c="crimson", s=80, zorder=5)

        def init():
            agent_scatter.set_offsets(np.empty((0, 2)))
            return (agent_scatter,)

        def update(_frame):
            self.sim.step()
            positions = np.array([a.position for a in self.sim.agents])
            agent_scatter.set_offsets(positions)
            return (agent_scatter,)

        _anim = animation.FuncAnimation(
            self.fig, update, init_func=init, frames=steps, interval=interval, blit=True
        )
        plt.show()
