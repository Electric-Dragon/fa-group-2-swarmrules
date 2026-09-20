import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import numpy as np
from .simulation import Simulation


class SimulationVisualizer:
    def __init__(self, sim: Simulation, fig: plt.Figure | None = None, ax: plt.Axes | None = None):
        self.sim = sim
        self.fig = fig if fig is not None else plt.figure(figsize=(7, 7))
        self.ax = ax if ax is not None else self.fig.gca()
        self.dynamic_objects = []
        self.dynamic_patches = []
        self.setup_plot()

    def setup_plot(self) -> None:
        self.ax.clear()
        max_x, max_y = self.sim.env.bounds
        self.ax.set_xlim(0, max_x)
        self.ax.set_ylim(0, max_y)
        self.ax.set_aspect("equal")
        self.ax.set_title(f"Time: {self.sim.time:.2f}s")

        # Pheromone grid background
        extent = (0, max_x, 0, max_y)
        self.phero_img = self.ax.imshow(
            self.sim.env.pheromone_grid,
            origin="lower",
            extent=extent,
            cmap="YlGnBu",
            alpha=0.6,
            vmin=0.0,
            vmax=5.0,
            zorder=1,
        )

        # Draw walls
        mask = self.sim.env.wall_mask
        cell = self.sim.env.cell_size
        for iy in range(mask.shape[0]):
            for ix in range(mask.shape[1]):
                if mask[iy, ix]:
                    rect = patches.Rectangle(
                        (ix * cell, iy * cell), cell, cell, facecolor="black", edgecolor="none"
                    )
                    self.ax.add_patch(rect)

        # Objects
        self.dynamic_objects = []
        self.dynamic_patches = []
        for obj in self.sim.objects:
            if obj.is_static:
                circle = patches.Circle(obj.position, obj.radius, color="green", alpha=0.7, zorder=3)
                self.ax.add_patch(circle)
            else:
                circle = patches.Circle(obj.position, obj.radius, color="royalblue", alpha=0.85, zorder=4)
                self.ax.add_patch(circle)
                self.dynamic_objects.append(obj)
                self.dynamic_patches.append(circle)

        # Agents
        positions = np.array([a.position for a in self.sim.agents]) if self.sim.agents else np.empty((0, 2))
        u = np.array([np.cos(a.theta) * a.radius for a in self.sim.agents])
        v = np.array([np.sin(a.theta) * a.radius for a in self.sim.agents])

        x_coords = positions[:, 0] if len(positions) else []
        y_coords = positions[:, 1] if len(positions) else []
        self.agent_scatter = self.ax.scatter(x_coords, y_coords, c="crimson", s=80, zorder=5)
        self.quiver = self.ax.quiver(
            x_coords, y_coords, u, v,
            color="white", angles="xy", scale_units="xy", scale=1, zorder=6
        )

    def update_frame(self) -> tuple:
        self.sim.step()
        positions = np.array([a.position for a in self.sim.agents]) if self.sim.agents else np.empty((0, 2))
        u = np.array([np.cos(a.theta) * a.radius for a in self.sim.agents])
        v = np.array([np.sin(a.theta) * a.radius for a in self.sim.agents])

        self.agent_scatter.set_offsets(positions)
        self.quiver.set_offsets(positions)
        self.quiver.set_UVC(u, v)
        self.phero_img.set_data(self.sim.env.pheromone_grid)
        self.ax.set_title(f"Time: {self.sim.time:.2f}s")

        for obj, patch_circle in zip(self.dynamic_objects, self.dynamic_patches):
            patch_circle.center = obj.position

        return (self.phero_img, self.agent_scatter, self.quiver, *self.dynamic_patches)

    def animate(self, steps: int = 500, interval: int = 30) -> animation.FuncAnimation:
        return animation.FuncAnimation(
            self.fig, lambda _: self.update_frame(), frames=steps, interval=interval, blit=False
        )
