import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
from matplotlib.widgets import Button
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

        # Pheromone heatmap
        extent = (0, max_x, 0, max_y)
        phero_img = self.ax.imshow(
            self.sim.env.pheromone_grid,
            origin="lower",
            extent=extent,
            cmap="YlGnBu",
            alpha=0.6,
            vmin=0.0,
            vmax=5.0,
            zorder=1,
        )

        self._draw_walls()

        # Render objects
        static_patches = []
        dynamic_objects = []
        dynamic_patches = []
        for obj in self.sim.objects:
            if obj.is_static:
                circle = patches.Circle(obj.position, obj.radius, color="green", alpha=0.7, zorder=3)
                self.ax.add_patch(circle)
                static_patches.append(circle)
            else:
                circle = patches.Circle(obj.position, obj.radius, color="royalblue", alpha=0.85, zorder=4)
                self.ax.add_patch(circle)
                dynamic_objects.append(obj)
                dynamic_patches.append(circle)

        # Agent scatter and orientation quiver
        init_pos = np.array([a.position for a in self.sim.agents]) if self.sim.agents else np.empty((0, 2))
        u = np.array([np.cos(a.theta) * a.radius for a in self.sim.agents])
        v = np.array([np.sin(a.theta) * a.radius for a in self.sim.agents])

        agent_scatter = self.ax.scatter(init_pos[:, 0] if len(init_pos) else [], init_pos[:, 1] if len(init_pos) else [], c="crimson", s=80, zorder=5)
        quiver = self.ax.quiver(init_pos[:, 0] if len(init_pos) else [], init_pos[:, 1] if len(init_pos) else [], u, v, color="white", angles="xy", scale_units="xy", scale=1, zorder=6)

        def update(_frame):
            self.sim.step()
            positions = np.array([a.position for a in self.sim.agents]) if self.sim.agents else np.empty((0, 2))
            u = np.array([np.cos(a.theta) * a.radius for a in self.sim.agents])
            v = np.array([np.sin(a.theta) * a.radius for a in self.sim.agents])
            agent_scatter.set_offsets(positions)
            quiver.set_offsets(positions)
            quiver.set_UVC(u, v)
            phero_img.set_data(self.sim.env.pheromone_grid)

            for obj, patch_circle in zip(dynamic_objects, dynamic_patches):
                patch_circle.center = obj.position

            return (phero_img, agent_scatter, quiver, *dynamic_patches)

        anim = animation.FuncAnimation(
            self.fig, update, frames=steps, interval=interval, blit=True
        )

        # End Simulation button
        plt.subplots_adjust(bottom=0.15)
        button_ax = self.fig.add_axes([0.35, 0.03, 0.3, 0.06])
        btn_stop = Button(button_ax, "End Simulation", color="lightgray", hovercolor="salmon")

        def on_stop(_event):
            anim.event_source.stop()
            total_agents = len(self.sim.agents)
            in_bounds = sum(
                0 <= a.position[0] <= max_x and 0 <= a.position[1] <= max_y
                for a in self.sim.agents
            )
            print(f"\n[Simulation Ended] Agents inside simulation box: {in_bounds}/{total_agents}")
            if in_bounds < total_agents:
                print(f"Warning: {total_agents - in_bounds} agent(s) escaped the bounds!")
            else:
                print("All agents remained inside the bounds.")
            plt.close(self.fig)

        btn_stop.on_clicked(on_stop)
        plt.show()
