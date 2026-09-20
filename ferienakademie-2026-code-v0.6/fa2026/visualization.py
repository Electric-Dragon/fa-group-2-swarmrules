from __future__ import annotations

import time

import matplotlib.pyplot as plt
from matplotlib import colors, patches
import numpy as np


class Visualizer:
    """Small Matplotlib visualizer used by ``execute(..., visualize=True)``."""

    def __init__(self, engine):
        self.engine = engine
        self.fig, self.ax = plt.subplots(figsize=(9, 7))
        self._display_handle = None
        self._notebook = self._running_in_notebook()
        self._setup()

    @staticmethod
    def _running_in_notebook() -> bool:
        try:
            from IPython import get_ipython

            shell = get_ipython()
            return shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell"
        except Exception:
            return False

    def _setup(self) -> None:
        e = self.engine
        h, w = e.rules.grid_shape
        self.ax.set_aspect("equal")
        self.ax.set_xlim(0, w)
        self.ax.set_ylim(0, h)
        self.ax.set_xticks(np.arange(0, w + 1, 1), minor=True)
        self.ax.set_yticks(np.arange(0, h + 1, 1), minor=True)
        self.ax.grid(which="minor", linewidth=0.2, alpha=0.15)

        target = np.ma.masked_where(~e.target_mask, e.target_mask)
        self.ax.imshow(
            target,
            origin="lower",
            extent=(0, w, 0, h),
            interpolation="nearest",
            alpha=0.25,
            cmap="Greens",
            vmin=0,
            vmax=1,
            zorder=0,
        )

        self.phero_img = None
        if any(p.color is not None for p in e.config.pheromones):
            self.phero_img = self.ax.imshow(
                self._pheromone_rgba(),
                origin="lower",
                extent=(0, w, 0, h),
                interpolation="nearest",
                zorder=1,
            )

        ys, xs = np.nonzero(e.walls)
        for y, x in zip(ys, xs):
            self.ax.add_patch(
                patches.Rectangle((x, y), 1, 1, facecolor="black", edgecolor="none", zorder=3)
            )

        positions = e.agent_positions
        agent_types = np.array([agent.agent_type for agent in e.agents], dtype=int)
        unique_types = np.unique(agent_types)
        if unique_types.size == 1:
            agent_colors = "crimson"
        else:
            cmap = plt.get_cmap("tab10")
            agent_colors = [cmap(int(t) % 10) for t in agent_types]
            from matplotlib.lines import Line2D

            handles = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="",
                    markersize=5,
                    color=cmap(int(t) % 10),
                    label=f"type {int(t)}",
                )
                for t in unique_types
            ]
            self.ax.legend(handles=handles, loc="upper left", framealpha=0.8, fontsize=8)

        self.agent_scatter = self.ax.scatter(
            positions[:, 0], positions[:, 1], s=10, c=agent_colors, zorder=5
        )

        size = e.rules.cargo_size
        self.cargo_patch = patches.Rectangle(
            (e.cargo_position[0] - size / 2, e.cargo_position[1] - size / 2),
            size,
            size,
            facecolor="royalblue",
            edgecolor="navy",
            linewidth=1.5,
            zorder=4,
        )
        self.ax.add_patch(self.cargo_patch)
        self._update_title()
        self.fig.tight_layout()

    def _pheromone_rgba(self) -> np.ndarray:
        """Blend all visible pheromone channels for plotting only.

        Each visible channel is normalized independently by its current maximum,
        so channels with different numerical scales remain distinguishable.
        Channel colors never affect the simulation or the values seen by agents.
        """
        e = self.engine
        h, w = e.rules.grid_shape
        weighted_rgb = np.zeros((h, w, 3), dtype=float)
        weight = np.zeros((h, w), dtype=float)

        for channel, pheromone in enumerate(e.config.pheromones):
            if pheromone.color is None:
                continue
            rgb = np.asarray(colors.to_rgb(pheromone.color), dtype=float)
            field = e.pheromone_fields[channel]
            maximum = float(np.max(field))
            if maximum <= 0.0:
                continue
            intensity = np.clip(field / maximum, 0.0, 1.0)
            weighted_rgb += intensity[..., None] * rgb
            weight += intensity

        rgba = np.zeros((h, w, 4), dtype=float)
        mask = weight > 0.0
        if np.any(mask):
            rgba[mask, :3] = weighted_rgb[mask] / weight[mask, None]
            rgba[..., 3] = 0.50 * np.clip(weight, 0.0, 1.0)
        return rgba

    def _update_title(self) -> None:
        e = self.engine
        mean_energy = float(np.mean(e.agent_energies)) if e.agents else 0.0
        self.ax.set_title(f"turn {e.turn}  |  mean energy {mean_energy:.2f}")

    def draw(self, *, delay: float = 0.0) -> None:
        e = self.engine
        self.agent_scatter.set_offsets(e.agent_positions)
        size = e.rules.cargo_size
        self.cargo_patch.set_xy(
            (e.cargo_position[0] - size / 2, e.cargo_position[1] - size / 2)
        )
        if self.phero_img is not None:
            self.phero_img.set_data(self._pheromone_rgba())
        self._update_title()

        if self._notebook:
            try:
                from IPython.display import display

                if self._display_handle is None:
                    self._display_handle = display(self.fig, display_id=True)
                else:
                    self._display_handle.update(self.fig)
            except Exception:
                self.fig.canvas.draw_idle()
        else:
            plt.show(block=False)
            self.fig.canvas.draw_idle()
            plt.pause(max(delay, 1e-6))

        if delay > 0 and self._notebook:
            time.sleep(delay)

    def finish(self) -> None:
        # In notebooks, keeping the Matplotlib figure open makes the inline
        # backend display the final frame a second time when the cell ends.
        if self._notebook:
            plt.close(self.fig)
        else:
            self.fig.canvas.draw_idle()
