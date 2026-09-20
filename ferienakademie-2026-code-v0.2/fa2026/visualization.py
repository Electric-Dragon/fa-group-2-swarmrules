from __future__ import annotations

import time

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


class Visualizer:
    """Small Matplotlib visualizer used by execute()."""

    def __init__(self, engine, pheromone_channel: int = 0):
        self.engine = engine
        self.pheromone_channel = pheromone_channel
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
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
        self.ax.grid(which="minor", linewidth=0.2, alpha=0.2)

        # Target field.
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

        # Optional pheromone heatmap.
        self.phero_img = None
        if e.pheromone_fields.shape[0] > 0:
            ch = min(max(self.pheromone_channel, 0), e.pheromone_fields.shape[0] - 1)
            self.phero_img = self.ax.imshow(
                e.pheromone_fields[ch],
                origin="lower",
                extent=(0, w, 0, h),
                interpolation="nearest",
                cmap="YlOrRd",
                alpha=0.45,
                zorder=1,
            )

        # Walls.
        ys, xs = np.nonzero(e.walls)
        for y, x in zip(ys, xs):
            self.ax.add_patch(
                patches.Rectangle((x, y), 1, 1, facecolor="black", edgecolor="none", zorder=3)
            )

        positions = e.agent_positions
        self.agent_scatter = self.ax.scatter(
            positions[:, 0], positions[:, 1], s=12, c="crimson", zorder=5
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
        self._update_title_and_background()
        self.fig.tight_layout()

    def _update_title_and_background(self) -> None:
        e = self.engine
        phase = "day" if e.is_day else "night"
        mean_energy = float(np.mean(e.agent_energies)) if e.agents else 0.0
        self.ax.set_title(
            f"turn {e.turn}  |  {phase}  |  mean energy {mean_energy:.2f}"
        )
        self.ax.set_facecolor("#fbfbf3" if e.is_day else "#202733")

    def draw(self, *, delay: float = 0.0) -> None:
        e = self.engine
        self.agent_scatter.set_offsets(e.agent_positions)
        size = e.rules.cargo_size
        self.cargo_patch.set_xy((e.cargo_position[0] - size / 2, e.cargo_position[1] - size / 2))
        if self.phero_img is not None:
            ch = min(max(self.pheromone_channel, 0), e.pheromone_fields.shape[0] - 1)
            data = e.pheromone_fields[ch]
            self.phero_img.set_data(data)
            vmax = max(float(np.max(data)), 1e-12)
            self.phero_img.set_clim(0.0, vmax)
        self._update_title_and_background()

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
        if not self._notebook:
            self.fig.canvas.draw_idle()
