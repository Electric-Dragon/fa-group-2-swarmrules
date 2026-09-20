import io
import threading
import time
import ipywidgets as widgets
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from IPython import get_ipython
from src.environment import GridEnvironment
from src.visualization import SimulationVisualizer

env = None

def _render_env(env_obj):
    fig, ax = plt.subplots(figsize=(6, 6))
    max_x, max_y = env_obj.bounds
    ax.set_xlim(0, max_x)
    ax.set_ylim(0, max_y)
    ax.set_aspect("equal")

    mask = env_obj.wall_mask
    cell = env_obj.cell_size
    for iy in range(mask.shape[0]):
        for ix in range(mask.shape[1]):
            if mask[iy, ix]:
                rect = patches.Rectangle(
                    (ix * cell, iy * cell), cell, cell, facecolor="black", edgecolor="none"
                )
                ax.add_patch(rect)

    plt.close(fig)
    return fig, ax

def create_environment_configurator():
    """
    Creates and returns an interactive widget configurator for GridEnvironment.
    Updates the global variable `env` whenever parameters change.
    """
    global env

    width_slider = widgets.IntSlider(value=30, min=10, max=60, step=2, description='Width:')
    height_slider = widgets.IntSlider(value=30, min=10, max=60, step=2, description='Height:')
    cell_size_slider = widgets.FloatSlider(value=1.0, min=0.1, max=5.0, step=0.1, description='Cell Size:')
    decay_rate_slider = widgets.FloatSlider(value=0.05, min=0.0, max=0.5, step=0.01, description='Decay Rate:')

    mode_dropdown = widgets.Dropdown(
        options=['random', 'labyrinth', 'hybrid'],
        value='random',
        description='Mode:'
    )
    seed_slider = widgets.IntSlider(value=42, min=0, max=9999, step=1, description='Seed:')

    wall_density_slider = widgets.FloatSlider(value=0.2, min=0.0, max=0.8, step=0.05, description='Wall Density:')
    field_density_slider = widgets.FloatSlider(value=0.15, min=0.0, max=0.8, step=0.05, description='Field Density:')
    maze_thresh_slider = widgets.FloatSlider(value=0.5, min=0.0, max=1.0, step=0.05, description='Maze Thresh:')

    mode_controls = widgets.VBox([])

    output = widgets.Output()

    def update_env(*args):
        global env
        env = GridEnvironment(
            width=width_slider.value,
            height=height_slider.value,
            cell_size=cell_size_slider.value,
            decay_rate=decay_rate_slider.value
        )
        mode = mode_dropdown.value
        if mode == 'random':
            env.generate_random_walls(wall_density=wall_density_slider.value, seed=seed_slider.value)
        elif mode == 'labyrinth':
            env.generate_labyrinth(seed=seed_slider.value)
        elif mode == 'hybrid':
            env.generate_hybrid_environment(
                maze_threshold=maze_thresh_slider.value,
                field_obstacle_density=field_density_slider.value,
                seed=seed_slider.value
            )

        ip = get_ipython()
        if ip is not None:
            ip.user_ns['env'] = env

        with output:
            clear_output(wait=True)
            fig, ax = _render_env(env)
            display(fig)

    def on_mode_change(*args):
        mode = mode_dropdown.value
        if mode == 'random':
            mode_controls.children = [wall_density_slider]
        elif mode == 'labyrinth':
            mode_controls.children = []
        elif mode == 'hybrid':
            mode_controls.children = [widgets.HBox([maze_thresh_slider, field_density_slider])]

    mode_dropdown.observe(on_mode_change, names='value')
    on_mode_change()

    all_widgets = [
        width_slider, height_slider, cell_size_slider, decay_rate_slider,
        mode_dropdown, seed_slider, wall_density_slider, field_density_slider, maze_thresh_slider
    ]
    for w in all_widgets:
        w.observe(update_env, names='value')

    update_env()

    controls = widgets.VBox([
        widgets.HBox([width_slider, height_slider]),
        widgets.HBox([cell_size_slider, decay_rate_slider]),
        widgets.HBox([mode_dropdown, seed_slider]),
        mode_controls
    ])

    return widgets.VBox([controls, output])


def create_simulation_viewer(sim, fps: int = 20):
    """
    Creates an interactive viewer for a Simulation instance with Start and Stop controls.
    Uses an ipywidgets.Image widget to stream live frames smoothly across threads.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    visualizer = SimulationVisualizer(sim, fig=fig, ax=ax)
    plt.close(fig)

    image_widget = widgets.Image(format="png")
    btn_start = widgets.Button(description="Start", button_style="success")
    btn_stop = widgets.Button(description="Stop", button_style="danger", disabled=True)

    is_running = False
    lock = threading.Lock()

    def update_image():
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        image_widget.value = buf.getvalue()

    update_image()

    def run_loop():
        nonlocal is_running
        delay = 1.0 / fps
        while True:
            with lock:
                if not is_running:
                    break
                visualizer.update_frame()
                update_image()
            time.sleep(delay)

    def on_start_clicked(b):
        nonlocal is_running
        with lock:
            if not is_running:
                is_running = True
                btn_start.disabled = True
                btn_stop.disabled = False
                t = threading.Thread(target=run_loop, daemon=True)
                t.start()

    def on_stop_clicked(b):
        nonlocal is_running
        with lock:
            is_running = False
            btn_start.disabled = False
            btn_stop.disabled = True

    btn_start.on_click(on_start_clicked)
    btn_stop.on_click(on_stop_clicked)

    controls = widgets.HBox([btn_start, btn_stop])
    return widgets.VBox([controls, image_widget])
