import numpy as np
from src.environment import GridEnvironment
from src.simulation import Simulation
from src.visualization import SimulationVisualizer
from src.entities.agent import BaseAgent
from src.entities.objects import Attractor


def main():
    # 1. Initialize environment
    grid_size = 20
    env = GridEnvironment(width=grid_size, height=grid_size, cell_size=1.0)
    env.generate_random_walls(wall_density=0.15, seed=42)

    # 2. Initialize simulation
    sim = Simulation(env, dt=0.05, seed=123)

    # 3. Add an attractor in the center
    center = np.array([grid_size * 0.5, grid_size * 0.5])
    sim.add_object(Attractor(obj_id=0, position=center, strength=15.0, cutoff=12.0))

    # 4. Spawn agents in non-wall areas
    agent_count = 15
    for i in range(agent_count):
        pos = np.random.uniform(1.0, grid_size - 1.0, size=2)
        while env.is_position_in_wall(pos, radius=0.3):
            pos = np.random.uniform(1.0, grid_size - 1.0, size=2)
        
        agent = BaseAgent(
            agent_id=i + 1,
            position=pos,
            radius=0.3,
            mobility=1.0,
            diffusion_coeff=0.8,
        )
        sim.add_agent(agent)

    # 5. Visualize
    visualizer = SimulationVisualizer(sim)
    visualizer.run(steps=400, interval=30)


if __name__ == "__main__":
    main()
