import numpy as np
from .environment import GridEnvironment
from .entities.base import BaseObject
from .entities.agent import BaseAgent


class Simulation:
    def __init__(self, env: GridEnvironment, dt: float = 0.05, seed: int | None = None):
        self.env = env
        self.dt = dt
        self.time = 0.0
        self.agents: list[BaseAgent] = []
        self.objects: list[BaseObject] = []
        self.rng = np.random.default_rng(seed)

    def add_agent(self, agent: BaseAgent) -> None:
        self.agents.append(agent)

    def add_object(self, obj: BaseObject) -> None:
        self.objects.append(obj)

    def compute_forces_on_agent(self, agent: BaseAgent) -> np.ndarray:
        total_force = agent.compute_self_propulsion()
        
        # Forces from passive/interactive static objects
        for obj in self.objects:
            total_force += obj.force_on(agent)
            
        # Forces from other agents
        for other in self.agents:
            if other.id != agent.id:
                total_force += other.force_on(agent)
                
        return total_force

    def step(self) -> None:
        for agent in self.agents:
            force = self.compute_forces_on_agent(agent)
            
            # Deterministic drift: \mu * F * dt
            drift = agent.mobility * force * self.dt
            
            # Stochastic Wiener displacement: \sqrt{2 * D * dt} * N(0, 1)
            noise_scale = np.sqrt(2.0 * agent.diffusion_coeff * self.dt)
            noise = self.rng.standard_normal(2) * noise_scale
            
            proposed_pos = agent.position + drift + noise
            agent.position = self.env.resolve_collision(agent.position, proposed_pos, agent.radius)
            agent.update_internal_state(self.dt)

        self.time += self.dt
