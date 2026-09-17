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
            
            # Deterministic drift: F * dt
            drift = force * self.dt
            
            # Stochastic Wiener displacement: \sqrt{2 * D * dt} * N(0, 1)
            noise_scale = np.sqrt(2.0 * agent.diff * self.dt)
            noise = self.rng.standard_normal(2) * noise_scale

            # Rotational diffusion: \sqrt{2 * D_r * dt} * N(0, 1)
            agent.theta += np.sqrt(2.0 * agent.rotdiff * self.dt) * self.rng.standard_normal()
            agent.theta %= 2 * np.pi

            proposed_pos = agent.position + drift + noise
            agent.position = self.env.resolve_collision(agent.position, proposed_pos, agent.radius)
            agent.update_internal_state(self.dt, self.env)

        # Update dynamic (non-static) objects like Cargo
        for obj in self.objects:
            if not obj.is_static:
                obj_force = np.zeros(2, dtype=float)
                # Forces exerted by agents on this object (Newton's 3rd law)
                for agent in self.agents:
                    obj_force -= obj.force_on(agent)
                # Forces from other objects
                for other in self.objects:
                    if other.id != obj.id:
                        obj_force += other.force_on(obj)

                drift = obj_force * self.dt
                proposed_pos = obj.position + drift
                obj.position = self.env.resolve_collision(obj.position, proposed_pos, obj.radius)

        self.env.update_pheromones(self.dt)
        self.time += self.dt
