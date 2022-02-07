from typing import Protocol

from agents.agent_typing import Agent


class EvolutionaryAlgorithm(Protocol):
    def generation(self): ...

    elite: Agent
    elite_fitness: float
