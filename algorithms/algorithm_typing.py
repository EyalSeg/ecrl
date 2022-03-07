from numpy import array

from dataclasses import dataclass

from typing import Protocol, Callable, TypeVar, List
from agents.agent_typing import Agent


class EvolutionaryAlgorithm(Protocol):
    def generation(self): ...

    elite: Agent
    elite_fitness: float


@dataclass
class Trajectory:
    observations: array
    actions: array
    rewards: array


BC = TypeVar('BC')

Initializer = Callable[[], Agent]
Rollout = Callable[[Agent], Trajectory]
Fitness = Callable[[BC], float]
BehaviorCharacteristic = Callable[[Trajectory], BC]
SurvivorSelector = Callable[[List[Agent], List[float]], List[Agent]]
Mutator = Callable[[Agent], None]


