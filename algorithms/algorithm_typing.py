from numpy import array

from dataclasses import dataclass

from typing import Protocol, Callable, TypeVar, List, Generic, Tuple
from agents.agent_typing import Agent


@dataclass
class Trajectory:
    observations: array
    actions: array
    rewards: array


BC = TypeVar('BC')

Initializer = Callable[[], Agent]
Rollout = Callable[[Agent], Trajectory]
FitnessMeasure = Callable[[BC], float]
BatchFitnessMeasure = Callable[[List[BC]], List[float]]
ArchiveToBatchFitness = Callable[[List[BC]], BatchFitnessMeasure]
BehaviorCharacteristic = Callable[[Trajectory], BC]
SurvivorSelector = Callable[[List[Agent], List[float]], List[Agent]]
Mutator = Callable[[Agent], None]
EliteExtractor = Callable[[List[Agent], List[float]], Tuple[Agent, float]]


class EvolutionaryAlgorithm(Protocol):
    def generation(self): ...

    elite: Agent
    elite_fitness: float


class Archive(Protocol[BC]):
    def store(self, item: List[BC]) -> None: ...
    def retrieve(self) -> List[BC]: ...

