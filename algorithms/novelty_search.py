from typing import List, Union

import numpy as np
import toolz

from copy import deepcopy

from agents.agent_typing import Agent
from algorithms.algorithm_typing import Initializer, Rollout, SurvivorSelector, Mutator, Archive, \
    BehaviorCharacteristic, FitnessMeasure, ArchiveToBatchFitness, EliteExtractor


class NoveltySearch:
    def __init__(self, pop_size, initializer: Initializer, rollout: Rollout,
                 novelty_from_archive: ArchiveToBatchFitness, elite_extractor: EliteExtractor,
                 fitness: FitnessMeasure, survivors_selector: SurvivorSelector, mutator: Mutator,
                 archive: Archive, behavior_characteristic: BehaviorCharacteristic):
        self.initializer = initializer
        self.pop_size = pop_size
        self.rollout = rollout
        self.behavior_characteristic = behavior_characteristic
        self.novelty_from_archive = novelty_from_archive
        self.fitness_measure = fitness
        self.survivors_selector = survivors_selector
        self.mutator = mutator
        self.elite_extractor = elite_extractor

        self.population: List[Agent] = []
        self.population_novelties: List[float] = []
        self.population_fitness: List[float] = []
        self.archive: Archive = archive

        self.elite: Union[Agent, None] = None
        self.elite_fitness: Union[float, None] = None

    def generation(self):
        if not self.population:
            self.population = [self.initializer() for _ in range(self.pop_size)]
        else:
            survivors = self.survivors_selector(self.population, self.population_novelties)
            parents = np.random.choice(survivors, self.pop_size - 1, replace=True)  # one for the elite

            children = [toolz.do(self.mutator, deepcopy(parent)) for parent in parents]

            self.population = [self.elite] + children

        trajectories = [self.rollout(specimen) for specimen in self.population]
        bcs = [self.behavior_characteristic(traj) for traj in trajectories]
        self.population_fitness = [self.fitness_measure(traj) for traj in trajectories]

        batch_novelty = self.novelty_from_archive(self.archive)
        self.population_novelties = batch_novelty(bcs)

        self.archive.store(bcs)

        self.elite, self.elite_fitness = self.elite_extractor(self.population, self.population_fitness)







