from typing import List, Union

import numpy as np

from copy import deepcopy
from toolz import do

from agents.agent_typing import Agent
from algorithms.algorithm_typing import SurvivorSelector, Mutator, Initializer, FitnessMeasure
from algorithms.operators.selection import truncated_selection


class GeneticAlgorithm:
    def __init__(self, pop_size: int, survivors_selector: SurvivorSelector, mutator: Mutator,
                 initializer: Initializer, fitness: FitnessMeasure, elitism: int):
        self.pop_size = pop_size
        self.initializer = initializer
        self.mutator = mutator
        self.fitness = fitness
        self.survivors_selector = survivors_selector
        self.n_elites = elitism
        self.elitism = truncated_selection(self.n_elites)

        self.population: List[Agent] = []
        self.population_fitness: List[float] = []
        self.elite: Union[Agent, None] = None
        self.elite_fitness: Union[float, None] = None

    def generation(self):
        if not self.population:
            self.population = [self.initializer() for _ in range(self.pop_size)]
        else:
            survivors = self.survivors_selector(self.population, self.population_fitness)
            parents = np.random.choice(survivors, self.pop_size - self.n_elites, replace=True)

            children = [do(self.mutator, deepcopy(parent)) for parent in parents]

            elites = self.elitism(self.population, self.population_fitness)
            self.population = elites + children

        self.population_fitness = [self.fitness(specimen) for specimen in self.population]

        elite_idx = np.argmax(self.population_fitness)
        self.elite = self.population[elite_idx]
        self.elite_fitness = self.population_fitness[elite_idx]