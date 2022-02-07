import numpy as np

from copy import deepcopy
from toolz import do


class GeneticAlgorithm:
    def __init__(self, pop_size, survivors_selector, mutator, initializer, fitness):
        self.pop_size = pop_size
        self.initializer = initializer
        self.mutator = mutator
        self.fitness = fitness
        self.survivors_selector = survivors_selector

        self.population = None
        self.population_fitness = None
        self.elite = None
        self.elite_fitness = None

    def generation(self):
        if not self.population:
            self.population = [self.initializer() for _ in range(self.pop_size)]
        else:
            survivors = self.survivors_selector(self.population, self.population_fitness)
            parents = np.random.choice(survivors, self.pop_size - 1, replace=True)  # 1 for the elite

            children = [do(self.mutator, deepcopy(parent)) for parent in parents]
            self.population = [self.elite] + children  # elitism - elite stays as is without mutations

        self.population_fitness = [self.fitness(specimen) for specimen in self.population]

        elite_idx = np.argmax(self.population_fitness)
        self.elite = self.population[elite_idx]
        self.elite_fitness = self.population_fitness[elite_idx]