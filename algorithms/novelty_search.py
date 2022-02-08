import numpy as np
import toolz

from copy import deepcopy

from sklearn.neighbors import NearestNeighbors


class NoveltySearch:
    def __init__(self, pop_size, initializer, rollout, novelty_measure, archive_pr, survivors_selector, mutator):
        self.initializer = initializer
        self.pop_size = pop_size
        self.rollout = rollout
        self.novelty_measure = novelty_measure
        self.archive_pr = archive_pr
        self.survivors_selector = survivors_selector
        self.mutator = mutator

        self.population = None
        self.population_novelties = None
        self.population_fitness = None
        self.archive = None

        self.elite = None
        self.elite_fitness = None

    def generation(self):
        if not self.population:
            self.population = [self.initializer() for _ in range(self.pop_size)]
        else:
            survivors = self.survivors_selector(self.population, self.population_novelties)
            parents = np.random.choice(survivors, self.pop_size - 1, replace=True)

            self.population = [self.elite] + [toolz.do(self.mutator, deepcopy(parent)) for parent in parents]

        fit_bcs = [self.rollout(specimen) for specimen in self.population]
        self.population_fitness = [fitness for fitness, bc in fit_bcs]
        bcs = [bc for fitness, bc in fit_bcs]

        # measure the novelty vs the archive + current generation
        archive_ = np.concatenate([self.archive, np.array(bcs)]) if self.archive is not None else np.array(bcs)
        self.population_novelties = [self.novelty_measure(bc, archive_) for bc in bcs]

        # update the archive
        newly_archived = [bc for bc in bcs if np.random.rand() < self.archive_pr]
        if newly_archived:
            newly_archived = np.stack(newly_archived)
            self.archive = np.concatenate(
                [self.archive, newly_archived]) if self.archive is not None else newly_archived

        elite_idx = np.argmax(self.population_fitness)
        self.elite = self.population[elite_idx]
        self.elite_fitness = self.population_fitness[elite_idx]


@toolz.curry
def knn_novelty(n, bc, archive):
    knn = NearestNeighbors(
        n_neighbors=n + 1)  # +1 because the archive will contain bc itself, which will add a 0 distance
    knn.fit(archive)

    distances, indices = knn.kneighbors(bc.reshape(1, -1))
    avg = np.sum(distances) / n

    return avg

