import numpy as np
import toolz

from copy import deepcopy

from sklearn.neighbors import NearestNeighbors


class NoveltySearch:
    def __init__(self, pop_size, initializer, rollout, novelty_measure, archive_pr, survivors_selector, mutator, robustness):
        self.initializer = initializer
        self.pop_size = pop_size
        self.rollout = rollout
        self.novelty_measure = novelty_measure
        self.archive_pr = archive_pr
        self.survivors_selector = survivors_selector
        self.mutator = mutator
        self.robustness = robustness

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

        fit_bcs = [self._robust_rollout(specimen) for specimen in self.population]
        self.population_fitness = [fitness for fitness, bc in fit_bcs]
        pop_bcs = [bc for fitness, bc in fit_bcs]

        # measure the novelty vs the archive + current generation
        archive_ = np.concatenate([self.archive, np.concatenate(pop_bcs)]) if self.archive is not None\
            else np.concatenate(pop_bcs)

        self.population_novelties = [[self.novelty_measure(bc, archive_) for bc in bcs] for bcs in pop_bcs]
        self.population_novelties = [sum(novelties) / len(novelties) for novelties in self.population_novelties]

        # update the archive
        newly_archived = [bc for bc in np.concatenate(pop_bcs) if np.random.rand() < self.archive_pr]
        if newly_archived:
            newly_archived = np.stack(newly_archived)
            self.archive = np.concatenate(
                [self.archive, newly_archived]) if self.archive is not None else newly_archived

        elite_idx = np.argmax(self.population_fitness)
        self.elite = self.population[elite_idx]
        self.elite_fitness = self.population_fitness[elite_idx]

    def _robust_rollout(self, specimen):
        '''
        performs self.robustness rollouts

        :param specimen:
        :return:
        average of their fitnesses and a list of their behaviour characteristics
        '''
        fit_bcs = [self.rollout(specimen) for _ in range(self.robustness)]

        avg_fitness = sum([fitness for fitness, bc in fit_bcs]) / self.robustness
        bcs = [bc for fitness, bc in fit_bcs]

        return avg_fitness, bcs


def _get_knn(n, archive):
    if _get_knn.archive_cache is None or not np.all(archive == _get_knn.archive_cache):
        # +1 because the archive will contain bc itself, which will add a 0 distance
        _get_knn.knn_cache = NearestNeighbors(n_neighbors=n + 1)
        _get_knn.knn_cache.fit(archive)

        _get_knn.archive_cache = archive

    return _get_knn.knn_cache


_get_knn.archive_cache = None
_get_knn.knn_cache = None


@toolz.curry
def knn_novelty(n, bc, archive):
    knn = _get_knn(n, archive)

    distances, indices = knn.kneighbors(bc.reshape(1, -1))
    avg = np.sum(distances) / n

    return avg

