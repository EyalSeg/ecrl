from copy import deepcopy

import numpy as np

from toolz import do
from algorithms.utilities.replay_buffer import ReplayBuffer


class SurpriseSearch:
    def __init__(self, popsize, initializer, rollout, replay_buffer_size, train_learner, fitness,
                 surprise, survivors_selector, mutator, elite_children=1, train_validate_ratio=0.8):
        self.popsize = popsize
        self.initializer = initializer
        self.rollout = rollout
        self.train_learner = train_learner
        self.fitness = fitness
        self.surprise = surprise
        self.survivors_selector = survivors_selector
        self.mutator = mutator

        self.population = None
        self.pop_fitnesses = None
        self.pop_surprises = None

        self.elite = None
        self.elite_fitness = None
        self.elite_children = elite_children

        # self.replay_buffer = None
        self.buffer_size = replay_buffer_size
        self.train_buffer = None
        self.validate_buffer = None
        self.train_validate_ratio = train_validate_ratio

    def generation(self):
        if not self.population:
            self.population = [self.initializer() for _ in range(self.popsize)]
            trajectories = [self.rollout(specimen) for specimen in self.population]

            for observations, actions, rewards in trajectories:
                self._store(observations, actions, rewards)

            self.train_learner(self.train_buffer.retrieve(), self.validate_buffer.retrieve())

            self.pop_fitnesses = [self.fitness(*trajectory) for trajectory in trajectories]
            self.pop_surprises = [self.surprise(*trajectory) for trajectory in trajectories]

        else:
            survivors = self.survivors_selector(self.population, self.pop_surprises)
            parents = np.random.choice(survivors, self.popsize - (self.elite_children + 1), replace=True)
            parents = list(parents) + [self.elite for _ in range(self.elite_children)]
            children = [do(self.mutator, deepcopy(parent)) for parent in parents]

            self.population = [self.elite] + children

            trajectories = [self.rollout(specimen) for specimen in self.population]
            for observations, actions, rewards in trajectories:
                self._store(observations, actions, rewards)

            self.pop_fitnesses = [self.fitness(*trajectory) for trajectory in trajectories]
            self.pop_surprises = [self.surprise(*trajectory) for trajectory in trajectories]

            self.train_learner(self.train_buffer.retrieve(), self.validate_buffer.retrieve())

        elite_idx = np.argmax(self.pop_fitnesses)
        self.elite = self.population[elite_idx]
        self.elite_fitness = self.pop_fitnesses[elite_idx]

    def _store(self, observations, actions, rewards):
        if not self.train_buffer:
            self.train_buffer = \
                ReplayBuffer(observations.shape[1:], int(self.buffer_size * self.train_validate_ratio))
        if not self.validate_buffer:
            self.validate_buffer = \
                ReplayBuffer(observations.shape[1:], int(self.buffer_size * (1 - self.train_validate_ratio)))

        train_mask = np.full(observations.shape[0], False)
        train_mask[:int(observations.shape[0] * self.train_validate_ratio)] = True
        np.random.shuffle(train_mask)

        validate_mask = np.invert(train_mask)

        self.train_buffer.store(observations[train_mask], actions[train_mask], rewards[train_mask])
        self.validate_buffer.store(observations[validate_mask], actions[validate_mask], rewards[validate_mask])