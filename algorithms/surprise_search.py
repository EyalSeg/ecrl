from copy import deepcopy

import numpy as np

from toolz import do
from algorithms.utilities.replay_buffer import SplitReplayBuffer


class SurpriseSearch:
    def __init__(self, popsize, initializer, rollout, replay_buffer_size, train_learner, fitness,
                 surprise, survivors_selector, mutator, train_validate_ratio=0.8):
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

        self.replay_buffer = None
        self.buffer_size = replay_buffer_size
        self.train_validate_ratio = train_validate_ratio

    def generation(self):
        if not self.population:
            self.population = [self.initializer() for _ in range(self.popsize)]
            trajectories = [self.rollout(specimen) for specimen in self.population]

            for observations, actions, rewards in trajectories:
                self._store(observations, actions, rewards)

            self.train_learner(self.replay_buffer.buffers[0].retrieve(), self.replay_buffer.buffers[1].retrieve())

            self.pop_fitnesses = [self.fitness(*trajectory) for trajectory in trajectories]
            self.pop_surprises = [self.surprise(*trajectory) for trajectory in trajectories]

        else:
            survivors = self.survivors_selector(self.population, self.pop_surprises)
            parents = np.random.choice(survivors, self.popsize, replace=True)
            children = [do(self.mutator, deepcopy(parent)) for parent in parents]

            self.population = children

            trajectories = [self.rollout(specimen) for specimen in self.population]
            for observations, actions, rewards in trajectories:
                self._store(observations, actions, rewards)

            self.pop_fitnesses = [self.fitness(*trajectory) for trajectory in trajectories]
            self.pop_surprises = [self.surprise(*trajectory) for trajectory in trajectories]

            self.train_learner(self.replay_buffer.buffers[0].retrieve(), self.replay_buffer.buffers[1].retrieve())

        elite_idx = np.argmax(self.pop_fitnesses)
        self.elite = self.population[elite_idx]
        self.elite_fitness = self.pop_fitnesses[elite_idx]

    def _store(self, observations, actions, rewards):
        if not self.replay_buffer:
            ratios = [self.train_validate_ratio, 1 - self.train_validate_ratio]
            self.replay_buffer = SplitReplayBuffer(observations.shape[1:], self.buffer_size, ratios)

        self.replay_buffer.store(observations, actions, rewards)