from functools import partial

import gym
import toolz

from agents.pytorch import LinearTorchPolicy, TorchPolicyAgent
from algorithms.random_search import RandomSearch
from experiments.scripts.experiment_base import ExperimentBase


class RandomSearchExperiment(ExperimentBase):
    def __init__(self, initializer, fit_robustness, train_env, validation_env, max_train_steps: int, validation_episodes: int):
        super().__init__(train_env, validation_env, max_train_steps, validation_episodes)

        fitness = self.episodic_rewards(self.train_env, n_episodes=fit_robustness)
        self.alg = RandomSearch(initializer, fitness)

    def start(self):
        super().start(self.alg)


if __name__ == "__main__":
    env_name = "Acrobot-v1"
    env = gym.make(env_name)

    policy_dims = [sum(env.observation_space.shape),
                   256,
                   512,
                   env.action_space.n]

    initializer = partial(toolz.compose_left(LinearTorchPolicy, TorchPolicyAgent), policy_dims)

    random_search = RandomSearchExperiment(initializer, 20, gym.make(env_name), gym.make(env_name), int(1e6), 100)
    random_search.start()
