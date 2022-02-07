from functools import partial

import gym
import toolz

from agents.pytorch import LinearTorchPolicy, TorchPolicyAgent
from algorithms.random_search import RandomSearch
from experiments.scripts.experiment_base import ExperimentBase
from experiments.scripts.loggers.composite_logger import CompositeLogger
from experiments.scripts.loggers.console_logger import ConsoleLogger
from experiments.scripts.loggers.logger_typing import Logger
from experiments.scripts.loggers.wandb_log import WandbLogger


class RandomSearchExperiment(ExperimentBase):
    def __init__(self, initializer, fit_robustness, train_env, validation_env, max_train_steps: int,
                 validation_episodes: int, logger:Logger):

        super().__init__(train_env, validation_env, max_train_steps, validation_episodes, logger=logger)

        fitness = self.episodic_rewards(self.train_env, n_episodes=fit_robustness)
        self.alg = RandomSearch(initializer, fitness)

    def start(self):
        super().start(self.alg)


if __name__ == "__main__":
    env_name = "Acrobot-v1"
    env = gym.make(env_name)
    validation_episodes = 100
    fit_robustness = 20

    policy_dims = [sum(env.observation_space.shape),
                   256,
                   512,
                   env.action_space.n]

    initializer = partial(toolz.compose_left(LinearTorchPolicy, TorchPolicyAgent), policy_dims)

    logger = CompositeLogger([
        ConsoleLogger(),
        WandbLogger("ecrl", "eyal-segal", config={
            "Algorithm": "Random Search",
            "Environment": env_name,
            "Validation Episodes": validation_episodes,
            "Fitness Robustness": 10,
        })
    ])

    random_search = RandomSearchExperiment(initializer=initializer,
                                           train_env=gym.make(env_name),
                                           validation_env=gym.make(env_name),
                                           max_train_steps=int(1e6),
                                           fit_robustness=fit_robustness,
                                           validation_episodes=validation_episodes,
                                           logger=logger)
    random_search.start()
