from functools import partial

import toolz

from agents.pytorch import LinearTorchPolicy, TorchPolicyAgent
from algorithms.random_search import RandomSearch
from algorithms.trainer import Trainer
from loggers.composite_logger import CompositeLogger
from loggers.console_logger import ConsoleLogger
from loggers.wandb_log import WandbLogger


if __name__ == "__main__":
    env_name = "Acrobot-v1"
    validation_episodes = 100
    fit_robustness = 20

    logger = CompositeLogger([
        ConsoleLogger(),
        WandbLogger("ecrl", "eyal-segal", config={
            "Algorithm": "Random Search",
            "Environment": env_name,
            "Validation Episodes": validation_episodes,
            "Fitness Robustness": fit_robustness,
        })
    ])

    trainer = Trainer(env_name, max_train_steps=int(1e6), validation_episodes=validation_episodes, logger=logger)

    policy_dims = [sum(trainer.train_env.observation_space.shape),
                   256,
                   512,
                   trainer.train_env.action_space.n]

    initializer = partial(toolz.compose_left(LinearTorchPolicy, TorchPolicyAgent), policy_dims)
    fitness = trainer.episodic_rewards(trainer.train_env, n_episodes=fit_robustness)

    rs = RandomSearch(initializer, fitness)
    trainer.fit(rs)
