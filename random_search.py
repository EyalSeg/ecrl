from argparse import ArgumentParser
from functools import partial

import toolz

from agents.pytorch import LinearTorchPolicy, TorchPolicyAgent
from algorithms.random_search import RandomSearch
from algorithms.trainer import Trainer
from loggers.composite_logger import CompositeLogger
from loggers.console_logger import ConsoleLogger
from loggers.wandb_log import WandbLogger


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--env", type=str)
    parser.add_argument("--validation_episodes", type=int)
    parser.add_argument("--fitness_robustness", type=int)
    parser.add_argument("--train_steps", type=int)

    args = parser.parse_args()

    logger = CompositeLogger([
        ConsoleLogger(),
        WandbLogger("ecrl", "eyal-segal", config={
            "Algorithm": "Random Search",
            "env": args.env,
            "validation_episodes": args.validation_episodes,
            "fitness_robustness": args.fitness_robustness,
        })
    ])

    trainer = Trainer(args.env, max_train_steps=args.train_steps, validation_episodes=args.validation_episodes, logger=logger)

    policy_dims = [sum(trainer.train_env.observation_space.shape),
                   256,
                   512,
                   trainer.train_env.action_space.n]

    initializer = partial(toolz.compose_left(LinearTorchPolicy, TorchPolicyAgent), policy_dims)
    fitness = trainer.episodic_rewards(trainer.train_env, n_episodes=args.fitness_robustness)

    rs = RandomSearch(initializer, fitness)
    trainer.fit(rs)
