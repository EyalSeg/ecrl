from argparse import ArgumentParser
from functools import partial

from gym.spaces import Box, Discrete

import toolz
import mlflow

from agents.pytorch import LinearTorchPolicy, TorchPolicyAgent
from algorithms.random_search import RandomSearch
from algorithms.trainer import Trainer
from loggers.composite_logger import CompositeLogger
from loggers.console_logger import ConsoleLogger


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--env", type=str)
    parser.add_argument("--algorithm", type=str, default="Random Search")
    parser.add_argument("--validation_episodes", type=int)
    parser.add_argument("--fitness_robustness", type=int)
    parser.add_argument("--train_steps", type=int)

    args = parser.parse_args()

    mlflow.log_params(args.__dict__)

    mlflow_logger = type("Object", (), {"log": lambda metrics: mlflow.log_metrics(metrics, step=metrics["train_step"])})

    logger = CompositeLogger([
        ConsoleLogger(),
        mlflow_logger
    ])

    trainer = Trainer(args.env, max_train_steps=args.train_steps, validation_episodes=args.validation_episodes, logger=logger)

    is_discrete = isinstance(trainer.train_env.action_space, Discrete)

    policy_dims = [sum(trainer.train_env.observation_space.shape),
                   256,
                   256,
                   trainer.train_env.action_space.n if is_discrete else trainer.train_env.action_space.shape[0]]

    def initializer():
        policy = LinearTorchPolicy(policy_dims)
        if is_discrete:
            agent = TorchPolicyAgent(policy, mode="discrete-deterministic")
        else:
            agent = TorchPolicyAgent(policy, mode="continuous")

        return agent

    fitness = trainer.episodic_rewards(trainer.train_env, n_episodes=args.fitness_robustness)

    rs = RandomSearch(initializer, fitness)
    trainer.fit(rs)
