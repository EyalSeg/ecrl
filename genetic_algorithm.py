from argparse import ArgumentParser
from functools import partial

import toolz
import mlflow

from gym.spaces import Box, Discrete

from agents.pytorch import LinearTorchPolicy, TorchPolicyAgent, add_gaussian_noise
from algorithms.genetic_algorithm import GeneticAlgorithm
from algorithms.operators.selection import truncated_selection, find_true_elite
from loggers.composite_logger import CompositeLogger
from loggers.console_logger import ConsoleLogger
from algorithms.trainer import Trainer


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--algorithm", type=str, default="Genetic Algorithm")
    parser.add_argument("--popsize", type=int, required=True)
    parser.add_argument("--validation_episodes", type=int, required=True)
    parser.add_argument("--elite_robustness", type=int, required=True)
    parser.add_argument("--elite_candidates", type=int, required=True)
    parser.add_argument("--mutation_strength", type=float, required=True)
    parser.add_argument("--truncation_size", type=int, required=True)
    parser.add_argument("--train_steps", type=int, default=int(1e6))
    parser.add_argument("--elitism", type=int, default=1,)

    args = parser.parse_args()

    mlflow.log_params(args.__dict__)

    mlflow_logger = type("Object", (), {"log": lambda metrics: mlflow.log_metrics(metrics, step=metrics["train_step"])})

    logger = CompositeLogger([
        ConsoleLogger(),
        mlflow_logger,
    ])

    trainer = Trainer(env_name=args.env,
                      max_train_steps=args.train_steps,
                      validation_episodes=args.validation_episodes,
                      logger=logger)

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

    fitness = trainer.episodic_rewards(trainer.train_env, n_episodes=1)
    mutator = add_gaussian_noise(args.mutation_strength)
    selector = truncated_selection(args.truncation_size)
    elite_extractor = find_true_elite(args.elite_candidates, fitness, args.elite_robustness)

    ga = GeneticAlgorithm(
        pop_size=args.popsize,
        initializer=initializer,
        fitness=fitness,
        mutator=mutator,
        survivors_selector=selector,
        elite_extractor=elite_extractor,
        elitism=args.elitism
    )

    trainer.fit(ga)

