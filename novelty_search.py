from argparse import ArgumentParser
from functools import partial

import toolz
import mlflow

from gym.spaces import Box, Discrete

from agents.agent_typing import Agent
from agents.pytorch import LinearTorchPolicy, TorchPolicyAgent, add_gaussian_noise
from algorithms.algorithm_typing import Trajectory, FitnessMeasure
from algorithms.novelty_search import NoveltySearch
from algorithms.operators.archive import ProbabilisticArchive
from algorithms.operators.behavior_characteristic import last_observation_bc, last_position_bc
from algorithms.operators.knn_novelty import archive_to_knn_novelty
from algorithms.operators.selection import truncated_selection, find_true_elite
from algorithms.trainer import Trainer
from loggers.composite_logger import CompositeLogger
from loggers.console_logger import ConsoleLogger


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--algorithm", type=str, default="Novelty Search")
    parser.add_argument("--standardize", type=bool, default=False)
    parser.add_argument("--popsize", type=int, required=True)
    parser.add_argument("--validation_episodes", type=int, required=True)
    parser.add_argument("--elite_robustness", type=int, required=True)
    parser.add_argument("--elite_candidates", type=int, required=True)
    parser.add_argument("--mutation_strength", type=float, required=True)
    parser.add_argument("--truncation_size", type=int, required=True)
    parser.add_argument("--train_steps", type=int, default=int(1e6))
    parser.add_argument("--novelty_neighbors", type=int, required=True)
    parser.add_argument("--archive_pr", type=float, required=True)

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

    rollout = trainer.rollout(trainer.train_env, visualize=False)
    mutator = add_gaussian_noise(args.mutation_strength)
    selector = truncated_selection(args.truncation_size)
    fitness: FitnessMeasure[Agent] = trainer.episodic_rewards(trainer.train_env, n_episodes=1)
    elite_extractor = find_true_elite(
        args.elite_candidates,
        trainer.episodic_rewards(trainer.train_env, n_episodes=args.elite_robustness),
        1)

    bc = last_position_bc if "PyBulletEnv" in args.env else last_observation_bc(add_timestep=True)

    ns = NoveltySearch(
        pop_size=args.popsize,
        initializer=initializer,
        rollout=rollout,
        novelty_from_archive=archive_to_knn_novelty(args.novelty_neighbors, standardize=args.standardize),
        survivors_selector=selector,
        mutator=mutator,
        archive=ProbabilisticArchive(args.archive_pr),
        behavior_characteristic=bc,
        elite_extractor=elite_extractor,
        fitness=lambda traj: sum(traj.rewards),
    )
    trainer.fit(ns)

