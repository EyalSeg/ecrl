from argparse import ArgumentParser
from functools import partial

import toolz
import numpy as np

from agents.agent_typing import Agent
from agents.pytorch import LinearTorchPolicy, TorchPolicyAgent, add_gaussian_noise
from algorithms.algorithm_typing import Trajectory, FitnessMeasure
from algorithms.novelty_search import NoveltySearch
from algorithms.operators.knn_novelty import archive_to_knn_novelty
from algorithms.operators.selection import truncated_selection, find_true_elite
from algorithms.trainer import Trainer
from loggers.composite_logger import CompositeLogger
from loggers.console_logger import ConsoleLogger
from loggers.wandb_log import WandbLogger


class ProbabilisticArchive:
    def __init__(self, archive_pr: float):
        self.pr = archive_pr
        self._archive = []

    def store(self, items):
        to_add = list(toolz.random_sample(self.pr, items))

        self._archive.extend(to_add)

    def retrieve(self):
        return self._archive.copy()

@toolz.curry
def last_observation_bc(env, traj: Trajectory, add_timestep=False):
    last_observation = traj.observations[-1]

    if add_timestep:
        last_observation = np.append(last_observation, len(traj.observations) / env._max_episode_steps)

    return last_observation


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--env", type=str, required=True)
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

    logger = CompositeLogger([
        ConsoleLogger(),
        WandbLogger("ecrl", "eyal-segal", config={
            "Algorithm": "Novelty Search",
            "env": args.env,
            "popsize": args.popsize,
            "validation_episodes": args.validation_episodes,
            "elite_robustness": args.elite_robustness,
            "elite_candidates": args.elite_candidates,
            "mutation_strength": args.mutation_strength,
            "truncation_size": args.truncation_size,
            "novelty_neighbors": args.novelty_neighbors,
            "archive_pr": args.archive_pr,
        })
    ])

    trainer = Trainer(env_name=args.env,
                      max_train_steps=args.train_steps,
                      validation_episodes=args.validation_episodes,
                      logger=logger)

    policy_dims = [sum(trainer.train_env.observation_space.shape),
                   56,
                   56,
                   56,
                   trainer.train_env.action_space.n]

    initializer = partial(toolz.compose_left(LinearTorchPolicy, TorchPolicyAgent), policy_dims)
    rollout = trainer.rollout(trainer.train_env, visualize=False)
    mutator = add_gaussian_noise(args.mutation_strength)
    selector = truncated_selection(args.truncation_size)
    fitness: FitnessMeasure[Agent] = trainer.episodic_rewards(trainer.train_env, n_episodes=1)
    elite_extractor = find_true_elite(
        args.elite_candidates,
        trainer.episodic_rewards(trainer.train_env, n_episodes=args.elite_robustness),
        1)

    ns = NoveltySearch(
        pop_size=args.popsize,
        initializer=initializer,
        rollout=rollout,
        novelty_from_archive=archive_to_knn_novelty(args.novelty_neighbors),
        survivors_selector=selector,
        mutator=mutator,
        archive=ProbabilisticArchive(args.archive_pr),
        behavior_characteristic=last_observation_bc(trainer.train_env, add_timestep=True),
        elite_extractor=elite_extractor,
        fitness=lambda traj: sum(traj.rewards),
    )
    trainer.fit(ns)

