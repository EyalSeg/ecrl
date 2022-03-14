from argparse import ArgumentParser

import toolz
import numpy as np

from functools import partial

from agents.pytorch import LinearTorchPolicy, TorchPolicyAgent, add_gaussian_noise
from algorithms.adaptive_explore_exploit import AdaptiveExploreExploit
from algorithms.operators.archive import ProbabilisticArchive
from algorithms.operators.behavior_characteristic import last_observation_bc
from algorithms.operators.knn_novelty import archive_to_knn_novelty
from algorithms.operators.selection import truncated_selection, find_true_elite
from algorithms.trainer import Trainer
from loggers.composite_logger import CompositeLogger
from loggers.console_logger import ConsoleLogger
from loggers.wandb_log import WandbLogger


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--env", type=str)
    parser.add_argument("--popsize", type=int)
    parser.add_argument("--validation_episodes", type=int)
    parser.add_argument("--mutation_strength", type=float)
    parser.add_argument("--truncation_size", type=int)
    parser.add_argument("--train_steps", type=int, default=int(1e6))
    parser.add_argument("--novelty_neighbors", type=int)
    parser.add_argument("--archive_pr", type=float)
    parser.add_argument("--elite_candidates", type=int)
    parser.add_argument("--elite_robustness", type=int)
    parser.add_argument("--ratio", type=float)
    parser.add_argument("--ratio_growth", type=float)
    parser.add_argument("--ratio_decay", type=float)
    parser.add_argument("--group", type=str, default=None)

    args = parser.parse_args()

    logger = CompositeLogger([
        ConsoleLogger(),
        WandbLogger(
            "ecrl", "eyal-segal",
            config={
                "Algorithm": "Explore Exploit Novelty Search",
                "env": args.env,
                "popsize": args.popsize,
                "validation_episodes": args.validation_episodes,
                "mutation_strength": args.mutation_strength,
                "truncation_size": args.truncation_size,
                "novelty_neighbors": args.novelty_neighbors,
                "archive_pr": args.archive_pr,
                "elite_candidates": args.elite_candidates,
                "elite_robustness": args.elite_robustness,
                "ratio": args.ratio,
                "ratio_growth": args.ratio_growth,
                "ratio_decay": args.ratio_decay,
            },
            group=args.group)
    ])

    trainer = Trainer(env_name=args.env,
                      max_train_steps=args.train_steps,
                      validation_episodes=args.validation_episodes,
                      logger=logger)

    def last_observation_and_timestep_bc(trajectory):
        observations, _, _ = zip(trajectory)
        observations = observations[0]

        last_obs = observations[-1]
        last_timestep = len(observations) / trainer.train_env._max_episode_steps

        bc = np.append(last_obs, last_timestep)
        return bc

    policy_dims = [sum(trainer.train_env.observation_space.shape),
                   56,
                   56,
                   56,
                   trainer.train_env.action_space.n]

    init_agent = toolz.compose_left(LinearTorchPolicy, TorchPolicyAgent)
    initializer = partial(toolz.compose_left(LinearTorchPolicy, TorchPolicyAgent), policy_dims)
    mutator = add_gaussian_noise(args.mutation_strength)
    selector = truncated_selection(args.truncation_size)
    rollout = trainer.rollout(trainer.train_env, visualize=False)


    alg = AdaptiveExploreExploit(
        popsize=args.popsize,
        initializer=initializer,
        mutator=mutator,
        survivor_selection=selector,
        rollout=rollout,
        archive=ProbabilisticArchive(args.archive_pr),
        behavior_characteristic=last_observation_bc(trainer.train_env, add_timestep=True),
        elite_extractor=find_true_elite(
            args.elite_candidates,
            trainer.episodic_rewards(trainer.train_env, n_episodes=args.elite_robustness),
            1),
        exploit_fitness=lambda traj: sum(traj.rewards),
        explore_fitness_from_archive=archive_to_knn_novelty(args.novelty_neighbors),
        ratio_decay=args.ratio_decay,
        ratio_growth=args.ratio_growth,
        explore_exploit_ratio=args.ratio,
    )

    trainer.fit(alg)