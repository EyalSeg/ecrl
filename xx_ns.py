from argparse import ArgumentParser

from gym.spaces import Box, Discrete
import toolz
import mlflow
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

    args = parser.parse_args()

    mlflow.log_params(args.__dict__)
    mlflow.log_param("algorithm", "Explore Exploit Novelty Search")

    mlflow_logger = type("Object", (), {"log": lambda metrics: mlflow.log_metrics(metrics, step=metrics["train_step"])})

    logger = CompositeLogger([
        ConsoleLogger(),
        mlflow_logger,
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
