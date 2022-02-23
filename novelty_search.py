from argparse import ArgumentParser
from functools import partial

import toolz
import numpy as np

from agents.pytorch import LinearTorchPolicy, TorchPolicyAgent, add_gaussian_noise
from algorithms.novelty_search import NoveltySearch
from algorithms.operators.knn_novelty import knn_novelty
from algorithms.operators.selection import truncated_selection
from algorithms.trainer import Trainer
from loggers.composite_logger import CompositeLogger
from loggers.console_logger import ConsoleLogger
from loggers.wandb_log import WandbLogger


@toolz.curry
def last_observation_behaviour_characteristic(env, agent):
    timestep = 0
    rewards = 0
    observation = env.reset()
    done = False

    while not done:
        action = agent.act(observation)
        observation, reward, done, info = env.step(action)
        timestep += 1
        rewards += reward

    return rewards, np.array(observation + [timestep / env._max_episode_steps])


@toolz.curry
def robust_characteristic(k, bc_func, agent):
    rewards_bcs = [bc_func(agent) for _ in range(k)]
    rewards, bcs = [reward for reward, bc in rewards_bcs], [bc for reward, bc in rewards_bcs]

    return sum(rewards) / k, sum(bcs) / 5


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--env", type=str)
    parser.add_argument("--popsize", type=int)
    parser.add_argument("--validation_episodes", type=int)
    parser.add_argument("--fitness_robustness", type=int)
    parser.add_argument("--mutation_strength", type=float)
    parser.add_argument("--truncation_size", type=int)
    parser.add_argument("--train_steps", type=int, default=int(1e6))
    parser.add_argument("--novelty_neighbors", type=int)
    parser.add_argument("--archive_pr", type=float)

    args = parser.parse_args()

    logger = CompositeLogger([
        ConsoleLogger(),
        WandbLogger("ecrl", "eyal-segal", config={
            "Algorithm": "Novelty Search",
            "env": args.env,
            "popsize": args.popsize,
            "validation_episodes": args.validation_episodes,
            "fitness_robustness": args.fitness_robustness,
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
    rollout = last_observation_behaviour_characteristic(trainer.train_env)
    robust_rollout = robust_characteristic(args.fitness_robustness, rollout)
    mutator = add_gaussian_noise(args.mutation_strength)
    selector = truncated_selection(args.truncation_size)
    novelty_measure = knn_novelty(args.novelty_neighbors)

    ns = NoveltySearch(
        pop_size=args.popsize,
        initializer=initializer,
        rollout=robust_rollout,
        novelty_measure=novelty_measure,
        archive_pr=args.archive_pr,
        survivors_selector=selector,
        mutator=mutator,
        robustness=args.fitness_robustness,
    )
    trainer.fit(ns)

