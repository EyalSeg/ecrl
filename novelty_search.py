from argparse import ArgumentParser
from functools import partial

import toolz
import numpy as np

from agents.pytorch import LinearTorchPolicy, TorchPolicyAgent, add_gaussian_noise
from algorithms.novelty_search import NoveltySearch, knn_novelty
from algorithms.operators import truncated_selection
from algorithms.trainer import Trainer
from loggers.console_logger import ConsoleLogger


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

    return rewards, np.array(observation + [timestep / 500])


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

    logger = ConsoleLogger()

    trainer = Trainer(env_name=args.env,
                      max_train_steps=args.train_steps,
                      validation_episodes=args.validation_episodes,
                      logger=logger)

    policy_dims = [sum(trainer.train_env.observation_space.shape),
                   256,
                   512,
                   trainer.train_env.action_space.n]

    initializer = partial(toolz.compose_left(LinearTorchPolicy, TorchPolicyAgent), policy_dims)
    rollout = last_observation_behaviour_characteristic(trainer.train_env)
    mutator = add_gaussian_noise(args.mutation_strength)
    selector = truncated_selection(args.truncation_size)
    novelty_measure = knn_novelty(args.novelty_neighbors)

    ns = NoveltySearch(
        pop_size=args.popsize,
        initializer=initializer,
        rollout=rollout,
        novelty_measure=novelty_measure,
        archive_pr=args.archive_pr,
        survivors_selector=selector,
        mutator=mutator,
        robustness=args.fitness_robustness,
    )
    trainer.fit(ns)

