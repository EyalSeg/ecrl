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
    logger = ConsoleLogger()

    args = {}

    trainer = Trainer(env_name="Acrobot-v1",
                      max_train_steps=int(1e6),
                      validation_episodes=100,
                      logger=logger)

    policy_dims = [sum(trainer.train_env.observation_space.shape),
                   256,
                   512,
                   trainer.train_env.action_space.n]

    initializer = partial(toolz.compose_left(LinearTorchPolicy, TorchPolicyAgent), policy_dims)
    rollout = last_observation_behaviour_characteristic(trainer.train_env)
    mutator = add_gaussian_noise(0.8)
    selector = truncated_selection(6)
    novelty_measure = knn_novelty(40)

    ns = NoveltySearch(
        pop_size=50,
        initializer=initializer,
        rollout=rollout,
        novelty_measure=novelty_measure,
        archive_pr=0.1,
        survivors_selector=selector,
        mutator=mutator,
        robustness=5
    )
    trainer.fit(ns)

