from typing import Callable, List

import gym
import toolz
import numpy as np

from agents.agent_typing import Agent
from algorithms.algorithm_typing import EvolutionaryAlgorithm, Trajectory
from loggers.logger_typing import Logger


class StepsMonitoringWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        self.step_count = 0
        self._max_episode_steps = env._max_episode_steps

    def step(self, action):
        self.step_count += 1

        return super().step(action)


class Trainer:
    def __init__(self, env_name: str, max_train_steps: int, validation_episodes: int,
                 logger: Logger = None, log_callbacks: List[Callable[[EvolutionaryAlgorithm], dict]] = ()):
        self.env_name = env_name

        self.max_train_steps = max_train_steps
        self.validation_episodes = validation_episodes

        self.train_env = StepsMonitoringWrapper(gym.make(env_name))
        self.validation_env = gym.make(env_name)

        self.logger = logger
        self.log_callbacks = log_callbacks

    def fit(self, algorithm: EvolutionaryAlgorithm):
        gen = 0
        elite = None
        elite_value = None

        while self.train_env.step_count < self.max_train_steps:
            algorithm.generation()

            if elite != algorithm.elite:
                elite = algorithm.elite
                elite_value = self.validate(elite)

            if self.logger:
                logs = {
                    "train_step": self.train_env.step_count,
                    "generation": gen,
                    "train_fitness": algorithm.elite_fitness,
                    "validation_fitness": elite_value,
                }

                callbacked = [callback(algorithm) for callback in self.log_callbacks]
                for sublog in callbacked:
                    logs.update(sublog)

                self.logger.log(logs)

            gen += 1

        return elite, elite_value

    @toolz.curry
    def rollout(self, env, agent, visualize=False) -> Trajectory:
        return self._episode(env, agent, log_trajectory=True, visualize=visualize)

    @toolz.curry
    def episodic_rewards(self, env, agent, n_episodes=1) -> float:
        rewards = [self._episode(env, agent, log_trajectory=False) for _ in range(n_episodes)]
        return sum(rewards) / n_episodes

    def validate(self, agent: Agent) -> float:
        return self.episodic_rewards(self.validation_env, agent, self.validation_episodes)

    @toolz.curry
    def _episode(self, env, agent, log_trajectory=False, visualize=False):
        '''
        :param env:
        :param agent:
        :param log_trajectory:
        :return: Runs a single episode.
        If logging the trajectory, will return an np arrays of states, actions, and rewards
         of the entire rollout.
        Otherwise, will return the sum of rewards.
        '''
        if log_trajectory:
            rewards = np.full(env._max_episode_steps, np.nan)
            observations = np.full(
                (env._max_episode_steps, *env.observation_space.shape),
                np.nan)

            actions = np.full(env._max_episode_steps, np.nan)

            def on_timestep(t, s, a, r):
                observations[t, :] = s
                actions[t] = a
                rewards[t] = r

            retval = lambda t: (observations[:t], actions[:t], rewards[:t])

        else:
            rewards = [0]

            def on_timestep(t, s, a, r):
                rewards[0] += r

            retval = lambda t: rewards[0]

        observation = env.reset()
        done = False
        timestep = 0

        while not done:
            if visualize:
                env.render()

            action = agent.act(observation)
            observation_, reward, done, info = env.step(action)

            on_timestep(timestep, observation, action, reward)

            observation = observation_
            timestep += 1

        return retval(timestep)