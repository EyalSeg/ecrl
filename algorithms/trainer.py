from typing import Callable, List
from math import inf

import gym
import pybulletgym  # register PyBullet enviroments with open ai gym

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

        self.local_elite = None
        self.local_elite_value = None

        self.global_elite = None
        self.global_elite_value = -inf

    def fit(self, algorithm: EvolutionaryAlgorithm):
        gen = 0

        while self.train_env.step_count < self.max_train_steps:
            algorithm.generation()

            if self.local_elite != algorithm.elite:
                self.local_elite = algorithm.elite
                self.local_elite_value = self.validate(self.local_elite)

            if self.local_elite_value > self.global_elite_value:
                self.global_elite = self.local_elite
                self.global_elite_value = self.local_elite_value

            if self.logger:
                logs = {
                    "train_step": self.train_env.step_count,
                    "generation": gen,
                    "train_fitness": algorithm.elite_fitness,
                    "validation_fitness": self.local_elite_value,
                    "cummulative_validation_fitness": self.global_elite_value,
                }

                callbacked = [callback(algorithm) for callback in self.log_callbacks]
                for sublog in callbacked:
                    logs.update(sublog)

                self.logger.log(logs)

            gen += 1

    @toolz.curry
    def rollout(self, env, agent, visualize=False) -> Trajectory:
        observations, actions, rewards, positions = self._episode(env, agent, log_trajectory=True, visualize=visualize)

        return Trajectory(observations=observations, actions=actions, rewards=rewards, positions=positions)

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
                np.nan
            )
            positions = np.full(
                (env._max_episode_steps, 3),
                np.nan
            )

            # actions = np.full(env._max_episode_steps, np.nan)
            actions = []

            def on_timestep(t, s, a, r, p):
                observations[t, :] = s
                actions.append(a)
                rewards[t] = r
                positions[t, :] = p

            retval = lambda t: (observations[:t], np.array(actions[:t]), rewards[:t], positions[:t])

        else:
            rewards = [0]

            def on_timestep(t, s, a, r, p):
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
            xyz = env.robot.body_xyz if hasattr(env, "robot") else np.full(3, np.nan)

            on_timestep(timestep, observation, action, reward, xyz)

            observation = observation_
            timestep += 1

        return retval(timestep)