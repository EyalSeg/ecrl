import gym
import toolz

from agents.agent_typing import Agent
from algorithms.algorithm_typing import EvolutionaryAlgorithm


class StepsMonitoringWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        self.step_count = 0

    def step(self, action):
        self.step_count += 1

        return super().step(action)


class ExperimentBase:
    def __init__(self, train_env: gym.Env, validation_env: gym.Env, max_train_steps: int, validation_episodes: int):
        self.max_train_steps = max_train_steps
        self.validation_episodes = validation_episodes

        self.train_env = StepsMonitoringWrapper(train_env)
        self.validation_env = validation_env

    def start(self, algorithm: EvolutionaryAlgorithm):
        gen = 0
        while self.train_env.step_count < self.max_train_steps:
            algorithm.generation()

            elite = algorithm.elite
            elite_value = self.validate(elite)

            print(f"gen {gen} \t\ttimesteps {self.train_env.step_count} \t\trewards {elite_value}")

            gen += 1

    @toolz.curry
    def episodic_rewards(self, env, agent, n_episodes=1):
        rewards = 0

        for _ in range(n_episodes):
            observation = env.reset()
            done = False

            while not done:
                action = agent.act(observation)
                observation, reward, done, info = env.step(action)
                rewards += reward

        return rewards / n_episodes

    def validate(self, agent: Agent):
        return self.episodic_rewards(self.validation_env, agent, self.validation_episodes)
