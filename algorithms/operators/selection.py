from typing import List
from numpy import argmax

import toolz

from agents.agent_typing import Agent
from algorithms.algorithm_typing import FitnessMeasure


@toolz.curry
def truncated_selection(truncation_len: int, population: List[Agent], fitnesses: List[Agent]):
    fit_pop = [{"specimen": specimen, "fitness": fit} for specimen, fit in zip(population, fitnesses)]
    selected = list(toolz.topk(truncation_len, fit_pop, key=lambda x: x["fitness"]))

    return selected

@toolz.curry
def find_true_elite(candidates_len: int, fitness: FitnessMeasure, robustness: int,
                    population: List[Agent], fitnesses: List[Agent]) -> (Agent, float):

    candidates = truncated_selection(candidates_len, population, fitnesses)
    agent_episodes = [[fitness(agent) for _ in range(robustness)] for agent in population]
    fitnesses = [sum(episodes) / len(episodes) for episodes in agent_episodes]

    elite_idx = argmax(fitnesses)

    return candidates[elite_idx], fitnesses[elite_idx]


