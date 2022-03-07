from typing import List

import toolz

from agents.agent_typing import Agent


@toolz.curry
def truncated_selection(truncation_len: int, population: List[Agent], fitnesses: List[Agent]):
    fit_pop = [{"specimen": specimen, "fitness": fit} for specimen, fit in zip(population, fitnesses)]
    selected = list(toolz.topk(truncation_len, fit_pop, key=lambda x: x["fitness"]))

    return selected

