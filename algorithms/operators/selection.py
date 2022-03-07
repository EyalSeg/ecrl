import toolz


@toolz.curry
def truncated_selection(truncation_len, population, fitnesses):
    fit_pop = [{"specimen": specimen, "fitness": fit} for specimen, fit in zip(population, fitnesses)]
    selected = list(toolz.topk(truncation_len, fit_pop, key=lambda x: x["fitness"]))

    return selected

