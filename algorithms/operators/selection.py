import toolz


@toolz.curry
def truncated_selection(truncation_len, population, fitnesses):
    # Sort the populaiton by the fitness
    fitness_pop = sorted(zip(fitnesses, population), reverse=True, key=lambda fit_pop: fit_pop[0])
    pop = [specimen for fit, specimen in fitness_pop]

    return pop[:truncation_len]