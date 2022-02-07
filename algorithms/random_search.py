
class RandomSearch:
    def __init__(self, initializer, fitness):
        self.fitness = fitness
        self.initializer = initializer

        self.elite = None
        self.elite_fitness = None

    def generation(self):
        specimen = self.initializer()
        fitness = self.fitness(specimen)

        if not self.elite or fitness > self.elite_fitness:
            self.elite = specimen
            self.elite_fitness = fitness
