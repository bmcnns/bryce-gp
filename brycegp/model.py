import random
import tqdm

from brycegp.pareto import pareto_front
from brycegp.program import Program
from brycegp.selection import tournament_selection
from brycegp.variation import crossover

class SymbolicRegressor:
    def __init__(self, num_generations, num_cascades,
                 population_size, max_complexity,
                 archive_tournament_size, population_tournament_size):
        self.num_generations = num_generations
        self.num_cascades = num_cascades
        self.population_size = population_size
        self.max_complexity = max_complexity
        self.archive_tournament_size = archive_tournament_size
        self.population_tournament_size = population_tournament_size

    def fit(self, X, y):
        # Population Generation
        archive = []

        for cascade in range(1, self.num_cascades + 1):
            print("beginning cascade", cascade)

            population = [Program(num_features=1, num_outputs=1, depth=random.randint(3, 6)) for i in
                          range(self.population_size)]

            for generation in tqdm.trange(self.num_generations):
                pareto_front(archive, len(population) * 0.10, population, X, y)
                reproduced, crossed_over, mutated = [], [], []
                while len(crossed_over) < len(population):
                    parent1 = tournament_selection(archive, X, y, tournament_size=4)
                    parent2 = tournament_selection(population, X, y, tournament_size=4)
                    child1, child2 = crossover(parent1, parent2)

                    while child1.complexity() >= self.max_complexity:
                        child1, _ = crossover(parent1, parent2)
                    while child2.complexity() >= self.max_complexity:
                        child2, _ = crossover(parent1, parent2)

                    crossed_over.extend([child1, child2])
                population = reproduced + crossed_over

        return archive