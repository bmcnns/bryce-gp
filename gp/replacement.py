import random
from copy import deepcopy

from gp.variation import crossover, mutation

def replacement(population, population_size=100):

    offspring = []
    while len(offspring) < population_size:
        parent1, parent2 = random.sample(population, 2)
        parent1 = deepcopy(parent1)
        parent2 = deepcopy(parent2)

        probability = random.random()
        if 0.0 < probability < 0.85:
            child1, child2 = crossover(parent1, parent2)
        elif 0.85 < probability < 0.95:
            child1, child2 = parent1, parent2
        else:
            child1 = mutation(parent1, 3)
            child2 = mutation(parent2, 3)

        offspring.append(child1)

        if len(offspring) == population_size:
            break
        else:
            offspring.append(child2)

    return offspring