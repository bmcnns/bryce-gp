import random

import numpy as np
from sklearn.metrics import mean_squared_error

def dominates(candidate, other):
    cand_program, cand_error = candidate
    other_program, other_error = other

    # Candidate is no worse than other in all objectives.
    no_worse_in_error = cand_error <= other_error
    no_worse_in_complexity = cand_program.complexity() <= other_program.complexity()

    # Candidate is strictly better in at least one objective.
    strictly_better = (cand_error < other_error) or (cand_program.complexity() < other_program.complexity())

    return no_worse_in_error and no_worse_in_complexity and strictly_better

    #return cand_program.complexity() < other_program.complexity() and cand_error < other_error

def tournament_selection(population, X, y, tournament_size=4):
    tournament = random.sample(population, k=tournament_size)
    winner = sorted(tournament, key=lambda program: mean_squared_error(program(X), y))[0]
    return winner