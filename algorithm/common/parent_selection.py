from typing import Callable

import numpy as np


def roulette_selection(
    population: np.ndarray,
    fitness_function: Callable[[np.ndarray], float],
) -> tuple[np.ndarray, np.ndarray]:
    fitness_values = np.array([fitness_function(individual) for individual in population])
    total_fitness = fitness_values.sum()
    probabilities = fitness_values / total_fitness

    indices = np.random.choice(len(population), size=2, p=probabilities)
    return population[indices[0]], population[indices[1]]


def rank_selection(
    population: np.ndarray,
    fitness_function: Callable[[np.ndarray], float],
) -> tuple[np.ndarray, np.ndarray]:
    fitness_values = np.array([fitness_function(individual) for individual in population])
    # Rank from 1 (worst) to N (best)
    ranks = np.argsort(np.argsort(fitness_values)) + 1
    # Selection probability proportional to rank
    probabilities = ranks / ranks.sum()

    indices = np.random.choice(len(population), size=2, p=probabilities)
    return population[indices[0]], population[indices[1]]


def tournament_selection(
    population: list[np.ndarray],
    fitness_function: Callable[[np.ndarray], float],
    tournament_size: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    def select_one() -> np.ndarray:
        indices = np.random.choice(len(population), size=tournament_size, replace=False)
        competitors = [population[i] for i in indices]
        fitness_values = [fitness_function(ind) for ind in competitors]
        winner_idx = np.argmax(fitness_values)
        return competitors[winner_idx]

    return select_one(), select_one()
