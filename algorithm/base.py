from abc import ABC, abstractmethod
from random import random
from typing import Any, TypeVar

import numpy as np

T = TypeVar('T')


class GeneticAlgorithm(ABC):
    def __init__(self, population: list[T], crossover_probability: float, mutation_probability: float):
        self.population = population
        self.generation = 0
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability

    def step(self) -> None:
        offsprings = []

        while len(offsprings) < len(self.population):
            parent1, parent2 = self._select_parents()

            if random() < self.crossover_probability:
                child = self._crossover(parent1, parent2)
            else:
                child = parent1

            if random() < self.mutation_probability:
                child = self._mutate(child)
            offsprings.append(child)

        self.population = offsprings[:len(self.population)]
        self.generation += 1

    def get_best(self) -> tuple[T, float]:
        fitness_values = [self._fitness(ind) for ind in self.population]
        best_idx = int(np.argmax(fitness_values))
        return self.population[best_idx], fitness_values[best_idx]

    @abstractmethod
    def termination_condition(self) -> bool:
        pass

    @abstractmethod
    def _fitness(self, individual: T) -> float:
        pass

    @abstractmethod
    def _select_parents(self) -> tuple[T, T]:
        pass

    @abstractmethod
    def _crossover(self, parent1: T, parent2: T) -> T:
        pass

    @abstractmethod
    def _mutate(self, individual: T) -> T:
        pass
