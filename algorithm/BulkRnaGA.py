import numpy as np

from algorithm.ElitistGA import ElitistGA
from algorithm.common.crossover import blend_crossover
from algorithm.common.parent_selection import tournament_selection


class BulkRnaGA(ElitistGA):
    def __init__(
        self,
        population_size: int,
        crossover_probability: float,
        mutation_probability: float,
        m_matrix: np.ndarray,
        h_matrix: np.ndarray,
        initial_mutation_strength: float = 0.5,
        decay_rate: float = 0.99,
        elite_count: int = 2,
    ):
        self.m_matrix = m_matrix  # genes x samples
        self.h_matrix = h_matrix  # genes x celltypes
        self.num_celltypes = h_matrix.shape[1]
        self.num_samples = m_matrix.shape[1]
        self.initial_mutation_strength = initial_mutation_strength
        self.decay_rate = decay_rate
        self.m_mean = np.mean(np.abs(m_matrix))  # for relative error visualization

        population = []
        for _ in range(population_size):
            x = np.random.uniform(0, 1, (self.num_celltypes, self.num_samples))
            x = x / x.sum(axis=0, keepdims=True)
            population.append(x)

        super().__init__(population, crossover_probability, mutation_probability, elite_count)

    def termination_condition(self) -> bool:
        _, best_fitness = self.get_best()
        return abs(best_fitness) <= 5.5

    def _fitness(self, individual: np.ndarray) -> float:
        predicted = self.h_matrix @ individual
        return -float(np.mean(np.abs(self.m_matrix - predicted)))

    def _select_parents(self) -> tuple[np.ndarray, np.ndarray]:
        return tournament_selection(self.population, self._fitness)

    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        child = blend_crossover(parent1, parent2)
        return child / child.sum(axis=0, keepdims=True) # normalize

    def _mutate(self, x: np.ndarray) -> np.ndarray:
        strength = self.initial_mutation_strength * (self.decay_rate ** self.generation)
        noise = np.random.uniform(-strength, strength, x.shape)
        return np.maximum(x + noise, 0) # trim