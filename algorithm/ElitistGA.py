from abc import ABC
from algorithm.base import GeneticAlgorithm, T


class ElitistGA(GeneticAlgorithm, ABC):
    def __init__(
        self,
        population: list[T],
        crossover_probability: float,
        mutation_probability: float,
        elite_count: int = 2,
    ):
        super().__init__(population, crossover_probability, mutation_probability)
        self.elite_count = elite_count

    def step(self) -> None:
        elites = self._get_elites()
        super().step()
        self._replace_worst_with_elites(elites)

    def _get_elites(self) -> list[T]:
        fitness_values = [(i, self._fitness(ind)) for i, ind in enumerate(self.population)]
        fitness_values.sort(key=lambda x: x[1], reverse=True)
        return [self.population[i] for i, _ in fitness_values[:self.elite_count]]

    def _replace_worst_with_elites(self, elites: list[T]) -> None:
        fitness_values = [(i, self._fitness(ind)) for i, ind in enumerate(self.population)]
        fitness_values.sort(key=lambda x: x[1])
        for j, elite in enumerate(elites):
            worst_idx = fitness_values[j][0]
            self.population[worst_idx] = elite
