"""
algorithms/genetic_algorithm.py
Population-based optimization using selection, recombination, and mutation.
"""

from typing import Tuple
from algorithms.base import LocalSearchSolver
from core.state import State
from core.problem import CommuteProblem
from algorithms.operators import get_neighbors, crossover
import random


class GeneticAlgorithmSolver(LocalSearchSolver):
    def __init__(self, problem: CommuteProblem, **kwargs):
        super().__init__(problem, **kwargs)
        self.population_size: int = self.params.get("population_size", 30)
        self.generations: int = self.params.get("generations", 100)
        self.crossover_rate: float = self.params.get("crossover_rate", 0.8)
        self.mutation_rate: float = self.params.get("mutation_rate", 0.2)
        self.elitism: int = self.params.get("elitism", 2)

    def solve(self) -> Tuple[State, float]:
        """
        Executes Genetic Algorithm optimization.

        Guidelines:
        - Initialize a population of valid candidate solutions.
        - Design a fitness metric suitable for minimizing commute travel time (lower cost -> higher fitness).
        - Use selection operators (such as tournament or roulette selection) to pick parents.
        - Pair parents to produce offspring via `crossover()`, and apply stochastic mutation.
        - Preserve the top candidates across generations if using elitism.
        - Track best population performance per generation in self.history.

        Encouraged:
        - Experimenting with population size, selection pressure, and operator rates to balance exploration vs. exploitation.
        """
        # TODO: Implement Genetic Algorithm
        population = []
        for _ in range(self.population_size):
            s = self.problem.generate_random_state() if hasattr(self.problem, "generate_random_state") else State([self.problem.origin, self.problem.destination], self.problem.t_min)
            c = self.problem.evaluate(s) if hasattr(self.problem, "evaluate") else s.evaluate(self.problem)
            population.append((s, c))

        population.sort(key=lambda x: x[1])
        best_state, best_cost = population[0]

        for gen in range(self.generations):
            self.history.append(best_cost)
            population.sort(key=lambda x: x[1])

            new_population = population[:self.elitism]

            while len(new_population) < self.population_size:
                p1 = min(random.sample(population, 3), key=lambda x: x[1])[0]
                p2 = min(random.sample(population, 3), key=lambda x: x[1])[0]

                if random.random() < self.crossover_rate:
                    child = crossover(p1, p2, self.problem)
                else:
                    child = p1

                if random.random() < self.mutation_rate:
                    neighbors = get_neighbors(child, self.problem)
                    if neighbors:
                        child = random.choice(neighbors)

                cc = self.problem.evaluate(child) if hasattr(self.problem, "evaluate") else child.evaluate(self.problem)
                new_population.append((child, cc))

            population = new_population
            population.sort(key=lambda x: x[1])
            if population[0][1] < best_cost:
                best_state, best_cost = population[0]

        return best_state, best_cost
