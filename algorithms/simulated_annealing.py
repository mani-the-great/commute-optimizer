"""
algorithms/simulated_annealing.py
Stochastic optimization governed by a temperature schedule.
"""

from typing import Tuple
from algorithms.base import LocalSearchSolver
from core.state import State
from core.problem import CommuteProblem
from algorithms.operators import get_neighbors
import math
import random

class SimulatedAnnealingSolver(LocalSearchSolver):
    def __init__(self, problem: CommuteProblem, **kwargs):
        super().__init__(problem, **kwargs)
        self.initial_temp: float = self.params.get("initial_temp", 100.0)
        self.cooling_rate: float = self.params.get("cooling_rate", 0.995)
        self.min_temp: float = self.params.get("min_temp", 1e-3)
        self.max_iterations: int = self.params.get("max_iterations", 2000)

    def solve(self) -> Tuple[State, float]:
        """
        Executes Simulated Annealing optimization.

        Guidelines:
        - Start from a high temperature where worse moves are accepted with significant probability.
        - Lower temperature over iterations according to a cooling schedule.
        - Downhill moves (improvements) must always be accepted.
        - Uphill moves (worse costs) should be accepted probabilistically according to temperature.
        - Record objective values throughout the run in self.history.

        Prohibited:
        - Ignoring the temperature parameter or turning this into pure hill climbing.
        """
        # TODO: Implement Simulated Annealing
        current_state = self.problem.generate_random_state() if hasattr(self.problem, "generate_random_state") else State([self.problem.origin, self.problem.destination], self.problem.t_min)
        current_cost = self.problem.evaluate(current_state) if hasattr(self.problem, "evaluate") else current_state.evaluate(self.problem)

        best_state = current_state
        best_cost = current_cost

        temp = self.initial_temp

        for iteration in range(self.max_iterations):
            self.history.append(current_cost)
            if temp < self.min_temp:
                break

            neighbors = get_neighbors(current_state, self.problem)
            if not neighbors:
                break

            next_state = random.choice(neighbors)
            next_cost = self.problem.evaluate(next_state) if hasattr(self.problem, "evaluate") else next_state.evaluate(self.problem)

            delta_e = next_cost - current_cost  

            if delta_e < 0 or random.random() < math.exp(-delta_e / max(temp, 1e-9)):
                current_state, current_cost = next_state, next_cost
                if current_cost < best_cost:
                    best_state, best_cost = current_state, current_cost

            temp *= self.cooling_rate

        return best_state, best_cost
