"""
algorithms/hill_climbing.py
Hill Climbing search with sideways move limits and random restarts.
"""

from typing import Tuple
from algorithms.base import LocalSearchSolver
from core.state import State
from core.problem import CommuteProblem
from algorithms.operators import get_neighbors


class HillClimbingSolver(LocalSearchSolver):
    def __init__(self, problem: CommuteProblem, **kwargs):
        super().__init__(problem, **kwargs)
        self.max_sideways: int = self.params.get("max_sideways", 100)
        self.max_restarts: int = self.params.get("max_restarts", 5)

    def solve(self) -> Tuple[State, float]:
        """
        Executes Hill Climbing optimization.

        Guidelines:
        - Search through the neighborhood greedily for candidates that minimize total travel time.
        - Handle flat local maxima / shoulders by allowing up to `max_sideways` non-improving steps.
        - Apply up to `max_restarts` to explore different starting basins.
        - Track the best objective cost seen at each major step in self.history.

        Prohibited:
        - Looping indefinitely on flat plateaus.
        """
        # TODO: Implement Hill Climbing with sideways moves and restarts
        best_overall_state = None
        best_overall_cost = float('inf')

        for restart in range(self.max_restarts):
            current_state = self.problem.generate_random_state() if hasattr(self.problem, "generate_random_state") else State([self.problem.origin, self.problem.destination], self.problem.t_min)
            current_cost = self.problem.evaluate(current_state) if hasattr(self.problem, "evaluate") else current_state.evaluate(self.problem)

            sideways_steps = 0
            
            while True:
                self.history.append(current_cost)
                neighbors = get_neighbors(current_state, self.problem)
                if not neighbors:
                    break

                evaluated_neighbors = []
                for n in neighbors:
                    cost = self.problem.evaluate(n) if hasattr(self.problem, "evaluate") else n.evaluate(self.problem)
                    evaluated_neighbors.append((n, cost))

                evaluated_neighbors.sort(key=lambda x: x[1])
                next_state, next_cost = evaluated_neighbors[0]

                if next_cost < current_cost:
                    current_state, current_cost = next_state, next_cost
                    sideways_steps = 0  # Reset sideways counter on strict improvement
                elif next_cost == current_cost and sideways_steps < self.max_sideways:
                    current_state, current_cost = next_state, next_cost
                    sideways_steps += 1
                else:
                    break

            if current_cost < best_overall_cost:
                best_overall_cost = current_cost
                best_overall_state = current_state

        return best_overall_state, best_overall_cost
