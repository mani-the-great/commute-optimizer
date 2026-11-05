"""
algorithms/beam_search.py
Local Beam Search maintaining K parallel candidate states.
"""

from typing import Tuple
from algorithms.base import LocalSearchSolver
from core.state import State
from core.problem import CommuteProblem
from algorithms.operators import get_neighbors


class LocalBeamSearchSolver(LocalSearchSolver):
    def __init__(self, problem: CommuteProblem, **kwargs):
        super().__init__(problem, **kwargs)
        self.beam_width: int = self.params.get("beam_width", 5)
        self.max_iterations: int = self.params.get("max_iterations", 150)

    def solve(self) -> Tuple[State, float]:
        """
        Executes Local Beam Search.

        Guidelines:
        - Maintain a set of `beam_width` parallel configurations.
        - In each iteration, generate neighbors from all active states in the beam.
        - Pool and rank all generated successors collectively, then retain the top `beam_width` states.
        - Halt when the beam ceases to improve or reaches max_iterations.
        - Record the best cost of the population at each step in self.history.

        Encouraged:
        - Ensuring diversity across states in the beam to prevent collapse into a single redundant path.

        Prohibited:
        - Treating this as K independent, non-communicating searches.
        """
        # TODO: Implement Local Beam Search
        beam = []
        for _ in range(self.beam_width):
            s = self.problem.generate_random_state() if hasattr(self.problem, "generate_random_state") else State([self.problem.origin, self.problem.destination], self.problem.t_min)
            c = self.problem.evaluate(s) if hasattr(self.problem, "evaluate") else s.evaluate(self.problem)
            beam.append((s, c))

        best_state, best_cost = min(beam, key=lambda x: x[1])

        for iteration in range(self.max_iterations):
            self.history.append(best_cost)
            all_candidates = []

            for state, cost in beam:
                neighbors = get_neighbors(state, self.problem)
                for n in neighbors:
                    nc = self.problem.evaluate(n) if hasattr(self.problem, "evaluate") else n.evaluate(self.problem)
                    all_candidates.append((n, nc))

            if not all_candidates:
                break

            # Remove duplicates based on path and time representation if needed
            unique_candidates = {tuple(s.path) + (s.departure_time,): (s, c) for s, c in all_candidates}
            sorted_candidates = sorted(unique_candidates.values(), key=lambda x: x[1])

            beam = sorted_candidates[:self.beam_width]
            current_best_state, current_best_cost = beam[0]

            if current_best_cost < best_cost:
                best_state, best_cost = current_best_state, current_best_cost

        return best_state, best_cost
