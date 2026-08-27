import math
from typing import Tuple
from algorithms import utils
from algorithms.problems import SystemRepairProblem


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def manhattanHeuristic(state, problem):
    """
    The Manhattan distance heuristic.

    Baseline rule for this workshop: estimate the direct distance to the next
    mandatory target:
    - K if the robot does not have the kit yet.
    - the nearest pending T if the robot has the kit and systems remain.
    - C if all systems have been repaired.
    """
    position, hasKit, pendingSystems = state

    if not hasKit:
        target = problem.kitPosition
    elif pendingSystems:
        target = min(
            pendingSystems,
            key=lambda system: abs(position[0] - system[0])
            + abs(position[1] - system[1]),
        )
    else:
        target = problem.controlPosition

    return abs(position[0] - target[0]) + abs(position[1] - target[1])


def euclideanHeuristic(state, problem):
    """
    The Euclidean distance heuristic.

    Baseline rule for this workshop: estimate the direct distance to the next
    mandatory target:
    - K if the robot does not have the kit yet.
    - the nearest pending T if the robot has the kit and systems remain.
    - C if all systems have been repaired.
    """
    position, hasKit, pendingSystems = state

    if not hasKit:
        target = problem.kitPosition
    elif pendingSystems:
        target = min(
            pendingSystems,
            key=lambda system: math.hypot(
                position[0] - system[0], position[1] - system[1]
            ),
        )
    else:
        target = problem.controlPosition

    return math.hypot(position[0] - target[0], position[1] - target[1])


def systemRepairHeuristic(
    state: Tuple[Tuple, bool, Tuple], problem: SystemRepairProblem
):
    """
    Your heuristic for the SystemRepairProblem.

    state: (position, hasKit, pendingSystems)
    problem: SystemRepairProblem instance

    This must be admissible and preferably consistent.

    Hints:
    - Use problem.heuristicInfo to cache expensive computations
    - Go with some simple heuristics first, then build up to more complex ones
    - Consider the kit, pending systems, and the final return to control center
    - Balance heuristic strength vs. computation time (do experiments!)
    """
    position, hasKit, pendingSystems = state

    requiredPositions = []
    if not hasKit:
        requiredPositions.append(problem.kitPosition)
    requiredPositions.extend(pendingSystems)
    requiredPositions.append(problem.controlPosition)

    targets = list(dict.fromkeys(requiredPositions))
    if not targets:
        return 0

    # The MST is a lower bound on the cost of visiting every remaining target.
    connected = {position}
    remainingTargets = set(targets) - connected
    totalCost = 0
    while remainingTargets:
        cheapestTarget = None
        cheapestCost = float("inf")

        for target in remainingTargets:
            distance = min(
                abs(node[0] - target[0]) + abs(node[1] - target[1])
                for node in connected
            )
            if distance < cheapestCost:
                cheapestTarget = target
                cheapestCost = distance

        connected.add(cheapestTarget)
        remainingTargets.remove(cheapestTarget)
        totalCost += cheapestCost

    return totalCost
