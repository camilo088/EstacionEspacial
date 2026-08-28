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
    
    distCache = problem.heuristicInfo.setdefault("pairDist", {})

    def cachedDist(a, b):
        if a == b:
            return 0
        key = (a, b) if a < b else (b, a)
        d = distCache.get(key)
        if d is None:
            d = abs(a[0] - b[0]) + abs(a[1] - b[1])
            distCache[key] = d
        return d
    
    remaining = targets
    bestDist = [cachedDist(position, t) for t in remaining]
    totalCost = 0

    while remaining:
        bestIdx = 0
        for i in range(1, len(remaining)):
            if bestDist[i] < bestDist[bestIdx]:
                bestIdx = i

        totalCost += bestDist[bestIdx]
        newNode = remaining[bestIdx]
        remaining[bestIdx] = remaining[-1]
        remaining.pop()
        bestDist[bestIdx] = bestDist[-1]
        bestDist.pop()

        for i, t in enumerate(remaining):
            d = cachedDist(newNode, t)
            if d < bestDist[i]:
                bestDist[i] = d

    return totalCost