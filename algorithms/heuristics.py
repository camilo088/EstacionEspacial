from collections import deque
from math import hypot
from typing import Tuple
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
    isPosition = (
        isinstance(state, tuple)
        and len(state) == 2
        and all(isinstance(value, int) for value in state)
    )
    position = state if isPosition else state[0]

    if hasattr(problem, "kitPosition") and hasattr(problem, "controlPosition"):
        if len(state) == 3:
            _, hasKit, pendingSystems = state
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
        else:
            hasKit = state[1]
            target = problem.controlPosition if hasKit else problem.kitPosition
    elif hasattr(problem, "modulePosition"):
        target = problem.controlPosition if state[1] else problem.modulePosition
    else:
        target = problem.goal

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
    isPosition = (
        isinstance(state, tuple)
        and len(state) == 2
        and all(isinstance(value, int) for value in state)
    )
    position = state if isPosition else state[0]

    if hasattr(problem, "kitPosition") and hasattr(problem, "controlPosition"):
        if len(state) == 3:
            _, hasKit, pendingSystems = state
            if not hasKit:
                target = problem.kitPosition
            elif pendingSystems:
                target = min(
                    pendingSystems,
                    key=lambda system: hypot(
                        position[0] - system[0], position[1] - system[1]
                    ),
                )
            else:
                target = problem.controlPosition
        else:
            hasKit = state[1]
            target = problem.controlPosition if hasKit else problem.kitPosition
    elif hasattr(problem, "modulePosition"):
        target = problem.controlPosition if state[1] else problem.modulePosition
    else:
        target = problem.goal

    return hypot(position[0] - target[0], position[1] - target[1])


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
    controlPosition = problem.controlPosition

    distanceCache = problem.heuristicInfo.setdefault("distances", {})

    def distance(start, goal):
        key = (start, goal)
        if key in distanceCache:
            return distanceCache[key]

        frontier = deque([(start, 0)])
        visited = {start}
        while frontier:
            current, cost = frontier.popleft()
            if current == goal:
                distanceCache[key] = cost
                distanceCache[(goal, start)] = cost
                return cost

            x, y = current
            for nextPosition in (
                (x + 1, y),
                (x - 1, y),
                (x, y + 1),
                (x, y - 1),
            ):
                if not problem.walls[nextPosition[0]][nextPosition[1]] and nextPosition not in visited:
                    visited.add(nextPosition)
                    frontier.append((nextPosition, cost + 1))

        return 0

    def minimumSpanningTree(points):
        points = tuple(dict.fromkeys(points))
        if len(points) < 2:
            return 0

        tree = {points[0]}
        total = 0
        while len(tree) < len(points):
            edgeCost, nextPoint = min(
                (
                    (distance(point, candidate), candidate)
                    for point in tree
                    for candidate in points
                    if candidate not in tree
                ),
                key=lambda edge: edge[0],
            )
            total += edgeCost
            tree.add(nextPoint)
        return total

    if not hasKit:
        mandatoryPoints = (problem.kitPosition,) + pendingSystems + (controlPosition,)
        return distance(position, problem.kitPosition) + minimumSpanningTree(mandatoryPoints)

    mandatoryPoints = (position,) + pendingSystems + (controlPosition,)
    return minimumSpanningTree(mandatoryPoints)
