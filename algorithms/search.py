from algorithms.problems import SearchProblem
import algorithms.utils as utils
from world.game import Directions
from algorithms.heuristics import nullHeuristic


def tinyDiagnosticSearch(problem: SearchProblem):
    """
    Returns a hard-coded sequence of moves for the tinyDiagnostic layout.
    For any other station layout, the sequence of moves will be incorrect.
    """
    s = Directions.SOUTH
    e = Directions.EAST
    return [s, e, s, e, e, e, e, s, e, e, s, s, e, s, s, e, s, e, e, e, e, e, e, e]


def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    startState = problem.getStartState()
    frontier = utils.Stack()
    frontier.push((startState, []))
    visited = {startState}

    while not frontier.isEmpty():
        state, actions = frontier.pop()

        if problem.isGoalState(state):
            return actions

        for successor, action, stepCost in problem.getSuccessors(state):
            if successor not in visited:
                visited.add(successor)
                frontier.push((successor, actions + [action]))

    return []


def breadthFirstSearch(problem: SearchProblem):
    """
    Search the shallowest nodes in the search tree first.
    """
    startState = problem.getStartState()
    frontier = utils.Queue()
    frontier.push((startState, []))
    visited = {startState}

    while not frontier.isEmpty():
        state, actions = frontier.pop()

        if problem.isGoalState(state):
            return actions

        for successor, action, stepCost in problem.getSuccessors(state):
            if successor not in visited:
                visited.add(successor)
                frontier.push((successor, actions + [action]))

    return []



def uniformCostSearch(problem: SearchProblem):
    """
    Search the node of least total cost first.
    """

    startState = problem.getStartState()
    frontier = utils.PriorityQueue()
    frontier.push((startState, [], 0), 0)
    bestCosts = {startState: 0}

    while not frontier.isEmpty():
        state, actions, cost = frontier.pop()

        if cost != bestCosts[state]:
            continue

        if problem.isGoalState(state):
            return actions

        for successor, action, stepCost in problem.getSuccessors(state):
            newCost = cost + stepCost
            if successor not in bestCosts or newCost < bestCosts[successor]:
                bestCosts[successor] = newCost
                frontier.push((successor, actions + [action], newCost), newCost)

    return []


def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """
    Search the node that has the lowest combined cost and heuristic first.
    """
    startState = problem.getStartState()
    startCost = 0
    frontier = utils.PriorityQueue()
    frontier.push((startState, [], startCost), heuristic(startState, problem))
    bestCosts = {startState: startCost}

    while not frontier.isEmpty():
        state, actions, cost = frontier.pop()

        if cost != bestCosts[state]:
            continue

        if problem.isGoalState(state):
            return actions

        for successor, action, stepCost in problem.getSuccessors(state):
            newCost = cost + stepCost
            if successor not in bestCosts or newCost < bestCosts[successor]:
                bestCosts[successor] = newCost
                priority = newCost + heuristic(successor, problem)
                frontier.push((successor, actions + [action], newCost), priority)

    return []


# Abbreviations (you can use them for the -f option in main.py)
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
