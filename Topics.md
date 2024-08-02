# Artificial Intelligence Course: Optimizing, Constraints, and Search Techniques 🧠✨

---

## Week 0: Introduction 🚀

### Topics:
- History of AI
- Can Machines Think?
- Turing Test
- Winograd Schema Challenge
- Language and Thought
- Wheels & Gears

---

## Week 1: Philosophy and Foundations 📚

### Topics:
- Philosophy of AI
- Mind and Reasoning
- Computation
- Dartmouth Conference
- The Chess Saga
- Epiphenomena

---

## Week 2: State Space Search 🔍

### Topics:
- Depth First Search (DFS)
- Breadth First Search (BFS)
- Depth First Iterative Deepening (DFID)

### Example: Depth First Search
```python
def dfs(graph, start):
    visited, stack = set(), [start]
    while stack:
        vertex = stack.pop()
        if vertex not in visited:
            visited.add(vertex)
            stack.extend(set(graph[vertex]) - visited)
    return visited

graph = {'A': ['B', 'C'], 'B': ['A', 'D', 'E'], 'C': ['A', 'F'], 'D': ['B'], 'E': ['B', 'F'], 'F': ['C', 'E']}
print(dfs(graph, 'A'))
```

---

## Week 3: Heuristic Search 🌟

### Topics:
- Best First Search
- Hill Climbing
- Solution Space
- Traveling Salesman Problem (TSP)
- Escaping Local Optima
- Stochastic Local Search

### Example: Hill Climbing
```python
def hill_climbing(problem, max_iterations=1000):
    current = problem.initial_state()
    for _ in range(max_iterations):
        neighbors = problem.neighbors(current)
        if not neighbors:
            break
        neighbor = max(neighbors, key=problem.value)
        if problem.value(neighbor) <= problem.value(current):
            break
        current = neighbor
    return current

# Define a sample problem with initial state and value function
class SimpleProblem:
    def initial_state(self):
        return 0
    def neighbors(self, state):
        return [state - 1, state + 1]
    def value(self, state):
        return -abs(state - 10)

problem = SimpleProblem()
solution = hill_climbing(problem)
print(f"Solution: {solution}")
```

---

## Week 4: Population Based Methods 🌐

### Topics:
- Genetic Algorithms
- SAT
- TSP
- Emergent Systems
- Ant Colony Optimization

### Example: Genetic Algorithm
```python
import random

def genetic_algorithm(population, fitness_fn, generations=1000):
    for _ in range(generations):
        population = sorted(population, key=fitness_fn, reverse=True)
        next_generation = population[:2]  # Elitism
        for _ in range(len(population) // 2 - 1):
            parent1, parent2 = random.sample(population[:10], 2)
            child1, child2 = crossover(parent1, parent2)
            next_generation.extend([mutate(child1), mutate(child2)])
        population = next_generation
    return max(population, key=fitness_fn)

def crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 1)
    return parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]

def mutate(individual, mutation_rate=0.01):
    return [gene if random.random() > mutation_rate else random.choice([0, 1]) for gene in individual]

# Example fitness function and initial population
fitness_fn = lambda x: sum(x)
population = [[random.choice([0, 1]) for _ in range(10)] for _ in range(20)]
solution = genetic_algorithm(population, fitness_fn)
print(f"Solution: {solution}")
```

---

## Week 5: Finding Optimal Paths 🛤️

### Topics:
- Branch & Bound
- A*
- Admissibility of A*
- Informed Heuristic Functions

### Example: A* Algorithm
```python
import heapq

def a_star_search(graph, start, goal, h):
    open_list = []
    heapq.heappush(open_list, (0, start))
    came_from = {}
    g_score = {node: float('inf') for node in graph}
    g_score[start] = 0

    while open_list:
        _, current = heapq.heappop(open_list)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path

        for neighbor, weight in graph[current]:
            tentative_g_score = g_score[current] + weight
            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score = tentative_g_score + h[neighbor]
                heapq.heappush(open_list, (f_score, neighbor))

    return None

graph = {'A': [('B', 1), ('C', 3)], 'B': [('A', 1), ('D', 3), ('E', 1)], 'C': [('A', 3), ('F', 5)], 'D': [('B', 3)], 'E': [('B', 1), ('F', 1)], 'F': [('C', 5), ('E', 1)]}
h = {'A': 5, 'B': 2, 'C': 4, 'D': 6, 'E': 1, 'F': 0}

path = a_star_search(graph, 'A', 'F', h)
print(f"Path: {path}")
```

---

## Week 6: Space-Saving Versions of A* 🔧

### Topics:
- Weighted A*
- IDA*
- RBFS
- Monotone Condition
- Sequence Alignment
- DCFS
- SMGS
- Beam Stack Search

---

## Week 7: Game Playing 🎮

### Topics:
- Game Theory
- Board Games and Game Trees
- Algorithm Minimax
- Alpha-Beta Pruning
- SSS*

### Example: Minimax Algorithm
```python
def minimax(node, depth, maximizingPlayer):
    if depth == 0 or is_terminal(node):
        return evaluate(node)

    if maximizingPlayer:
        maxEval = float('-inf')
        for child in get_children(node):
            eval = minimax(child, depth - 1, False)
            maxEval = max(maxEval, eval)
        return maxEval
    else:
        minEval = float('inf')
        for child in get_children(node):
            eval = minimax(child, depth - 1, True)
            minEval = min(minEval, eval)
        return minEval

# Placeholder functions for example purposes
def is_terminal(node): return False
def evaluate(node): return 0
def get_children(node): return []

# Example usage
root_node = {}
best_value = minimax(root_node, 3, True)
print(f"Best value: {best_value}")
```

---

## Week 8: Automated Planning 🗂️

### Topics:
- Domain Independent Planning
- Blocks World
- Forward & Backward Search
- Goal Stack Planning
- Plan Space Planning

---

## Week 9: Problem Decomposition 🔍

### Topics:
- Means-Ends Analysis
- Algorithm Graphplan
- Algorithm AO*

---

## Week 10: Rule-Based Expert Systems 🧩

### Topics:
- Production Systems
- Inference Engine
- Match-Resolve-Execute
- Rete Net

---

## Week 11: Deduction as Search 🔍

### Topics:
- Logic
- Soundness
- Completeness
- First Order Logic
- Forward Chaining
- Backward Chaining

---

## Week 12: Constraint Processing 🔧

### Topics:
- CSPs
- Consistency-Based Diagnosis
- Algorithm Backtracking
- Arc Consistency
- Algorithm Forward Checking

### Example: Backtracking Search
```python
def backtracking_search(csp):
    def backtrack(assignment):
        if len(assignment) == len(csp.variables):
            return assignment
        var = select_unassigned_variable(csp.variables, assignment)
        for value in csp.domains[var]:
            if csp.is_consistent(var, value, assignment):
                assignment[var] = value
                result = backtrack(assignment)
                if result:
                    return result
                assignment.pop(var)
        return None
    return backtrack({})

def select_unassigned_variable(variables, assignment):
    return random.choice([var for var in variables if var not in assignment])

variables = ['A', 'B', 'C']
domains = {'A': [1, 2, 3], 'B': [1, 2, 3], 'C': [1, 2, 3]}
constraints = [lambda A, a, assignment: assignment.get('B', a) != a, lambda B, b, assignment: assignment.get('A', b) != b, lambda C, c, assignment: assignment.get('A', c) != c]

csp = CSP(variables, domains, constraints)
solution = backtracking_search(csp)
print(f"Solution: {solution}")
```

---

## Week 13: Constraint Optimization 🛠️

### Topics:
- Arc Consistency
- Conflict-Driven Clause Learning (CDCL)
- Active Constraint Acquisition

### Example: Arc Consistency
```python
def arc_consistency

(csp):
    queue = [(xi, xj) for xi in csp.variables for xj in csp.neighbors[xi]]
    while queue:
        (xi, xj) = queue.pop(0)
        if revise(csp, xi, xj):
            if not csp.domains[xi]:
                return False
            for xk in csp.neighbors[xi]:
                if xk != xj:
                    queue.append((xk, xi))
    return True

def revise(csp, xi, xj):
    revised = False
    for x in csp.domains[xi]:
        if not any(csp.is_consistent(xi, x, {xj: y}) for y in csp.domains[xj]):
            csp.domains[xi].remove(x)
            revised = True
    return revised

class CSP:
    def __init__(self, variables, domains, constraints):
        self.variables = variables
        self.domains = domains
        self.constraints = constraints
        self.neighbors = {var: [v for v in variables if v != var] for var in variables}

    def is_consistent(self, var, value, assignment):
        assignment[var] = value
        for constraint in self.constraints:
            if not constraint(assignment):
                return False
        return True

variables = ['A', 'B', 'C']
domains = {'A': [1, 2, 3], 'B': [1, 2, 3], 'C': [1, 2, 3]}
constraints = [lambda a: a.get('A', 0) != a.get('B', 0), lambda a: a.get('B', 0) != a.get('C', 0)]

csp = CSP(variables, domains, constraints)
arc_consistency(csp)
print(f"Domains after arc consistency: {csp.domains}")
```

---

## Week 14: Real-World Applications 🌍

### Topics:
- Scheduling
- Resource Allocation
- Vehicle Routing Problem (VRP)
- Hybrid Methods
- Future Trends

### Example: Vehicle Routing Problem
```python
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

def create_data_model():
    data = {}
    data['distance_matrix'] = [
        [0, 2, 9, 10],
        [1, 0, 6, 4],
        [15, 7, 0, 8],
        [6, 3, 12, 0],
    ]
    data['num_vehicles'] = 1
    data['depot'] = 0
    return data

def print_solution(manager, routing, solution):
    print('Objective: {} miles'.format(solution.ObjectiveValue()))
    index = routing.Start(0)
    plan_output = 'Route for vehicle 0:\n'
    route_distance = 0
    while not routing.IsEnd(index):
        plan_output += ' {} ->'.format(manager.IndexToNode(index))
        previous_index = index
        index = solution.Value(routing.NextVar(index))
        route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
    plan_output += ' {}\n'.format(manager.IndexToNode(index))
    plan_output += 'Distance of the route: {}miles\n'.format(route_distance)
    print(plan_output)

def main():
    data = create_data_model()
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']), data['num_vehicles'], data['depot'])
    routing = pywrapcp.RoutingModel(manager)
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    solution = routing.SolveWithParameters(search_parameters)
    if solution:
        print_solution(manager, routing, solution)

if __name__ == '__main__':
    main()
```

---

**This comprehensive course equips you with foundational knowledge and practical skills in AI, covering essential topics like state space search, heuristic search, population-based methods, game playing, automated planning, problem decomposition, rule-based expert systems, deduction as search, and constraint processing. By the end of this course, you'll be adept at applying these techniques to solve real-world problems effectively. Happy learning! 🎓📘**
