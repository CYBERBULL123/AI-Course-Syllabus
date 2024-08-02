### Week 0: Introduction

**1. History**
- **Definition**: The evolution of AI from early symbolic AI to modern machine learning and deep learning approaches.
- **Example**: Early AI research focused on symbolic reasoning and expert systems, while modern AI emphasizes data-driven approaches like deep learning.
- **Uses**: Understanding AI's history helps contextualize current trends and future directions in the field.
- **Architecture**: Not applicable.
- **Codebase**: Not applicable.

**2. Can Machines Think?**
- **Definition**: A philosophical question posed by Alan Turing in his 1950 paper, questioning the nature of machine intelligence.
- **Example**: The Turing Test, which assesses a machine's ability to exhibit intelligent behavior equivalent to, or indistinguishable from, that of a human.
- **Uses**: Sparks discussions on the potential and limitations of AI, guiding research and ethical considerations.
- **Architecture**: Not applicable.
- **Codebase**: Not applicable.

**3. Turing Test**
- **Definition**: A test proposed by Alan Turing to determine if a machine's behavior is indistinguishable from that of a human.
- **Example**: A chatbot that can engage in a conversation without the user realizing it's a machine.
- **Uses**: Measures progress in natural language processing and AI's ability to mimic human behavior.
- **Architecture**: Typically involves a conversational AI system.
- **Codebase**:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# Let's chat for 5 lines
for step in range(5):
    # Take user input
    user_input = input("User: ")
    # Encode the new user input, add the eos_token and return a tensor in Pytorch
    new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

    # Append the new user input tokens to the chat history
    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids

    # Generate a response
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    # Convert the tokens to text, and then split the responses into lines
    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    print(f"Bot: {response}")
```

**4. Winograd Schema Challenge**
- **Definition**: An alternative to the Turing Test that involves resolving pronoun references in sentences.
- **Example**: "The city councilmen refused the demonstrators a permit because they feared violence." Who feared violence? The answer requires understanding context.
- **Uses**: Tests AI's natural language understanding and reasoning capabilities.
- **Architecture**: Natural Language Processing (NLP) systems with contextual understanding.
- **Codebase**:
```python
import spacy
from neuralcoref import Coref

nlp = spacy.load("en_core_web_sm")
coref = Coref(nlp=nlp)

sentence = "The city councilmen refused the demonstrators a permit because they feared violence."
doc = coref(sentence)
resolved = doc.get_resolved_utterances()
print(resolved)
```

**5. Language and Thought**
- **Definition**: The relationship between linguistic capabilities and cognitive processes.
- **Example**: How language influences perception, memory, and reasoning.
- **Uses**: Insights into human cognition and improving AI's natural language understanding.
- **Architecture**: NLP and Cognitive Science integration.
- **Codebase**:
```python
import spacy

nlp = spacy.load("en_core_web_sm")

text = "Language shapes the way we think."
doc = nlp(text)

for token in doc:
    print(token.text, token.pos_, token.dep_)
```

**6. Wheels & Gears**
- **Definition**: Basic mechanisms and concepts underlying AI systems.
- **Example**: Understanding algorithms, data structures, and basic AI principles.
- **Uses**: Foundation for building more complex AI systems.
- **Architecture**: Core AI principles and techniques.
- **Codebase**:
```python
# Simple example of a basic algorithm - Sorting
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr

array = [64, 34, 25, 12, 22, 11, 90]
sorted_array = bubble_sort(array)
print(sorted_array)
```

### Week 1: Introduction

**1. Philosophy**
- **Definition**: Philosophical foundations and implications of AI.
- **Example**: Questions about consciousness, free will, and the ethics of AI.
- **Uses**: Guides ethical AI development and deployment.
- **Architecture**: Not applicable.
- **Codebase**: Not applicable.

**2. Mind**
- **Definition**: Concepts of consciousness and intelligence.
- **Example**: Distinguishing between human-like and machine-like intelligence.
- **Uses**: Understanding and replicating cognitive processes in AI.
- **Architecture**: Cognitive architectures.
- **Codebase**: Not applicable.

**3. Reasoning**
- **Definition**: How machines can perform logical reasoning.
- **Example**: Deductive, inductive, and abductive reasoning in AI.
- **Uses**: Enhances decision-making capabilities in AI systems.
- **Architecture**: Logical reasoning systems.
- **Codebase**:
```python
from pyDatalog import pyDatalog

pyDatalog.create_terms('X, Y, Z, father, grandfather')

+father('john', 'doe')
+father('doe', 'jane')

grandfather(X, Y) <= father(X, Z) & father(Z, Y)

print(grandfather(X, Y))
```

**4. Computation**
- **Definition**: Basics of computational theory.
- **Example**: Understanding algorithms, complexity, and computational limits.
- **Uses**: Fundamental knowledge for AI development.
- **Architecture**: Computational models and algorithms.
- **Codebase**: Not applicable.

**5. Dartmouth Conference**
- **Definition**: The 1956 conference that marked the birth of AI as a field.
- **Example**: Proposals and discussions from early AI pioneers.
- **Uses**: Historical context for AI research.
- **Architecture**: Not applicable.
- **Codebase**: Not applicable.

**6. The Chess Saga**
- **Definition**: Milestones in AI's development through chess-playing programs.
- **Example**: IBM's Deep Blue defeating Garry Kasparov in 1997.
- **Uses**: Demonstrates advances in search algorithms and game theory.
- **Architecture**: Game-playing AI architectures.
- **Codebase**:
```python
import chess
import chess.engine

engine = chess.engine.SimpleEngine.popen_uci("/usr/games/stockfish")

board = chess.Board()
result = engine.play(board, chess.engine.Limit(time=2.0))
board.push(result.move)

print(board)
engine.quit()
```

**7. Epiphenomena**
- **Definition**: Secondary effects or by-products of AI systems.
- **Example**: Unintended consequences of AI decisions.
- **Uses**: Understanding and mitigating side effects in AI deployment.
- **Architecture**: Not applicable.
- **Codebase**: Not applicable.

### Week 2: State Space Search

**1. Depth First Search (DFS)**
- **Definition**: Explores as far as possible along each branch before backtracking.
- **Example**: Navigating a maze by following one path to its end, then backtracking.
- **Uses**: Solving problems with deep, but not necessarily wide, search spaces.
- **Architecture**: Tree or graph traversal algorithms.
- **Codebase**:
```python
def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()
    visited.add(start)
    print(start)
    for next in graph[start] - visited:
        dfs(graph, next, visited)
    return visited

graph = {'A': {'B', 'C'},
         'B': {'A', 'D', 'E'},
         'C': {'A', 'F'},
         'D': {'B'},
         'E': {'B', 'F'},
         'F': {'C', 'E'}}

dfs(graph, 'A')
```

**2. Breadth First Search (BFS)**
- **Definition**: Explores all nodes at the present depth before moving on to nodes at the next depth level.
- **Example**: Finding the shortest path in an unweighted graph.
- **Uses**: Solving problems with wide, but not necessarily deep, search spaces.
- **Architecture**: Tree or graph traversal algorithms.
- **Codebase**:
```python
from collections import deque

def bfs(graph, start):
    visited = set()
    queue = deque([start])
    visited.add(start)

    while queue:
        vertex = queue.popleft()
        print(vertex)
        for neighbor in graph[vertex]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

graph = {'A': ['B', 'C'],
         'B': ['A', 'D', 'E'],
         'C': ['A', 'F'],
         'D': ['B'],
         'E': ['B', 'F'],
         'F': ['C', 'E']}

bfs(graph, 'A')
```

**3. Depth First Iterative Deepening (DFID)**
- **Definition**: Combines the space efficiency of DFS with the completeness of BFS.
- **Example**: Search algorithm that repeatedly applies DFS with increasing depth limits.
- **Uses**: Finding solutions in problems with large search spaces without excessive memory usage.
- **Architecture**: Tree or graph traversal algorithms.
- **Codebase**:
```python
def dls(graph, node, depth, visited=None):
    if visited is None:
        visited = set()
    if depth == 0:
        return
    print(node)
    visited.add(node)
    for neighbor in graph[node]:
        if neighbor not in visited:
            dls(graph, neighbor, depth - 1, visited)

def dfid(graph, start, max_depth):
    for depth in range(max_depth):
        visited = set()
        dls(graph, start, depth, visited)

graph = {'A': ['B', 'C'],
         'B': ['A', 'D', 'E'],
         'C': ['A', 'F'],
         'D': ['B'],
         'E': ['B', 'F'],
         'F': ['C', 'E']}

dfid(graph, 'A', 3)
```

### Week 3: Heuristic Search

**1. Best First Search**
- **Definition**: Uses heuristics to guide the search process towards the goal.
- **Example**: A* search algorithm.
- **Uses**: Finding the shortest path in weighted graphs.
- **Architecture**: Graph traversal with priority queues.
- **Codebase**:
```python
import heapq

def best_first_search(graph, start, goal, h):
    open_list = []
    heapq.heappush(open_list, (h[start], start))
    closed_list = set()

    while open_list:
        _, current = heapq.heappop(open_list)
        if current == goal:
            return True
        closed_list.add(current)
        for neighbor, weight in graph[current]:
            if neighbor not in closed_list:
                heapq.heappush(open_list, (h[neighbor], neighbor))
    return False

graph = {
    'A': [('B', 1), ('C', 3)],
    'B': [('A', 1), ('D', 3), ('E', 1)],
    'C': [('A', 3), ('F', 5)],
    'D': [('B', 3)],
    'E': [('B', 1), ('F', 1)],
    'F': [('C', 5), ('E', 1)]
}

h = {'A': 5, 'B': 2, 'C': 4, 'D': 6, 'E': 1, 'F': 0}

best_first_search(graph, 'A', 'F', h)
```

**2. Hill Climbing**
- **Definition**: An iterative algorithm that starts with an arbitrary solution and makes small changes to improve it.
- **Example**: Optimizing a mathematical function.
- **Uses**: Local search problems where a global optimum is not required.
- **Architecture**: Simple iterative improvement.
- **Codebase**:
```python
import random

def hill_climbing(f, domain, steps=1000):
    current = random.choice(domain)
    for _ in range(steps):
        neighbors = [current + delta for delta in [-1, 1] if current + delta in domain]
        next_step = max(neighbors, key=f)
        if f(next_step) <= f(current):
            break
        current = next_step
    return current

def objective_function(x):
    return -x**2 + 4

domain = list(range(-10, 11))

best_solution = hill_climbing(objective_function, domain)
print(f"Best solution: {best_solution}, Objective value: {objective_function(best_solution)}")
```

**3. Solution Space**
- **Definition**: The domain of all possible solutions to a problem.
- **Example**: All possible assignments of values to variables in a constraint satisfaction problem.
- **Uses**: Identifying and exploring potential solutions.
- **Architecture**: Problem-specific structures.
- **Codebase**: Not directly applicable; domain-specific.

**4. Traveling Salesman Problem (TSP)**
- **Definition**: Finding the shortest possible route visiting a set of nodes and returning to the origin node.
- **Example**: An optimal route for a salesperson to visit multiple cities.
- **Uses**: Logistics, route planning, and optimization.
- **Architecture**: Combinatorial optimization problem.
- **Codebase**:
```python
import itertools

def calculate_distance(route, distances):
    return sum(distances[route[i]][route[i + 1]] for i in range(len(route) - 1)) + distances[route[-1]][route[0]]

def tsp_brute_force(cities, distances):
    shortest_route = None
    min_distance = float('inf')
    for perm in itertools.permutations(cities):
        current_distance = calculate_distance(perm, distances)
        if current_distance < min_distance:
            min_distance = current_distance
            shortest_route = perm
    return shortest_route, min_distance

cities = ['A', 'B', 'C', 'D']
distances = {
    'A': {'A': 0, 'B': 10, 'C': 15, 'D': 20},
    'B': {'A': 10, 'B': 0, 'C': 35, 'D': 25},
    'C': {'A': 15, 'B': 35, 'C': 0, 'D': 30},
    'D': {'A': 20, 'B': 25, 'C': 30, 'D': 0}
}

shortest_route, min_distance = tsp_brute_force(cities, distances)
print(f"Shortest route: {shortest_route}, Distance: {min_distance}")
```

**5. Escaping Local Optima**
- **Definition**: Techniques to avoid being trapped in local optima during optimization.
- **Example**: Simulated annealing.
- **Uses**: Global optimization problems.
- **Architecture**: Stochastic optimization methods.
- **Codebase**:
```python
import math
import random

def simulated_annealing(f, domain, temp, cooling_rate=0.99, steps=1000):
    current = random.choice(domain)
    for _ in range(steps):
        next_step = random.choice(domain)
        delta_f = f(next_step) - f(current)
        if delta_f > 0 or math.exp(delta_f / temp) > random.random():
            current = next_step
        temp *= cooling_rate
    return current

def objective_function(x):
    return -x**2 + 4

domain = list(range(-10, 11))

best_solution = simulated_annealing(objective_function, domain, temp=100)
print(f"Best solution: {best_solution}, Objective value: {objective_function(best_solution)}")
```

**6. Stochastic Local Search**
- **Definition**: Uses randomness to escape local optima and explore the solution space.
- **Example**: Random restart hill climbing.
- **Uses**: Optimization problems where deterministic methods fail.
- **Architecture**: Stochastic methods.
- **Codebase**:
```python
def random_restart_hill_climbing(f, domain, restarts=10, steps=1000):
    best_solution = None
    best_value = float('-inf')
    for _ in range(restarts):
        current = hill_climbing(f, domain, steps)
        current_value = f(current)
        if current_value > best_value:
            best_value = current_value
            best_solution = current
    return best_solution

best_solution = random_restart_hill_climbing(objective_function, domain)
print(f"Best solution: {best_solution}, Objective value: {objective_function(best_solution)}")
```

### Week 4: Population-Based Methods

**1. Genetic Algorithms**
- **Definition**: Optimization algorithms inspired by the process of natural selection.
- **Example**: Evolving solutions to optimization problems.
- **Uses**: Complex optimization problems, machine learning model tuning.
- **Architecture**: Population of candidate solutions, selection, crossover, mutation.
- **Codebase**:
```python
import random

def genetic_algorithm(population, fitness_fn, mutate_fn, crossover_fn, generations=100):
    for generation in range(generations):
        population = sorted(population, key=fitness_fn, reverse=True)
        next_generation = population[:2]  # Keep top 2
        for _ in range(len(population) // 2 - 1):
            parents = random.sample(population[:10], 2)
            child = crossover_fn(*parents)
            if random.random() < 0.1:
                child = mutate_fn(child)
            next_generation.append(child)
        population = next_generation
    return sorted(population, key=fitness_fn, reverse=True)[0]

def fitness_fn(x):
    return -x**2 + 4

def mutate_fn(x):
    return x + random.randint(-1, 1)

def crossover_fn(parent1, parent2):
    return (parent1 + parent2) // 2

population = [random.randint(-10, 10) for _ in range(10)]
best_solution = genetic_algorithm(population, fitness_fn, mutate_fn, crossover_fn)
print(f"Best solution: {best_solution}, Fitness value: {fitness_fn(best_solution)}")
```

**2. SAT (Boolean Satisfiability Problem)**
- **Definition**: Determining if there exists an interpretation that satisfies a given Boolean formula.
- **Example**: Finding variable assignments that make a complex logical formula true.
- **Uses**: Hardware verification, software testing, AI planning.
- **Architecture**: SAT solvers using algorithms like DPLL or CDCL.
- **Codebase**:
```python
from pysat.solvers import Glucose3

# Example CNF: (A or B) and (not A or C) and (not B or not C)
solver = Glucose3()
solver.add_clause([1, 2])
solver.add_clause([-1, 3])
solver.add_clause([-2, -3])

is_sat = solver.solve()
solution = solver.get_model()

print(f"Satisfiable: {is_sat}, Solution: {solution}")
```

**3. Traveling Salesman Problem (TSP)**
- **Definition**: Already defined in Week 3.
- **Example**: Already defined in Week 3.
- **Uses**: Already defined in Week 3.
- **Architecture**: Already defined in Week 3.
- **Codebase**: Already provided in Week 3.

**4. Emergent Systems**
- **Definition**: Systems in which complex behaviors emerge from simple rules.
- **Example**: Flocking behavior of birds or traffic flow.
- **Uses**: Modeling natural systems, AI simulations.
- **Architecture**: Agent-based models.
- **Codebase**:
```python
import matplotlib.pyplot as plt
import numpy as np

class Boid:
    def __init__(self, position, velocity):
        self.position = position
        self.velocity = velocity

    def update(self, boids):
        self.velocity += self.flock(boids)
        self.position += self.velocity

    def flock(self, boids):
        # Simple rules for alignment, cohesion, and separation
        alignment = np.mean([boid.velocity for boid in boids], axis=0) - self.velocity
        cohesion = np.mean([boid.position for boid in boids], axis=0) - self.position
        separation = np.sum([self.position - boid.position for boid in boids], axis=0)
        return alignment + cohesion + separation

boids = [Boid(np.random.rand(2) * 100, np.random.rand(2) * 10 - 5) for _ in range(30)]

for _ in range(100):
    plt.clf()
    for boid in boids:
        boid.update(boids)
        plt.scatter(boid.position[0], boid.position[1])
    plt.pause(0.1)
```

**5. Ant Colony Optimization**
- **Definition**: Optimization technique inspired by the behavior of ants finding paths to food.
- **Example**: Finding the shortest path in a graph.
- **Uses**: Network routing, scheduling, TSP.
- **Architecture**: Pheromone trails and heuristic information.
- **Codebase**:
```python
import numpy as np

def ant_colony_optimization(distances, num_ants=10, num_iterations=100, alpha=1, beta=5, evaporation_rate=0.5):
    num_nodes = len(distances)
    pheromones = np.ones((num_nodes, num_nodes)) / num_nodes
    best_solution = None
    best_length = float('inf')

    def calculate_path_length(path):
        return sum(distances[path[i]][path[i + 1]] for i in range(num_nodes - 1)) + distances[path[-1]][path[0]]

    for _ in range(num_iterations):
        all_paths = []
        for _ in range(num_ants):
            path = [np.random.randint(num_nodes)]
            while len(path) < num_nodes:
                i = path[-1]
                probabilities = [
                    (pheromones[i][j] ** alpha) * ((1.0 / distances[i][j]) ** beta)
                    for j in range(num_nodes) if j not in path
                ]
                next_node = np.random.choice([j for j in range(num_nodes) if j not in path], p=probabilities / np.sum(probabilities))
                path.append(next_node)
            all_paths.append((path, calculate_path_length(path)))

        all_paths.sort(key=lambda x: x[1])
        if all_paths[0][1] < best_length:
            best_solution, best_length = all_paths[0]

        for path, length in all_paths:
            for i in range(num_nodes):
                pheromones[path[i]][path[(i + 1) % num_nodes]] *= (1 - evaporation_rate)
                pheromones[path[i]][path[(i + 1) % num_nodes]] += 1.0 / length

    return best_solution, best_length

distances = np.array([
    [0, 2, 2, 3, 1],
    [2, 0, 4, 4, 2],
    [2, 4, 0, 3, 3],
    [3, 4, 3, 0, 5],
    [1, 2, 3, 5, 0]
])

best_solution, best_length = ant_colony_optimization(distances)
print(f"Best solution: {best_solution}, Distance: {best_length}")
```

### Week 5: Finding Optimal Paths

**1. Branch & Bound**
- **Definition**: A search algorithm that systematically enumerates candidate solutions by branching, pruning suboptimal solutions.
- **Example**: Solving combinatorial optimization problems.
- **Uses**: Optimization problems, including TSP and knapsack problem.
- **Architecture**: Tree search with bounds.
- **Codebase**:
```python
def branch_and_bound(values, weights, capacity):
    n = len(values)
    best_value = 0
    best_combination = []
    def dfs(i, current_value, current_weight, combination):
        nonlocal best_value, best_combination
        if current_weight > capacity:
            return
        if i == n:
            if current_value > best_value:
                best_value = current_value
                best_combination = combination
            return
        dfs(i + 1, current_value + values[i], current_weight + weights[i], combination + [i])
        dfs(i + 1, current_value, current_weight, combination)
    dfs(0, 0, 0, [])
    return best_value, best_combination

values = [60, 100, 120]
weights = [10, 20, 30]
capacity = 50
max_value, items = branch_and_bound(values, weights, capacity)
print(f"Max value: {max_value}, Items: {items}")
```

**2. A\***
- **Definition**: A search algorithm that finds the shortest path to the goal by combining the cost to reach a node and the estimated cost to reach the goal.
- **Example**: Pathfinding in games.
- **Uses**: Pathfinding and graph traversal.
- **Architecture**: Graph search with priority queue.
- **Codebase**:
```python
import heapq

def a_star_search(graph, start, goal, h):
    open_list = []
    heapq.heappush(open_list, (h[start], start))
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
            return path[::-1]
        for neighbor, cost in graph[current]:
            tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score = tentative_g_score + h[neighbor]
                heapq.heappush(open_list, (f_score, neighbor))

    return None

graph = {
    'A': [('B', 1), ('C', 3)],
    'B': [('A', 1), ('D', 3), ('E', 1)],
    'C': [('A', 3), ('F', 5)],
    'D': [('B', 3)],
    'E': [('B', 1), ('F', 1)],
    'F': [('C', 5), ('E', 1)]
}

h = {'A': 5, 'B': 2, 'C': 4, 'D': 6, 'E': 1, 'F': 0}

path = a_star_search(graph, 'A', 'F', h)
print(f"Path: {path}")
```

### Week 6: Modeling & Constraints

**1. Modeling Constraint Problems**
- **Definition**: Defining a problem using variables, domains, and constraints.
- **Example**: Sudoku puzzle.
- **Uses**: Scheduling, resource allocation, planning.
- **Architecture**: Constraint satisfaction problems (CSPs).
- **Codebase**:
```python
from ortools.sat.python import cp_model

model = cp_model.CpModel()
n = 9
cells = [[model.NewIntVar(1, n, f'cell{i}{j}') for j in range(n)] for i in range(n)]

# Example constraint: cells[0][0] + cells[1][1] == cells[2][2]
model.Add(cells[0][0] + cells[1][1] == cells[2][2])

solver = cp_model.CpSolver()
status = solver.Solve(model)

if status == cp_model.FEASIBLE:
    for i in range(n):
        for j in range(n):
            print(solver.Value(cells[i][j]), end=' ')
        print()
```

**2. Constraint Propagation**
- **Definition**: Reducing the search space by iteratively enforcing constraints.
- **Example**: Arc consistency in CSPs.
- **Uses**: CSPs like scheduling, planning, and resource allocation.
- **Architecture**: Constraint networks.
- **Codebase**:
```python
from ortools.sat.python import cp_model

model = cp_model.CpModel()
x = model.NewIntVar(0, 10, 'x')
y = model.NewIntVar(0, 10, 'y')

model.Add(x + y <= 10)
model.Add(x - y >= 2)

solver = cp_model.CpSolver()
status = solver.Solve(model)

if status == cp_model.FEASIBLE:
    print(f"x = {solver.Value(x)}, y = {solver.Value(y)}")
```

**3. Conflict-Driven Clause Learning (CDCL)**
- **Definition**: A SAT solver technique that learns from conflicts to avoid similar conflicts in the future.
- **Example**: Modern SAT solvers.
- **Uses**: Efficiently solving SAT problems.
- **Architecture**: CDCL-based SAT solvers.
- **Codebase**:
```python
from pysat.solvers import Glucose3

solver = Glucose3()
solver.add_clause([1, -2])
solver.add_clause([-1, 2])
solver.add_clause([1, 2, -3])
solver.add_clause([-1, -2, 3])

is_sat = solver.solve()
solution = solver.get_model()

print(f"Satisfiable: {is_sat}, Solution: {solution}")
```

### Week 7: Learning

**1. Learning Constraint Problems**
- **Definition**: Using machine learning to improve constraint solving.
- **Example**: Learning variable ordering heuristics.
- **Uses**: Enhancing efficiency of solving CSPs.
- **Architecture**: Integration of ML with CSPs.
- **Codebase**:
```python
# Placeholder code for learning-enhanced CSP solving
import random

class CSP:
    def __init__(self, variables, domains, constraints):
        self.variables = variables
        self.domains = domains
        self.constraints = constraints

    def is_consistent(self, var, value, assignment):
        for constraint in self.constraints:
            if not constraint(var, value, assignment):
                return False
        return True

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
constraints = [lambda A, a, assignment: assignment.get('B', a) != a,
               lambda B, b, assignment: assignment.get('A', b) != b,
               lambda C, c, assignment: assignment.get('A', c) != c]

csp = CSP(variables, domains, constraints)
solution = backtracking_search(csp)
print(f"Solution: {solution}")
```

**2. Active Constraint Acquisition**
- **Definition**: Actively learning constraints through queries.
- **Example**: Interactive scheduling systems.
- **Uses**: Dynamic environments requiring adaptive constraint handling.
- **Architecture**: User-interactive constraint learning.
- **Codebase**: Not directly applicable; domain-specific.

### Week 8: Algorithms for Constraints

**1. Backtracking Search**
- **Definition**: A depth-first search algorithm for CSPs that incrementally builds candidates.
- **Example**: Solving a Sudoku puzzle.
- **Uses**: Constraint satisfaction problems.
- **Architecture**: Recursive search with pruning.
- **Codebase**:
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
constraints = [lambda A, a, assignment: assignment.get('B', a) != a,
               lambda B, b, assignment: assignment.get('A', b) != b,
               lambda C, c, assignment: assignment.get('A', c) != c]

csp = CSP(variables, domains, constraints)
solution = backtracking_search(csp)
print(f"Solution: {solution}")
```

**2. Arc Consistency**
- **Definition**: A variable in a CSP is arc-consistent if every value in its domain satisfies the variable's binary constraints.
- **Example**: Enforcing consistency in Sudoku.
- **Uses**: Preprocessing step to reduce domains in CSPs.
- **Architecture**: Constraint propagation.
- **Codebase**:
```python
def arc_consistency(csp):
    queue = [(xi, xj) for xi in csp.variables for xj in csp.variables if xi != xj]
    while queue:
        (xi, xj) = queue.pop(0)
        if revise(csp, xi, xj):
            if len(csp.domains[xi]) == 0:
                return False
            for xk in csp.variables:
                if xk != xi and xk != xj:
                    queue.append((xk, xi))
    return True

def revise(csp, xi, xj):
    revised = False
    for x in csp.domains[xi]:
        if all(not csp.is_consistent(xi, x, {xj: y}) for y in csp.domains[xj]):
            csp.domains[xi].remove(x)
            revised = True
    return revised

variables = ['A', 'B', 'C']
domains = {'A': [1, 2, 3], 'B': [1, 2, 3], 'C': [1, 2, 3]}
constraints = [lambda A, a, assignment: assignment.get('B', a) != a,
               lambda B, b, assignment: assignment.get('A', b) != b,
               lambda C, c, assignment: assignment.get('A', c) != c]

csp = CSP(variables, domains, constraints)
arc_consistency(csp)
print(f"Domains after arc consistency: {csp.domains}")
```

### Week 9: Case Study

**1. Constraint Satisfaction Application**
- **Definition**: Applying CSP techniques to a real-world problem, such as scheduling or resource allocation.
- **Example**: Employee shift scheduling.
- **Uses**: Any domain requiring optimal assignment under constraints.
- **Architecture**: Define variables, domains, constraints, and use CSP algorithms to find solutions.
- **Codebase**:
```python
from ortools.sat.python import cp_model

model = cp_model.CpModel()
num_employees = 5
num_shifts = 3
num_days = 7

shifts = {}
for e in range(num_employees):
    for d in range(num_days):
        for s in range(num_shifts):
            shifts[(e, d, s)] = model.NewBoolVar(f'shift_e{e}_d{d}_s{s}')

# Constraints: Each shift is assigned to exactly one employee per day
for d in range(num_days):
    for s in range(num_shifts):
        model.AddExactlyOne(shifts[(e, d, s)] for e in range(num_employees))

# Constraints: Each employee works at most one shift per day
for e in range(num_employees):
    for d in range(num_days):
        model.AddAtMostOne(shifts[(e, d, s)] for s in range(num_shifts))

solver = cp_model.CpSolver()
status = solver.Solve(model)

if status == cp_model.FEASIBLE:
    for d in range(num_days):
        print(f"Day {d + 1}")
        for s in range(num_shifts):
            for e in range(num_employees):
                if solver.Value(shifts[(e, d, s)]):
                    print(f"  Shift {s}: Employee {e}")
```

**2. Vehicle Routing Problem (VRP)**
- **Definition**: A combinatorial optimization problem which seeks to service a number of customers with a fleet of vehicles.
- **Example**: Optimizing delivery routes for a logistics company.
- **Uses**: Transportation, logistics, and supply chain management.
- **Architecture**: Graph-based model, heuristic or metaheuristic optimization.
- **Codebase**:
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

### Week 10: Advanced Topics and Conclusion

**1. Hybrid Methods**
- **Definition**: Combining different algorithms or techniques to solve CSPs more efficiently.
- **Example**: Combining SAT solvers with local search.
- **Uses**: Complex CSPs where a single method is insufficient.
- **Architecture**: Integration of multiple solving techniques.
- **Codebase**:
```python
# Example: Combining genetic algorithms with SAT solving

# Simplified genetic algorithm initialization
def genetic_algorithm(csp, population_size=50, generations=100):
    population = [random_individual(csp) for _ in range(population_size)]
    for _ in range(generations):
        population = evolve_population(csp, population)
    return best_individual(csp, population)

def random_individual(csp):
    return {var: random.choice(csp.domains[var]) for var in csp.variables}

def evolve_population(csp, population):
    # Selection, crossover, and mutation operations
    new_population = []
    for _ in range(len(population)):
        parent1, parent2 = select_parents(csp, population)
        child = crossover(csp, parent1, parent2)
        new_population.append(mutate(csp, child))
    return new_population

def select_parents(csp, population):
    # Selection strategy (e.g., tournament selection)
    return random.sample(population, 2)

def crossover(csp, parent1, parent2):
    # Single-point crossover
    child = {}
    crossover_point = random.randint(0, len(csp.variables) - 1)
    for i, var in enumerate(csp.variables):
        if i < crossover_point:
            child[var] = parent1[var]
        else:
            child[var] = parent2[var]
    return child

def mutate(csp, individual):
    # Mutation operation
    var = random.choice(csp.variables)
    individual[var] = random.choice(csp.domains[var])
    return individual

def best_individual(csp, population):
    # Fitness evaluation (e.g., based on constraint satisfaction)
    return max(population, key=lambda ind: fitness(csp, ind))

def fitness(csp, individual):
    return sum(1 for constraint in csp.constraints if constraint(individual))

# Sample CSP definition
variables = ['A', 'B', 'C']
domains = {'A': [1, 2, 3], 'B': [1, 2, 3], 'C': [1, 2, 3]}
constraints = [lambda v: v['A'] != v['B'], lambda v: v['B'] != v['C']]

csp = CSP(variables, domains, constraints)
solution = genetic_algorithm(csp)
print(f"Solution: {solution}")
```

**2. Future Trends**
- **Definition**: Emerging research directions and applications in constraint solving and optimization.
- **Example**: Quantum computing for CSPs.
- **Uses**: Expanding the capabilities and efficiency of solving complex problems.
- **Architecture**: Integration with cutting-edge technologies.
- **Codebase**: Depends on the specific trend or technology (e.g., quantum computing libraries).

### Conclusion
The course has covered fundamental and advanced topics in AI for optimization, constraints, and search techniques. Through hands-on examples and practical applications, you've gained a solid foundation in solving real-world problems using AI techniques.

---

This detailed 10-week course outline, including architectures and codebases, should provide a comprehensive understanding of key AI concepts and practical skills for solving complex optimization and constraint satisfaction problems. If you have any specific questions or need further details on any topic, feel free to ask!
   
