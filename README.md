### Week 0: Introduction
**History, Can Machines Think?, Turing Test, Winograd Schema Challenge, Language and Thought, Wheels & Gears**
- **History**: Overview of AI's evolution from early computing to modern AI.
- **Can Machines Think?**: Philosophical question raised by Alan Turing in his 1950 paper.
- **Turing Test**: A test proposed by Turing to determine if a machine exhibits intelligent behavior indistinguishable from a human.
- **Winograd Schema Challenge**: An alternative to the Turing Test focusing on understanding and reasoning.
- **Language and Thought**: Connection between linguistic capabilities and cognitive processes.
- **Wheels & Gears**: Basic mechanisms and concepts underlying AI systems.

### Week 1: Introduction
**Philosophy, Mind, Reasoning, Computation, Dartmouth Conference, The Chess Saga, Epiphenomena**
- **Philosophy**: Philosophical foundations and implications of AI.
- **Mind**: Concepts of consciousness and intelligence.
- **Reasoning**: How machines can perform logical reasoning.
- **Computation**: Basics of computational theory.
- **Dartmouth Conference**: The 1956 conference that marked the birth of AI as a field.
- **The Chess Saga**: Milestones in AI's development through chess-playing programs.
- **Epiphenomena**: Secondary effects or by-products of AI systems.

### Week 2: State Space Search
**Depth First Search, Breadth First Search, Depth First Iterative Deepening**
- **Depth First Search (DFS)**: Explores as far as possible along each branch before backtracking.
- **Breadth First Search (BFS)**: Explores all nodes at the present depth before moving on to nodes at the next depth level.
- **Depth First Iterative Deepening (DFID)**: Combines DFS's space efficiency and BFS's optimality.

### Week 3: Heuristic Search
**Best First Search, Hill Climbing, Solution Space, TSP, Escaping Local Optima, Stochastic Local Search**
- **Best First Search**: Uses heuristics to guide search.
- **Hill Climbing**: An iterative algorithm that starts with an arbitrary solution and makes small changes to improve it.
- **Solution Space**: The domain of all possible solutions.
- **TSP (Traveling Salesman Problem)**: Finding the shortest possible route visiting a set of nodes and returning to the origin node.
- **Escaping Local Optima**: Techniques like simulated annealing to avoid being trapped in local optima.
- **Stochastic Local Search**: Uses randomness to escape local optima.

### Week 4: Population-Based Methods
**Genetic Algorithms, SAT, TSP, Emergent Systems, Ant Colony Optimization**
- **Genetic Algorithms (GA)**: Optimization algorithms inspired by the process of natural selection.
- **SAT (Boolean Satisfiability Problem)**: Problem of determining if there is a truth assignment that satisfies a given Boolean formula.
- **TSP**: Already defined.
- **Emergent Systems**: Systems where complex behavior emerges from simple rules.
- **Ant Colony Optimization (ACO)**: Optimization algorithm inspired by the behavior of ants finding paths to food.

### Week 5: Finding Optimal Paths
**Branch & Bound, A*, Admissibility of A*, Informed Heuristic Functions**
- **Branch & Bound**: Algorithm design paradigm for solving combinatorial optimization problems.
- **A\***: Widely used pathfinding and graph traversal algorithm.
- **Admissibility of A\***: Ensures that A\* finds the least-cost path.
- **Informed Heuristic Functions**: Heuristics that provide estimates to guide the search process efficiently.

### Week 6: Space Saving Versions of A*
**Weighted A\*, IDA\*, RBFS, Monotone Condition, Sequence Alignment, DCFS, SMGS, Beam Stack Search**
- **Weighted A\***: Variation of A\* that uses a weight to speed up the search.
- **IDA\***: Iterative Deepening A\* combines the benefits of DFS and BFS.
- **RBFS (Recursive Best First Search)**: Memory-efficient version of A\*.
- **Monotone Condition**: Ensures consistent heuristics.
- **Sequence Alignment**: Technique to identify regions of similarity in sequences.
- **DCFS, SMGS, Beam Stack Search**: Advanced search algorithms (specific details depend on context).

### Week 7: Game Playing
**Game Theory, Board Games and Game Trees, Algorithm Minimax, AlphaBeta and SSS***
- **Game Theory**: Study of mathematical models of strategic interaction.
- **Board Games and Game Trees**: Representation of possible moves in games.
- **Algorithm Minimax**: Decision rule for minimizing the possible loss.
- **AlphaBeta Pruning**: Reduces the number of nodes evaluated in the minimax algorithm.
- **SSS***: Further optimization of game tree search.

### Week 8: Automated Planning
**Domain Independent Planning, Blocks World, Forward & Backward Search, Goal Stack Planning, Plan Space Planning**
- **Domain Independent Planning**: Creating plans that apply to various domains.
- **Blocks World**: Simplified domain used in AI planning.
- **Forward & Backward Search**: Approaches to planning.
- **Goal Stack Planning**: A form of planning using goal decomposition.
- **Plan Space Planning**: Involves searching through the space of possible plans.

### Week 9: Problem Decomposition
**Means Ends Analysis, Algorithm Graphplan, Algorithm AO***
- **Means Ends Analysis**: Problem-solving technique that involves breaking down the difference between the current state and the goal state.
- **Algorithm Graphplan**: Planning algorithm that uses a graph-based representation.
- **Algorithm AO***: Algorithm for AND-OR graph search.

### Week 10: Rule Based Expert Systems
**Production Systems, Inference Engine, Match-Resolve-Execute, Rete Net**
- **Production Systems**: Set of rules and a database of facts.
- **Inference Engine**: Applies logical rules to the knowledge base.
- **Match-Resolve-Execute**: Process used in rule-based systems.
- **Rete Net**: Efficient algorithm for matching patterns in production systems.

### Week 11: Deduction as Search
**Logic, Soundness, Completeness, First Order Logic, Forward Chaining, Backward Chaining**
- **Logic**: Formal systems for reasoning.
- **Soundness**: If an inference system is sound, every derived statement is true.
- **Completeness**: If an inference system is complete, it can derive every true statement.
- **First Order Logic**: A form of predicate logic.
- **Forward Chaining**: Data-driven inference technique.
- **Backward Chaining**: Goal-driven inference technique.

### Week 12: Constraint Processing
**CSPs, Consistency Based Diagnosis, Algorithm Backtracking, Arc Consistency, Algorithm Forward Checking**
- **CSPs (Constraint Satisfaction Problems)**: Problems defined by constraints that must be satisfied.
- **Consistency Based Diagnosis**: Identifying inconsistencies in systems.
- **Algorithm Backtracking**: Algorithm for finding solutions by exploring possible options and backtracking upon failure.
- **Arc Consistency**: Ensuring consistency in binary constraints.
- **Algorithm Forward Checking**: Optimizes backtracking by checking constraints ahead of time.
