# PART A: AI Lab (any FOUR) - Code + Logic + Viva

All code below was run and tested (Python 3).

---
## A1. Simple Reflex Agent (Vacuum Cleaner World)

**Logic:** Agent decides ONLY from the current percept (location, status). No memory. Rule: if Dirty then Suck, else move to the other square (condition-action rules).

```python
def reflex_vacuum_agent(percept):
    location, status = percept
    if status == "Dirty":
        return "Suck"
    return "Right" if location == "A" else "Left"

env = {"A": "Dirty", "B": "Dirty"}
loc = "A"
for step in range(6):
    action = reflex_vacuum_agent((loc, env[loc]))
    print(f"Step {step}: at {loc}, env={env}, action={action}")
    if action == "Suck":
        env[loc] = "Clean"
    elif action == "Right":
        loc = "B"
    else:
        loc = "A"
```

**Explain:** percept = (location, dirt status). The function is the "agent program". The loop is the "environment" that applies the action and changes the world.

**Rational?** Rational = maximizes performance measure (e.g., +1 per clean square per time step) given what it knows. Our agent is rational for the 2-square world. But it keeps moving forever after everything is clean (wastes energy). Fix needs memory (model-based agent).

**Viva:**
1. What is a PEAS description? Performance (clean squares), Environment (2 squares), Actuators (suck, left, right), Sensors (location, dirt).
2. Types of agents? Simple reflex, model-based reflex, goal-based, utility-based, learning.
3. Limitation of simple reflex? Fails in partially observable environments, can loop forever.
4. Is the environment fully observable? Here yes (it knows location and dirt of the current square only, so technically partially observable for the other square).
5. Rational vs omniscient? Rational maximizes expected performance; omniscient knows actual outcomes (impossible).

---
## A2. Model-Based Agent in Grid with Obstacles

**Logic:** The agent keeps an INTERNAL STATE (known obstacles, visited cells). Each step: sense neighbours, update model, plan a path (BFS) on the model assuming unknown cells are free, take one step, repeat. Replans when it discovers obstacles.

```python
from collections import deque

GRID = ["S..#....",
        ".#.#.##.",
        ".#...#..",
        ".####.#.",
        "......#G"]
R, C = len(GRID), len(GRID[0])
DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))

def find(ch):
    for r in range(R):
        for c in range(C):
            if GRID[r][c] == ch:
                return (r, c)

class ModelBasedAgent:
    def __init__(self, start, goal):
        self.pos, self.goal = start, goal
        self.obstacles = set()      # internal model
        self.visited = {start}      # internal state

    def sense(self):                # percept: blocked neighbours
        r, c = self.pos
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < R and 0 <= nc < C and GRID[nr][nc] == '#':
                self.obstacles.add((nr, nc))

    def plan(self):                 # BFS on the internal model
        q = deque([(self.pos, [])])
        seen = {self.pos}
        while q:
            (r, c), path = q.popleft()
            if (r, c) == self.goal:
                return path
            for dr, dc in DIRS:
                n = (r + dr, c + dc)
                if 0 <= n[0] < R and 0 <= n[1] < C and n not in self.obstacles and n not in seen:
                    seen.add(n)
                    q.append((n, path + [n]))
        return None

    def step(self):
        self.sense()
        path = self.plan()
        if not path:
            return False
        self.pos = path[0]
        self.visited.add(self.pos)
        return True

agent = ModelBasedAgent(find('S'), find('G'))
trail = [agent.pos]
while agent.pos != agent.goal:
    if not agent.step():
        print("No path"); break
    trail.append(agent.pos)
print("Path:", trail)
print("Steps:", len(trail) - 1, "| Obstacles discovered:", len(agent.obstacles))
```

**Viva:**
1. Difference from reflex agent? Maintains internal state/model of the world; works in partially observable environments.
2. What is the "model"? Knowledge of how the world evolves + what actions do. Here: set of known obstacles + visited cells.
3. Why BFS? Gives the shortest path in an unweighted grid.
4. Why replan every step? New obstacles may be discovered.
5. What if goal is unreachable? plan() returns None, agent stops.

---
## A3. A* Search for 8-Puzzle

**Logic:** f(n) = g(n) + h(n). g = moves so far, h = Manhattan distance of tiles from goal positions (admissible: never overestimates). Priority queue always expands lowest f. Gives optimal solution.

```python
import heapq

GOAL = (1, 2, 3, 4, 5, 6, 7, 8, 0)      # 0 = blank

def h(s):                                # Manhattan distance
    return sum(abs(i // 3 - (v - 1) // 3) + abs(i % 3 - (v - 1) % 3)
               for i, v in enumerate(s) if v)

def neighbors(s):
    z = s.index(0)
    r, c = divmod(z, 3)
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < 3 and 0 <= nc < 3:
            n = nr * 3 + nc
            l = list(s)
            l[z], l[n] = l[n], l[z]
            yield tuple(l)

def solvable(s):
    a = [x for x in s if x]
    inv = sum(1 for i in range(8) for j in range(i + 1, 8) if a[i] > a[j])
    return inv % 2 == 0

def astar(start):
    pq = [(h(start), 0, start)]
    g = {start: 0}
    parent = {start: None}
    expanded = 0
    while pq:
        f, cost, s = heapq.heappop(pq)
        if cost > g[s]:
            continue
        expanded += 1
        if s == GOAL:
            path = []
            while s:
                path.append(s)
                s = parent[s]
            return path[::-1], expanded
        for n in neighbors(s):
            if cost + 1 < g.get(n, 10**9):
                g[n] = cost + 1
                parent[n] = s
                heapq.heappush(pq, (cost + 1 + h(n), cost + 1, n))
    return None, expanded

start = (1, 2, 3, 4, 0, 6, 7, 5, 8)
if solvable(start):
    path, exp = astar(start)
    for st in path:
        print(st[:3], st[3:6], st[6:])
        print()
    print("Moves:", len(path) - 1, "| Nodes expanded:", exp)
else:
    print("Unsolvable")
```

**Viva:**
1. What is an admissible heuristic? Never overestimates real cost. Manhattan distance and misplaced tiles are both admissible; Manhattan is better (more informed).
2. Is A* complete/optimal? Yes (finite branching, admissible h). Optimal for tree search with admissible h; graph search needs consistent h.
3. Time/space? O(b^d) worst case; space is the main problem (stores all nodes).
4. How do you check solvability? Count inversions; even inversions means solvable (for 3x3).
5. A* vs Greedy best-first vs UCS? UCS uses only g, Greedy only h, A* both. UCS = A* with h=0.
6. Why the priority queue? To pop the min f-value node efficiently.

---
## A4. Adversarial Search with Alpha-Beta Pruning

**Logic:** Minimax + pruning. Alpha = best value MAX can guarantee so far, Beta = best value MIN can guarantee. If beta <= alpha, the remaining children cannot affect the result, so prune. Same answer as minimax, fewer nodes.

```python
import math
pruned = 0

def alphabeta(node, depth, alpha, beta, maximizing):
    global pruned
    if depth == 0 or not isinstance(node, list):
        return node
    if maximizing:
        best = -math.inf
        for child in node:
            best = max(best, alphabeta(child, depth - 1, alpha, beta, False))
            alpha = max(alpha, best)
            if beta <= alpha:
                pruned += 1
                break
        return best
    else:
        best = math.inf
        for child in node:
            best = min(best, alphabeta(child, depth - 1, alpha, beta, True))
            beta = min(beta, best)
            if beta <= alpha:
                pruned += 1
                break
        return best

tree = [[[3, 5], [6, 9]], [[1, 2], [0, -1]]]
print("Optimal value:", alphabeta(tree, 3, -math.inf, math.inf, True))
print("Branches pruned:", pruned)
```
Output: value 5, 2 cut-offs (once root MAX already has 5, the right MIN subtree is cut as soon as it sees a value below 5, e.g. 1 and 0).

**Viva:**
1. Does pruning change the result? No, only efficiency.
2. Best-case complexity? O(b^(d/2)) with perfect move ordering (can search twice as deep). Worst case O(b^d).
3. Initial alpha, beta? -infinity, +infinity.
4. Why is move ordering important? Good ordering causes earlier cutoffs.
5. What kind of games? Two-player, zero-sum, deterministic, perfect information (chess, tic-tac-toe).

---
## A5. Tower of Hanoi by State Space Search

**Logic:** State = tuple giving the peg (0,1,2) of each disk; disk 0 is smallest. Start (0,0,0), goal (2,2,2). A disk can move only if no smaller disk is on top of it or on the target peg. BFS over the state space finds the shortest solution (2^n - 1 moves).

```python
from collections import deque

N = 3

def moves(state):
    for d in range(N):
        src = state[d]
        if any(state[k] == src for k in range(d)):      # smaller disk on top
            continue
        for dst in range(3):
            if dst == src:
                continue
            if any(state[k] == dst for k in range(d)):  # smaller on target
                continue
            new = list(state)
            new[d] = dst
            yield (d, src, dst), tuple(new)

def bfs(start, goal):
    q = deque([start])
    parent = {start: None}
    while q:
        s = q.popleft()
        if s == goal:
            break
        for mv, n in moves(s):
            if n not in parent:
                parent[n] = (s, mv)
                q.append(n)
    path = []
    while parent[s] is not None:
        prev, mv = parent[s]
        path.append(mv)
        s = prev
    return path[::-1]

sol = bfs((0,) * N, (2,) * N)
names = "ABC"
for i, (d, a, b) in enumerate(sol, 1):
    print(f"{i}. Move disk {d + 1} from {names[a]} to {names[b]}")
print("Total moves:", len(sol), "(2^N - 1 =", 2**N - 1, ")")

# Recursive version (for comparison)
def hanoi(n, a, c, b):
    if n == 0: return
    hanoi(n - 1, a, b, c)
    print(f"Move disk {n} from {a} to {c}")
    hanoi(n - 1, b, c, a)
```

**Viva:**
1. Number of states? 3^N. Minimum moves? 2^N - 1.
2. State, operators, goal test, path cost? State = disk positions; operator = legal move; goal = all on peg C; cost = 1 per move.
3. Recursion idea? Move n-1 disks to auxiliary, move largest to target, move n-1 on top.
4. Why BFS? Guarantees shortest path.
5. Time complexity of recursive solution? O(2^n).

---
## A6. Sudoku Solver (Backtracking + Constraint Propagation)

**Logic:** CSP: variables = 81 cells, domain 1-9, constraints = row/col/box all-different.
- **Constraint propagation:** if a cell has only ONE possible value (naked single), fill it, repeat.
- **MRV heuristic:** pick the empty cell with fewest candidates.
- **Backtracking:** try each candidate; undo if it fails.

```python
def candidates(b, r, c):
    used = set(b[r]) | {b[i][c] for i in range(9)}
    br, bc = 3 * (r // 3), 3 * (c // 3)
    used |= {b[i][j] for i in range(br, br + 3) for j in range(bc, bc + 3)}
    return [v for v in range(1, 10) if v not in used]

def solve(b):
    filled = []
    # --- constraint propagation (naked singles) ---
    while True:
        progress = False
        for r in range(9):
            for c in range(9):
                if b[r][c] == 0:
                    cand = candidates(b, r, c)
                    if not cand:                       # dead end
                        for rr, cc in filled: b[rr][cc] = 0
                        return False
                    if len(cand) == 1:
                        b[r][c] = cand[0]
                        filled.append((r, c))
                        progress = True
        if not progress:
            break
    # --- MRV: choose most constrained empty cell ---
    best = None
    for r in range(9):
        for c in range(9):
            if b[r][c] == 0:
                cand = candidates(b, r, c)
                if best is None or len(cand) < len(best[2]):
                    best = (r, c, cand)
    if best is None:
        return True                                    # solved
    r, c, cand = best
    for v in cand:                                     # --- backtracking ---
        b[r][c] = v
        if solve(b):
            return True
    b[r][c] = 0
    for rr, cc in filled: b[rr][cc] = 0
    return False

puzzle = [[5,3,0,0,7,0,0,0,0],
          [6,0,0,1,9,5,0,0,0],
          [0,9,8,0,0,0,0,6,0],
          [8,0,0,0,6,0,0,0,3],
          [4,0,0,8,0,3,0,0,1],
          [7,0,0,0,2,0,0,0,6],
          [0,6,0,0,0,0,2,8,0],
          [0,0,0,4,1,9,0,0,5],
          [0,0,0,0,8,0,0,7,9]]
if solve(puzzle):
    for row in puzzle:
        print(row)
else:
    print("No solution")
```

**Viva:**
1. What is a CSP? (Variables, domains, constraints.)
2. Difference between plain backtracking and with propagation? Propagation prunes the domains early, so far fewer dead ends.
3. What is MRV / degree heuristic / LCV? MRV = fewest legal values; degree = most constraints on others; LCV = value that rules out the fewest options.
4. Forward checking vs arc consistency (AC-3)? FC removes values from neighbours' domains after assignment; AC-3 enforces consistency between all arcs.
5. Complexity? Sudoku is NP-complete in general.

---
## A7. Local Beam Search for TSP

**Logic:** Keep K best states (tours) at a time. Generate ALL neighbours of all K states (2-opt: reverse a segment), pick the K best overall from the pool, repeat until no improvement. Unlike K independent hill climbs, states share information (good ones take over the beam).

```python
import random, math

random.seed(1)
n = 10
cities = [(random.randint(0, 100), random.randint(0, 100)) for _ in range(n)]

def cost(t):
    return sum(math.dist(cities[t[i]], cities[t[(i + 1) % n]]) for i in range(n))

def neighbors(t):
    out = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            nb = t[:]
            nb[i:j + 1] = reversed(nb[i:j + 1])     # 2-opt move
            out.append(nb)
    return out

def local_beam_search(k=3, iters=200):
    beam = [random.sample(range(n), n) for _ in range(k)]
    best = min(beam, key=cost)
    for _ in range(iters):
        pool = [nb for s in beam for nb in neighbors(s)]
        pool.sort(key=cost)
        new, seen = [], set()
        for p in pool:
            if tuple(p) not in seen:
                seen.add(tuple(p))
                new.append(p)
            if len(new) == k:
                break
        if cost(new[0]) >= cost(best):              # no improvement: stop
            break
        beam, best = new, new[0]
    return best

best = local_beam_search(k=3)
print("Best tour:", best + [best[0]])
print("Cost:", round(cost(best), 2))
```

**Viva:**
1. Beam search vs hill climbing? Hill climbing keeps 1 state; beam keeps K. Beam with K=1 is hill climbing.
2. Local beam vs K parallel random restarts? In local beam, information is shared (best successors from ANY state are chosen).
3. Problem? Beam can lose diversity (all K states cluster in the same region). Fix: stochastic beam search.
4. Is it complete / optimal? No, it can get stuck in local optimum.
5. TSP search space size? (n-1)!/2.
6. What is state, neighbour? State = permutation of cities; neighbour = 2-opt/swap.
