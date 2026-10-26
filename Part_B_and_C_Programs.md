# PART B: AI Lab (any FOUR) - Code + Logic + Viva

---
## B1. BFS for a Planning Problem (8-Puzzle)

**Logic:** BFS explores level by level using a FIFO queue, so the first time it reaches the goal is via the SHORTEST path (uniform step cost). Keep a `visited` set to avoid repeats.

```python
from collections import deque

GOAL = (1, 2, 3, 4, 5, 6, 7, 8, 0)

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

def bfs(start):
    q = deque([start])
    parent = {start: None}
    while q:
        s = q.popleft()
        if s == GOAL:
            path = []
            while s:
                path.append(s)
                s = parent[s]
            return path[::-1], len(parent)
        for n in neighbors(s):
            if n not in parent:
                parent[n] = s
                q.append(n)
    return None, len(parent)

start = (1, 2, 3, 4, 0, 6, 7, 5, 8)
path, explored = bfs(start)
for st in path:
    print(st[:3], st[3:6], st[6:])
print("Moves:", len(path) - 1, "| States generated:", explored)
```

**Viva:**
1. Complete? Yes (finite b). Optimal? Yes if all step costs are equal.
2. Time/space? O(b^d) both. Space is the bottleneck.
3. Data structure? Queue (FIFO).
4. BFS vs DFS? BFS = shortest path, more memory; DFS = less memory, may not be optimal, can loop without visited.
5. Planning problem examples? Robot path planning, blocks world, 8-puzzle.

---
## B2. DFS for Problem Solving (Water Jug Problem)

**Logic:** State = (x, y) litres in jugs A (4L) and B (3L). Goal: 2L in A. Operators: fill A/B, empty A/B, pour A→B, pour B→A. DFS goes deep along one branch with a stack (recursion); `visited` prevents infinite loops. It finds *a* solution, not necessarily shortest.

```python
CAP_A, CAP_B, TARGET = 4, 3, 2

def successors(x, y):
    yield (CAP_A, y), "Fill A"
    yield (x, CAP_B), "Fill B"
    yield (0, y), "Empty A"
    yield (x, 0), "Empty B"
    t = min(x, CAP_B - y)
    yield (x - t, y + t), "Pour A -> B"
    t = min(y, CAP_A - x)
    yield (x + t, y - t), "Pour B -> A"

def dfs(state, visited, path):
    if state[0] == TARGET:
        return path
    visited.add(state)
    for nxt, action in successors(*state):
        if nxt not in visited:
            res = dfs(nxt, visited, path + [(action, nxt)])
            if res:
                return res
    return None

sol = dfs((0, 0), set(), [])
print("Start (0, 0)")
for action, st in sol:
    print(f"{action:12s} -> {st}")
```

*(Also works for Missionaries & Cannibals, maze: only `successors` and the goal test change.)*

**Viva:**
1. Is DFS complete/optimal? Not optimal. Complete only with a visited set/finite space.
2. Space complexity? O(b·m) (linear), big advantage over BFS.
3. Why does the water jug have a solution? Target must be a multiple of gcd(4,3)=1 and ≤ max(capacities).
4. DFS data structure? Stack (or recursion).
5. Depth-limited search / IDDFS? Limit depth to avoid infinite paths; IDDFS combines BFS optimality with DFS memory.

---
## B3. Backtracking for 8-Queens (CSP)

**Logic:** Place one queen per row. For each column, check it is not attacked (same column or diagonal). If no safe column, backtrack to the previous row and try the next column.

```python
N = 8
solutions = []

def safe(cols, r, c):
    for pr in range(r):
        pc = cols[pr]
        if pc == c or abs(pc - c) == r - pr:
            return False
    return True

def solve(r, cols):
    if r == N:
        solutions.append(cols[:])
        return
    for c in range(N):
        if safe(cols, r, c):
            cols.append(c)
            solve(r + 1, cols)
            cols.pop()                     # backtrack

solve(0, [])
print("Total solutions:", len(solutions))   # 92
first = solutions[0]
for r in range(N):
    print(" ".join("Q" if first[r] == c else "." for c in range(N)))
```

**Viva:**
1. Number of solutions? 92 (12 distinct under rotation/reflection).
2. Variables, domain, constraints? Variable = queen per row, domain = columns 0-7, constraints = no same column/diagonal.
3. Diagonal check? |r1-r2| == |c1-c2|.
4. Why better than brute force? Brute force = 8^8 or C(64,8) ≈ 4.4 billion; backtracking prunes early.
5. Improvements? Forward checking, MRV, min-conflicts (local search).

---
## B4. Rule-Based Chatbot (Pattern Matching)

**Logic:** List of (regex pattern, responses). Take the user input, match the first pattern, reply. Fallback reply for no match (like ELIZA).

```python
import re, random

rules = [
    (r"\b(hi|hello|hey)\b", ["Hello! How can I help you?", "Hi there!"]),
    (r"my name is (\w+)", ["Nice to meet you, {0}!"]),
    (r"\bhow are you\b", ["I'm doing great, thanks! And you?"]),
    (r"\b(i feel|i am) (sad|tired|stressed)\b", ["Sorry to hear you are {1}. Want to talk?"]),
    (r"\b(exam|viva|practical)\b", ["All the best for your {0}!"]),
    (r"\bwhat is (ai|artificial intelligence)\b",
        ["AI is the study of intelligent agents that perceive and act to achieve goals."]),
    (r"\b(bye|exit|quit)\b", ["Goodbye! Have a nice day."]),
]

def respond(text):
    text = text.lower().strip()
    for pattern, replies in rules:
        m = re.search(pattern, text)
        if m:
            return random.choice(replies).format(*m.groups())
    return "Sorry, I didn't understand. Can you rephrase?"

print("Chatbot (type 'bye' to exit)")
while True:
    msg = input("You: ")
    reply = respond(msg)
    print("Bot:", reply)
    if re.search(r"\b(bye|exit|quit)\b", msg.lower()):
        break
```

**Viva:**
1. What is the Turing Test? A machine passes if a human judge can't tell it from a human in conversation.
2. Rule-based vs ML chatbot? Rule-based = fixed patterns, predictable, no learning; ML/NLP = learns from data, handles variations.
3. What is ELIZA? 1966 chatbot by Weizenbaum using pattern matching + substitution.
4. Limitations? No context memory, can't handle unseen inputs.
5. What is a regex group? Captured part of the match, used in the reply.

---
## B5. Number Guessing Agent (Higher/Lower Feedback)

**Logic:** Binary search! Maintain range [low, high]. Guess the midpoint. Feedback "higher" sets low = guess+1, "lower" sets high = guess-1. Needs at most ceil(log2 N) guesses.

```python
def guessing_agent(secret, low=1, high=100):
    steps = 0
    while low <= high:
        guess = (low + high) // 2
        steps += 1
        if guess == secret:
            print(f"Guess #{steps}: {guess} -> Correct!")
            return steps
        elif guess < secret:
            print(f"Guess #{steps}: {guess} -> Higher")
            low = guess + 1
        else:
            print(f"Guess #{steps}: {guess} -> Lower")
            high = guess - 1

import random
secret = random.randint(1, 100)
print("Secret (hidden from agent):", secret)
n = guessing_agent(secret)
print("Solved in", n, "guesses (max 7 for 1-100)")
```

**Viva:**
1. Which algorithm? Binary search, O(log n).
2. Agent type? Goal-based agent using feedback (percept) to shrink its belief range (internal state).
3. Max guesses for 1-100? 7 (2^7 = 128 ≥ 100).
4. Why midpoint? Halves the search space every step (worst-case optimal).

---
## B6. Goal Stack Planning (Blocks World)

**Logic (STRIPS):** Use a stack. Push the goal. Loop: pop top.
- If it's a compound goal, push its sub-goals.
- If it's an unsatisfied predicate, push an action that achieves it, then its preconditions.
- If it's an action, apply it (update state, record in plan).

Predicates: `ON(x,y)`, `ONTABLE(x)`, `CLEAR(x)`, `HOLDING(x)`, `ARMEMPTY`.
Actions: `STACK(x,y)`, `UNSTACK(x,y)`, `PICKUP(x)`, `PUTDOWN(x)`.

```python
# Initial: B on A, A on table, C on table      Goal: A on B, B on C (tower: A over B over C)
init = {("ON","B","A"), ("ONTABLE","A"), ("ONTABLE","C"),
        ("CLEAR","B"), ("CLEAR","C"), ("ARMEMPTY",)}
goal = [("ON","B","C"), ("ON","A","B")]   # order matters: build from the bottom up
ACTIONS = {"PICKUP", "PUTDOWN", "STACK", "UNSTACK"}

def spec(act):
    n, *a = act
    x = a[0]; y = a[1] if len(a) > 1 else None
    if n == "PICKUP":
        return ([("ONTABLE",x), ("CLEAR",x), ("ARMEMPTY",)],      # preconditions
                [("HOLDING",x)],                                  # add list
                [("ONTABLE",x), ("CLEAR",x), ("ARMEMPTY",)])      # delete list
    if n == "PUTDOWN":
        return ([("HOLDING",x)],
                [("ONTABLE",x), ("CLEAR",x), ("ARMEMPTY",)],
                [("HOLDING",x)])
    if n == "STACK":
        return ([("HOLDING",x), ("CLEAR",y)],
                [("ON",x,y), ("CLEAR",x), ("ARMEMPTY",)],
                [("HOLDING",x), ("CLEAR",y)])
    if n == "UNSTACK":
        return ([("ON",x,y), ("CLEAR",x), ("ARMEMPTY",)],
                [("HOLDING",x), ("CLEAR",y)],
                [("ON",x,y), ("CLEAR",x), ("ARMEMPTY",)])

def achiever(p, state):
    """Pick an action whose add-list contains the unsatisfied predicate p."""
    n = p[0]
    if n == "ON":       return ("STACK", p[1], p[2])
    if n == "ONTABLE":  return ("PUTDOWN", p[1])
    if n == "CLEAR":
        for s in state:
            if s[0] == "ON" and s[2] == p[1]:
                return ("UNSTACK", s[1], p[1])
    if n == "HOLDING":
        if ("ONTABLE", p[1]) in state:
            return ("PICKUP", p[1])
        for s in state:
            if s[0] == "ON" and s[1] == p[1]:
                return ("UNSTACK", p[1], s[2])
    if n == "ARMEMPTY":
        for s in state:
            if s[0] == "HOLDING":
                return ("PUTDOWN", s[1])

state, plan = set(init), []
stack = list(reversed(goal))              # GOAL STACK (top = end of list)
for _ in range(500):                      # safety guard
    if not stack:
        break
    top = stack.pop()
    if top[0] in ACTIONS:                 # ---- it is an ACTION
        pre, add, dele = spec(top)
        missing = [p for p in pre if p not in state]
        if not missing:                   # all preconditions true -> apply
            state -= set(dele)
            state |= set(add)
            plan.append(top)
        else:                             # re-push action, then its preconditions
            stack.append(top)
            stack.extend(missing)
    elif top in state:                    # ---- predicate already true
        continue
    else:                                 # ---- unsatisfied predicate
        act = achiever(top, state)
        stack.append(top)                 # re-check it after the action
        stack.append(act)

print("PLAN:")
for i, p in enumerate(plan, 1):
    print(f"  {i}. {p[0]}({', '.join(p[1:])})")
print("Goal reached:", all(g in state for g in goal))
```

**Note:** goals are listed bottom-up (B on C first, then A on B). Reverse the order and the planner undoes work: that is the Sussman anomaly (see viva Q3).

**Sample output:** UNSTACK(B,A), STACK(B,C), PICKUP(A), STACK(A,B).

**Viva:**
1. What is STRIPS? Planning language: state = set of predicates; action = preconditions, add list, delete list.
2. Why a stack? Handles subgoals in order; depth-first style planning.
3. Limitation? Sussman anomaly: goal stack planners can produce non-optimal plans when sub-goals interact. Fix: partial-order planning.
4. Frame problem? What stays unchanged after an action (STRIPS assumes everything not in add/delete is unchanged).
5. Forward vs backward planning? Goal stack is backward (goal-directed).

---
## B7. Minimax Game-Playing Agent (Tic-Tac-Toe)

**Logic:** MAX (X) picks the move with the highest value, MIN (O) picks the lowest. Recursively evaluate until terminal: +1 X wins, -1 O wins, 0 draw. Agent plays the move giving the best guaranteed outcome.

```python
WINS = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]

def winner(b):
    for a, c, d in WINS:
        if b[a] != " " and b[a] == b[c] == b[d]:
            return b[a]
    return None

def minimax(b, is_max):
    w = winner(b)
    if w == "X": return 1
    if w == "O": return -1
    if " " not in b: return 0
    scores = []
    for i in range(9):
        if b[i] == " ":
            b[i] = "X" if is_max else "O"
            scores.append(minimax(b, not is_max))
            b[i] = " "
    return max(scores) if is_max else min(scores)

def best_move(b):                     # computer plays X (MAX)
    best, move = -2, None
    for i in range(9):
        if b[i] == " ":
            b[i] = "X"
            s = minimax(b, False)
            b[i] = " "
            if s > best:
                best, move = s, i
    return move

def show(b):
    for i in range(0, 9, 3):
        print(" | ".join(b[i:i+3]))
        if i < 6: print("--+---+--")
    print()

board = [" "] * 9
show(board)
turn = "O"                            # human starts as O
while winner(board) is None and " " in board:
    if turn == "O":
        m = int(input("Your move (0-8): "))
        if board[m] != " ":
            print("Taken!"); continue
        board[m] = "O"
    else:
        board[best_move(board)] = "X"
    show(board)
    turn = "X" if turn == "O" else "O"
print("Winner:", winner(board) or "Draw")
```

**Viva:**
1. Minimax assumption? Opponent plays optimally.
2. Complexity? O(b^d); tic-tac-toe has 255,168 games (~5478 states).
3. How to improve? Alpha-beta pruning, depth limit + evaluation function.
4. Zero-sum game? One player's gain = other's loss.
5. Minimax vs alpha-beta? Same result; alpha-beta prunes.
6. Can the human ever beat it? No, best they can do is draw.

---

# PART C: Mini-Project Options

## C1. Medical Diagnosis Expert System (Forward + Backward Chaining)

**Concepts:**
- **Knowledge base:** IF-THEN rules (symptoms → disease).
- **Forward chaining (data-driven):** start with known facts (symptoms), fire rules whose conditions are satisfied, add conclusions, repeat until nothing new.
- **Backward chaining (goal-driven):** start with a hypothesis ("Does the patient have flu?"), find rules that conclude it, then try to prove their conditions (asking the user if unknown).
- **Inference engine** = chaining logic. **Explanation** = which rules fired.

```python
rules = [
    ({"fever", "cough", "body_ache"},       "flu"),
    ({"fever", "rash", "joint_pain"},       "dengue"),
    ({"sneezing", "runny_nose"},            "common_cold"),
    ({"high_sugar", "frequent_urination"},  "diabetes"),
    ({"flu", "breathing_difficulty"},       "see_doctor_urgently"),
    ({"dengue"},                            "see_doctor_urgently"),
]

# ---------- FORWARD CHAINING ----------
def forward_chain(facts):
    facts = set(facts)
    trace = []
    changed = True
    while changed:
        changed = False
        for cond, concl in rules:
            if cond <= facts and concl not in facts:
                facts.add(concl)
                trace.append((sorted(cond), concl))
                changed = True
    return facts, trace

# ---------- BACKWARD CHAINING ----------
def backward_chain(goal, facts, asked=None):
    asked = asked if asked is not None else {}
    if goal in facts:
        return True
    rules_for_goal = [c for c, g in rules if g == goal]
    if not rules_for_goal:                         # basic symptom: ask user
        if goal not in asked:
            asked[goal] = input(f"  Does patient have {goal}? (y/n): ").lower() == "y"
            if asked[goal]:
                facts.add(goal)
        return asked[goal]
    return any(all(backward_chain(c, facts, asked) for c in cond)
               for cond in rules_for_goal)

# ---------- DEMO ----------
facts, trace = forward_chain({"fever", "cough", "body_ache", "breathing_difficulty"})
print("FORWARD CHAINING")
for cond, concl in trace:
    print(f"  {cond} => {concl}")
print("Diagnoses:", [f for f in facts if f in {g for _, g in rules}])

print("\nBACKWARD CHAINING: hypothesis = flu")
print("Result:", backward_chain("flu", set()))
```

**Viva:**
1. Forward vs backward chaining? Data-driven (facts → conclusions) vs goal-driven (goal → needed facts). Forward good for monitoring/diagnosis with many facts; backward efficient when testing a specific hypothesis.
2. Parts of an expert system? Knowledge base, inference engine, user interface, explanation facility, knowledge acquisition.
3. Example expert systems? MYCIN (bacterial infections), DENDRAL.
4. Limitations? No learning, hard to maintain large rule sets, can't handle uncertainty (fix: certainty factors/fuzzy/Bayes).
5. Conflict resolution? Choose which rule to fire when several match (priority, specificity, recency).
6. Add a disclaimer: educational only, not real medical advice.

---
## C2. Rule-Based Recommendation System (Movies)

**Logic:** Rules map user preferences (genre, mood, rating threshold) to movies. Score each movie by how many preference rules it satisfies, then rank.

```python
movies = [
    {"title": "Inception",       "genre": "scifi",   "rating": 8.8, "year": 2010},
    {"title": "3 Idiots",        "genre": "comedy",  "rating": 8.4, "year": 2009},
    {"title": "The Dark Knight", "genre": "action",  "rating": 9.0, "year": 2008},
    {"title": "Interstellar",    "genre": "scifi",   "rating": 8.7, "year": 2014},
    {"title": "Dangal",          "genre": "drama",   "rating": 8.3, "year": 2016},
    {"title": "Hera Pheri",      "genre": "comedy",  "rating": 8.2, "year": 2000},
]

mood_to_genre = {"happy": ["comedy"], "excited": ["action", "scifi"],
                 "thoughtful": ["scifi", "drama"], "sad": ["comedy", "drama"]}

def recommend(genre=None, mood=None, min_rating=0, recent=False, top=3):
    liked = set()
    if genre: liked.add(genre)
    if mood:  liked.update(mood_to_genre.get(mood, []))
    scored = []
    for m in movies:
        score = 0
        if m["genre"] in liked:         score += 3      # rule 1
        if m["rating"] >= min_rating:   score += 2      # rule 2
        if recent and m["year"] >= 2010: score += 1     # rule 3
        if m["rating"] >= 8.8:          score += 1      # rule 4: popular
        if score:
            scored.append((score, m))
    scored.sort(key=lambda x: (-x[0], -x[1]["rating"]))
    return [(s, m["title"]) for s, m in scored[:top]]

print(recommend(genre="scifi", mood="thoughtful", min_rating=8.5, recent=True))
```

**Viva:**
1. Types of recommenders? Content-based, collaborative filtering, hybrid, knowledge/rule-based.
2. Cold-start problem? New users/items have no data. Rule-based avoids it.
3. Limitation? Rules are manual, don't adapt, no personalization from history.
4. How to improve? Cosine similarity on user vectors, matrix factorization.

---

# GENERAL VIVA QUESTIONS (Whole Subject)

1. **Define AI.** Study/design of intelligent agents that perceive their environment and take actions to maximize success.
2. **Four approaches (Russell & Norvig):** thinking humanly, thinking rationally, acting humanly (Turing test), acting rationally (rational agent).
3. **Agent vs environment; PEAS.** Performance measure, Environment, Actuators, Sensors.
4. **Environment properties:** observable/partial, deterministic/stochastic, episodic/sequential, static/dynamic, discrete/continuous, single/multi-agent.
5. **Uninformed vs informed search:** BFS, DFS, UCS, IDDFS vs Greedy, A*, hill-climbing.
6. **Search metrics:** completeness, optimality, time, space.
7. **Heuristic?** Estimate of cost to goal. **Admissible** = never overestimates; **consistent** = h(n) ≤ c(n,n') + h(n').
8. **Hill climbing problems:** local maxima, plateau, ridges. Fix: random restart, simulated annealing.
9. **Simulated annealing:** accept worse moves with probability e^(-ΔE/T), T decreases.
10. **Genetic algorithm:** population, selection, crossover, mutation, fitness.
11. **CSP:** variables, domains, constraints; backtracking, forward checking, AC-3, MRV.
12. **Knowledge representation:** propositional logic, FOL, semantic nets, frames, production rules.
13. **Planning:** STRIPS, state-space vs partial-order planning, goal stack, frame problem.
14. **Complete vs sound inference:** sound = only derives true statements; complete = derives all true statements.
15. **Resolution, Modus Ponens, unification:** know one-line definitions.
16. **Supervised vs unsupervised vs reinforcement learning.**
17. **Tricks to remember:** BFS = queue; DFS = stack; A* = priority queue by f=g+h.

## Exam-day tips
- Know ONE program from each part *very* well (type it from memory, explain every function).
- Always state: Problem → State representation → Operators → Goal test → Algorithm → Complexity.
- Be ready to modify input (different start state, grid, N queens N=4/6) and re-run.
- Know sample output and what it means.
