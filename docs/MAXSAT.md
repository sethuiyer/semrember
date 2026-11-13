# The MaxSAT core, in Crystal

[README](../README.md) · [Core mathematics](MATH.md) · [Engine design](DESIGN.md)

`src/maxsat.cr` is an independent Crystal implementation of the NuWLS-style
local search used by SUTRA. It is a port of the algorithm, not a wrapper:
there is no C linkage, no subprocess call, and no copied source. The solver is
an **approximator**; it returns the best assignment it found and never claims
optimality.

The external contract is a weighted CNF:

```
p wcnf <vars> <clauses> <top>
<weight> <lit> ... 0      soft clause (weight < top)
h <lit> ... 0             hard clause
```

A soft weight of `0` is treated as hard, matching SUTRA's convention. The
implementation also reads plain `p cnf` where every clause is hard.

---

## 1. The objective

For an assignment `A`:

```
hard(A) = number of violated hard clauses
soft(A) = sum of weights of violated soft clauses
```

Comparisons are lexicographic:

```
A better than B  ⇔  hard(A) < hard(B)
                    or hard(A) = hard(B) and soft(A) < soft(B)
```

Hard clauses are promises, soft clauses are preferences. The search may
temporarily violate hard clauses, but it never prefers a soft gain over a
smaller hard violation.

---

## 2. Scores

For each variable `v`:

```
score[v] = weight gained by flipping v
         = +w  for every violated clause containing v
           -w  for every satisfied clause where v is its only true literal
```

Positive means an improving flip. The search maintains `score` incrementally
and recomputes it from scratch only for tests and after weight movements.

### The four clause transitions

When `v` flips, each clause containing it moves between states:

| transition | meaning | score updates |
|---|---|---|
| `0 → 1` | violated becomes satisfied | subtract `w` from every literal in the clause; leave the violated stack |
| `1 → 2` | one true literal becomes two | add `w` to the old unique true variable |
| `2 → 1` | two true literals become one | subtract `w` from the remaining true variable |
| `1 → 0` | satisfied becomes violated | add `w` to every literal in the clause; enter the violated stack |

The final invariant:

```
score[v] = -score_before(v)
```

Flipping back must undo the gain. Without this line the incremental scores
drift and the "improving" moves stop improving.

---

## 3. The good-variable stack

`good` is a dense list of variables with `score[v] > 0`; `good_pos[v]` makes
removal `O(1)` by swapping with the last element. While the list is non-empty,
every move descends.

The next variable is chosen by **BMS** (best from multiple selections): sample
22 entries, keep the best score; ties go to the variable with the older
`stamp` (least recently flipped). With probability `0.036`, take a random good
variable instead, to avoid determinism.

---

## 4. Weights and escape

When no positive-score variable exists, the search is at a local minimum. It
first updates the tuned clause weights:

- soft weights are rescaled so their average is `1000`; hard clauses get
  `1e9`;
- every violated soft clause gets `+3`, capped at `tw[c] + 10`;
- with probability `1e-3`, a smoothing pass takes `3` back from satisfied
  heavy clauses.

Growth is **additive**, not multiplicative. Additive growth plus a cap keeps
a heavy clause heavier than a light one; multiplying everything by `1.01`
eventually saturates the objective and destroys the ordering.

Then it picks a random violated clause (hard first if any), and either:

- flips a random literal (`0.48`), or
- flips the best-scoring literal in that clause.

That random walk carries the search out of the local minimum. Ties use
`stamp`, which prevents two-variable oscillation.

---

## 5. Continuous warm start

Before the discrete repair, each variable is relaxed to `x[v] ∈ [0,1]` and
descended with Adam. The energy has three pieces.

### Hard clauses: log barrier

Let `p_c` be the probability the clause is false:

```
p_c = ∏_{l ∈ c} (l > 0 ? 1 - x[var(l)] : x[var(l)])
```

The penalty is `-log(1 - p_c)`. Its gradient pushes away from `p_c → 1`. With
`slack = max(1 - p_c, ε)`:

```
coef = 64 / slack
∂/∂x[var] += coef * (p_c / p_false(lit)) * (lit > 0 ? -1 : +1)
```

### Soft clauses: squared violation

For width 2, the product form is used directly:

```
p_c = ∏ false-probability
coef = 2 * w * p_c
∂/∂x[var] += coef * (p_c / p_false(lit)) * (lit > 0 ? -1 : +1)
```

For general width, a stable softplus hinge keeps the gradient bounded:

```
u = 1 - Σ (lit > 0 ? x : 1 - x)
τ = max(tau, 1e-3)
soft_viol = u + τ log(1 + exp(-u/τ))   if u > 0
          = τ log(1 + exp(u/τ))        otherwise
σ = sigmoid(u/τ)
coef = 2 * w * soft_viol
∂/∂x[var] += coef * (lit > 0 ? -σ : +σ)
```

### Adam and decimation

```
mu = 0.05 * (step / steps)²
lr = lr0 * (0.5 + 0.5 * (1 - step/steps))
m = 0.9 m + 0.1 g
v = 0.999 v + 0.001 g²
x -= lr * (m / (1 - 0.9^(step+1))) / (sqrt(v / (1 - 0.999^(step+1))) + 1e-8)
x = clamp(x, 1e-4, 1 - 1e-4)
```

Every `dec_freq` steps, variables with `|x - 0.5| > dec_threshold` are
snapped to `0.999` or `0.001`. The rounded point seeds the discrete repair.

The continuous phase is an initializer, not a correctness requirement. It is
skipped when `steps = 0`, which is also how a warm-start witness is used.

---

## 6. The repair loop

```
state = SearchState(formula, assignment)
best = assignment
while flips < budget:
    if all hard and soft clauses satisfied: converged
    v = pick()
    flip(v)
    if hard becomes feasible: recount and record best immediately
    every 256 flips: recount and record the best lexicographic assignment
return best
```

The extra "hard becomes feasible" check matters: waiting 256 flips can miss a
short-lived hard-feasible assignment in tiny formulas. Soft improvements are
still checked every 256 flips to keep the recount cost bounded.

The returned cost is always recomputed from the returned assignment.

---

## 7. What is different from SUTRA

This is the same search math, not the whole SUTRA product:

- no parity preprocessing, CDCL, DRAT/GFP certificates, or portfolio mode;
- no `UNSAT` claim; hard-unsatisfied is reported as `UNKNOWN`;
- wall-clock time budgets rather than CPU `clock()`;
- deterministic per-run seeding, including the NuWLS RNG;
- zero-based assignment indexing internally, one-based DIMACS at the boundary;
- an immediate hard-feasible incumbent check in the repair loop.

The C solver remains the reference for the full pipeline. This Crystal core is
for embedding the approximation in the shard-architecture planner without a
foreign binary dependency.

---

## 8. Complexity

Let `L` be the average clause width and `F` the number of flips.

| phase | cost |
|---|---|
| rebuild scores | `O(total literals)` |
| one flip | `O(L · occurrences of v)` |
| pick (BMS) | `O(22)` |
| weight update | `O(literals in violated soft clauses)` |
| recount | `O(total literals)` every 256 flips |
| continuous step | `O(total literals + vars)` |

Memory is `O(vars + clauses + literals)`.

---

## 9. Usage

```bash
# build
make maxsat

# solve
build/maxsat instance.wcnf --time-budget-seconds 30 --solution witness.sol

# warm start from a previous witness
build/maxsat instance.wcnf --warm-start witness.sol --repair-flips 5000000
```

The output is one JSON line:

```json
{"engine":"MaxSatCr","status":"FEASIBLE","hard_unsatisfied":0,
 "soft_cost":6.0,"flips":8144896,"verified":"recount_from_witness"}
```

`--solution` writes one `v <lit>` line per variable. `--warm-start` reads the
same format.

---

## 10. Source map

| Math | Code |
|---|---|
| WCNF/CNF parsing | `src/maxsat.cr`, `Wcnf.parse` |
| scores, flip transitions, good stack | `src/maxsat.cr`, `SearchState#flip`, `#rebuild_scores` |
| weighting and random walk | `src/maxsat.cr`, `#update_weights`, `#pick` |
| repair loop | `src/maxsat.cr`, `NuWLS.repair` |
| continuous Adam phase | `src/maxsat.cr`, `Continuous.gradient_step` |
| lexicographic solve and recount | `src/maxsat.cr`, `Solver.solve`, `Solver.cost` |
| CLI | `src/maxsat_cli.cr` |
| tests | `spec/maxsat_spec.cr` |
