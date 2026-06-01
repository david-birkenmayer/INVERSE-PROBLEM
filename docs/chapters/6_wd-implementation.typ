#import "../setup/preamble.typ": template
#import "../setup/commands.typ": *
#import "../setup/ctheorems.typ": *

#show: template

#let chapter = [

= Implemented W_d Model (Code-to-Math)

== Scope
This chapter documents the currently implemented $W_d$ pipeline in the codebase.
We focus on the a-priori $W_d$ branch (not $W_d(M)$), which is the default in the solver tab.

== Execution flow for a-priori $W_d$
For a fixed site set $M$:

- The GUI payload sets $"MODE" = "W_d"$ and force-overrides measurement source to base
  (no custom measurement instance is used in a-priori $W_d$).
- The backend builds measurement metadata from base heads and base total demand,
  but the optimization is solved in equivalence-class mode.
- The solver chooses the xd backend when $"METHOD" = "xd"$.
- Multi-start or Dynamic Multi-Start (DMS) runs several starts and keeps a selected run.
- The reported radius is
$
  W_d(M) = norm(d^A - d^B)
$
with the configured norm.

== Decision variables (xd backend)
For each pipe $e$ and junction $j$:

- $q_e^A, q_e^B$: flows in scenarios $A, B$.
- $x_e^A, x_e^B$: headloss variables with nonlinear coupling to flows.
- $d_j^A, d_j^B$: implied nodal demands from flow balance.

No explicit nodal head variables are optimized in xd mode.

== Implemented optimization problem for $W_d$
Given network $(V, E)$, junction set $J$, measurement sites $M subs V$, resistance $r_e$, and exponent $n$:

Maximize demand separation:
$
  max norm(d^A - d^B)
$

subject to (for both scenarios):

1. Pipe law
$
  x_e^s = r_e q_e^s abs(q_e^s)^(n-1), quad s in {A,B}, e in E.
$

2. Cycle consistency (one equation per cycle row $c$)
$
  sum_(e in E) sigma_(c,e) x_e^A = 0,
$
$
  sum_(e in E) sigma_(c,e) x_e^B = 0.
$

3. Measurement equivalence (head-equality-only form)
For each measurement node $m in M$ a path row is built from reservoir to $m$:
$
  sum_(e in E) pi_(m,e) (x_e^A - x_e^B) = 0.
$
This enforces equal head *differences* to the reservoir, hence equal heads at measured nodes between $A$ and $B$.

4. Junction demand definitions and lower bounds
$
  d_j^s = "inflow"_j (q^s) - "outflow"_j (q^s), quad s in {A,B},
$
$
  d_j^s >= ell_j,
$
where
$
  ell_j = max("DEMAND_LB", d_j^("ref")).
$

Exact implementation of inflow/outflow (per junction $j$):
$
  "inflow"_j (q^s) = sum_(e in E: "end"(e)=j) q_e^s,
$
$
  "outflow"_j (q^s) = sum_(e in E: "start"(e)=j) q_e^s.
$
Hence each pipe contributes with a sign induced by its stored orientation
$("start"(e), "end"(e))$.

If a pipe is incident to the reservoir and to a junction $j$, it still appears in the equation for
$d_j^s$ with the corresponding sign. So reservoir-incident edges are part of the demand definition.


5. A-priori total-demand budget (upper bound)
$
  sum_(j in J) d_j^A <= D_max,
$
$
  sum_(j in J) d_j^B <= D_max,
$
with
$
  D_max = sum_(j in J) d_j^("ref") + "EXTRA_DEMAND".
$

6. Pairwise reservoir outflow equality
$
  "netOut"(q^A) = "netOut"(q^B).
$

with
$
  "netOut"(q^s) = sum_(e in E: "start"(e)=r) q_e^s - sum_(e in E: "end"(e)=r) q_e^s,
$
where $r$ is the selected reservoir node.

== Objective realization
- For finite $p$: maximize $sum_j abs(d_j^A - d_j^B)^p$.
- For $p = infinity$: maximize epigraph $t$ with $t >= abs(d_j^A - d_j^B)$ for all $j$.

The saved radius is then converted back to $norm(d^A - d^B)$.

== DMS layer (selection over starts)
When DMS is enabled, the above optimization is solved repeatedly with perturbed starts.
The algorithm then assigns one of:

- no-improvement certificate,
- improvement certificate,
- inconclusive certificate,

based on consistency/deviation thresholds and reference radius $r$.

This layer does not change the per-start NLP constraints; it changes acceptance and caching logic.

== Reservoir handling audit
This section answers whether reservoir conditions are correctly represented.

1. Is reservoir outflow implemented?
- Yes, as a side condition between paired scenarios in a-priori $W_d$:
$
  "netOut"(q^A) = "netOut"(q^B).
$
- The net outflow expression explicitly sums all reservoir-incident edges with orientation signs.
- In a-priori $W_d$, reservoir outflow is *not* fixed to a measured scalar value by default; only pair equality is enforced.
- Total demand is also controlled separately through the upper bound $D_max$.

1b. Is "where the flow comes from" preserved when flow is unknown?
- Yes, in the implemented equations: the origin information is carried by
  - directed pipe incidence in $d_j = "inflow"_j - "outflow"_j$,
  - nonlinear pipe laws, and
  - cycle/path constraints.
- So this is not a demand-only reduced model; flow variables are explicit and source-adjacent edges are represented.
- What is *not* enforced in this branch is a fixed absolute reservoir outflow value (unless explicitly requested), only pairwise equality and global total-demand cap.

== Small worked example: where does source-flow information appear?
Consider one reservoir $r$ and two junctions $1,2$ with two pipes:

- $e_1: r -> 1$
- $e_2: 1 -> 2$

Assume these directions are exactly the stored pipe orientations.
For one scenario $s$, the implemented demand definitions are:
$
  d_1^s = q_(e_1)^s - q_(e_2)^s,
$
$
  d_2^s = q_(e_2)^s.
$
Hence
$
  q_(e_2)^s = d_2^s,
$
$
  q_(e_1)^s = d_1^s + d_2^s.
$

So the flow on the reservoir-incident edge $e_1$ is directly tied to downstream demands.
This is exactly the information "where the flow comes from" in this oriented model.

The implemented reservoir outflow expression is
$
  "netOut"(q^s) = q_(e_1)^s,
$
because $e_1$ starts at $r$ and no edge ends at $r$ in this toy network.
Therefore
$
  "netOut"(q^s) = d_1^s + d_2^s.
$

In the two-scenario $W_d$ model, the code enforces
$
  "netOut"(q^A) = "netOut"(q^B),
$
so
$
  d_1^A + d_2^A = d_1^B + d_2^B.
$

This shows two things clearly:

- Reservoir-incident edges are part of the model and carry source-flow information.
- The current a-priori branch constrains this information *relatively* (between $A$ and $B$), not to a fixed absolute measured value.

How do measurement path constraints interact?
If node $2$ is a measurement site, xd uses one path equation from $r$ to $2$:
$
  (x_(e_1)^A + x_(e_2)^A) - (x_(e_1)^B + x_(e_2)^B) = 0.
$
Because $x_e = r_e q_e abs(q_e)^(n-1)$, this constraint still depends on flows on both
the reservoir edge and the downstream edge. So reservoir-edge flow is not discarded there either.

What is *not* encoded by that equation alone is an external absolute reference for
"how much must leave the reservoir". That requires either:

- explicit fixed reservoir outflow, or
- an equivalent absolute condition (if available from data).

2. Is "reservoir is a measurement site" implemented?
- In xd mode, measurement equalities are encoded by path equations from reservoir to each site.
- If the reservoir itself is listed as a measurement site, that path is trivial and contributes no extra equation.
- This is mathematically consistent: head equality at the reference node is tautological.

3. Is absolute reservoir head enforced in xd mode?
- Not as a hard optimization constraint in the xd model.
- xd enforces only head *differences* via $x$-path equations; the absolute head level is a gauge.
- Reservoir head is used for reconstructing/reporting heads after optimization, not for tightening the feasible set in xd.

4. Important structural note
- Junction demand equations are written only for non-reservoir nodes.
- Reservoir balance is represented only through the explicit net-outflow expression above.
- In the current implementation, one reservoir node is selected as reference in this xd branch.

== Conclusion on reservoir treatment
- The implementation treats reservoir outflow as a paired side condition (plus global demand budget), which is coherent with the intended equivalence-class variant.
- "Reservoir as measurement site" is handled implicitly and is redundant in xd path-equality formulation.
- If one wants absolute-head anchoring at the reservoir as a strict constraint in xd, that would require an additional explicit anchoring mechanism not currently present in the xd NLP itself.

]

#chapter
