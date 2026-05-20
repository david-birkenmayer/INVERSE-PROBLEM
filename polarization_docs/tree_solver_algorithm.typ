#import "setup/commands.typ": *

= Gradient/Subgradient Solver for Polarization on Metric Trees

This note describes a first-order solver for the polarization objective on a finite metric tree.

== Problem
Let $T$ be a finite metric tree and $p in NN$. For center configuration
$X = (x_1, ..., x_p) in T^p$ define
$
F(X,t) := sum_(i=1)^p K(d(t,x_i)),
$
where $K$ is a kernel (strictly decreasing and convex on $(0, infinity)$, with $K(0)=infinity$).

Let $T without {x_1,...,x_p}$ have connected components $C_1(X),...,C_k (X)$. Define
$
m_j (X) := min_(t in C_j(X)) F(X,t),
quad
underline(m)(X) := min_j m_j (X).
$
Polarization is
$
max_(X in T^p) underline(m)(X).
$

== Core idea
The objective is a lower envelope of component minima, so it is typically non-smooth. We optimize it via subgradient ascent.

For fixed $j$, choose a minimizer
$
t_j^* in "argmin"_(t in C_j(X)) F(X,t).
$
By the envelope theorem (for fixed component structure and active minimizer),
$
(partial m_j)/(partial x_i)
=
K'(d(t_j^*,x_i)) dot (partial d(t_j^*,x_i))/(partial x_i).
$
Thus differentiation of $m_j$ does not require differentiating $t_j^*(X)$.

== Tree coordinates and directional derivatives
Represent each center as $x_i=(e_i,u_i)$ on an oriented edge $e_i=[a_i,b_i]$ with local coordinate $u_i in [0,L_{e_i}]$.

Define unit directions at $x_i$:
- $+$: increasing $u_i$ along $a_i -> b_i$
- $-$: decreasing $u_i$

For any fixed $t in T$, the distance derivative along a direction is
$
D_v d(t,x_i) =
cases(
-1 & "if moving in direction " v " goes toward " t,
+1 & "if moving in direction " v " goes away from " t.
)
$
At interior points of an edge this is unambiguous due to unique geodesics in trees.
At vertices this becomes multi-directional and is handled by branch selection.

== Active set and subgradient
Let
$
A(X) := {j : m_j(X)=underline(m)(X)}
$
be the active component index set.

A valid subgradient direction for center $x_i$ is the average active directional derivative:
$
g_i := (1/(|A(X)|)) sum_(j in A(X)) K'(d(t_j^*,x_i)) dot sigma_(i,j),
$
where $sigma_(i,j)$ is the signed derivative of distance with respect to the current local coordinate direction on edge $e_i$.

If multiple minimizers exist in a component, pick one minimizer or average over several sampled minimizers to stabilize updates; both yield valid subgradient-like schemes in practice.

== One iteration
Given current centers $X^k$ and step size $alpha_k$:

1. Build components $C_j (X^k)$ of $T without {x_1^k,...,x_p^k}$.
2. For each component, compute
   $t_j^* in "argmin"_(t in C_j) F(X^k,t)$ and value $m_j(X^k)$.
3. Determine active set $A(X^k)$.
4. Compute per-center subgradient $g_i$ from active components.
5. Propose local-coordinate updates
   $u_i^("trial") = u_i^k + alpha_k g_i$.
6. Project to feasibility on the current edge interval.
7. If projection hits a vertex, perform branch selection:
   evaluate directional derivatives on each adjacent outgoing edge and continue on the edge with best ascent prediction.
8. Accept step only if
   $underline(m)(X^("trial")) >= underline(m)(X^k) - "tol"$;
   otherwise reduce $alpha_k$ and retry.

== Step-size policy
Practical robust schedule:
- Initial step $alpha_0 > 0$.
- Backtracking on rejection: update $alpha = beta alpha$ with $beta in (0,1)$.
- Optional global decay each outer iteration.
- Stop when any of the following holds:
   + $alpha < alpha_("min")$,
  + $|underline(m)(X^(k+1)) - underline(m)(X^k)| < epsilon$,
  + subgradient norm below threshold,
  + max iteration reached.

== Component minimization on trees
For each component, minimize $t mapsto F(X,t)$ by edgewise 1D convex minimization:
- Restrict $F$ to each open edge segment in the component.
- Use safeguarded Newton or bisection on derivative to find interior minimizers.
- Also evaluate admissible boundary points (vertices or cut points).
- Take the minimum over all candidates in the component.

This generalizes the star implementation directly: replace star-specific distance/component routines by generic tree path and cut routines.

== Numerical safeguards
- Clip all distances by small $eps > 0$ before evaluating $K$ and $K'$.
- Use tolerance-based active set: $m_j <= underline(m)+tau_("active")$.
- Deduplicate minimizers that are geometrically identical up to tolerance.
- Log diagnostics per iteration:
  + $underline(m)$,
  + active-set size,
  + step size,
  + subgradient norm,
  + equioscillation gap $max_j m_j - min_j m_j$.

== Complexity notes
For $E$ edges and $p$ centers:
- One objective evaluation at one point costs $O(p)$.
- One full component-minimum pass is roughly $O(E dot p)$ plus 1D minimization overhead.
- Caching shortest-path decomposition and path-incidence signs reduces repeated cost significantly.

== Scope and limitations
Included in this method:
- finite trees,
- kernels $1/r^s$ and $log(1/r)$,
- first-order ascent with subgradients.

Not included (first version):
- graphs with cycles,
- second-order methods,
- global optimality certification.

== Pseudocode
1. Initialize $X^0$ (initial centers) and $alpha = alpha_0$.
2. For $k = 0,1,2,...,k_("max")$ repeat:
    + Compute components and minimizer/value pairs $(m_j, t_j^*)_j$.
    + Set active indices as
      $A = {j : m_j <= min_l m_l + tau_("active")}$.
    + Compute subgradients $(g_i)_i$ from active components.
    + Propose and project $X^("trial")$ with vertex branch selection.
    + If $underline(m)(X^("trial")) >= underline(m)(X^k) - "tol"$:
       set $X^(k+1) = X^("trial")$ and optionally update
       $alpha = eta alpha$ with $eta approx 1$.
    + Otherwise update $alpha = beta alpha$.
    + Apply stopping test.

This yields a practical envelope-theorem-based solver architecture for polarization on general metric trees.
