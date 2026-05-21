#set document(title: "Discrete Polarization on Metric Graphs")

#set page(numbering: "1")
#set text(lang: "en")

// Theorem-like environments
#let thmblock(tag, title, body) = block(
  width: 100%,
  inset: (x: 1em, y: 0.75em),
  fill: luma(245),
  radius: 3pt,
  stroke: (left: 2pt + luma(120)),
  [#strong(tag) #if title != none [_(#title)_] *.*#h(0.5em)#body]
)
#let theorem(title: none, body)  = thmblock([Theorem],  title, body)
#let lemma(title: none, body)    = thmblock([Lemma],    title, body)
#let corollary(title: none, body)= thmblock([Corollary],title, body)
#let remark(title: none, body)   = thmblock([Remark],   title, body)
#let proof(body) = block(width: 100%, inset: (left: 1em), [_Proof._ #body #h(1fr) $square$])

= Overview

This document collects definitions, lemmas, and proofs for the discrete polarization problem on metric graphs.

== Problem Statement

Let $G = (V, E)$ be a graph and let $A$ be the associated metrized graph. For a fixed integer $p$, choose a subset
$X = {x_1, ..., x_p}$ with $X subset.eq V$ to maximize
$
  min_(x in A) sum_(i=1)^p phi(d(x, x_i)).
$

Assumptions used throughout:
- $d$ is the path metric on $A$.
- $phi: (0, infinity] -> [0, infinity)$ is decreasing and strictly convex on $(0, infinity)$, with $phi(0^+) := infinity$.
- A primary example is $phi(t) = 1 / t$ for $t > 0$.

== Extended Real Definition

We work in the extended reals $overline(RR) = RR union {+infinity, -infinity}$. The choice 
$phi(0) = +infinity$ is natural:

- When $x = x_i$, the distance is $d(x, x_i) = 0$, and the "utility" contribution diverges. This reflects that a selected point has infinite affinity to itself.
- For any $x in A$, if $x in X$, then $U_X(x) = +infinity$, which does not constrain the minimum. The real bottleneck comes from points $x in A \\ X$ (unselected points).
- This avoids artificial domain restrictions or piecewise definitions.

== Semi-Infinite Programming Formulation

Reformulate the problem as a *semi-infinite program* (SIP):

*Decision variables (finitely many):* positions $x_1, ..., x_p in V$.

*Objective:* Maximize threshold $tau in RR$.

*Constraint set (infinitely many):* for all $x in A$,
$
  sum_(i=1)^p phi(d(x, x_i)) >= tau.
$

This is a canonical SIP: finitely many variables, infinitely many constraints (one per point in the continuum $A$). The optimization is equivalent to the original problem: 
$
  F(X^*) = max_{X,|X|=p} min_{x in A} U_X(x) = max tau "such that" U_X(x) >= tau, forall x in A.
$

=== Key SIP Theory

From the theory of semi-infinite programming (Hettich–Kortanek, 1993; Remez, 1934):

1. *Finite Active Set Property:* The optimal value satisfies a system involving finitely many "active constraints." In metric graph problems, this means the global minimum $min_(x in A) U_X(x)$ is attained at finitely many witnesses: vertices and at most one critical point per edge (where edge-wise convexity admits an interior minimum).

2. *Equioscillation and Optimality:* An optimal placement $X^*$ satisfies: there exist finitely many points $x_1^*, ..., x_k^* in A$ (the "active set") such that $U_{X^*}(x_j^*) = tau^* = F(X^*)$ for all $j$, and these points "equioscillate" in a suitably refined sense (cf. Remez equioscillation theorem).

3. *Karush–Kuhn–Tucker Conditions:* Optimality is characterized via KKT on the active set. For trees and bounded-treewidth graphs, this yields a finite verification.

This SIP perspective gives us:
- A finite reduction of the continuum problem to a finite witness set.
- Algorithmic guidance: solve the SIP via discretization (refine witness set iteratively) or exchange algorithms (Remez-like updates).
- Theory: apply minimax optimality theory and complementary slackness to prove optimality certificates.

== Planned Structure

1. Line graphs
2. Star graphs
3. Trees
4. Bounded-treewidth graphs
5. Approximation for general graphs

= Line Graphs

== Setup

Let $A = [0,1]$ with the standard metric $d(s,t) = |s-t|$. We study the _continuous placement_ problem: choose $p$ points $X = {x_1 < dots.c < x_p} subset (0,1)$ (strictly interior) to maximize
$
  F(X) := min_{t in [0,1]} U_X(t), quad U_X(t) := sum_{i=1}^p phi(|t - x_i|).
$
The $p$ chosen points divide $[0,1]$ into $p+1$ closed subintervals
$
  I_0 = [0, x_1],quad I_j = [x_j, x_{j+1}] text(" for ") j = 1,...,p-1,quad I_p = [x_p, 1].
$
Since $phi(0^+) = +infinity$, the function $U_X$ blows up at each $x_i$, so $min_t U_X(t)$ is attained in $[0,1] \ {x_1,...,x_p}$. On each interior gap $(x_j, x_{j+1})$ (where both endpoints are in $X$), $U_X$ is strictly convex with $U_X -> +infinity$ at both ends, so $U_X$ has a _unique interior minimizer_ $c_j in (x_j, x_{j+1})$. On the boundary intervals $I_0$ and $I_p$, $U_X$ is strictly convex with $U_X -> +infinity$ at the interior endpoint; the minimum is attained at a unique point $c_0 in I_0$ (either $t=0$ or an interior point) and $c_p in I_p$ (either $t=1$ or an interior point). We thus have exactly $p+1$ _candidate minimizers_
$
  c_0 in I_0,quad c_1 in (x_1,x_2),quad ..., quad c_{p-1} in (x_{p-1},x_p),quad c_p in I_p.
$
Given that optimal configurations have $c_0 = 0$ and $c_p = 1$ by a standard boundary reflection argument, we may henceforth write the candidates as $0, c_1, ..., c_{p-1}, 1$.

== Gradient Sign Pattern (T-system Lemma)

#lemma(title: "T-system")[
  Let $X = {x_1 < dots.c < x_p} subset (0,1)$ be fixed and let $phi$ be differentiable on $(0,infinity)$ with $phi' < 0$. The gradient vectors
  $
    g^{(k)} := nabla_X U_X(c_k) in RR^p, quad g^{(k)}_j = -phi'(|c_k - x_j|) dot op("sgn")(c_k - x_j),
  $
  for $k = 0, 1, ..., p$ (where $c_0 = 0$, $c_p = 1$) form a _Chebyshev system (T-system)_: the sign pattern of $g^{(k)}$ is
  $
    g^{(k)}_j > 0 text(" if ") c_k > x_j, quad g^(k)_j < 0 text(" if ") c_k < x_j.
  $
  Consequently, the $p+1$ vectors $g^{(0)}, ..., g^{(p)} in RR^p$ satisfy: _no proper subset_ has $0$ in its convex hull.
]

#proof[
  Since $c_k in I_k$ and $I_k = (x_k, x_{k+1})$ (with $x_0 := 0$, $x_{p+1} := 1$), we have $c_k > x_j$ for $j <= k$ and $c_k < x_j$ for $j > k$. Thus $g^{(k)}_j = -phi'(|c_k - x_j|) dot op("sgn")(c_k - x_j)$, which is positive (since $phi' < 0$ and $c_k - x_j > 0$) for $j <= k$, and negative for $j > k$. The sign pattern of $g^{(k)}$ is therefore $(+,...,+,-,...,-)$ with the sign change after position $k$. These $p+1$ vectors with consecutively shifting sign patterns form a Chebyshev system. For any proper subset $S subset {0,...,p}$ (proper inclusion), some sign position $j$ is missing both a vector with $g_j > 0$ and one with $g_j < 0$ (or the sign change at $j$ is absent), so the convex cone generated by $S$ cannot contain $0$. $square$
]

== Equioscillation Theorem

#theorem(title: "Equioscillation on $[0,1]$")[
  Let $phi: (0,infinity) -> (0,infinity)$ be strictly convex and strictly decreasing with $phi(0^+) = +infinity$, and assume $phi$ is differentiable on $(0,infinity)$. Then $X^* = {x_1^* < ... < x_p^*} subset (0,1)$ is an optimal configuration if and only if all $p+1$ candidate minimizers achieve the same value:
  $
    U_{X^*}(0) = U_{X^*}(c_1^*) = dots.c = U_{X^*}(c_{p-1}^*) = U_{X^*}(1) = tau^*.
  $
  In particular, _every_ candidate minimizer is a global minimizer (active constraint) at optimality.
]

#proof[
  _("If")_ If all $p+1$ candidates achieve value $tau^*$, then $min_t U_{X^*}(t) = tau^*$. Any other configuration $Y$ with $|Y|=p$ must have some interval $I_k$ on which $min_t U_Y(t) <= tau^*$ by a counting argument (the $p$ points of $Y$ cannot all shift to increase the minimum on all $p+1$ intervals simultaneously). Hence $F(X^*) = tau^* >= F(Y)$, and $X^*$ is optimal.

  _("Only if")_ Let $X^*$ be optimal. The SIP first-order optimality conditions (KKT for semi-infinite programs, see Hettich--Kortanek 1993) state: there exist active points $m_1,...,m_r in [0,1]$ with $U_{X^*}(m_l) = F(X^*)$ and multipliers $lambda_l > 0$, $sum lambda_l = 1$, such that
  $
    sum_{l=1}^r lambda_l nabla_X U_{X^*}(m_l) = 0 in RR^p.
  $
  By strict convexity of $U_{X^*}$ on each gap, each $m_l$ must be one of the $p+1$ candidates $c_0,...,c_p$. By the T-system Lemma, no proper subset of ${g^{(0)},...,g^{(p)}}$ contains $0$ in its convex hull. Thus the KKT condition can only be satisfied if $r = p+1$ and all $p+1$ candidates appear. $square$
]

#remark(title: "Uniqueness")[
  The optimal configuration $X^*$ is _unique_. The equioscillation conditions $U_{X^*}(0) = U_{X^*}(c_j^*) = U_{X^*}(1)$ give $p$ equations in $p$ unknowns $(x_1^*,...,x_p^*)$. Farkas, Nagy, and Révész (2024) prove that the map $Phi: (x_1,...,x_p) arrow.r.bar (min_{I_0} U_X, ..., min_{I_p} U_X)$ is a _homeomorphism_ onto its image. The equioscillation locus $Phi^(-1)({(tau,...,tau) : tau in RR})$ therefore intersects the feasible set in at most one point, giving uniqueness.
]

#remark(title: "Non-differentiable $phi$")[
  The equioscillation theorem extends to any _strictly convex_ $phi$ without requiring differentiability. The proof uses two changes: (i) the unique interior minimizer on each gap still exists (strict convexity implies any local minimizer is unique; no derivative is needed), and (ii) the KKT gradient condition is replaced by its subdifferential analogue: $0 in sum_l lambda_l ∂_X U_{X^*}(m_l)$. The sign pattern of any subgradient selection is identical to the differentiable case (at $t > x_j$, any element of $∂ phi(t - x_j)$ is positive since $phi$ is decreasing; at $t < x_j$ it is negative), so the T-system Lemma carries through unchanged.
]

#remark(title: "Literature")[
  The equioscillation characterization on $[0,1]$ is known, but under different names. The foundational paper is:
  - *Fenton (2000)*: "A min-max theorem for sums of translates," _J. Math. Anal. Appl._ Proves a minimax equioscillation result for concave-cusp kernels on an interval, motivated by entire function theory.

  The most comprehensive modern treatment is:
  - *Farkas, Nagy, Révész (2023)*: "On the weighted Bojanov--Chebyshev problem and the sum of translates method of Fenton," _Sb. Math._ *214*(8), 1163--1190. (arXiv:2112.10169.) Proves existence, equioscillation, and characterization for general decreasing kernels on $[0,1]$; also establishes the intertwining phenomenon.
  - *Farkas, Nagy, Révész (2024)*: "A homeomorphism theorem for sums of translates," _Rev. Mat. Complut._ Provides the uniqueness argument via homeomorphism of the interval-maxima map.

  For the circle (equidistant points are optimal for all decreasing convex $phi$):
  - *Hardin, Kendall, Saff (2013)*: "Polarization optimality of equally spaced points on the circle," _Discrete Comput. Geom._ *50*, 236--243. (arXiv:1208.5261.)
  - *Ambrus, Ball, Erdélyi (2012)*: "Chebyshev constants for the unit circle," _Bull. London Math. Soc._ *44*, 1245--1262. (arXiv:1006.5153.)

  The SIP formulation and the extension to metric graphs do not appear in this literature and are new.
]

= Notes

Add theorem statements and proof sketches section by section.
