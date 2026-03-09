= Inverse WDN Optimization Model (Typst)

This document summarizes the current inverse problem formulation implemented in the codebase.

== Sets and indices
- $G=(V,E)$: directed graph of the water distribution network
- $w in V$: reservoir node
- $J = V setminus {w}$: junction nodes
- $E_1 subset.eq E$: primary pipes
- $E_2 subset.eq E$: secondary pipes, $E_1 cap E_2 = emptyset$, $E_1 cup E_2 = E$
- $S subset.eq V$: measurement nodes with known head

== Parameters
- $h_w$: reservoir head (known)
- $h_s$: measured heads for $s \in S$
- $C_e >= 0$: minimum flow magnitude for primary pipes
- $c_e >= 0$: maximum flow magnitude for secondary pipes
- $r_e > 0$: Darcy–Weisbach resistance per pipe

== Decision variables
- $q_e$: flow on pipe $e in E$
- $h_v$: head at node $v in J$
- $d_v <= 0$: demand at node $v in J$ (negative = consumption)

== Hydraulic constraints

=== Mass balance (junctions)
For each junction $v \in J$:

$sum_(e in delta^-(v)) q_e - sum_(e in delta^+(v)) q_e = d_v$

=== Headloss (Darcy–Weisbach)
For each pipe $e=(u,v) \in E$:

$h_u - h_v = r_e |q_e| q_e$

=== Fixed heads (reservoir + measurements)
- $h_w$ is fixed at the reservoir
- $h_s$ is fixed for each $s \in S$

== Pipe bounds

Primary pipes:

$|q_e| >= C_e, e in E_1$

Secondary pipes:

$|q_e| <= c_e, e in E_2$

== Global energy bound
Using the reservoir head $h_w$ and total demand $d_w = -sum_(v in J) d_v$:

$sum_(e in E) r_e |q_e|^3 <= h_w d_w$

== Inverse problem objectives
The code currently solves *separate* bound problems for each junction $v$:

- Minimum demand at node $v$:
  $min d_v$ subject to all constraints

- Maximum demand at node $v$:
  $max d_v$ subject to all constraints

This yields demand bounds $[d_v^min, d_v^max]$.

== Notes
- Demands are negative for consumption (no supply at junctions).
- Reservoir is modeled as a fixed-head boundary (no demand variable).
- The solver uses nonlinear constraints with Darcy–Weisbach headloss.
