#import "../setup/preamble.typ": template
#import "../setup/commands.typ": *
#import "../setup/ctheorems.typ": *

#show: template

#let chapter = [

= Dirichlet model for demand scenarios

Assume a WDN with junction set $J$ and base demand vector
$
  d^0 = (d_j^0)_(j in J),
$
with base total demand
$
  D_0 := sum_(j in J) d_j^0.
$

The generator takes parameters
- $Delta_max > 0$: maximum extra demand,
- $a, b$ with $0 <= a <= b$: min_deviation and max_deviation,
- a random seed.

For each scenario $s$, it samples:

1. A global deviation factor
$
  xi_s ~ U([a,b]).
$

2. A share vector on the simplex
$
  alpha_s ~ "Dirichlet"(1, dots, 1) in Delta^(|J|-1),
$
so $alpha_(s,j) >= 0$ and $sum_(j in J) alpha_(s,j) = 1$.

Then the extra mass and scenario demands are
$
  Delta_s := xi_s Delta_max,
$
$
  d_(s,j) = d_j^0 + alpha_(s,j) Delta_s, quad j in J.
$

Therefore, total demand is exactly
$
  sum_(j in J) d_(s,j) = D_0 + Delta_s in [D_0 + a Delta_max, D_0 + b Delta_max].
$

Important consequence: extra demand is global, not per-node. Because the Dirichlet shares sum to one, the model distributes one global amount $Delta_s$ across all junctions.

Under this symmetric Dirichlet model,
$
  bb(E)[alpha_(s,j)] = 1/abs(J),
$
so the expected added demand at a node is
$
  bb(E)[d_(s,j)-d_j^0] = bb(E)[xi_s] Delta_max / abs(J) = (a+b)/(2|J|) Delta_max.
$

== Sampling algorithm
1. Draw $xi_s ~ U([a,b])$ and set $Delta_s = xi_s Delta_max$.
2. Draw $alpha_s ~ "Dirichlet"(1, dots, 1)$.
3. Set $d_(s,j) = d_j^0 + alpha_(s,j) Delta_s$ for all $j in J$.
4. Run one hydraulic simulation with this demand vector.
5. Store resulting demand/flows/heads triple.

]
#chapter
