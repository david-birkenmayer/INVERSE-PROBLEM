#import "../setup/preamble.typ": template
#import "../setup/commands.typ": *
#import "../setup/ctheorems.typ": *

#show: template

#let chapter = [

= The line case

In this chapter we assume the simplest case: $G$ only consists of two vertices connected by an edge, that is, homeomorphic to the unit interval [0,1]. Thus we can identify $cal(V)$ with the unit interval and $d$ with the usual euclidean distance. In this case, the connected components of $cal(V) without {x_1,...,x_p}$ are just the intervals between the points $x_i$ and the endpoints 0 and 1.

We can formulate the problem as the following semi-finite program (SIP):
$
  max quad& m \
  s.t. quad & m <= F(x,t) "for all" t in [0,1]\
  & m in RR \
  & x in [0,1]^p
$

The following strong optimality criterion was shown (under stronger assumptions) in @fenton and later generalized in @FNR23.

#thm[Optimality criterion for Polarization on the Line][
  let $K_i$ be a kernels. Then the following are equivalent for $x in [0,1]^p$:
  + $x$ is optimal for the polarization problem 
  + $x$ is optimal for the dual polarization problem
  + $x$ is an equioscillation point
  The optimizer is unique up to permutation, that is, it is unique after imposing $x_1 < ... < x_p$. It fulfills $x_i != x_i'$ for all $i != i'$.
]

In fact, this result was not only shown for the "pure" sum of translates function $f$, but also for the more general weighted sum of translates
$
  F(x,t) :=& J(t) + sum_(i=1)^p r_i K (d(t, x_i))
$
which involves scaling kernels by scalars $r_i>0$ and adding an arbitary field function $J: Gamma -> (-infinity, infinity]$, which has to be real-valued at at least $p$ points.
The @FNR23 paper also shows that the optimality criterion fails if the kernel is not strictly decreasing or not convex, so the assumptions on $K$ are necessary for the result to hold. Note that they have swapped signs and are working with a minimization problem, so their assumptions on $K$ are that it is strictly increasing and concave, which is equivalent to our assumptions.

]
#chapter
