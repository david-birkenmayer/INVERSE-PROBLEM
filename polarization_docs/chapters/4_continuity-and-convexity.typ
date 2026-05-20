#import "../setup/preamble.typ": template
#import "../setup/commands.typ": *
#import "../setup/ctheorems.typ": *

#show: template

#let chapter = [

= Continuity and convexity

Trees have the special property that there is a unique simple path between any two points. In turn, the geodesic curves are also unique and coincide with these simple paths. This has the following important consequence:

#lem[
  Let $scp(e,t)$ and $scp(e,t')$ be two different points on the same edge $e$ and let $gamma: [0,L] -> Gamma$ be the unique geodesic between them, parametrized as $gamma(s) = scp(e, (1 - s/L) t + (s/L) t')$, where $L = |t-t'|$. Then the function $phi_y: [0,L] -> RR_(> 0), s mapsto d(gamma(s), y)$ is affine and strictly monotone for any fixed $y in Gamma$ that does not lie on $e$.
]
#pf[
  Assume $d(scp(e,t), y) > d(scp(e,t'), y)$. By uniqueness of paths in trees, the geodesic $eta$ from $scp(e,t)$ to $y$ must pass through $scp(e,t')$ (otherwise there would be two distinct paths from $scp(e,t)$ to $y$). Thus the geodesic $eta'$ from $scp(e,t')$ to $y$ is the tail-restriction of $eta$, hence for any $s in [0,L]$:
  $
  phi_y (s) = d(gamma(s), y) = d(gamma(s), scp(e,t')) + d(scp(e,t'),y) = (L-s) + d(scp(e,t'),y),
  $
  which is affine and strictly decreasing in $s$. If instead $d(scp(e,t), y) < d(scp(e,t'), y)$, the same argument with roles swapped shows $phi_y$ is affine and strictly increasing.
] 


This does not hold for graphs with cycles, as the following picture shows: TODO
\ \

The previous lemma has the following consequence for the sum of translates function:

#lem[Convexity in the second component][
  For fixed $x in Gamma^p$, the function $f(x,dot)$ is strictly convex on each connected component of $Gamma without (X cup V)$.
]
#pf[
  Any two points in the same connected component of $Gamma without (X cup V)$ lie on the same edge interval. Let $gamma: [0,L] -> Gamma$ be the unique geodesic between them along that edge.
  
  By the previous lemma, for each $i in {1,...,p}$, the function $phi_(x_i): [0,L] -> RR_(>0), s mapsto d(gamma(s), x_i)$ is affine and strictly monotone.
  Since $K$ is strictly convex and decreasing, the composition $K compose phi_(x_i)$ is strictly convex. The function $s mapsto f(x, gamma(s)) = sum_(i=1)^p K(phi_(x_i)(s))$ is a sum of strictly convex functions, hence strictly convex on $[0,L]$.
] 



Next, a continuity result:

#cor[Continuity of edge-minima][
  The function $x mapsto min_t f(x,t)$ is continuous on each connected component of $Gamma without (X cup V)$
]

It is an immediate consequence of the following standard result:

#lem[Minimum over a Fixed Compact Set is Continuous][
  Let $X$ be a metric space, $T$ a compact metric space, and $f: X times T -> (-infinity, infinity]$ a continuous function. Define
  $
    v(x) := min_(t in T) f(x,t).
  $
  Then $v: X -> RR$ is continuous.
]

#pf[
  Let $x_n -> x$. Choose $t_n in T$ with $v(x_n) = f(x_n,t_n)$.
  Since $T$ is compact, after passing to a subsequence we may assume $t_n -> t_* in T$.
  By continuity of $f$,
  $
    liminf_(n->infinity) v(x_n)
    = liminf_(n->infinity) f(x_n,t_n)
    = f(x,t_*)
    >= v(x).
  $
  On the other hand, for any fixed minimizer $hat(t) in arg min_(t in T) f(x,t)$,
  $
    limsup_(n->infinity) v(x_n)
    <= limsup_(n->infinity) f(x_n,hat(t))
    = f(x,hat(t))
    = v(x).
  $
  Hence $v(x_n) -> v(x)$, so $v$ is continuous. $square$
]


]
#chapter