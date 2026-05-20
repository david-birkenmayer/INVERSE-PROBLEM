#import "../setup/preamble.typ": template
#import "../setup/commands.typ": *
#import "../setup/ctheorems.typ": *

#show: template

#let chapter = [
  
= The Polarization Problem

#def[Kernel][
  A _kernel_ is a map $K: (0,infinity) -> RR$ s.t:
  + $lim_(x -> 0) K(x) = infinity$
  + $K$ is strictly decreasing and convex
  In the following we will often abuse notation and denote with $K$ its extension to a function $[0,infinity) -> (- infinity, infinity]$ by setting $K(0) = infinity$.
]
Note that kernels are countinuous functions on $(0, infinity)$, and extended continuous on $[0, infinity)$ respectively.
Clearly kernels are closed under multiplication with positive scalars and under additions with decreasing convex functions. The latter includes constants, other kernels, and left-translated kernels, that is, functions of the form $x mapsto K(x+t)$ for some $t>=0$.

The following gives more examples of kernels:
#cor[
  Let $D subs RR$ with $0 in D$ and $g:D -> RR$ be strictly increasing and convex fulfilling 
  $lim_(x -> infinity) g(x) = infinity$.
  Then if $K: (0,infinity) -> D$ is a kernel, so is $g compose K$.
]
#pf[
  Follows from the fact [TODO: ANHANG] that if $g$ is strictly increasing and convex and 
  $K$ is strictly decreasing and convex, then $g compose K$ is also strictly decreasing and convex.
]

The primary example is $g(x)= exp(x)$: Hence $K(x) = exp(1/x^p)$ is a kernel, as well as $K(x)=exp(exp(1/x^p))$ etc.

Here a list of prominent kernels:
$
  K_s (x) &= 1/x^s  && quad quad "Riesz Kernel" \  
  K_log (x) &= log(1/x)  && quad quad "Logarithmic Kernel" \
  K_(exp, s) (x) &= exp(1/x^p)-1  && quad quad "Exponential Riesz Kernel" \
  K_(exp, L) (x) &= exp(log(1/x) - x)  && quad quad "Lower Exponential Kernel" \
  K_(exp, F) (x) &= exp(1/x - x)  && quad quad "Fully Exponential Kernel" \
$


#def[Sum of Translates][
  Let $G$ be a metric graph, $p in NN$ and let $K$ be a kernel. 
  Given $x in Gamma^p$ and $t in Gamma$ define the _pure sum of translates_ as
  $ 
  f(x,t) :=& sum_(i=1)^p K (d(t, x_i)) \
  $
]

#def[Polarization Problem][
  Let $cal(X) subs Gamma^p$ be a set of feasible center placements and $x = (x_1,...,x_p) in cal(X)$ be a vector in that set. Denote with $X :={x_1,...,x_p}$ the corresponding set, and the connected components of $Gamma without X$ as $Gamma_1(x),...,Gamma_k (x)$.\
  The _valleys_ of $f$ are the minima on the connected components:
  $ m_j (x) := min_(t in Gamma_j (x)) f(x,t) quad quad quad "for" j = 1,...,k $
  We further define
  $ 
  underline(m)(x) &:= min_(1<= j <= k) m_j (x) = inf_(t in Gamma) f(x,t) \
  overline(m)(x) &:= max_(1<= j <= k) m_j (x)
  $
  We call $j$ _active_ if $m_j (x) = underline(m)(x)$, that is, if the minimum is attained on $Gamma_j$. Clearly $underline(m)(x) <= overline(m)(x)$ and equality occurs if and only if all $j$ are active. If equality occurs, we call $x$ an _equioscillation point_.
  Now we can define the problems we are interested in:
  $ 
  m(cal(X)) &:= sup_(x in cal(X)) underline(m)(x) &= sup_x inf_j m_j (x) & quad quad ("Polarization problem") \
  M(cal(X)) &:= inf_(x in cal(X)) overline(m)(x) &= inf_x sup_j m_j (x) & quad quad ("dual Polarization problem")
  $
]

We are mainly interested in computing the polarization problem for $cal(X) = Gamma^p$, but it is also interesting to consider the problem for smaller sets, especially finite sets, like for example $V$.

In the following we always assume that $Gamma$ is induced by a tree. This is a natural assumption, since the presence of cycles would allow for more complicated behavior of the sum of translates function, losing important properties like convexity on certain subsets -- more on this later.


]
#chapter

