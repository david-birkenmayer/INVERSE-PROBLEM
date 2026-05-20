#import "../setup/preamble.typ": template
#import "../setup/commands.typ": *
#import "../setup/ctheorems.typ": *

#show: template

#let chapter = [

= Vertex-discrete Polarization

In this chapter we restrict center locations to vertices. Let
$
	cal(X)_c := Gamma^p,
	quad
	cal(X)_v := V^p.
$
The corresponding optimal values are
$
	m_c := m(cal(X)_c),
	quad
	m_v := m(cal(X)_v).
$
Since $V^p subs Gamma^p$, we always have
$
	m_v <= m_c.
$

== Naive rounding of a continuous optimizer

Let $x^* = (x_1^*,...,x_p^*) in Gamma^p$ be an optimizer of the continuous problem, i.e. $underline(m)(x^*)=m_c$.
For each center choose a nearest vertex
$
	hat(x)_i in arg min_(v in V) d(x_i^*, v),
$
and write $hat(x)=(hat(x)_1,...,hat(x)_p) in V^p$.
Define
$
	ell_(max) := max_(e in E) ell(e),
	quad
	delta := max_i d(x_i^*, hat(x)_i).
$
By construction of metric graphs,
$
	delta <= ell_(max)/2.
$

#lem[Pointwise distance perturbation bound][
For every $y in Gamma$ and each $i$,
$
	|d(y,x_i^*) - d(y,hat(x)_i)| <= delta.
$
]
#pf[
Triangle inequality gives
$
	d(y,hat(x)_i) <= d(y,x_i^*) + d(x_i^*,hat(x)_i) <= d(y,x_i^*) + delta,
$
and symmetrically
$
	d(y,x_i^*) <= d(y,hat(x)_i) + delta.
$
]

The bound above is purely geometric. To convert it into an objective-value estimate, we use one more quantity.

#def[Rounding clearance][
Let $hat(y)$ be a minimizer of $y mapsto f(hat(x),y)$ and define
$
	rho(hat(x)) := min_(1<=i<=p) d(hat(y), hat(x)_i).
$
]

Note that $rho(hat(x))>0$ because $K(0)=infinity$ and minimizers of $f(hat(x), dot)$ cannot occur at centers.

#thm[A-posteriori rounding estimate][
Assume $K in C^1(0,infinity)$, strictly decreasing and convex.
If $rho(hat(x)) > delta$, then
$
	0 <= m_v - underline(m)(hat(x)) <= m_c - underline(m)(hat(x))
	<= p (-K'(rho(hat(x)) - delta)) delta.
$
Using $delta <= ell_(max)/2$,
$
	m_v - underline(m)(hat(x))
	<= -p K'(rho(hat(x)) - ell_(max)/2) ell_(max)/2
$
whenever $rho(hat(x)) > ell_(max)/2$.
]
#pf[
The first inequality is trivial, and the second follows from $m_v <= m_c$.
For the third, let $hat(y) in arg min_y f(hat(x),y)$, so
$
	underline(m)(hat(x)) = f(hat(x),hat(y)).
$
Since $m_c = inf_y f(x^*,y) <= f(x^*,hat(y))$,
$
	m_c - underline(m)(hat(x))
	<= f(x^*,hat(y)) - f(hat(x),hat(y))
	= sum_(i=1)^p [K(d(hat(y),x_i^*)) - K(d(hat(y),hat(x)_i))].
$
By the previous lemma,
$
	|d(hat(y),x_i^*) - d(hat(y),hat(x)_i)| <= delta.
$
Also $d(hat(y),hat(x)_i) >= rho(hat(x))$, hence
$
	d(hat(y),x_i^*) >= rho(hat(x)) - delta > 0.
$
Therefore both radii lie in $[rho(hat(x))-delta, infinity)$.
By the mean-value theorem, $|K(r_1)-K(r_2)| <= sup_(r >= rho(hat(x))-delta) |K'(r)| |r_1-r_2|$.
Since $K$ is convex, $K'$ is increasing, so $-K'$ is decreasing and this supremum is attained at $r = rho(hat(x))-delta$:
$
	|K(r_1)-K(r_2)| <= -K'(rho(hat(x))-delta) dot |r_1-r_2|
	<= -K'(rho(hat(x))-delta) dot delta.
$
Summing over $i=1,...,p$ gives
$
	m_c - underline(m)(hat(x)) <= -K'(rho(hat(x))-delta) dot p delta.
$
Finally use $delta <= ell_(max)/2$.
]

#cor[Power kernel $K(r)=1/r^s$][
For $K(r)=1/r^s$ with $s>0$, we have $-K'(r)=s/r^(s+1)$ and thus
$
	m_v - underline(m)(hat(x))
	<= p s delta / (rho(hat(x)) - delta)^(s+1)
	<= p s (ell_(max)/2) / (rho(hat(x)) - ell_(max)/2)^(s+1)
$
whenever $rho(hat(x)) > ell_(max)/2$.
]

#cor[Log kernel $K(r)=log(1/r)$][
For $K(r)=log(1/r)$, $-K'(r)=1/r$, so
$
	m_v - underline(m)(hat(x))
	<= p delta / (rho(hat(x)) - delta)
	<= p (ell_(max)/2) / (rho(hat(x)) - ell_(max)/2)
$
whenever $rho(hat(x)) > ell_(max)/2$.
]

== Interpretation
+ The geometric discretization scale enters only through $ell_(max)$.
+ The sensitivity enters through the singularity strength of $K$ near $0$
+ The key stability parameter is the clearance $rho(hat(x))$: if valley minimizers for the rounded solution stay away from centers, naive rounding is quantitatively close to vertex-discrete optimality.

A priori estimate needs a lower bound on $rho(hat(x))$, which is not available in general.

]
#chapter
