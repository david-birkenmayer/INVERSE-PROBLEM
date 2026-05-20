#import "../setup/preamble.typ": template
#import "../setup/commands.typ": *
#import "../setup/ctheorems.typ": *

#show: template

#let chapter = [

= Equioscillation

To prove equioscillation necessity, we first explain why a single global perturbation direction is too weak, and then build a local first-order framework based on active valleys.

== Why the naive perturbation can fail

#obs[Pertubation argument fails on trees][
	Consider a star with $n>=2$ arms of length $1+c$. Place $n$ centers so that each center lies on one arm at distance $1$ from the midpoint.

	Consider a perturbation that moves the center on arm $i$ outward with speed $eps_i>0$.
	Fix the leaf $y$ on arm $j$ where $eps_j = min_(1 <= i <= n) eps_i =: eps$.
  Relative to $y$, one center has distance $c$ and the other $n-1$ centers have distance $2+c$.
	Thus, when perturbing, the value at $y$ is
	$
		g(t) = 1/(c-t eps) + sum_(i != j) 1/(2+c+t eps_i).
	$
	Differentiate at $t=0$:
	$
		g'(0) = eps/c^2 - sum_(i != j) eps_i/(2+c)^2.
	$
	Since $eps_i >= eps$ for all $i$, we get
	$
		g'(0) <= eps (1/c^2 - (n-1)/(2+c)^2).
	$
	Hence a sufficient condition for $g'(0)<0$ is
	$
		(n-1)c^2 > (2+c)^2.
	$
	For example, for $(n,c)=(6,6)$:
	$
		g'(0) <= eps (1/36 - 5/64) = - 29 / 576 eps < 0,
	$
	so $g$ is decreasing near $0$.

	Therefore, for such parameters, every outward perturbation decreases the value at at least one leaf. This shows that this perturbation template cannot prove equioscillation necessity by itself.
]

The key lesson is that we need a _local active-set condition_ at an optimizer, not a fixed geometric push direction.

== Local setup

#def[Valley functions and active set][
	Let $x in Gamma^p$ be a regular configuration (all centers in edge interiors, and the component decomposition of $Gamma without X$ is locally stable).
	Write the components as $Gamma_1 (x), ..., Gamma_k (x)$ and define
	$
		m_j (x) := min_(t in Gamma_j (x)) f(x,t),
		quad
		Phi(x) := min_(1 <= j <= k) m_j (x).
	$
	The active set is
	$
		A(x) := {j in {1,...,k}: m_j (x)=Phi(x)}.
	$
]

From Chapter 4 (continuity and convexity), each valley map $m_j$ is continuous on a regularity neighborhood where labels are fixed.

To obtain a first-order condition, we now use two standard external results.

#lem[Danskin's Theorem][
	Let $U subs RR^p$ be open, let $psi_1,...,psi_k in C^1(U)$, and define
	$
		Psi(x) := min_(1 <= j <= k) psi_j (x),
		quad
		A(x):=arg min_(1 <= j <= k) psi_j (x).
	$
	For $x in U$ and $h in RR^p$, define the one-sided directional derivative by
	$
		partial_h Psi(x) := lim_(tau -> 0^+) (Psi(x+tau h)-Psi(x))/tau.
	$
	Then for every $x in U$ and direction $h in RR^p$,
	$
		partial_h Psi(x) = min_(j in A(x)) (nabla psi_j (x) dot h).
	$
]
#pf[
	This is the minimum-analogue of the standard max-envelope directional derivative formula (often called a Danskin-type formula). It follows immediately by applying the max formula to $-Psi = max_j (-psi_j)$; see @rockafellarwets1998 (variational analysis, max/min envelope rules).
]

#lem[Fermat rule via directional derivatives][
	Let $Psi: U -> RR$ and $x^* in U$. If $x^*$ is a local maximizer of $Psi$ and all one-sided directional derivatives exist at $x^*$, then
	$
		partial_h Psi(x^*) <= 0 quad "for all" h in RR^p.
	$
]
#pf[
	For fixed $h$, local maximality gives $Psi(x^*+tau h) <= Psi(x^*)$ for all sufficiently small $tau >0$. Divide by $tau$ and let $tau -> 0^+$.
]

#lem[Active-gradient convex-hull balance][
	In the setting above, suppose each active valley map is $C^1$ near $x^*$ and $x^*$ is a local maximizer of $Phi = min_j m_j$.
	Then
	$
		0 in op("conv")({nabla m_j (x^*): j in A(x^*)}).
	$
]
#pf[
	By the two previous lemmas,
	$
		0 >= partial_h Phi(x^*) = min_(j in A(x^*)) (nabla m_j (x^*) dot h)
	$
	for every $h in RR^p$.

	Assume for contradiction that
	$
		0 nin op("conv")({nabla m_j (x^*): j in A(x^*)}).
	$
	By strict separation of a point and a compact convex set (Hahn--Banach separation theorem, see @rockafellar1970), there exists $h in RR^p$ such that
	$
		nabla m_j (x^*) dot h > 0
	$
	for all $j in A(x^*)$.
	Hence
	$
		min_(j in A(x^*)) (nabla m_j (x^*) dot h) > 0,
	$
	contradicting $partial_h Phi(x^*) <= 0$.
]

== Why this is the right bridge to equioscillation

The lemma above is the precise replacement for the failed global perturbation argument. It says:

+ At a local optimizer, active valleys must be in first-order balance.
+ This balance is encoded by a convex-hull condition on active gradients.

Therefore, to prove equioscillation necessity, it suffices to prove a geometric statement on trees showing that this balance is impossible unless all components are active.

One convenient formulation is:

#lem[Reduction of equioscillation to a geometric nondegeneracy check][
	Assume $x^*$ is a regular local maximizer of $Phi$ and active valley maps are $C^1$ near $x^*$.
	If every strict subset $B subs {1,...,k}$ violates the balance condition
	$
		0 nin op("conv")({nabla m_j (x^*): j in B}),
	$
	then necessarily $A(x^*) = {1,...,k}$, i.e. $x^*$ is equioscillating.
]
#pf[
	The previous lemma gives
	$
		0 in op("conv")({nabla m_j (x^*): j in A(x^*)}).
	$
	By assumption this can only happen when $A(x^*)$ is the full index set.
]

This is where tree geometry should enter next: identify the structure of $nabla m_j$ and prove the stated nondegeneracy.
We move a detailed line-case sanity check (including $p=1$, $p=2$, and arbitrary-$p$ balance equations) to Chapter 6.

]
#chapter
