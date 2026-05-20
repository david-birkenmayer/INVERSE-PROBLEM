#import "../setup/preamble.typ": template
#import "../setup/commands.typ": *
#import "../setup/ctheorems.typ": *

#show: template

#let chapter = [

= NP-hardness

#def[vanishing Kernels][
	A Kernel $K$ is _vanishing_ if there exists an $x>0$ such that $K(y) = 0$ for all $y >= x$.
	We call $xi := inf {x > 0 : K(x) = 0}$ the _vanishing point_.
]
#thm[
	The vertex-discrete polarization with a vanishing kernel problem is NP-hard, and it is not possible to approximate it within any factor, unless P=NP.
]
#pf[
	Assume w.l.o.g. that $xi = 3/2$, otherwise $ell$ can be rescaled by $1/xi$. Further let $kappa = K(1)$.

	We carry out a reduction from #problem-name([Dominating Set]) on 3-regular graphs, which is known to be NP-complete. We show that if a 3-regular graph $G=(V,E)$ has a dominating set of size $p$, then the continuous polarization problem with on the graph $Gamma$ induced by $G$ with edge lengths $ell = kappa$ has a solution with objective value $m >= kappa$. On the other hand we show that if $G$ does not have a dominating set of size $p$, then the continuous polarization problem on $Gamma$ has objective value $0$.
	
	Claim: For any $x$ we have $f(x, y) >= kappa$ if and only if $f(x, v) >= kappa$ holds for all $v in V$.

	Proof of claim: If $f(x, v) >= kappa$ for all $v in V$, then for any $y in Gamma$ there is a vertex $v$ with $d(y,v) <= 1$ (since $Gamma$ is the metric graph induced by $G$ with edge lengths $kappa < 3/2$), hence

	Claim: There always exists an optimal solution with centers placed on vertices, and that we can obtain such a solution by rounding centers to their nearest vertex.

]


]
#chapter
