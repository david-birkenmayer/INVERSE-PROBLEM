#import "../setup/preamble.typ": template
#import "../setup/commands.typ": *
#import "../setup/ctheorems.typ": *

#show: template

#let chapter = [

= Experimenting with the First-Order Condition

The active-set gradient balance condition (from the previous chapter) provides a systematic framework, but it is not clear whether it forces full equioscillation for general kernels. This chapter experiments with two specific kernels to test the strength of this criterion.

== Sanity check: line case

We now test the criterion on $Gamma=[0,1]$ with ordered centers
$
	0 < x_1 < ... < x_p < 1.
$
Let $I_0=[0,x_1)$, $I_j=(x_j,x_(j+1))$ for $1<=j<=p-1$, and $I_p=(x_p,1]$.
For $j=1,...,p-1$, strict convexity in the $t$-variable gives a unique minimizer
$
	t_j (x) in I_j,
	quad
	m_j (x)=f(x,t_j (x)).
$
At the boundary components,
$
	m_0 (x)=f(x,0),
	quad
	m_p (x)=f(x,1).
$

#lem[Gradient sign pattern on the line][
	Assume the valley maps are differentiable at $x$. Then:
	$
		partial_(x_i) m_0 (x) = K'(x_i) < 0,
		quad
		partial_(x_i) m_p (x) = -K'(1-x_i) > 0.
	$
	For $j in {1,...,p-1}$,
	$
		partial_(x_i) m_j (x)
		=
		cases(
			-K'(t_j (x)-x_i) > 0 & "if" i<=j,
			K'(x_i-t_j (x)) < 0 & "if" i>j,
		).
	$
	Hence $nabla m_j$ has a single sign change: positive entries up to index $j$, negative entries after index $j$.
]
#pf[
	For $m_0$ and $m_p$, differentiate $f(x,0)$ and $f(x,1)$ directly.
	For interior $j$, apply the envelope identity
	$
		partial_(x_i) m_j (x) = partial_(x_i) f(x,t_j (x)),
	$
	which is valid because $t_j (x)$ is an interior minimizer of $t mapsto f(x,t)$, so $partial_t f(x,t_j (x))=0$. Then differentiate $K(abs(t_j-x_i))$ with respect to $x_i$.
]

#lem[Endpoint activity is forced][
	Let $x^*$ be a regular local maximizer of $Phi(x)=min_j m_j (x)$ on the line, and assume the active valley maps are differentiable at $x^*$.
	Then $0 in A(x^*)$ and $p in A(x^*)$.
]
#pf[
	If $0 nin A(x^*)$, then every active index satisfies $j>=1$.
	Take direction $h=e_1$. By the previous lemma, for every active $j>=1$ we have
	$
		nabla m_j (x^*) dot h = partial_(x_1) m_j (x^*) > 0.
	$
	Hence
	$
		partial_h Phi(x^*) = min_(j in A(x^*)) (nabla m_j (x^*) dot h) > 0,
	$
	contradicting the Fermat directional condition.

	If $p nin A(x^*)$, then every active index satisfies $j<=p-1$.
	Take direction $h=-e_p$. For every such active $j$,
	$
		nabla m_j (x^*) dot h = -partial_(x_p) m_j (x^*) > 0
	$
	because $partial_(x_p) m_j (x^*)<0$ for $j<=p-1$. Again $partial_h Phi(x^*)>0$, contradiction.
]

#cor[Complete check for one center][
	For $p=1$ on the line, every regular local maximizer is equioscillating.
]
#pf[
	There are only two valleys, indices $0$ and $1$. By the previous lemma both must be active.
]

#lem[The case $p=2$][
	Assume $0 < x_1 < x_2 < 1$ and write $q := -K' > 0$.
	Then the three valley gradients are
	$
		
		g_0 &:= nabla m_0 (x) = (-q(x_1), -q(x_2)),
		\
		g_1 &:= nabla m_1 (x) = (q(d), -q(d)),
		\
		g_2 &:= nabla m_2 (x) = (q(1-x_1), q(1-x_2)),
	$
	where $d = (x_2-x_1)/2$ is the midpoint distance in the middle interval.
	If $x^*$ is a regular local maximizer, then $0 in op("conv")({g_j: j in A(x^*)})$.
	In particular, the endpoint-only active set ${0,2}$ is feasible if and only if
	$
		q(x_1)/q(1-x_1) = q(x_2)/q(1-x_2).
	$
]
#pf[
	The formula for $g_0$ and $g_2$ comes from direct differentiation of the boundary valleys.
	For the middle valley, the unique minimizer on $(x_1,x_2)$ is the midpoint $t=(x_1+x_2)/2$, because the two summands are symmetric and their sum is strictly convex.

	Now suppose only the endpoints are active.
	Then there exists $lambda in [0,1]$ such that
	$
		lambda g_0 + (1-lambda) g_2 = 0.
	$
	Comparing coordinates gives
	$
		lambda q(x_1) = (1-lambda) q(1-x_1),
		quad
		lambda q(x_2) = (1-lambda) q(1-x_2).
	$
	Eliminating $lambda$ yields the ratio condition.
]

#obs[What the p=2 check shows][
	The active-gradient criterion is already more rigid than mere endpoint activity: for $p=2$ it reduces the remaining obstruction to a single ratio identity involving $q=-K'$. 
	So the criterion does not settle the line case by sign considerations alone; it needs a kernel-specific monotonicity input on that ratio to force the middle valley to be active as well.
]

The previous corollary settles $p=1$. We now write the arbitrary-$p$ consequence in explicit algebraic form.

#lem[Arbitrary-$p$ balance equations on the line][
	Let $x^*$ be a regular local maximizer on $[0,1]$ and let $A:=A(x^*)$.
	Then there exist weights $lambda_j >=0$ ($j in A$), with $sum_(j in A) lambda_j = 1$, such that
	$
		sum_(j in A) lambda_j nabla m_j (x^*) = 0.
	$
	Equivalently, for each coordinate $i=1,...,p$,
	$
		sum_(j in A, j>=i) lambda_j a_(i,j)
		=
		sum_(j in A, j<i) lambda_j b_(i,j),
	$
	where all coefficients are strictly positive and given by
	$
		a_(i,j) := partial_(x_i) m_j (x^*) quad (j>=i),
		quad
		b_(i,j) := -partial_(x_i) m_j (x^*) quad (j<i).
	$
]
#pf[
	The existence of $(lambda_j)$ is exactly the convex-hull condition
	$
		0 in op("conv")({nabla m_j (x^*): j in A}).
	$
	Taking the $i$-th coordinate gives
	$
		sum_(j in A) lambda_j partial_(x_i) m_j (x^*) = 0.
	$
	By the sign pattern lemma, $partial_(x_i) m_j (x^*)>0$ for $j>=i$ and $partial_(x_i) m_j (x^*)<0$ for $j<i$, yielding the displayed equality with positive coefficients.
]

#cor[What must be checked for general $p$][
	For a proposed active set $A subs {0,...,p}$, the first-order criterion is equivalent to feasibility of the above weighted balance system.
	In particular, if for some $i$ one side of the split is empty,
	$
		A cap {0,...,i-1} = emptyset
		quad "or" quad
		A cap {i,...,p} = emptyset,
	$
	then $A$ is impossible.
]
#pf[
	If one side is empty, one side of the equality in the previous lemma is $0$ while the other is a sum of strictly positive terms, contradiction.
]

So for arbitrary $p$, the criterion gives a concrete finite-dimensional test: determine whether there are nonnegative weights balancing all $p$ coordinate equations.
This is a meaningful reduction, but by itself it does _not yet_ force full equioscillation for every $p$ without an extra structural argument on the coefficients $a_(i,j), b_(i,j)$.
That extra step is exactly the remaining nondegeneracy part.

== The Reciprocal Kernel: $K(x) = 1/x$

We analyze the p=2 case on $[0,1]$ with centers $0 < x_1 < x_2 < 1$ and reciprocal kernel $K(x) = 1/x$.

#def[The three valley minima][
	The sum-of-translates objective is
	$
		f(x,t) = K(|t-x_1|) + K(|t-x_2|) = 1/(abs(t-x_1)) + 1/(abs(t-x_2)).
	$
	The three valleys $I_0=[0,x_1)$, $I_1=(x_1,x_2)$, $I_2=(x_2,1]$ have minima:
	$
		m_0(x) &= f(x,0) = 1/x_1 + 1/x_2 \
		m_1(x) &= f(x, t_1^*(x)), quad t_1^*(x) = (x_1+x_2)/2 \
		m_2(x) &= f(x,1) = 1/(1-x_1) + 1/(1-x_2).
	$
]
#pf[
	For $I_1$: $f(x,t) = 1/(t-x_1) + 1/(x_2-t)$ for $t in (x_1, x_2)$.
	Set $partial_t f = 0$:
	$
		-1/(t-x_1)^2 + 1/(x_2-t)^2 = 0 quad => quad (t-x_1)^2 = (x_2-t)^2.
	$
	Since $t$ is interior, $t-x_1 = x_2-t$, giving $t^* = (x_1+x_2)/2$.
	At this point,
	$
		m_1(x) = 1/((x_2-x_1)/2) + 1/((x_2-x_1)/2) = 4/(x_2-x_1).
	$
]

#obs[Gradient computation][
	For the boundary valleys, direct differentiation gives
	$
		nabla m_0 &= partial_x (1/x_1 + 1/x_2) = (-1/x_1^2, -1/x_2^2), \
		nabla m_2 &= partial_x (1/(1-x_1) + 1/(1-x_2)) = (1/(1-x_1)^2, 1/(1-x_2)^2).
	$
	For the interior valley $m_1$, the envelope theorem gives
	$
		partial_(x_i) m_1(x) = partial_(x_i) f(x, t_1^*(x)) = -partial_(t_i) K(|t_1^*(x)-x_i|).
	$
	Since $t_1^*=(x_1+x_2)/2$, we have $t_1^*-x_1 = (x_2-x_1)/2 =: d$ and $x_2-t_1^* = d$.
	Thus
	$
		partial_(x_1) m_1(x) &= -partial_(x_1) K(d) = 1/d^2, \
		partial_(x_2) m_1(x) &= -partial_(x_2) K(d) = -1/d^2.
	$
	So $nabla m_1 = (-1/d^2, 1/d^2)$ where $d = (x_2-x_1)/2$.
]

#lem[Nondegeneracy: $A = \{0,2\}$ is impossible][
	For a local maximizer, if only endpoints were active, there would exist $lambda_0, lambda_2 >= 0$ with $lambda_0 + lambda_2 = 1$ such that
	$
		lambda_0 nabla m_0 + lambda_2 nabla m_2 = 0.
	$
	Coordinate 1: $lambda_0 (-1/x_1^2) + lambda_2 (1/(1-x_1)^2) = 0$.
	This gives
	$
		lambda_2 / lambda_0 = (1/x_1^2) / (1/(1-x_1)^2) = ((1-x_1)/x_1)^2.
	$
	Coordinate 2: $lambda_0 (-1/x_2^2) + lambda_2 (1/(1-x_2)^2) = 0$.
	This gives
	$
		lambda_2 / lambda_0 = ((1-x_2)/x_2)^2.
	$
	For both equations to hold simultaneously,
	$
		((1-x_1)/x_1)^2 = ((1-x_2)/x_2)^2 quad ==> quad (1-x_1)/x_1 = (1-x_2)/x_2 quad ==> quad x_1 = x_2
	$
	
	For generic center positions, the two ratios differ, so no such $lambda_0, lambda_2$ exist. Therefore $A = {0,2}$ is infeasible, and $m_1$ must be active at any local maximizer.
]

#cor[Reciprocal kernel: full equioscillation on the line][
	For $K(x)=1/x$ and $p=2$ on $[0,1]$, every regular local maximizer is fully equioscillating: $A(x^*)={0,1,2}$.
]

#obs[Why this works][
	The reciprocal kernel's special structure—that reciprocals of symmetric intervals around the midpoint are identical—forces the boundary gradients to have a very specific ratio structure. This ratio structure is rigid enough that it cannot balance without the interior valley's contribution.
]

== The Logarithmic Kernel: $K(x) = log(x)$

We now analyze the same setup with $K(x) = log(x)$, which has $K'(x) = 1/x$.

#def[The three valley minima (log case)][
	The objective is
	$
		f(x,t) = log(abs(t-x_1)) + log(abs(t-x_2)).
	$
	The minima are:
	$
		m_0(x) &= log(x_1) + log(x_2), \
		m_1(x) &= 2 log((x_2-x_1)/2), \
		m_2(x) &= log(1-x_1) + log(1-x_2).
	$
]
#pf[
	Interior minimizer: On $(x_1,x_2)$, $partial_t f = 1/(t-x_1) - 1/(x_2-t) = 0$ gives $t^*=(x_1+x_2)/2$. Evaluating yields $m_1(x) = 2 log((x_2-x_1)/2)$.
	
	Boundary minima follow by substitution.
]

#obs[Gradient computation (log case)][
	$
		nabla m_0 &= (1/x_1, 1/x_2), \
		nabla m_1 &= partial_x (2 log((x_2-x_1)/2)) = (-2/(x_2-x_1), 2/(x_2-x_1)), \
		nabla m_2 &= (-1/(1-x_1), -1/(1-x_2)).
	$
]

#lem[Nondegeneracy: $A=\{0,2\}$ is impossible (log case)][
	Suppose $lambda_0, lambda_2 >= 0$ with $lambda_0 + lambda_2=1$ satisfy $lambda_0 nabla m_0 + lambda_2 nabla m_2 = 0$.
	
	Coordinate 1:
	$
		lambda_0 (1/x_1) - lambda_2 (1/(1-x_1)) = 0,
		quad "hence" quad
		lambda_2/lambda_0 = (1-x_1)/x_1.
	$
	Coordinate 2 gives
	$
		lambda_0 (1/x_2) - lambda_2 (1/(1-x_2)) = 0,
		quad "hence" quad
		lambda_2/lambda_0 = (1-x_2)/x_2.
	$
	Therefore we would need
	$
		(1-x_1)/x_1 = (1-x_2)/x_2,
	$
	which implies $x_1=x_2$, impossible for a regular configuration with $x_1 < x_2$. Contradiction.
	
	Therefore $A={0,2}$ is never feasible for the log kernel.
]

#cor[Logarithmic kernel: full equioscillation is forced][
	For $K(x)=log(x)$ and $p=2$ on $[0,1]$, the active-set condition _always_ forces $A(x^*)={0,1,2}$, i.e., every regular local maximizer is fully equioscillating, regardless of center placement.
]

#obs[Contrast with reciprocal kernel][
	For the reciprocal kernel, nondegeneracy held on a dense open set (excluding a codimension-1 locus).
	For the logarithmic kernel, nondegeneracy is _unconditional_: the boundary-balance equations force an impossible ratio identity unless $x_1=x_2$, so $A={0,2}$ is provably impossible for every regular configuration.
]

== Interpretation

These two test cases suggest that the first-order active-set criterion is _strong enough to force equioscillation_ on the line, at least for p=2. The difference between the two kernels reveals an important point:

+ For kernels with a certain "repulsion" property (like $1/x$, where reciprocals grow quickly), nondegeneracy holds generically.
+ For kernels with a rigid boundary-ratio structure (like $log(x)$ in the p=2 line case), nondegeneracy can hold universally.

In both cases, full equioscillation follows. This suggests that the criterion may be sufficient more broadly, though a complete proof for arbitrary p and general trees remains open.

]
#chapter
