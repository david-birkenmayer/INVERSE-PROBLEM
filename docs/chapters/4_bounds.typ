#import "../setup/preamble.typ": template
#import "../setup/commands.typ": *
#import "../setup/ctheorems.typ": *

#show: template

#let chapter = [

= Performance bounds for the inverse problem

== Assumptions
- We fix a norm $norm(dot)$, which is used to measure the differences in demands or pressures. We mainly focus on the $L^2$-norm here, but the arguments adapt to other norms.
- We split the node set into junctions $J$ and a reservoir node $R$.

== Measure equivalence and performance bounds
Given two scenarios $Y$ and $Z$ and a set of measurement sites $S$ we define _measure equivalence at $S$_ as
$
	Y ~_( S) Z quad :<==> quad &  h_S^Y = h_S^Z "and" d_R^Y = d_R^Z
$
That is, they have the same pressures at the measurement sites $S$ and the same total demand at the reservoir $R$.
We may write $d^Y ~_( S) d^Z$ or $h^Y ~_( S) h^Z$ instead, since $d$ and $h$ fully determine the scenario. If $S$ is clear from the context, we may omit it. 
The equivalence class $[Y]_~$ is uniquely determined by the values of $h_S^Y$ and $d_R^Y$, thus we may identify it with the _measurement_ $M =(h_S^Y, d_R^Y)$. Hence, by abuse of notation, we write $Y in M$, if $Y$ is a scenario with measurements $M$.

Given a fixed measurement $M$, we define:
$
  W_d (M) &:= sup_(Y,Z in M) thick norm(d^Y - d^Z) quad quad quad quad  \
  C_d (M) &:= inf_(Y in M) thin sup_(Z in M) thick norm(d^Y - d^Z)
$

And if $M$ not specified, we define:
$
W_d &:= sup_(M) thick W_d (M) = sup_(Y ~ Z) thick norm(d^Y - d^Z) \
C_d &:= sup_(M) thick C_d (M)
$
and similarly define $W_h$ and $C_h$ for the pressures.






]
#chapter
