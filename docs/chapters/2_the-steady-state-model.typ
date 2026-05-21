#import "../setup/preamble.typ": template
#import "../setup/commands.typ": *
#import "../setup/ctheorems.typ": *

#show: template

#let chapter = [

= The steady-state model
We give a short summary of the steady-state model here, which is the most commonly used model for WDNs and the basis for our work.

In the steady-state model, a WDN is modeled as a directed graph $G=(V,E)$, where vertices represent junctions and edges represent pipes. For $e in E$ we denote by $r_e$ the pipe resistance, it is a function of the pipe's length, diameter, and roughness, we assume it to be given. Also, assume that there is a single reservoir vertex $w in V$, where the water comes from.

The dynamic values of the model are
- demands $d_v$ for $v in V$, which represent the amount of water that is consumed at vertex $v$ (negative demand means that water is injected at $v$)
- flows $q_e$ for $e=(u,v) in E$, which represent the amount of water flowing from $u$ to $v$ (negative sign indicates that the flow direction is opposite to the edge direction).
- pressure head $h_v$ for $v in V$, which represents the pressure the water exterts at vertex $v$.

They are related by the following equations:
  $ 
    forall v in V&:&      sum_((u,v) in E) q_(u,v) &= - d_v      quad && "(mass)" \
    forall (u,v) in E &:& h_u - h_v                &= r_(u,v) dot q_(u,v) dot |q_(u,v)|^(x-1) quad && "(energy)"
  $
where the exponent $x$ is a fixed parameter, $x=1.85$ is the "Hazen-Williams" model, $x=2$ is the "Darcy-Weisbach" model. We will focus on the Darcy-Weisbach model in this paper, but our results can be easily adapted to the other models as well.


Assume we are given pipe factors $r_e$ for $e in E$, and demands $d_v$ for $v in V$, which fulfill $sum_(v in V) d_v = 0$ (balanced demand). We want to solve the following problem (S)
$ d_v &= sum_((u,v) in E) q_e - sum_((v,u) in E) q_e  quad forall v in V quad quad && "(mass conservation)" \
h_u - h_v &= r_(u,v) dot q_(u,v) dot abs(q_(u,v)) quad forall (u,v) in E quad quad && "(energy conservation) " $
for the flow $q_e$, where $h_v$ are the pressure variables. A positive sign in $q_e$ indicates here that the flow direction aligns with the direction of the edge, a negative sign indicates that it doesn't. Thus a solution to this problem would also determine the flow directions.
A more compact formulation utilizes the incidence matrix $B$ of $G$:
$ 
B q &= d quad quad && "(mass conservation)" \
B^top h &= r q abs(q) quad quad && "(energy conservation) " 
$
where $r q abs(q) := (r_e q_e abs(q_e))_(e in E)$ denotes the vector received by component-wise multiplication.

= Reformulation (R) using cycles 
We can represent cycles $C$ in $G$ as vectors $z_C in {0,1}^E$ with $B z = 0$ via the relationship
$ e in C <==> z_C = 1 $
A _cycle basis_ is then a set of cycles $cal(C)$ such that the vectors $(z_C)_(C in cal(C))$ form a basis for $ker B$. If we collect these as rows of a cycle matrix $Z_cal(C) in RR^(cal(C) times E)$ we get:
$ 
& exists h: B^top h = r q abs(q) \
<==> quad & r q abs(q) in im(B^top) \
<==> quad & r q abs(q) perp ker(B) \
<==> quad & z_C^top (r q abs(q)) = 0 quad quad quad & forall C in cal(C) \
<==> quad & Z_cal(C) (r q abs(q)) = 0  
$
so we can reformulate (S) into (R):
$
B q &= d quad quad && "(mass conservation)" \
Z_cal(C) (r q abs(q)) &= 0  quad quad && "(energy conservation)"
$

= Reformulation (P) as convex minimization problem 
Now, consider the following optimization problem (P):
$ min_q sum_(e in E) r_e/3 |q_e|^3 "s.t." B q = d $
The function $x mapsto |x|^3$ is continuously differentiable and strictly convex, hence by using basic convex optimization theory we obtain that the problem has a unique solution. If we let $h$ be the Lagrangian multipliers, the Lagrangian is 
$ cal(L)(q,h)= sum_(e in E)​ r_e/3 abs(q_e)^3 + h^top (B q−d) $
and the KKT conditions $(partial cal(L)) / (partial q_e) = 0$ are equivalent to
$ & r_e q_e abs(q_e) + (B^top h)_e = 0 \
 <==> &  h_u - h_v = r_e q_e abs(q_e) "with" e=(u,v) $
But this is exactly the energy conservation! Thus the optimal solutions of (P) correspond one to one with optimal solutions of (S) -- and thus (S) also has a unique solution. Furthermore, the pressures $h$ are unique up to a constant, or unique if the reservoir pressure is known.


]
#chapter

