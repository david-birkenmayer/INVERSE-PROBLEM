#import "../setup/preamble.typ": template
#import "../setup/commands.typ": *
#import "../setup/ctheorems.typ": *

#show: template

#let chapter = [

= MCMC posteriori scenario generation

== Goal
We fix measurement data
$
  M = (M_S, D),
$
where $M_S$ denotes the measured pressures on the sensor set $S$ and $D$ denotes the
measured total reservoir demand.

Given a predictor $P$, we want to evaluate how good the prediction $P(M)$ is relative to the
conditional scenario law induced by the Dirichlet demand model under the measurement constraint $M$.

The target is therefore not a worst-case quantity, but a posterior distribution on feasible
scenarios:
$
  pi(h,d | M).
$

The practical plan is to sample this distribution by a Metropolis-Hastings chain whose state is a
reduced pressure vector.

== State parametrization
Let $V$ be the junction set and let $S subs V$ be the measurement set. The pressures on $S$ are fixed
by $M_S$, so only the unobserved pressures can vary.

Choose one additional node $v$ among the unobserved nodes, that is, among the nodes of $V$ that are
not in $S$, as an elimination node. For the first implementation, choosing a central node is a reasonable
default because it reduces the risk that the eliminated coordinate produces numerically degenerate
pressure differences across long thin parts of the network.

Write $U$ for the unobserved node set, namely the nodes of $V$ that are not in $S$, 
and decompose the unknown pressures as
$
  h_U = (h_v, h_F),
$
where
$
  F
$
denotes the free unobserved nodes after removing the chosen elimination node $v$ from $U$.

The chain state is then the reduced vector
$
  z := h_F in RR^(abs(F)).
$
For a given $z$, the full pressure vector is reconstructed as follows:

- on $S$, use the fixed measured values $M_S$;
- on $F$, use the proposed values $z$;
- at $v$, recover $h_v$ from the total-demand constraint.

So the full pressure vector has the form
$
  h(z) = (M_S, h_v(z), z).
$

== Recovering the missing pressure by Newton iteration
Let
$
  d(h) = (d_j(h))_(j in V)
$
denote the demand vector reconstructed from the pressure vector $h$. In the intended posterior
pipeline, this reconstruction is assumed to be uniquely determined once $h$ is known.

Define the scalar constraint function
$
  g(h_v; z) := sum_(j in V) d_j(M_S, h_v, z) - D.
$
We recover the missing pressure $h_v$ by solving
$
  g(h_v; z) = 0.
$

In practice:

1. start from an initial guess $h_v^(0)$, preferably taken from the predictor state $P(M)$;
2. run a few Newton iterations
   $
     h_v^(k+1) = h_v^(k) - g(h_v^(k); z) / partial_(h_v) g(h_v^(k); z);
   $
3. accept the reconstructed value if the residual is sufficiently small and the resulting demands
   satisfy the feasibility conditions.

The feasibility filter is:

- $d_j(h(z)) >= d_j^0$ for all junctions $j$;
- any additional hydraulic admissibility conditions required by the model.

If this fails, the proposal is rejected immediately.

== Posterior density in reduced coordinates
The original Dirichlet model is a distribution on demand scenarios. Hence the posterior density in
pressure coordinates must include the change-of-variables factor from pressures to demands.

Let
$
  F(h) := d(h)
$
be the pressure-to-demand map. The full Jacobian is
$
  J(h) := (partial F) / (partial h).
$

Because measured pressures are fixed and one extra coordinate $h_v$ is eliminated through the scalar
constraint, the actual chain runs in the reduced coordinates $z = h_F$. The induced demand map is
$
  F_("red")(z) := F(M_S, h_v(z), z).
$

Its Jacobian is the reduced Jacobian
$
  J_("red")(z) := (partial F_("red")) / (partial z).
$

The posterior density in reduced coordinates is then, up to normalization,
$
  pi_("red")(z mid M)
  = C(M)
  pi_("Dir")(F_("red")(z))
  times
  sqrt(det(J_("red")(z)^T J_("red")(z)))
  times
  1_("feasible")(z).
$

Here:

- $pi_("Dir")$ is the Dirichlet density expressed on the demand vector;
- $sqrt(det(J_("red")^T J_("red")))$ is the Gram-determinant factor for the immersion from reduced
  pressure coordinates into demand space;
- $1_("feasible")$ is the hard feasibility indicator.

== Exact reduced-Jacobian formula
Write the full Jacobian columns with respect to the free coordinates and the eliminated coordinate as
$
  J(h) = [J_F(h) quad J_v(h)],
$
where $J_F$ contains the columns corresponding to $z = h_F$ and $J_v$ is the column corresponding to
$h_v$.

Differentiate the scalar constraint
$
  g(h_v(z); z) = 0
$
with respect to $z$. This gives
$
  partial_z g + partial_(h_v) g partial_z h_v = 0,
$
so
$
  partial_z h_v = - (partial_z g) / (partial_(h_v) g).
$

By the chain rule,
$
  J_("red") = J_F + J_v (partial_z h_v).
$
Substituting the previous expression yields
$
  J_("red")
  = J_F - J_v (partial_z g) / (partial_(h_v) g).
$

This is the precise reduced-coordinate formula. Relative to the free-column Jacobian $J_F$, the
correction term is of rank one, because it is an outer product of the column vector $J_v$ with the
row vector
$
  (partial_z g) / (partial_(h_v) g).
$

Passing from the full Jacobian to the reduced Jacobian is a
rank-one modification once the eliminated pressure is recovered through the total-demand constraint.

== Removing one demand row
Often one does not want to use the full rectangular Jacobian from reduced pressure coordinates into
all demands, because the total-demand constraint introduces one scalar dependence between the demand
components. A convenient square version is obtained by deleting the row corresponding to the chosen
node $v$.

Let
$
  F^(-v)(h)
  := (d_j(h))_(j != v),
$
and let
$
  J_("red,-v")
$
be the corresponding reduced Jacobian. Then the same derivation gives
$
  J_("red,-v")
  = J_F^(-v) - J_v^(-v) (partial_z g) / (partial_(h_v) g).
$

Again, this is a rank-one modification of the Jacobian obtained after removing the row of the
chosen node and keeping only the free pressure coordinates.

This square reduced Jacobian is often the cleanest object for local volume factors, and its Gram
determinant is
$
  det((J_("red,-v"))^T J_("red,-v")).
$

== Metropolis-Hastings chain
With the reduced state $z$, one MCMC step has the following form.

1. Current state: $z_k$.
2. Proposal: sample
   $
     z' = z_k + eta,
   $
   where $eta$ is centered Gaussian noise.
3. Reconstruct $h_v (z')$ by Newton iteration.
4. Reconstruct the full pressure vector $h(z')$ and then the demand vector $d(z')$.
5. If feasibility fails, reject.
6. Otherwise compute the log target density
   $
     log pi_("red")(z' | M)
   $
   using the Dirichlet term and the reduced Gram determinant.
7. Accept with the usual Metropolis-Hastings probability.

For a symmetric Gaussian proposal, the acceptance ratio is simply
$
  alpha(z,z') = min(1, (pi_("red")(z'| M)) / (pi_("red")(z | M))).
$

== Initialization and burn-in
The natural initialization is the predictor output $P(M)$.

More precisely:

- use the predicted pressure vector to initialize the free coordinates $z$;
- use its value at the elimination node as the initial Newton guess for recovering $h_v$;
- start the chain there and discard an initial burn-in segment.

After burn-in, the retained samples provide posterior scenarios compatible with the measurement
instance $M$.

== Performance tricks (practical and standard)
The construction above is standard Metropolis-Hastings in transformed coordinates @metropolis1953 @hastings1970.
The following implementation tricks are typically decisive for robustness and speed.

1. Evaluate the target in log form.
  Instead of evaluating $pi_("red")(z | M)$ directly, evaluate
  $
    ell(z) := log pi_("red")(z | M)
  $
  and use
  $
    log alpha(z,z') = min(0, ell(z') - ell(z)).
  $
  This avoids underflow/overflow and turns products into sums, which is numerically much more stable @robertcasella2004 @brooks2011.

2. Use determinant computations through factorization, not through raw `det`.
  For square reduced Jacobian, compute $log |det J_("red,-v")|$ via an LU factorization with pivoting.
  For Gram forms, compute $log det((J^T J))$ via Cholesky of $(J^T J)$ if positive definite.
  These are standard stable linear algebra pathways @golubvanloan2013 @higham2002.

3. Exploit the rank-one structure.
  Since
  $
    J_("red,-v") = A - u r^T,
  $
  one may use the matrix determinant lemma
  $
    det(A - u r^T) = det(A) (1 - r^T A^(-1) u)
  $
  and evaluate the correction with one linear solve after reusing a factorization of $A$.
  This can reduce cost substantially when many proposals share the same structural sparsity pattern @harville1997 @golubvanloan2013.

4. Reuse local linearization artifacts across proposals.
  In each MH step, the expensive operations are typically Newton recovery of $h_v$ and Jacobian-related solves.
  Reusing previous iterate values as warm starts (for Newton and linear solves) is usually a large constant-factor gain.

5. Tune proposal scale to avoid random-walk degeneracy.
  For random-walk Metropolis in moderate/high dimension, too small variance mixes slowly and too large variance causes frequent rejection.
  A common target regime is the classical random-walk acceptance window around the 0.2 to 0.3 range (with the 0.234 asymptotic heuristic) @roberts1997.

6. Monitor effective sampling, not only raw sample count.
  Report acceptance rate, autocorrelation diagnostics, and effective sample size (ESS).
  This is standard MCMC practice and is usually more informative than the number of retained states alone @gelman2013 @brooks2011.

== Why this is the right first formulation
This construction separates the problem into three layers:

- a geometric layer: pressures are the sampling coordinates;
- a physical layer: demands are reconstructed from pressures and filtered by feasibility;
- a probabilistic layer: the Dirichlet prior on demands is pulled back to the reduced pressure space.

This is the correct starting point for posterior predictor evaluation, because it produces samples
from the conditional scenario law instead of only an extremal scenario.

Once this posterior sampler is available, one can compare $P(M)$ to posterior summaries such as:

- posterior mean,
- posterior quantiles,
- posterior credible bands,
- posterior log-density at $P(M)$.

]

#chapter