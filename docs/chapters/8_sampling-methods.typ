#import "../setup/preamble.typ": template
#import "../setup/commands.typ": *
#import "../setup/ctheorems.typ": *

#show: template

#let chapter = [

= Sampling the posterior: MCMC, ABC, and exact enumeration

The previous chapter constructed a Metropolis-Hastings sampler for the posterior
$pi(h,d | M)$. This chapter reports what happens when that sampler is applied to networks
larger than the small benchmarks, why it fails there, and which two cheaper methods replace
it. The conclusion is that the choice of method is governed by a single scalar --- the ratio
of the prior-predictive spread of the sensor readings to the assumed sensor noise --- and
that on the networks of interest that ratio almost always favours something other than MCMC.

== Setting

Throughout, $cal(J)$ is the junction set, $cal(Y) subs cal(J)$ the sensor set, and an
observation is $z = (D^z, h^z_cal(Y))$. The sensor likelihood is Gaussian of width $eps$,
$
  P(z | s) prop exp(-1/(2 eps^2) norm(h^s_cal(Y) - h^z_cal(Y))^2),
$
and the prior $P$ is one of: a Dirichlet split of an extra demand $Delta$ over all junctions,
a Gaussian $N(d^0, "diag"(sigma^2))$ around the base demands, or the *single-leak* prior
$
  v ~ "Unif"(cal(J)), quad lambda ~ "Unif"([0, Delta]), quad d = d^0 + lambda e_v.
$

#def[
  The *signal-to-noise ratio* of a sensor $y in cal(Y)$ under a prior $P$ is
  $
    "SNR"_y := ("std"_(s ~ P) [h^s_y]) / eps,
  $
  the spread of that sensor's reading across the prior, measured in units of the sensor noise.
]

The whole of this chapter can be read as a study of this quantity. When $"SNR" lt.tilde 1$
the likelihood is flatter than the prior and the measurement is weakly informative; when
$"SNR" gt.tilde 1$ the posterior concentrates into a thin sliver of the prior support.

== Why MCMC does not work on larger networks

=== A representative failure

Take L-TOWN_C (BattLeDIM Area C): $92$ junctions, $109$ pipes, one reservoir, three pressure
sensors. Sampling the Gaussian-prior posterior with the affine-invariant ensemble sampler,
$2000$ burn-in and $2500$ retained iterations with $2 dot 91 + 2 = 184$ walkers, gives

#table(
  columns: 2, stroke: none, align: (left, right),
  [wall clock],                  [$4568$ s],
  [minimum ESS],                 [$1010$],
  [ESS per second],              [$0.2$],
  [maximum split-$hat(R)$],      [$2.96$],
  [mean-disagreement],           [$0.90 sigma$],
)

A split-$hat(R)$ of $2.96$ after $76$ minutes is not a chain that needs a little more time;
it is a chain whose walkers have not mixed at all. Both diagnostics that were introduced
precisely to catch this --- $hat(R)$ and the per-chain mean disagreement --- fire.

=== The failure is structural, not a tuning problem

It is tempting to read this as insufficient tuning. It is not. Linearising the forward map
at the base state gives the Jacobian $J = partial h_cal(Y) \/ partial d in RR^(3 times 92)$, whose
singular values are $453, 102, 53$ and nothing else: $J$ has rank $3$. Conditioning a
Gaussian prior on three linear functionals leaves the remaining $89$ directions *exactly* at
their prior. The exact linear-Gaussian posterior covariance
$
  Sigma_"post" = (Sigma^(-1) + J^top J \/ eps^2)^(-1)
$
gives a total information content, summed over junctions, of $2.02$ out of a possible $92$
even in the noiseless limit $eps -> 0$, and $0.085$ at the realistic $eps = 0.015$ m.

So the posterior is, in $89$ of $92$ directions, identical to the prior. A sampler asked to
explore such a distribution in $91$ dimensions is doing an enormous amount of work to
reproduce something already known in closed form, and the thin remainder is what it must
resolve. That is a bad trade.

=== The dangerous part: a non-converged chain gives a plausible wrong answer

The failure would be benign if it were loud. It is not. The unconverged chain reported a
per-junction variance reduction of median $0.017$ with a maximum of $0.77$ --- that is, it
claimed one junction's demand had been pinned down to $23%$ of its prior uncertainty. The
exact linear-Gaussian computation gives median $0.0001$ and maximum $0.046$, and *no*
junction above $10%$.

#obs[
  A stuck ensemble under-disperses: its walkers have not explored the prior, so the empirical
  spread of the samples is too small, and the inferred variance reduction is correspondingly
  too large. Non-convergence therefore biases the reported uncertainty in the *optimistic*
  direction. A posterior width taken from an unconverged chain will overstate how much the
  sensors have taught us.
]

This is the strongest argument against relying on MCMC here: the failure mode produces a
confident, wrong, and superficially reasonable answer.

== Approximate Bayesian Computation

=== The method

Draw $s_1, dots, s_N$ from the prior, forward-solve each, and weight by the likelihood,
$
  w_i prop exp(-1/(2 eps^2) norm(h^(s_i)_cal(Y) - h^z_cal(Y))^2), quad sum_i w_i = 1 .
$
Posterior expectations are weighted sums, and the quality of the approximation is measured by
the effective sample size $"ESS" = 1 \/ sum_i w_i^2$.

The textbook objection is the curse of dimensionality: acceptance scales like
$(1\/eps)^(abs(cal(Y)))$. The point of the SNR is that this curse is *conditional*. If $eps$
exceeds the spread of the readings across the prior, almost every draw is consistent with the
observation and the weights are nearly uniform.

=== It is essentially free in the weakly-informative regime

On L-TOWN_C the prior-predictive spreads are $0.0034$, $0.0051$ and $0.0025$ m against
$eps = 0.015$ m, so $"SNR" in [0.17, 0.34]$. Running ABC on the identical model that the
MCMC above sampled:

#table(
  columns: 3, stroke: none, align: (left, right, right),
  table.header([], [ABC], [MCMC]),
  [draws / iterations],  [$8000$],  [$4500$],
  [wall clock],          [$147$ s], [$4568$ s],
  [ESS],                 [$7968$],  [$1010$],
  [ESS / N],             [$0.996$], [---],
  [ESS per second],      [$54.3$],  [$0.2$],
  [converged],           [yes],     [no ($hat(R) = 2.96$)],
)

ABC is $270 times$ more efficient per second, and it *converges*: an ESS of $7968$ from
$8000$ draws means the samples are essentially independent. It also reproduces the exact
linear-Gaussian answer, where the MCMC did not.

=== Where ABC does break down

The curse is real when the measurement is informative. On Kadu ($24$ junctions, $4$ sensors,
prior-predictive spread $0.426$ m) the realised efficiency across $eps$ is

#table(
  columns: 4, stroke: none, align: (right, right, right, right),
  table.header([$eps$ (m)], [SNR], [ESS (of $3000$)], [ESS / N]),
  [$0.5$],   [$0.85$],  [$833$],  [$0.28$],
  [$0.2$],   [$2.13$],  [$147$],  [$0.049$],
  [$0.1$],   [$4.26$],  [$44$],   [$0.015$],
  [$0.05$],  [$8.52$],  [$9.5$],  [$0.003$],
  [$0.02$],  [$21.3$],  [$1.1$],  [$0.0004$],
)

Below $eps approx 0.05$ m the weights collapse onto a single draw. Two remarks make this
less damaging than it appears.

First, *MCMC fails in the same place*: on Kadu with four sensors the ensemble sampler reaches
$hat(R) approx 3.4$ at $eps = 0.05$ and mixes acceptably only at $eps approx 0.5$. The wall at
small $eps$ is a property of the posterior, not of the algorithm; it is the statement that the
measurement-consistent demand set is genuinely tiny.

Second, the small-$eps$ regime is not the physically relevant one. Kadu's sensors sit at
$approx 98$ m of head; a pressure transducer at $0.1$--$0.5%$ of a $100$ m full scale
contributes $0.1$--$0.5$ m, and the nominal-versus-real model mismatch of $plus.minus 10%$ in
demands and pipe parameters contributes a further median $0.11$ m and $95$th percentile
$0.45$ m. A defensible budget is $eps in [0.2, 0.5]$ m, which is exactly the regime where
ABC is usable.

#obs[
  The honest reading of the table is uncomfortable and worth stating plainly: ABC is
  efficient at $eps = 0.5$ m *because* the measurement is barely informative there. Kadu's
  entire network head spread is $2.00$ m, so $eps = 0.5$ m is a quarter of the whole signal.
  Efficiency of the sampler and informativeness of the data trade off against one another;
  there is no configuration in which both are large.
]

=== ABC needs $eps > 0$

For a prior with a density, the set of demand vectors reproducing the sensor readings exactly
has measure zero, so at $eps = 0$ the acceptance probability is exactly zero. ABC therefore
*requires* a positive noise level. This is not a defect of the implementation; it is why the
Dirac likelihood is listed as inapplicable to ABC. It is also precisely the gap that the next
section fills.

== The exact solver

=== The single-leak prior collapses

Under the single-leak prior the parameter space is not high-dimensional: it is one discrete
label $v$ and one scalar $lambda$. Geometrically the support is a union of $abs(cal(J))$ line
segments radiating from $d^0$, a one-dimensional set with $abs(cal(J))$ branches, and one
can simply walk it. Two variants arise.

#alg[
  *Variant A --- total demand observed.* Since $d = d^0 + lambda e_v$ implies
  $bb(1)^top d = D^0 + lambda$, the observed total demand determines
  $lambda^* = D^z - D^0$ outright. Then for each candidate $v$:
  + forward-solve $h_cal(Y)(d^0 + lambda^* e_v)$;
  + form the residual $r_v = norm(h_cal(Y)(v) - h^z_cal(Y))$;
  + set $P(v | z) prop K_eps (r_v)$ and normalise over the $abs(cal(J))$ candidates.
  Cost: $abs(cal(J))$ forward solves.
]

#alg[
  *Variant B --- total demand unknown.* Now $lambda$ must be inferred. For each candidate $v$:
  + solve $h_cal(Y)(d^0 + lambda e_v) = h^z_cal(Y)$ for $lambda_v$ (the sensor head is
    monotone in $lambda$);
  + form $r_v$ as above;
  + set
    $
      P(v | z) prop p(lambda_v) dot norm(partial h_cal(Y) \/ partial lambda)^(-1) dot K_eps (r_v)
    $
    and normalise.
]

The Jacobian factor in Variant B is essential and easy to miss. Conditioning on
$h_cal(Y)(v, lambda) = h^z_cal(Y)$ pins $lambda$, and the induced density on the discrete
label $v$ picks up the change-of-variables term $norm(partial h_cal(Y) \/ partial lambda)^(-1)$:
a candidate whose sensor reading responds *slowly* to $lambda$ explains a whole neighbourhood
of observations, and so deserves more posterior mass than one that responds sharply. Since
$p(lambda_v)$ is constant under the uniform prior and $K_eps (r_v)$ is identical across all
exactly-consistent candidates, this factor is the *only* thing that makes the posterior
non-uniform. Omitting it silently returns a uniform distribution over the consistent
candidates. It is the same Gram-determinant correction that appears in the pressure-space and
exact-demand MCMC formulations.

=== Cost

Variant A costs $abs(cal(J))$ forward solves. Variant B costs $abs(cal(J))$ scalar root-solves;
with a secant iteration these converge in about $6$ solves each, against $62$ for a $60$-step
bisection, at a cost in $lambda$ of $approx 4 dot 10^(-6)$ --- far below the $approx 3 dot
10^(-5)$ m precision to which the hydraulic solver reports heads. On a $24$-junction network
one full posterior is a matter of seconds.

=== When it is justified

+ *It is exact.* There is no Monte-Carlo error, no burn-in, no proposal to tune, and no
  convergence diagnostic to interpret. The result is reproducible to solver precision.
+ *It works at $eps = 0$.* With a finite candidate set there is no measure-zero obstruction,
  so the noiseless case --- inaccessible to ABC --- is available. On L-TOWN_C with exact
  readings this yields $98%$ exact leak localisation even for the smallest BattLeDIM leak,
  against $13%$ with readings rounded to the challenge's two decimal places.
+ *It is an oracle.* Because it is exact, it is a strictly better reference for validating
  samplers than an ABC oracle, which carries its own noise.
+ *It makes sensor placement affordable.* Optimising over sensor sets requires one posterior
  per candidate set. At seconds per posterior that search is tractable; at one MCMC run per
  posterior it is not.

=== When it is not justified

The method buys its exactness by assuming a very restrictive prior, and the assumption should
be stated rather than hidden.

+ *It requires structure.* The prior must be discrete or have very few continuous parameters.
  A free demand field over all junctions returns us to the continuous case.
+ *It does not scale in the number of leaks.* For $k$ simultaneous leaks one enumerates
  $binom(abs(cal(J)), k)$ node sets. For $abs(cal(J)) = 92$, $k = 2$ is $4186$ pairs and still
  feasible; $k = 3$ is $125{,}000$ triples and the approach loses.
+ *The prior is an assumption about the world.* A single leak of bounded magnitude at a node
  is a modelling choice. The posterior is exact *given* that choice, and the exactness must
  not be mistaken for freedom from modelling error.
+ *Real leaks are on pipes, not nodes*, and the nominal model differs from the real network.
  In particular, on L-TOWN_C the $98%$ localisation above degrades to $26%$ once the
  observation is generated from a network perturbed by the $plus.minus 10%$ the challenge
  specifies --- so the exactness of the inference is not the binding constraint on accuracy;
  model calibration is.

== Perturbing the base demands: a hierarchical prior

Both priors used so far treat the base demands $d^0$ as known. They are not: the nominal
model differs from the real network, and on the BattLeDIM benchmark the base demands are
randomised by $plus.minus 10%$. A more honest prior perturbs them and then places a leak on
top,
$
  tilde(d)^0 ~ N(d^0, Sigma_0), quad v ~ "Unif"(cal(J)), quad lambda ~ "Unif"([0,Delta]),
  quad d = tilde(d)^0 + lambda e_v .
$
This is attractive because the enumeration of the previous section still applies *conditionally*:
for a fixed $tilde(d)^0$ the candidate set is again finite and $lambda$ is again recovered from
the sensor constraint. The question is how to combine the runs.

=== The correct estimator sums evidence, not posteriors

Marginalising the nuisance parameter,
$
  P(v | z) prop P(v) integral P(z | v, tilde(d)^0) thin P(tilde(d)^0) thin d tilde(d)^0 .
$
For a fixed draw $tilde(d)^0_m$, the inner $lambda$-integral is exactly the quantity the
enumeration already computes. Write that per-draw *evidence* for candidate $v$ as
$
  w_(m,v) := p(lambda_(m,v)) dot norm(partial h_cal(Y) \/ partial lambda)^(-1) dot K_eps (r_(m,v)),
$
with $lambda_(m,v)$ and $r_(m,v)$ the solved magnitude and residual for candidate $v$ under
base $tilde(d)^0_m$. Then the Monte-Carlo estimator of the posterior is

#alg[
  *Rao-Blackwellised enumeration.*
  + draw $tilde(d)^0_1, dots, tilde(d)^0_M ~ N(d^0, Sigma_0)$;
  + for each $m$, run the enumeration and record the *unnormalised* $w_(m,v)$;
  + set $ P(v | z) prop sum_(m=1)^M w_(m,v) $ and normalise once, at the end.
]

The tempting alternative is to normalise each run separately and average the resulting
posteriors,
$
  P_"naive" (v | z) := 1/M sum_(m=1)^M w_(m,v) / (sum_(u in cal(J)) w_(m,u)) .
$
This is wrong. Writing $E_m := sum_u w_(m,u)$ for the total evidence of draw $m$, the correct
estimator is
$
  P(v | z) prop sum_m E_m dot P(v | z, tilde(d)^0_m),
$
whereas the naive one is $M^(-1) sum_m P(v | z, tilde(d)^0_m)$.

#obs[
  The naive estimator is the correct one with the evidence weights $E_m$ replaced by uniform
  weights. It is therefore exact if and only if every perturbed base explains the observation
  equally well. That is precisely what $Sigma_0 > 0$ destroys: some draws of $tilde(d)^0$ fit
  the sensor readings far better than others, and normalising per draw discards exactly that
  information. The discrepancy grows with $Sigma_0$, i.e. with the very model uncertainty the
  construction was introduced to represent.
]

=== Why this is the collapsed Gibbs structure

The state is the pair $(v, tilde(d)^0)$: one discrete label and one continuous nuisance vector.
An ordinary Gibbs sampler would alternate

$
  v^((t+1)) ~ P(v | tilde(d)^(0,(t)), z), quad quad
  tilde(d)^(0,(t+1)) ~ P(tilde(d)^0 | v^((t+1)), z) .
$

*Collapsing* --- equivalently, Rao-Blackwellising --- means replacing the sampling of a
component by its analytic conditional wherever that conditional is available in closed form.
Here it is: $v$ ranges over the finite set $cal(J)$ and $P(v | tilde(d)^0, z)$ is exactly what
one enumeration pass returns. So $v$ never has to be sampled at all. For any statistic $f$,
$
  EE[f(v) | z] = EE_(tilde(d)^0 | z) [ EE[ f(v) | tilde(d)^0, z ] ],
$
and the inner expectation is evaluated exactly rather than estimated. By the Rao-Blackwell
theorem, replacing a sampled indicator by its conditional expectation cannot increase the
variance of the estimator, so this dominates any scheme that draws $v$ at random --- including
plain ABC over the joint prior, which must explore the discrete dimension by chance.

Two honest qualifications on the terminology. First, the scheme above draws $tilde(d)^0$ from
its *prior* and reweights by evidence; that is Rao-Blackwellised importance sampling, not a
Gibbs sampler, because no Markov chain is involved. It becomes collapsed Gibbs proper when the
prior draws stop being efficient --- large $Sigma_0$ or small $eps$ --- and one instead samples
$tilde(d)^0 tilde.op P(tilde(d)^0 | z)$ by MCMC while continuing to enumerate $v$ exactly at
every step. The discrete component is collapsed in both cases; only the machinery for the
continuous component changes. Second, the argument for exactness rests on the change-of-variables
factor of the previous section: without it each $w_(m,v)$ is wrong by a $v$-dependent constant,
and summing wrong weights does not produce a right answer.

=== Status

The construction is derived here but not yet measured. The experiment that settles it compares,
on identical observations whose truth is generated from a *perturbed* base so that the model
mismatch is real: (i) the naive average of normalised posteriors, (ii) the evidence-weighted
sum above, and (iii) ABC over the full hierarchical prior as an unbiased reference. That would
quantify both how far the naive estimator is off and whether the Rao-Blackwellised estimator
beats plain ABC per forward solve, which the Rao-Blackwell theorem predicts but does not
quantify.

== Cross-validation of the two methods

ABC and enumeration are, structurally, the same quadrature applied to
$
  "posterior weight"_k prop "prior weight"_k dot K_eps ("residual"_k),
$
differing only in whether the candidates are deterministic (solutions of the structural
constraint) or Monte-Carlo (prior draws). They therefore validate one another: agreement is
evidence that both are correct, since they share no failure mode.

On Alperovits, averaged over $30$ random observations under the single-leak prior with
$eps = 0.05$ m, the mean per-junction absolute error of the posterior mean is

#table(
  columns: 3, stroke: none, align: (left, right, right),
  table.header([method], [Alperovits, $eps = 0.05$], [Kadu, $eps = 0.2$]),
  [ABC posterior mean],   [$0.1670$], [$0.0027$],
  [exact posterior mean], [$0.1673$], [$0.0006$],
  [trained GNN],          [$0.3257$], [$0.0817$],
)

On Alperovits, ABC and the exact posterior agree to $0.2%$ --- the mutual validation one
hopes for. On Kadu they do *not*: ABC's mean error is more than four times the exact one.
This is not a contradiction but a direct illustration of the earlier efficiency table. With
$eps = 0.2$ m on Kadu the realised ESS is of order tens out of $4000$ draws, so the ABC
posterior mean is itself a noisy estimate; the enumeration, being exact, is not. The
disagreement is a measurement of ABC's Monte-Carlo error, not evidence against either method,
and it is precisely the regime in which the exact solver earns its place.

=== Reading these numbers: irreducible error versus excess

All three rows above are scored the same way --- each method is collapsed to a single
predicted demand vector and compared against the truth, with the posterior *mean* serving as
the prediction for the two Bayesian methods. The posterior mean is the Bayes-optimal point
estimate under squared loss, so this is the right way to ask how much worse a predictor is
than the best one available. But the rows do not mean the same thing.

The exact posterior's error is not a deficiency of the method. Several demand vectors explain
the same sensor reading equally well, so no estimator can be right; the best possible one sits
among them and is still wrong by some amount. That amount is the *irreducible* error --- a
property of the network, the sensor set and the prior, not of any algorithm. A predictor's
total error decomposes as
$
  underbrace(abs(F(z) - d), "total")
  quad approx quad underbrace(abs(EE[d | z] - d), "irreducible")
  quad + quad underbrace(abs(F(z) - EE[d | z]), "excess, the predictor's own"),
$
and only the second term is the predictor's responsibility.

#table(
  columns: 4, stroke: none, align: (left, right, right, right),
  table.header([network], [irreducible], [GNN total], [GNN excess]),
  [Alperovits], [$0.1671$], [$0.3257$], [$approx 0.159$],
  [Kadu],       [$0.0006$], [$0.0817$], [$approx 0.081$],
)

Quoting ratios --- "twice as bad", "thirty times as bad" --- misleads in both directions. On
Alperovits roughly half the GNN's error is ambiguity it could never have avoided, so the
factor of two overstates its shortcoming. On Kadu the data very nearly determines the answer
(irreducible error $0.0006$), so essentially *all* of the GNN's error is its own, and the
factor of thirty understates the situation rather than exaggerating it.

#obs[
  A second caveat attaches to the GNN row specifically, and it works in the predictor's
  favour. The trained model outputs *pressures*; its demands are obtained afterwards by the
  algebraic inversion $q_e = "sign"(Delta h)(abs(Delta h) \/ r_e)^(1\/kappa)$ followed by
  nodal mass balance. The two posterior methods never invert pressures --- they work in demand
  coordinates throughout. What is tabulated as "GNN error" is therefore the error of
  *predictor plus reconstruction*, and the reconstruction is the ill-conditioned direction
  documented earlier. It is a fair assessment of the deployed pipeline, but it is not an
  isolated measurement of the learned component, and the excess above should not be attributed
  to the network without separating the two.
]

The obvious objection to the comparison is that the model was trained on Dirichlet
scenarios and is therefore out of distribution on a single-leak test, so that these factors
merely measure distribution shift. Repeating the experiment under the *training* prior
$d = d^0 + f Delta alpha$, $f ~ "Unif"([0,1])$, $alpha ~ "Dir"(1, dots, 1)$, refutes it:

#table(
  columns: 4, stroke: none, align: (left, left, right, right),
  table.header([network], [prior], [ABC], [GNN]),
  [Alperovits], [single-leak],              [$0.1670$], [$0.3257$],
  [Alperovits], [Dirichlet (training law)], [$0.0700$], [$0.2821$],
  [Kadu],       [single-leak],              [$0.0027$], [$0.0817$],
  [Kadu],       [Dirichlet (training law)], [$0.0024$], [$0.0734$],
)

In distribution the GNN does not improve relative to the posterior; on Alperovits the gap
*widens*, from a factor of $1.95$ to a factor of $4.0$. The reason is visible in the columns:
the Dirichlet posterior is markedly tighter than the single-leak one ($0.1670 -> 0.0700$),
while the GNN barely moves ($0.3257 -> 0.2821$). The measurement carries more information
under the training law, and the predictor does not exploit it. The deficiency is therefore a
property of the predictor, not an artefact of the test distribution.

Note also that under a continuous prior there is no exact panel: the support is a continuum
rather than a finite candidate set, so ABC is the only available reference and nothing
independent validates it. Given the low ESS on Kadu, its Dirichlet figures should be read
with the corresponding caution.

One further observation supports the reconstruction caveat above: on Kadu the GNN's error is
concentrated at junction $3$, which is the *only reservoir-adjacent junction* in the network.
Its reconstructed demand is the difference of the two flows on either side of it, the first of
which is the entire network inflow --- so a small nodal demand is computed as a small
difference of large numbers, and any head error there is amplified by roughly the ratio of
inflow to nodal demand. Junction $3$ is also a sensor whose prior-predictive reading spread is
$0.000$ m, i.e. hydraulically insensitive to demand, contributing nothing to the measurement.
Neither posterior method shows any such spike, because neither inverts pressures.

== Summary

#table(
  columns: 3, stroke: none, align: (left, left, left),
  table.header([regime], [character], [method]),
  [$"SNR" lt.tilde 1$], [likelihood flatter than prior], [ABC, nearly free],
  [$"SNR" gt.tilde 1$], [posterior concentrated],        [enumeration if the prior has structure],
  [$eps = 0$],          [noiseless],                     [enumeration only],
  [high-dim. continuous prior, sharp likelihood], [genuinely hard], [MCMC, if anything],
)

The band in which MCMC is the right tool --- a high-dimensional continuous prior together
with a sharp likelihood --- is narrow, and it coincides with the band in which every method
struggles. On the networks studied here the measurement is either weak enough that ABC is
essentially free, or the prior is structured enough that the posterior can be written down.

]

#chapter
