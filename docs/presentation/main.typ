#import "university.typ": *
#let ub = "ub"
#let lb = "lb"
#let subs = $subset.eq$
#let sups = $supset.eq$
#let eps = $epsilon$
#let ni = $in.rev$
#let nin = $in.not$
#let nni = $in.not.rev$
#let bset(a, b) = ${#a | #b}$
#let time(a) = $cal(O)(#a)$
#let ip(x, y) = $lr(angle.l #x, #y angle.r)$
#let script-symbols = (sym.lt, sym.lt.eq, sym.lt.equiv, sym.gt, sym.gt.eq, sym.gt.equiv, sym.prec.eq, sym.prec.eq, sym.succ.eq, sym.prec, sym.succ, sym.prec.equiv, sym.succ.equiv, sym.prec.eq.not, sym.prec.neq, sym.prec.nequiv, sym.succ.neq, sym.succ.nequiv)
#show math.equation: eq => {
  set text(weight: 400)
  set block(breakable: true)
  show regex(script-symbols.join("|")): math.scripts
  eq
}

#show: university-theme.with(
  aspect-ratio: "16-9",
  main-color: "dunkelblau",
  secondary-color: "hellblau",
  affiliations: ("RPTU Kaiserslautern-Landau"),
  sponsor-logos: ("/logos/affiliations/RPTU_Minimal.svg"),
  config-info(
    title: "Sampling nonlinear posteriors in the steady state sensor model",
    subtitle: "a SMARTWINE presentation",
    author: ("David A. Birkenmayer"),
    institution: "AG Optimierung, RPTU Kaiserslautern                          Supervisor: Professor Sven O. Krumke",
  ),
)

// Restyle the two markup shortcuts. NOTE: two stacked `#show ...: set text(...)` rules (one
// carrying font: "Red Hat Text") make Touying collapse to a single slide — use the function
// form instead, which is robust.
//  *strong* -> real bold, black (LM Sans has no bold face, so use the bundled Red Hat Text).
//  _emph_   -> red italic (takes over the theme's old red accent).
#show strong: it => text(fill: black, font: "Red Hat Text", weight: "bold", it.body)
#show emph: it => text(fill: red, style: "italic", it.body)

== Steady state model

#slide(repeat: 5, self => [
  #let (uncover, only) = utils.methods(self)
  
  #v(0.5em)
  #columns(2, gutter: 8pt)[
  #uncover("1-")[
    *Fixed values*
    - _wdn_ as digraph $G = (V,E)$, where \
      $V = cal(J) union.dot {v_0}$ with $v_0$ reservoir node\
    - _pipe resistances_ $r_e>0$ for $e in E$
    ]
    #uncover("3-")[
    - _reservoir pressure_ $H_0 in RR$

      #v(2em)
  #uncover("4-")[
    *Physical equations:*
  $
     bb(1)^top d &= D  && ("total demand")\ 
     B q &= - d  && ("mass")\
     B^top h &= phi.alt(q) - B_0^top H_0 quad &&("energy")
  $
  - $B in RR^(cal(J) times E)$ is the $v_0$-reduced incidence matrix of $G$
  - $B_0$ is eliminated $v_0$-row of the incidence matrix 
  - $phi.alt(x)_e := r_e dot q_e dot |q_e|^(kappa-1)$ the nonlinear engery function

  ]
  ]
  #colbreak()
  #v(0.5em)
  #uncover("2-")[
    *Variables:*
    - _demand_ $d_v>0$ for $v in cal(J)$
    - _flow_ $q_e$ for $e in E$ with $q_(v,u) := -q_(u,v)$
    - _pressure head_ $h_v$ for $v in cal(J)$
  ]
  #uncover("3-")[
    - _total demand_ $D>0$
  ]

  #v(1.5em)
  #uncover("5-")[
  *What we know:*
  - Given demands $d$ and $D$, pressures $h$ are unique. \ Calculation is well-conditioned.
  - Given pressures $h$, demands $d$ and $D$ are unique. \ Calculation is ill-conditioned around low flows.

  ]
  ]
  ])

== Our model

#slide(repeat: 5, self => [
  #let (uncover, only) = utils.methods(self)

  #columns(2, gutter: 40pt)[

  #v(0.5em)
  #uncover("1-")[
     *Hydraulic states:*
  - a _hydraulic state_ is a tuple $s = (d^s,q^s,h^s, D^s)$ \ fulfilling the physical equations
  - $S$ is the set of all hydraulic states
  - we assume to have a _prior distribution_ $P$ on $S$
  ]

  #v(0.5em)
  #uncover("2-")[
    *Measurements*
    - a _site configuration_ is a subset $cal(Y) subs cal(J)$, where pressure measuring sensors are placed
    - an _observation_ $z = (D^z, h^z_cal(Y) )$ is a tuple of the observed total demand and the observed pressures at the sites
    - $Z$ is the set of all observations.
  ]
  #uncover("3-")[
    - assume a _sensor likelihood_ function $P(z | s)$
  ]

  #v(0.5em)
  #uncover("4-")[
    $-->$ Want to find a good predictor $F: Z -> S$ (e.g. GNN)\
  ]

  #uncover("5-")[
    $-->$ But how do we determine what "good" is?
  ]

  #colbreak()
  #uncover("1-")[
    #align(center)[#image("alperovits_measurement.png", width: 82%)]
  ]
]

])


== Posterior sampling

#slide(repeat: 6, self => [
  #let (uncover, only) = utils.methods(self)

  #columns(2, gutter: 8pt)[
  #v(0.5em)
  #uncover("1-")[
    *Assumptions:*
    - _prior distribution_ $P$ on the state space $S$
    - _sensor likelihood_ $P(z | s)$
    - _measurement sites_ $cal(Y) subs cal(J)$ (hexagons)
    - _observation_ $z = (D^z, h^z_cal(Y) )$
  ]
  
  #uncover("2-")[
    *What we want to do:* \
    $->$ sample _posterior distribution_ 
    $ P_z (s) := P(s | z) quad prop quad
     underbrace(P(s), "prior") dot underbrace(P(z | s), "sensor likelihood") 
    $
  ]

  #uncover("4-")[
    *Why do we want this:* \
    - Can measure how good a prediction $F(z)$ is, by comparing it to the sampled results
    - Shows how confident we can be in the prediction in certain areas
    - Averaging the samples yields a predictor itself!
  ]

  #colbreak()
  ]

])

== Prior and sensor likelihood

#slide(repeat: 4, self => [
  #let (uncover, only) = utils.methods(self)

  #v(0.5em)
  #uncover("1-")[
    *Bayes Rule:* the measurement _reweights_ the prior $P_z (s) prop underbrace(P(z | s), "sensor likelihood") dot underbrace(P(s), "prior")$
  ]

  #uncover("2-")[
    *Prior* defined on the demands
    - Dirichlet: Given $d^0$ and $Delta>0$, uniform distribution on simplex ${d>= d^0 : bb(1)^top d <= bb(1)^top d + Delta}$
    - Gaussian: Normal distribution around $d^0$ with a std
  ]

  #uncover("3-")[
    *Sensor likelihood function $K_epsilon$:*
    - $P(z | s)$: encodes what we believe about _sensor noise_
    - We split: $quad quad quad P(z | s) quad prop quad underbrace(delta(D^z - D^s), "total demand: exact")
      dot underbrace(K_eps (h^s_cal(Y) - h^z_cal(Y)), "pressures: soft") $
    - Three choices for sensor likelihood 
    #table(
      columns: 3,
      stroke: none,

      [_Dirac_],   [$delta(h^s_cal(Y) - h^z_cal(Y))$], [no sensor error -- hard constraint],
      [_Uniform_], [$bb(1)[||h^s_cal(Y) - h^z_cal(Y)|| < eps]$], [sensor error bunded by $eps$],
      [_Gaussian_],[$exp(-1/(2 eps^2) ||h^s_cal(Y) - h^z_cal(Y)||^2)$], [gaussian sensor error with std $eps$],
    )
  ]
  #uncover("4-")[
    - $eps > 0$ softens; both Uniform and Gaussian $->$ Dirac as $eps -> 0$
    - Question: Which combination of Prior and Sensor likelihood is best?
  ]
])

== Maximum A-Posteriori Estimation (MAP) @Shao2019

#slide(repeat: 5, self => [
  #let (uncover, only) = utils.methods(self)

  #v(0.5em)
  #uncover("1-")[
    - Steady-state-model is _linearized around a fixed reference point_ $hat(s)$
    - Demands are gaussian-distributed around $d^(hat(s))$
  ]
  #uncover("2-")[
    - Use gaussian sensor-likelihood function:
    $
      P(z | s) prop exp(-1/(2epsilon^2)||h_cal(Y)^s-h^z_cal(Y)||^2)
    $
  ]
  #uncover("3-")[
    - Idea: By linearity the mean is equal to the _mode_ (most likely state):
  ]
  #uncover("4-")[
    $
      arg max_s P(s | z) &= arg max_s log(P(z | s)) + log(P(s))\
      &= arg min_s 1/(2epsilon^2)||h_cal(Y)^s-h^z_cal(Y)||^2 + 1/2||d^s-d^hat(s)||^2 
    $
    - This yields a smallest squares problem which can be solved by Newton's method
    - posterior distribution is approximated with _laplace-approximation_ (gives covariance)
  ] 

  #uncover("5-")[
    #columns(2, gutter: 8pt)[
    #h(0.1em) *+* #h(0.3em) fast and reliable solution \
    #h(0.1em) *--* #h(0.3em) requires fixed reference and linearizes\
    #colbreak()
    #h(0.1em) *--* #h(0.3em) only works with Gaussian prior\
    #h(0.1em) *--* #h(0.3em) posterior distribution is only approximated
    ]
  ]

])


== Approximate Bayesian Computation (ABC) @Sunnaker2013

#slide(repeat: 3, self => [
  #let (uncover, only) = utils.methods(self)

  #v(0.5em)
  #uncover("1-")[
    - _Algorithm:_ Sample $s ~ P$, and accept it with probability $P(z | s)$
    - for uniform likelihood, would be accepted if $||h^s_cal(Y)-h^z_cal(Y)||<eps$
    - for gaussian likelihood, would be accepted with probability $exp(-1/(2eps^2)||h^s_cal(Y)-h^z_cal(Y)||^2)$
    - dirac not applicable, since acceptance rate would be zero.
  ]
  #uncover("2-")[
    #h(0.1em) *+* #h(0.3em) Works with any prior, does not require linearization\
    #h(0.1em) *--* #h(0.3em) Acceptance rate scales with $(1/eps)^(|cal(Y)|)$ (curse of dimensionality)
  ]

  #v(0.5em)
  #uncover("3-")[
   - Essentially uses uniform sensor-likelihood
   $ P_"unf" (z | s) prop cases(1 "  if" ||h^s_cal(Y)-h^z_cal(Y)||<eps, 0 "  otherwise") $
  ]
])


== Monte Carlo Markov Chain (MCMC)

#slide(repeat: 4, self => [
  #let (uncover, only) = utils.methods(self)

  #columns(2, gutter: 8pt)[
  #v(0.5em)
  #uncover("1-")[
    - similar to ABC method, but uses dependend gernerates $(s_0,s_1,...)$ instead of independent samples
  ]
  
  #uncover("2-")[
    *Algorithm:* \
    - Assume we have already chosen $s_1,...,s_n$
    - Propose candidate $s_(n+1)$ by adding noise to $s_n$
    - If $cal(L)(s_(n+1)) >= cal(L)(s_n)$, accept the candidate
    - If not, accept with probability $(cal(L)(s_(n+1))) / (cal(L)(s_n)) <1$
  ]

  #v(0.5em)
  #uncover("3-")[
    - produces correct posterior if noise is symmetric (e.g. gaussian)
    - discard around first half of samples (burn-in period)
  ]
    
  #uncover("4-")[
    #h(0.1em) *+* #h(0.3em) Works with any prior, does not require linearization\
    #h(0.1em) *+* #h(0.3em) stays in high-likelihood-regions\
    #h(0.1em) *-* #h(0.3em) samples are _autocorrelated_, need more than usual \
    #h(0.1em) *-* #h(0.3em) Algorithm only works if it actually _mixes_\
      $quad$ (chain eventually independent of starting point)\
  ]
]

])

== What I've been trying

#slide(repeat: 3, self => [
  #let (uncover, only) = utils.methods(self)


  #uncover("1-")[
    *Exact Pressure Space approach*
    - Sample pressure vectors $h_v$ for $v in V without cal(Y)$, (we know $h_cal(Y) = h_cal(Y)^z$)
    - Calculate corresponding $d$.
    - Enforce $bb(1)^top d = D$ by changing $h_(v')$ accordingly, leads to hydraulic state $s$
    - calculate likelihood of state occuring via Jacobain determinant
  ]
  
  #uncover("2-")[
    *Exact Demand Space approach*
    - Sample demand vectors $d$ (all of them)
    - Calculate corresponding $h$.
    - Enforce $bb(1)^top d = D$ and $h_cal(Y) = h_cal(Y)^z$ by changing $d_cal(Y)$ and $d_v'$ accordingly, leads to hydraulic state $s$
    - calculate likelihood of state occuring via Jacobain determinant
  ]


  #uncover("3-")[
    *Inexact Demand Space approach:*
    - Instead of adjusting $d$ w.r.t $h_cal(Y) = h_cal(Y)^z$, use gaussion sensor-likelihood function
  ]


])


== My results

#slide(repeat: 1, self => [
  #let (uncover, only) = utils.methods(self)

  #columns(2, gutter: 20pt)[
  #v(0.5em)
  #uncover("1-")[
    *Correctness:* every sampler cross-validated against an independent _ABC oracle_ \
    (Alperovits, agreement in posterior-$sigma$):
    #v(0.4em)
    #table(
      columns: 2,
      stroke: none,
      align: (left, center),
      table.header([sampler], [gap to oracle]),
      [_MAP_ (Gaussian, MAP)],                  [$0.04 sigma$],
      [_MCMC_ pressure-space (exact)],          [$0.08 sigma$],
      [_MCMC_ demand-space (gaussian sensor)],  [$0.07 sigma$],
      [_MCMC_ demand-space (exact)],            [$0.01 sigma$],
      [_MCMC_ Gaussian prior],                  [$0.07 sigma$],
    )
    #v(0.2em)
    $-->$ MAP $approx$ MCMC-mean when near-Gaussian; MCMC recovers the _non-Gaussian_ shape.
  ]

  #colbreak()
  #v(0.5em)
  #uncover("2-")[
    *Making MCMC mix:*
    - an _affine-invariant ensemble_ proposal mixes on thin, high-dimensional posteriors where plain random-walk freezes
  ]
  #uncover("3-")[
    *Conditioning:*
    - sampling in _demand coordinates_ (not pressures) removes the ill-conditioning on _low-flow_ networks
  ]
  #uncover("4-")[
    *Limits (honest):*
    - many tight sensors + low flow $-->$ a _genuinely_ thin posterior: the demands are weakly identified (physics, not a bug)
    - sensor noise $eps$ is the practical lever
  ]
]

])

#bibliography("refs.bib")

== This is the end
#v(10em)
#set text(30pt)
#align(center)[
Thank you for your attention!
]
#set text(15pt)
#align(center)[
Shoutout to Fabian vdW for the Slide design
]





== Slide

@Gelman2013
@Bishop2006

#slide(repeat: 3, self => [
  #let (uncover, only) = utils.methods(self)

  #columns(2, gutter: 8pt)[
  #v(0.5em)
  #uncover("1-")[
    *blah:* 
  ]
  
  #uncover("2-")[
    *blah:* \
  ]


  #colbreak()
  #uncover("3-")[
    *blah:* \
  ]
]

])
