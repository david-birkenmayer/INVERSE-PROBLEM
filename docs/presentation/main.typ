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
  - Givenpressures $h$, $d$ and $D$ are unique. \ Calculation is ill-conditioned around low flows.

  ]
  ]
  ])

== Our model

#slide(repeat: 4, self => [
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

  #v(0.5em)
  #uncover("3-")[
    $-->$ Want to find a good predictor $F: Z -> S$ (e.g. GNN)\
  ]

  #uncover("4-")[
    $-->$ But how do we determine what "good" is?
  ]

  #colbreak()
  #uncover("3-")[
    *Insert Image* \
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
    - _measurement sites_ $cal(Y) subs cal(J)$ (hexagons)
    - _observation_ $z = (D^z, h^z_cal(Y) )$
  ]
  
  #uncover("2-")[
    *What we want to do:* \
    $->$ sample _posterior distribution_ $P_z := s mapsto P(s | z)$\
    $quad$ That is, with conditions $D^s approx D^z$ and $h^s_cal(Y) approx h^z_cal(Y)$
  ]

  #uncover("3-")[
    *Why do we want this:* \
    - Can measure how good a prediction $F(s)$ is, by comparing it to the sampled results
    - Shows how confident we can be in the prediction in certain areas
    - Averaging the samples yields a predictor itself!
  ]

  #colbreak()
  [insert image here]
  ]

])

== Open questions

#slide(repeat: 3, self => [
  #let (uncover, only) = utils.methods(self)

  #columns(2, gutter: 8pt)[
  #v(0.5em)
  #uncover("1-")[
    *Questions:* \
    - resonable choice for $P$?
    - soften the conditions to account for sensor noise?
    - how do we calculate $P_z$?
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

== Maximum A-Posteriori Estimation (MAP)

#slide(repeat: 5, self => [
  #let (uncover, only) = utils.methods(self)

  #v(0.5em)
  #uncover("1-")[
    - Steady-state-model is _linearized around a fixed reference point_ $hat(s)$
    - Demands are gaussian-distributed around $d^(hat(s))$
  ]
  #uncover("2-")[
    - Use sensor-likelihood function:
    $
      P(z | s) prop exp(-1/(2epsilon^2)||h_cal(Y)^s-h^z_cal(Y)||^2)
    $
  ]
  #uncover("3-")[
    - Idea: By linearity the mean is equal to the _mode_ (most likely $s$ given $z$):
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
  #h(0.1em) *+* #h(0.3em) fast and reliable solution \
  #h(0.1em) *--* #h(0.3em) requires fixed reference and linearizes\
  #h(0.1em) *--* #h(0.3em) only works with Gaussian prior\
  #h(0.1em) *--* #h(0.3em) posterior distribution is only approximated
  ]

])


== Approximate Bayesian Computation (ABC)

#slide(repeat: 3, self => [
  #let (uncover, only) = utils.methods(self)

  #v(0.5em)
  #uncover("1-")[
    - _Algorithm:_ Sample $s ~ P$, and accept it only if $||h^s_cal(Y)-h^z_cal(Y)||<eps$
  ]
  #uncover("2-")[
    #h(0.1em) *+* #h(0.3em) Works with any prior and without a fixed reference point\
    #h(0.1em) *+* #h(0.3em) Approximates exact posterior as $eps-> 0$\
    #h(0.1em) *--* #h(0.3em) Acceptance rate scales with $(1/eps)^(|cal(Y)|)$ (curse of dimensionality)
  ]

  #v(0.5em)
  #uncover("3-")[
   - Essentially uses uniform sensor-likelihood
   $ P_"unf" (z | s) prop cases(1 "  if" ||h^s_cal(Y)-h^z_cal(Y)||<eps, 0 "  otherwise") $
  ]
  - Could also use different sensor-likelihood, but runs into same problem

])


== Slide

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


== This is the end
#v(10em)
#set text(30pt)
#align(center)[
Thank you for your attention!
]



== Slide

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
