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


// ============================================================
// Slide 1: Steady State model
// ============================================================


== Steady state model

#slide(repeat: 5, self => [
  #let (uncover, only) = utils.methods(self)
  
  #v(0.5em)
  #columns(2, gutter: 8pt)[
  #uncover("1-")[
    *Fixed values:*
    - *wdn* as digraph $G = (V,E)$, where \
      $V = cal(J) union.dot {v_0}$ with $v_0$ reservoir node\
    - *pipe resistances* $r_e>0$ for $e in E$
    ]
    #uncover("3-")[
    - *reservoir pressure* $H in RR$

      #v(0.3em)
  #uncover("4-")[
    *Physical equations:*
    #v(0.35em)
  $
     bb(1)^top d &= D  && ("total demand")\ 
     B q &= - d  && ("mass")\
     B^top h &= phi.alt(q) - B_0^T dot H quad &&("energy")
  $
  where 
  - $B in RR^(cal(J) times E)$ is the $v_0$-reduced incidence matrix of $G$
  - $B_0$ is eliminated $v_0$-row of the incidence matrix 
  - $phi.alt(x)_e := r_e dot q_e dot |q_e|^(kappa-1)$ the nonlinear engery function

  ]
  ]
  #colbreak()
  #v(0.5em)
  #uncover("2-")[
    *Variables:*
    - *demand* $d_v>0$ for $v in cal(J)$
    - *flow* $q_e$ for $e in E$ with $q_(v,u) = -q_(u,v)$
    - *pressure head* $h_v$ for $v in cal(J)$
  ]
  #uncover("3-")[
    - *total demand* $D>0$
  ]

  #v(2em)
  #uncover("5-")[
  *Hydraulic states:*
  - A _hydraulic state_ is a tuple $s = (d^s,q^s,h^s, D^s)$ \ fulfilling the physical equations
  - $S$ is the set of all possible hydraulic states

  ]
  ]
  ])


// ============================================================
// Slides 2,3: Forward Problem
// ============================================================

== Our a-posteriori setting

#slide(repeat: 6, self => [
  #let (uncover, only) = utils.methods(self)

  #columns(2, gutter: 8pt)[
  #v(0.5em)
  #uncover("1-")[
    *Assumptions:* 
    - prior distribution $P$ on the states $S$
    - pressure measurement sites $cal(Y) subs cal(J)$ (hexagons)
    - an observation $z = (D^z, h^z_cal(Y) )$
  ]
  
  #uncover("2-")[
    *What we want to do:* \
    $->$ sample a-posteriori distribution $P_z := s mapsto P(s | z)$\
    $quad$ That is, with conditions $D^s = D^z$ and $h^s_cal(Y) = h^z_cal(Y)$
  ]

  #uncover("3-")[
    *Questions:* \
    - resonable choice for $P$?
    - soften the conditions to account for sensor noise?
    - how do we calculate $P_z$?
  ]

  #colbreak()
  [insert image here]
  ]

])

== Slide

#slide(repeat: 6, self => [
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

])



== Slide

#slide(repeat: 6, self => [
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