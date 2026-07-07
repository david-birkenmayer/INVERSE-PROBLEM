#import "university.typ": *
#let ub = "ub"
#let lb = "lb"
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
    title: "Perturbationstheorie für Optimierungsprobleme",
    subtitle: "Anwendung auf Schüler-Zuweisungsprobleme",
    author: ("Fabian von der Warth"),
    institution: "AG Optimierung, RPTU Kaiserslautern",
  ),
)

== Schüler-Zuweisungsproblem mit Absagen (SACP)

#slide(repeat: 5, self => [
  #let (uncover, only) = utils.methods(self)

  #uncover("1-")[
    Gegeben: Eine Menge $I$ von *Schülern* und eine Menge $J$ von *Projekten*.
  ]

  #v(0.3em)

  #uncover("2-")[
    *Ziel:* Weise Schülern optimal Projekte zu, wobei Projekte auch abgesagt werden können.
  ]

  #v(0.3em)

  #uncover("3-")[
    *Nebenbedingungen:*
    - Jeder Schüler wird genau einem Projekt zugewiesen, Kosten für Zuweisung von Schüler $i$ zu Projekt $j$: $w_(i j)$
    - Projekte haben Unter- und Obergrenzen $lb(j), ub(j)$ für Teilnehmerzahlen
    - Projekte können abgesagt werden ($y_j$)
  ]

  #v(0.3em)

  #uncover("4-")[
    *Formal (SACP):*
  ]
  #uncover("5-")[
    $
    &"min" & sum_(i in I, j in J) w_(i j) x_(i j) \
    &"s.t."& sum_(j in J) x_(i j) &= 1 && forall i in I \
    && lb(j) dot y_j <= sum_(i in I) x_(i j) &<= ub(j) dot y_j & quad &forall j in J \
    && x in {0, 1}^(I times J),& y in {0, 1}^J
    $
  ]
])

== Perturbationsidee

#slide(repeat: 5, self => [
  #let (uncover, only) = utils.methods(self)

  #uncover("1-")[
    *Problem:* Die Parameter $lb(j)$ und $ub(j)$ sind vom Entscheider vorab festgelegt — teilweise arbiträr
  ]

  #v(0.2em)

  #uncover("2-")[
    #block(fill: blue.lighten(85%), inset: 10pt, radius: 4pt)[
      *Kernidee:* Wenn ich ein Zielfunktionswert $f^*$ erreichen will, wie muss ich die Grenzen minimal ändern, um dies zu erreichen?
    ]
  ]

  #v(0.3em)

  #uncover("3-")[
    - Pertubationsvariablen $u_j, v_j in NN_0$: Änderung der Grenzen von Projekt $j$ (Kosten $c_j$ / $d_j$)
  ]

  #v(0.3em)

  #uncover("4-")[
    *Perturbiertes Modell (PSACP):*
  ]
  #uncover("5-")[
    $
    &"min" & sum_(j in J) c_j dot u_j + d_j dot v_j \
    &"s.t." & sum_(i in I, j in J) w_(i j) x_(i j) &<= f^* \
    && sum_(j in J) x_(i j) &= 1 && forall i in I \
    && lb(j) y_j - u_j <= sum_(i in I) x_(i j) &<= ub(j) y_j + v_j && forall j in J \
    && x in {0, 1}^(I times J), &y in {0,1}^J, u,v in NN_0^J
    $
  ]
])

// ============================================================
// Slide 3: General Perturbation Model
// ============================================================
== Allgemeines Perturbationsmodell

#slide(repeat: 4, self => [
  #let (uncover, only) = utils.methods(self)

  #uncover("1-")[
    Die Idee ist nicht auf Zuweisungsprobleme beschränkt. Gegeben ein beliebiges MMILP:
    $
    "min" C x, quad A x <= b, quad x in RR^n times ZZ^p
    $
  ]

  #v(0.5em)

  #uncover("2-")[
    Perturbiere _alle_ Modellparameter ($A$, $b$, $C$) gleichzeitig:
  ]

  #uncover("3-")[
    $
    &"min" & (C + u_C) x \
    &"min" & "budget"(|u_A|, |u_b|, |u_C|) \
    &"s.t." & \
    && "budget"(|u_A|, |u_b|, |u_C|) &<= epsilon \
    && (C + u_C) x &<= f^* \
    && (A + u_A) x &<= (b + u_b) \
    && x in RR^n times ZZ^p, u_A in RR^(m times n), &u_b in RR^m, u_C in RR^(p times n)
    $
  ]

  #v(0.5em)

  #uncover("4-")[
    #block(fill: luma(240), inset: 10pt, radius: 4pt)[
      *Achtung:* Dieses Modell ist im Allgemeinen _nicht_ mehr linear (Terme $u_C x$, $u_A x$). Spezialfälle wie PSACP bleiben jedoch (gemischt-ganzzahlig) linear.
    ]
  ]
])


== Diskussion & Fragen

#v(1em)

#block(fill: blue.lighten(85%), inset: 15pt, radius: 6pt)[
  #text(size: 20pt)[
    *Wir würden gerne erfahren:*
  ]

  #v(0.8em)

  #enum(
    spacing: 1em,
    [Gibt es bei Fraunhofer Optimierungsprojekte, bei denen die Modellparameter nicht exakt feststehen — und eine Perturbationsanalyse wertvoll wäre?],
    [Wird so eine Idee bereits in Projekten eingesetzt?]
  )
]

#v(2em)

#align(center)[
  #text(size: 26pt, weight: "bold")[Vielen Dank für Ihre Zeit!]
]
