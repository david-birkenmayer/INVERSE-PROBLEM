#import "@preview/ctheorems:1.1.3": *

#let prp = thmbox("prop", "Proposition", within: "chapter")

#let thm = thmbox("theorem", "Theorem", 
  fill: rgb("#fc8383")
)

#let lem = thmbox("theorem","Lemma",
  titlefmt: strong,
  fill: rgb("#fac5f8"),
  base: "heading"
).with(numbering: none)

#let cl = thmbox("theorem","Claim",
  titlefmt: strong,
  fill: rgb("#fac5f8"),
  base: "heading"
).with(numbering: none)

#let obs = thmbox("theorem","Observation",
  titlefmt: strong,
  fill: rgb("#f7eaa8")
)

#let cor = thmbox("theorem","Corollary",
  titlefmt: strong,
  fill: rgb("#ffe7c7"),
  base: "heading"
)

#let def = thmbox("theorem", "Definition",
  fill: rgb("#c8ffc7")
).with(numbering: none)

#let example = thmbox("theorem", "Definition",
  fill: rgb("#73a9fa")
)//.with(numbering: none)

#let alg = thmbox("theorem", "Algorithm",
  fill: rgb("#73a9fa")
)//.with(numbering: none)

#let pf = thmproof("proof", "Proof")
