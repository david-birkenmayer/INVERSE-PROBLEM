#import "@preview/unequivocal-ams:0.1.2": ams-article
#import "@preview/lovelace:0.3.0": *

#import "commands.typ": *
#import "ctheorems.typ": *

#let template(doc) = {
  show: thmrules.with(qed-symbol: $square$)
  set page(width: 16cm, height: auto, margin: 1.5cm)
  set heading(numbering: "1.1.1")
  ams-article.with(
    title: [Polarization],
    bibliography: bibliography("../refs.bib"),
  )({
    show math.equation.where(block: false): box
    set par(first-line-indent: 0pt, spacing: 1em)
    doc
  })
}
