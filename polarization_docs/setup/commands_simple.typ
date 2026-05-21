//text commands
#let equivalence(a, b) = ["$==>$" \ #a \  "$<==$" \ #b]
#let backward = ["$<==$" \ ]
#let forward = ["$==>$" \ ]

//math shorthands
#let subs = $subset.eq$
#let sups = $supset.eq$
#let eps = $epsilon$
#let ni = $in.rev$
#let nin = $in.not$
#let nni = $in.not.rev$
#let cup = $union$
#let cap = $inter$
#let infty = $infinity$

//math operators
#let argmin = $"argmin"$
#let dist = $"dist"$
#let conv = $"conv"$


//math commands
#let bset(a, b) = ${#a | #b}$
#let time(a) = $cal(O)(#a)$

//problem-name formatting
#let problem-name(name) = text(font: "TeX Gyre Heros", weight: "semibold", smallcaps(name))