#import "../setup/preamble.typ": template
#import "../setup/commands.typ": *
#import "../setup/ctheorems.typ": *

#show: template

#let chapter = [

= Dynamic Multi-Start (DMS) for robust radius estimates

== Goal
For a fixed network and fixed measurement sites, the solver may return very small local optima for
$W_d$ although larger values exist. DMS replaces a single run by an adaptive sequence of starts and
stores a certificate interval in each artifact.

== Parameters
- $k$ (consistency): positive integer, default $k = 3$.
- $delta$ (deviation): real number in $(0, 1]$, default $delta = 0.95$.

== Inputs and outputs
- Input state: current reference radius $r$ from local search state (not stored in artifact).
- Single-start outputs: radii $r_1, r_2, dots$ from repeated starts for the same configuration.
- Artifact certificate: two fields
  - lower_bound
  - upper_bound

== One DMS run on a fixed configuration
Assume one DMS invocation produces single-start radii $r_1, r_2, dots$.
Sort descending as
$
  r_(1) >= r_(2) >= r_(3) >= dots
$.

DMS can produce two certificate types:

1. No-improvement certificate
If at least one start satisfies $r_i >= r$, then write
$
  "lower_bound" := r_i, quad "upper_bound" := -infinity.
$
This certifies that the current configuration is not better than the current reference
$r$ (for minimization). Hence this configuration can be discarded safely.

2. Improvement certificate
If at least $k$ starts were made and
$
  r_(k) >= delta dot r_(1),
$
then write
$
  "lower_bound" := r_(1), quad "upper_bound" := r_(1).
$
This marks a stable value under the DMS consistency rule and is treated as a relatively
confident improvement estimate.

== Cache interaction with current radius $r$
When a DMS artifact with $("lower_bound", "upper_bound")$ is considered for caching together with current
radius $r$, cache it only if
$
  r < "lower_bound" quad "or" quad r > "upper_bound".
$

Otherwise, run DMS again with the same current $r$. If a new artifact is produced, update bounds by
merging old and new certificates:
$
  "lower_bound" := min("lower_bound_old", "lower_bound_new"),
$
$
  "upper_bound" := max("upper_bound_old", "upper_bound_new").
$

== Local-search usage
For each measurement configuration:
- Start with $r = infinity$.
- Call DMS with this $r$.
- Use returned certificate and update current best radius as local search proceeds.
- For every new candidate configuration, call DMS with the current best $r$.

This gives an adaptive budget: configurations near the best known radius are re-tested more often,
while clearly bad or clearly better candidates are decided faster.

== Practical interpretation
- Single-start feasibility alone is not treated as fully reliable quality information.
- DMS promotes values that are reproducible across starts.
- The certificate interval separates
  - no-improvement certificates (configuration is worse and can be discarded), and
  - improvement certificates (configuration is likely better and stable under $k, delta$).

]

#chapter
