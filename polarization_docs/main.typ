#import "setup/preamble.typ": template
#import "chapters/1_metric-graphs.typ": chapter as metric-graphs-chapter
#import "chapters/2_polarization-problem.typ": chapter as polarization-problem-chapter
#import "chapters/3_line-case.typ": chapter as line-case-chapter
#import "chapters/4_continuity-and-convexity.typ": chapter as continuity-chapter
#import "chapters/5_equioscillation.typ": chapter as trees-chapter
#import "chapters/6_experimental-first-order.typ": chapter as experimental-chapter
#import "chapters/7_vertex-discrete-polarization.typ": chapter as vertex-discrete-chapter
#import "chapters/8_NP-hardness.typ": chapter as np-hardness-chapter

#let selected-chapter = sys.inputs.at("chapter", default: "all")
#let show-chapter(name) = selected-chapter == "all" or selected-chapter == name

#show: template

#if show-chapter("metric-graphs") [
  #metric-graphs-chapter
]

#if show-chapter("polarization-problem") [
  #polarization-problem-chapter
]

#if show-chapter("line-case") [
  #line-case-chapter
]

#if show-chapter("continuity") [
  #continuity-chapter
]

#if show-chapter("trees") [
  #trees-chapter
]

#if show-chapter("experimental") [
  #experimental-chapter
]

#if show-chapter("vertex-discrete") [
  #vertex-discrete-chapter
]
