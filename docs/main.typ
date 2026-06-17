#import "setup/preamble.typ": template
#import "chapters/1_introduction.typ": chapter as introduction
#import "chapters/5_dynamic-multistart.typ": chapter as dynamic_multistart
#import "chapters/6_wd-implementation.typ": chapter as wd_implementation
#import "chapters/7_mcmc-posteriori-scenario-generation.typ": chapter as mcmc_posteriori


#let selected-chapter = sys.inputs.at("chapter", default: "all")
#let show-chapter(name) = selected-chapter == "all" or selected-chapter == name

#show: template

#if show-chapter("metric-graphs") [
  #introduction
]

#if show-chapter("dms") [
  #dynamic_multistart
]

#if show-chapter("wd-implementation") [
  #wd_implementation
]

#if show-chapter("mcmc-posteriori") [
  #mcmc_posteriori
]

