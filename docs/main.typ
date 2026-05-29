#import "setup/preamble.typ": template
#import "chapters/1_introduction.typ": chapter as introduction
#import "chapters/5_dynamic-multistart.typ": chapter as dynamic_multistart


#let selected-chapter = sys.inputs.at("chapter", default: "all")
#let show-chapter(name) = selected-chapter == "all" or selected-chapter == name

#show: template

#if show-chapter("metric-graphs") [
  #introduction
]

#if show-chapter("dms") [
  #dynamic_multistart
]

