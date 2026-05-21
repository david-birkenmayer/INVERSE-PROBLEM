#import "setup/preamble.typ": template
#import "chapters/1_introduction.typ": chapter as introduction


#let selected-chapter = sys.inputs.at("chapter", default: "all")
#let show-chapter(name) = selected-chapter == "all" or selected-chapter == name

#show: template

#if show-chapter("metric-graphs") [
  #introduction
]

