// University theme

// Originally contributed by Pol Dellaiera - https://github.com/drupol

#import "@preview/touying:0.6.1": *

// RPTU Color Definitions (from Brand Manual)
#let rptu-colors = (
  // Primary colors
  dunkelblau: rgb(4, 44, 88),      // nacht
  hellblau: rgb(106, 178, 231),    // tag
  rot: rgb(227, 27, 76),            // himbeere
  orange: rgb(255, 162, 82),        // mango
  dunkelgruen: rgb(0, 107, 107),    // petrol
  hellgruen: rgb(38, 208, 124),     // apfel
  blaugrau: rgb(80, 114, 137),      // schiefer
  gruengrau: rgb(119, 182, 186),    // ozean
  violett: rgb(76, 53, 117),        // pflaume
  pink: rgb(209, 56, 150),          // fuchsia
  schwarz: rgb(0, 0, 0),
  weiss: rgb(255, 255, 255),
)

// Color pair mappings (main color -> secondary color)
#let rptu-color-pairs = (
  dunkelblau: rptu-colors.hellblau,
  hellblau: rptu-colors.dunkelblau,
  rot: rptu-colors.orange,
  orange: rptu-colors.rot,
  dunkelgruen: rptu-colors.hellgruen,
  hellgruen: rptu-colors.dunkelgruen,
  blaugrau: rptu-colors.gruengrau,
  gruengrau: rptu-colors.blaugrau,
  violett: rptu-colors.pink,
  pink: rptu-colors.violett,
)

/// Default slide function for the presentation.
///
/// - config (dictionary): is the configuration of the slide. Use `config-xxx` to set individual configurations for the slide. To apply multiple configurations, use `utils.merge-dicts` to combine them.
///
/// - repeat (int, auto): is the number of subslides. The default is `auto`, allowing touying to automatically calculate the number of subslides. The `repeat` argument is required when using `#slide(repeat: 3, self => [ .. ])` style code to create a slide, as touying cannot automatically detect callback-style `uncover` and `only`.
///
/// - setting (dictionary): is the setting of the slide, which can be used to apply set/show rules for the slide.
///
/// - composer (array, function): is the layout composer of the slide, allowing you to define the slide layout.
///
///   For example, `#slide(composer: (1fr, 2fr, 1fr))[A][B][C]` to split the slide into three parts. The first and the last parts will take 1/4 of the slide, and the second part will take 1/2 of the slide.
///
///   If you pass a non-function value like `(1fr, 2fr, 1fr)`, it will be assumed to be the first argument of the `components.side-by-side` function.
///
///   The `components.side-by-side` function is a simple wrapper of the `grid` function. It means you can use the `grid.cell(colspan: 2, ..)` to make the cell take 2 columns.
///
///   For example, `#slide(composer: 2)[A][B][#grid.cell(colspan: 2)[Footer]]` will make the `Footer` cell take 2 columns.
///
///   If you want to customize the composer, you can pass a function to the `composer` argument. The function should receive the contents of the slide and return the content of the slide, like `#slide(composer: grid.with(columns: 2))[A][B]`.
///
/// - bodies (arguments): is the contents of the slide. You can call the `slide` function with syntax like `#slide[A][B][C]` to create a slide.
#let slide(
  config: (:),
  repeat: auto,
  setting: body => body,
  composer: auto,
  align: auto,
  ..bodies,
) = touying-slide-wrapper(self => {
  if align != auto {
    self.store.align = align
  }
  let header(self) = {
    set std.align(top)
    let current-title = utils.call-or-display(self, self.store.header)
    if current-title != none and current-title != [] {
      block(
        width: 100%,
        inset: (x: 1em, top: 1em, bottom: 2em),
        breakable: false,
        [
          #grid(
            columns: (1fr, auto),
            align: (left, center),
            [
              #text(
                size: 1.3em,
                weight: "bold", 
                fill: self.colors.secondary,
                font: "Latin Modern Sans 12",
                current-title
              )
            ],
            [
              // Progress dots in upper right
              #box(
                inset: (left: 1em, top: 0.2em)
              )[
                #context {
                  let current-slide = utils.slide-counter.get().first()
                  let dot-count = utils.slide-counter.final().first()
                  
                  // One dot per content slide
                  let filled-dots =  calc.min(dot-count, current-slide )         
          
                  // Create progress dots
                  for i in range(dot-count) {
                    let dot_size = 12pt
                    if i < filled-dots {
                      // Filled dot
                      text(fill: self.colors.secondary, size: dot_size, font: "DejaVu Sans Mono")[●]
                    } else {
                      // Empty dot
                      text(fill: self.colors.secondary, size: dot_size, font: "DejaVu Sans Mono")[○]
                    }
                    if i < dot-count - 1 {
                      h(2pt)
                    }
                  }
                  h(20pt)
                }
              ]
            ]
          )
          #v(0.2em)
          #line(length: 6%, stroke: 1pt + self.colors.secondary)
          #v(0.5em)
        ]
      )
    }
  }
  //v(0.5em)
  let footer(self) = {
    set std.align(bottom)
    set text(size: 10pt, font: "Latin Modern Sans 12")
    block(
      width: 100%,
      inset: (x: 2em, y: 1em),
      [
        #grid(
          columns: (auto, 1fr, auto),
          align: (left, left, right),
          [
            #context utils.slide-counter.display()
            #h(1em) 
            #utils.call-or-display(self, self.store.footer-a)
          ],
          [
            #h(1em) #utils.call-or-display(self, self.store.footer-b)
          ],
          move(dy:16pt, dx:5pt)[
            #image("logos/Minimalversion/RPTU Logo_minimal_1cschwarz.svg", height: 4.2em)
          ]
        )
      ]
    )
  }
  let self = utils.merge-dicts(
    self,
    config-page(
      header: header,
      footer: footer,
    ),
  )
  let new-setting = body => {
    show: std.align.with(self.store.align)
    show: setting
    body
  }
  touying-slide(self: self, config: config, repeat: repeat, setting: new-setting, composer: composer, ..bodies)
})


/// Title slide for the presentation. You should update the information in the `config-info` function. You can also pass the information directly to the `title-slide` function.
///
/// Example:
///
/// ```typst
/// #show: university-theme.with(
///   config-info(
///     title: [Title],
///     logo: emoji.school,
///   ),
/// )
///
/// #title-slide(subtitle: [Subtitle])
/// ```
/// 
/// - config (dictionary): is the configuration of the slide. Use `config-xxx` to set individual configurations for the slide. To apply multiple configurations, use `utils.merge-dicts` to combine them.
///
/// - extra (string, none): is the extra information for the slide. This can be passed to the `title-slide` function to display additional information on the title slide.
#let title-slide(
  config: (:),
  extra: none,
  ..args,
) = touying-slide-wrapper(self => {
  self = utils.merge-dicts(
    self,
    config,
    config-common(freeze-slide-counter: true),
    config-page(
      fill: self.colors.primary,
      header: none,
      footer: none,
      margin: 0em,
    ),
  )
  let info = self.info + args.named()
  info.authors = {
    let authors = if "authors" in info {
      info.authors
    } else {
      info.author
    }
    if type(authors) == array {
      authors
    } else {
      (authors,)
    }
  }
  let body = {
    set text(fill: white)
    
    // Add RPTU logos in top right
    place(
      top + right,
      dx: -100pt,
      dy: 40pt,
      [
        #image("logos/U_Farben/RPTU U.png", height: 6cm)
      ]
    )
    place(
      top + right,
      dx: -180pt,
      dy: 120pt,
      [
        #image("logos/U_Farben/RPTU U12.png", height: 5cm)
      ]
    )
    place(
      bottom + right,
      dx: -50pt,
      dy: 30pt,
      [
        #image("logos/SVG/RPTU Logo_1cweiss-1.svg", height: 6cm)
      ]
    )
    // Main content grid
    box(width: 50%)[#grid(
      columns: (1fr),
      rows: (1fr, auto),
      
      // Main content area
      [
        #place(
          left + horizon,
          dx: 60pt,
          dy: -80pt,
          [
            #text(size: 32pt, weight: "bold")[
              #info.title
            ]
            #line(length: 5%, stroke: 1.5pt + self.colors.secondary)
            #if info.subtitle != none [
              #v(0.3em)
              #text(size: 20pt, weight: "medium")[
                #info.subtitle
              ]
            ]
          ]
        )
      ],
      
      // Bottom area with author info and RPTU logo
      [
        #grid(
          columns: (1fr, auto),
          [
            #place(
              left + bottom,
              dx: 60pt,
              dy: -40pt,
              [
                #if info.authors.len() > 0 [
                  #text(size: 18pt, weight: "medium")[#info.authors.first()]
                  #v(0.3em)
                ]
                #if info.institution != none [
                  #text(size: 16pt)[#info.institution]
                  #v(0.3em)
                ]
                #if info.date != none [
                  #text(size: 16pt)[#utils.display-info-date(self)]
                ]
              ]
            )
          ],
          [
            #place(
              right + bottom,
              dx: -40pt,
              dy: -40pt,
              image("logos/SVG/RPTU Logo_1cweiss-1.svg", height: 1.5em)
            )
          ]
        )
      ]
    )]
  }
  touying-slide(self: self, body)
})

/// RPTU-style theorem/lemma block
#let rptu-block(title: none, body, color: none) = {
  let bg-color = if color != none { color } else { rptu-colors.dunkelblau }
  block(
    width: 100%,
    fill: bg-color,
    inset: (x: 1em, y: 0.8em),
    radius: 2pt,
    [
      #set text(fill: white, font: "Latin Modern Sans 12")
      #if title != none [
        #text(size: 1em, weight: "bold", fill: white)[#title]
        #v(0.3em)
      ]
      #text(style: "italic")[#body]
    ]
  )
}

/// RPTU-style alert/attention block  
#let rptu-alert(body) = {
  text(size: 1em, weight: "bold", fill: rptu-colors.dunkelblau)[ACHTUNG! ] + body
}

/// Purple background slide (like "Bunt" slide in LaTeX template)
#let purple-slide(
  config: (:),
  ..args
) = touying-slide-wrapper(self => {
  self = utils.merge-dicts(
    self,
    config,
    config-page(
      fill: self.colors.primary,
      margin: (top: 4em, bottom: 2.5em, x: 2em),
    ),
  )
  let body = args.pos().join()
  let styled-body = {
    set text(fill: white, font: "Latin Modern Sans 12")
    
    // Style headings for colored slides
    show heading.where(level: 1): it => {
      set text(size: 1.4em, weight: "bold", fill: self.colors.secondary)
      block(below: 0.3em, it.body)
      line(length: 100%, stroke: 1pt + self.colors.secondary)
      v(0.8em)
    }
    
    body
  }
  touying-slide(self: self, styled-body)
})

/// Author funding slide for the presentation. Replicates the RPTU author slide design.
///
/// - config (dictionary): is the configuration of the slide.
/// - extra (string, none): is the extra information for the slide.
#let author-funding-slide(
  config: (:),
  extra: none,
  ..args,
) = touying-slide-wrapper(self => {
  self = utils.merge-dicts(
    self,
    config,
    config-common(freeze-slide-counter: true),
    config-page(
      fill: self.colors.primary,
      header: none,
      footer: none,
    ),
  )
  let info = self.info + args.named()
  info.authors = {
    let authors = if "authors" in info {
      info.authors
    } else {
      info.author
    }
    if type(authors) == array {
      authors
    } else {
      (authors,)
    }
  }
  
  // Get additional info from args
  let author-pictures = if "author-pictures" in args.named() { args.named().author-pictures } else { () }
  let author-affiliations = if "author-affiliations" in args.named() { args.named().author-affiliations } else { none }
  let affiliations = if "affiliations" in args.named() { args.named().affiliations } else { none }
  let sponsor-logos = if "sponsor-logos" in args.named() { args.named().sponsor-logos } else { none }
  
  let body = {
    set par(spacing: 12pt)
  }
  
  touying-slide(self: self, body)
})


/// New section slide for the presentation. You can update it by updating the `new-section-slide-fn` argument for `config-common` function.
///
/// Example: `config-common(new-section-slide-fn: new-section-slide.with(numbered: false))`
///
/// - config (dictionary): is the configuration of the slide. Use `config-xxx` to set individual configurations for the slide. To apply multiple configurations, use `utils.merge-dicts` to combine them.
/// 
/// - level (int, none): is the level of the heading.
///
/// - numbered (boolean): is whether the heading is numbered.
///
/// - body (auto): is the body of the section. This will be passed automatically by Touying.
#let new-section-slide(config: (:), level: 1, numbered: true, body) = touying-slide-wrapper(self => {
  let slide-body = {
    set std.align(horizon)
    show: pad.with(20%)
    set text(size: 1.5em, fill: self.colors.primary, weight: "bold")
    stack(
      dir: ttb,
      spacing: .65em,
      utils.display-current-heading(level: level, numbered: numbered),
      block(
        height: 2pt,
        width: 100%,
        spacing: 0pt,
        components.progress-bar(height: 2pt, self.colors.primary, self.colors.primary-light),
      ),
    )
    body
  }
  touying-slide(self: self, config: config, slide-body)
})


/// Focus on some content.
///
/// Example: `#focus-slide[Wake up!]`
/// 
/// - config (dictionary): is the configuration of the slide. Use `config-xxx` to set individual configurations for the slide. To apply multiple configurations, use `utils.merge-dicts` to combine them.
///
/// - background-color (color, none): is the background color of the slide. Default is the primary color.
///
/// - background-img (string, none): is the background image of the slide. Default is none.
#let focus-slide(config: (:), background-color: none, background-img: none, body) = touying-slide-wrapper(self => {
  let background-color = if background-img == none and background-color == none {
    rgb(self.colors.primary)
  } else {
    background-color
  }
  let args = (:)
  if background-color != none {
    args.fill = background-color
  }
  if background-img != none {
    args.background = {
      set image(fit: "stretch", width: 100%, height: 100%)
      background-img
    }
  }
  self = utils.merge-dicts(
    self,
    config-common(freeze-slide-counter: true),
    config-page(margin: 1em, ..args),
  )
  set text(fill: self.colors.neutral-lightest, weight: "bold", size: 2em)
  touying-slide(self: self, std.align(horizon, body))
})


// Create a slide where the provided content blocks are displayed in a grid and coloured in a checkerboard pattern without further decoration. You can configure the grid using the rows and `columns` keyword arguments (both default to none). It is determined in the following way:
///
/// - If `columns` is an integer, create that many columns of width `1fr`.
/// - If `columns` is `none`, create as many columns of width `1fr` as there are content blocks.
/// - Otherwise assume that `columns` is an array of widths already, use that.
/// - If `rows` is an integer, create that many rows of height `1fr`.
/// - If `rows` is `none`, create that many rows of height `1fr` as are needed given the number of co/ -ntent blocks and columns.
/// - Otherwise assume that `rows` is an array of heights already, use that.
/// - Check that there are enough rows and columns to fit in all the content blocks.
///
/// That means that `#matrix-slide[...][...]` stacks horizontally and `#matrix-slide(columns: 1)[...][...]` stacks vertically.
/// 
/// - config (dictionary): is the configuration of the slide. Use `config-xxx` to set individual configurations for the slide. To apply multiple configurations, use `utils.merge-dicts` to combine them.
#let matrix-slide(config: (:), columns: none, rows: none, ..bodies) = touying-slide-wrapper(self => {
  self = utils.merge-dicts(
    self,
    config-common(freeze-slide-counter: true),
    config-page(margin: 0em),
  )
  touying-slide(self: self, config: config, composer: components.checkerboard.with(columns: columns, rows: rows), ..bodies)
})


/// Touying university theme.
///
/// Example:
///
/// ```typst
/// #show: university-theme.with(aspect-ratio: "16-9", config-colors(primary: blue))`
/// ```
///
/// The default colors:
///
/// ```typ
/// config-colors(
///   primary: rgb("#04364A"),
///   secondary: rgb("#176B87"),
///   tertiary: rgb("#448C95"),
///   neutral-lightest: rgb("#ffffff"),
///   neutral-darkest: rgb("#000000"),
/// )
/// ```
///
/// - aspect-ratio (string): is the aspect ratio of the slides. Default is `16-9`.
/// 
/// - align (alignment): is the alignment of the slides. Default is `top`.
///
/// - progress-bar (boolean): is whether to show the progress bar. Default is `true`.
///
/// - header (content, function): is the header of the slides. Default is `utils.display-current-heading(level: 2)`.
///
/// - header-right (content, function): is the right part of the header. Default is `self.info.logo`.
///
/// - footer-columns (tuple): is the columns of the footer. Default is `(25%, 1fr, 25%)`.
///
/// - footer-a (content, function): is the left part of the footer. Default is `self.info.author`.
///
/// - footer-b (content, function): is the middle part of the footer. Default is `self.info.short-title` or `self.info.title`.
///
/// - footer-c (content, function): is the right part of the footer. Default is `self => h(1fr) + utils.display-info-date(self) + h(1fr) + context utils.slide-counter.display() + " / " + utils.last-slide-number + h(1fr)`.
#let university-theme(
  aspect-ratio: "16-9",
  align: top,
  progress-bar: true,
  header: self => {
    let current-slide-title = utils.display-current-heading(level: 2, style: auto)
    if current-slide-title != none and current-slide-title != [] {
      current-slide-title
    } else {
      // Fallback to presentation title if no slide title
      if self.info.short-title == auto {
        self.info.title
      } else {
        self.info.short-title
      }
    }
  },
  header-right: self => {
    let content-parts = ()
    
    // Add current section if available
    let current-section = utils.display-current-heading(level: 1)
    if current-section != none and current-section != [] {
      content-parts.push(current-section)
    }
    
    // Add RPTU logo if no custom logo is set
    if self.info.logo == none {
      content-parts.push(
        image("logos/SVG/RPTU Logo_1c-1.svg", height: 2em)
      )
    } else {
      content-parts.push(self.info.logo)
    }
    
    // Join parts with spacing
    content-parts.join(h(0.5em))
  },
  footer-columns: (25%, 1fr, 25%),
  footer-a: self => {
    let authors = if "authors" in self.info {
      self.info.authors
    } else {
      self.info.author
    }
    if type(authors) == array {
      authors.first()
    } else {
      authors
    }
  },
  footer-b: self => if self.info.short-title == auto {
    self.info.title
  } else {
    self.info.short-title
  },
  footer-c: self => {
    h(1fr)
    utils.display-info-date(self)
    h(1fr)
    context utils.slide-counter.display() + " / " + utils.last-slide-number
    h(1fr)
  },
  // RPTU specific parameters
  main-color: "dunkelblau",
  secondary-color: "hellblau",
  author-pictures: (),
  author-affiliations: none,
  affiliations: none,
  sponsor-logos: none,
  ..args,
  body,
) = {
  // Get RPTU colors based on main-color parameter
  let primary-color = rptu-colors.at(main-color, default: rptu-colors.dunkelblau)
  let secondary-color-val = rptu-color-pairs.at(main-color, default: rptu-colors.hellblau)
  
  show: touying-slides.with(
    config-page(
      paper: "presentation-" + aspect-ratio,
      header-ascent: 0em,
      footer-descent: 0em,
      margin: (top: 3.5em, bottom: 2em, x: 1em, y:1em),
      fill: white,
    ),
    config-common(
      slide-fn: slide,
      new-section-slide-fn: new-section-slide,
    ),
    config-methods(
      init: (self: none, body) => {
        set text(size: 16pt, font: "Latin Modern Sans 12")
        show math.equation: set text(font: "Latin Modern Math")
        show strong: set text(fill: red)
        set par(leading: 0.8em)
        
        // Style headings with reasonable sizes
        show heading.where(level: 1): it => {
          set text(size: 1.4em, weight: "bold", fill: self.colors.secondary)
          block(below: 0.3em, it.body)
          line(length: 100%, stroke: 1pt + self.colors.secondary)
          v(0.8em)
        }
        
        show heading.where(level: 2): it => {
          set text(size: 1.2em, weight: "bold", fill: self.colors.secondary)
          block(below: 0.2em, it.body)
          v(0.4em)
        }
        
        show heading.where(level: 3): set text(fill: self.colors.primary)
        show heading.where(level: 4): set text(fill: self.colors.primary)

        show sym.colon: $class("fence", colon)$
        // Style lists to match LaTeX design
        show list: it => {
          set text(size: 1em)
          it
        }
        
        show enum: it => {
          set text(size: 1em)
          it
        }

        body
      },
      // alert: utils.alert-with-primary-color, // Removed to stop bold text from using primary color
    ),
    config-colors(
      primary: primary-color,  // dunkelblau for backgrounds
      secondary: secondary-color-val,  // hellblau for accents
      tertiary: rptu-colors.blaugrau,
      neutral-lightest: rptu-colors.weiss,
      neutral-darkest: rptu-colors.schwarz,
    ),
    // save the variables for later use
    config-store(
      align: align,
      progress-bar: progress-bar,
      header: header,
      header-right: header-right,
      footer-columns: footer-columns,
      footer-a: footer-a,
      footer-b: footer-b,
      footer-c: footer-c,
      // Store RPTU specific parameters
      main-color: main-color,
      secondary-color: secondary-color,
      author-pictures: author-pictures,
      author-affiliations: author-affiliations,
      affiliations: affiliations,
      sponsor-logos: sponsor-logos,
    ),
    ..args,
  )

  // Generate title slide
  title-slide()
  
  // Generate author funding slide if we have the necessary info
  if author-pictures.len() > 0 or affiliations != none or sponsor-logos != none {
    author-funding-slide(
      author-pictures: author-pictures,
      author-affiliations: author-affiliations,
      affiliations: affiliations,
      sponsor-logos: sponsor-logos,
    )
  }

  body
}