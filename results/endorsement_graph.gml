graph [
  directed 1
  node [
    id 0
    label "gemini-2.0-flash-thinking-exp-1219"
  ]
  node [
    id 1
    label "gemini-exp-1206"
  ]
  node [
    id 2
    label "claude-3-5-sonnet-latest"
  ]
  node [
    id 3
    label "claude-3-opus-latest"
  ]
  node [
    id 4
    label "o1-preview"
  ]
  node [
    id 5
    label "gpt-4o"
  ]
  node [
    id 6
    label "deepseek-chat"
  ]
  edge [
    source 0
    target 6
    weight 5.0
  ]
  edge [
    source 0
    target 4
    weight 5.0
  ]
  edge [
    source 0
    target 5
    weight 5.0
  ]
  edge [
    source 1
    target 4
    weight 8.0
  ]
  edge [
    source 1
    target 0
    weight 5.0
  ]
  edge [
    source 1
    target 6
    weight 3.0
  ]
  edge [
    source 2
    target 3
    weight 9.0
  ]
  edge [
    source 2
    target 0
    weight 6.0
  ]
  edge [
    source 2
    target 1
    weight 7.0
  ]
  edge [
    source 3
    target 6
    weight 9.0
  ]
  edge [
    source 3
    target 0
    weight 6.0
  ]
  edge [
    source 3
    target 2
    weight 8.0
  ]
  edge [
    source 4
    target 5
    weight 9.0
  ]
  edge [
    source 4
    target 0
    weight 5.0
  ]
  edge [
    source 4
    target 2
    weight 9.0
  ]
  edge [
    source 5
    target 2
    weight 9.0
  ]
  edge [
    source 5
    target 1
    weight 8.0
  ]
  edge [
    source 5
    target 0
    weight 8.0
  ]
  edge [
    source 6
    target 4
    weight 9.0
  ]
  edge [
    source 6
    target 3
    weight 7.0
  ]
  edge [
    source 6
    target 0
    weight 8.0
  ]
]
