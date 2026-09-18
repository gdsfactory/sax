See the [native netlist guide](native-netlists.md) for extraction and migration examples.

::: sax

## Backend compatibility

The `forward` backend and its lower-level functions have been removed. Use
`backend="klu"` (the default) or `backend="fg"` for S-parameter simulation.
Connections are bidirectional; forward/backward probe measurements remain available.

Factory models may be keyed as `"library::component"` for native netlists.
Exact cell-specific models take precedence. Bare factory keys remain supported
when unambiguous across the supplied hierarchy.
