# KFNetlist through SAX's circuit path

**Status: implemented on the feature branch.**

## Intent

Replace every SAX-owned netlist representation and netlist parser/transform
contract with kfnetlist objects. Backends may retain solver-specific numerical
indices and evaluated S-matrices, but no second instance, net, port, connection,
array, or placement schema. Compatibility with old SAX dictionaries and lower
level backend signatures is out of scope for this migration pass.

## Proposed contract

`sax.circuit` consumes `kfnetlist.Netlist` or `HierarchicalNetlist`. Instance
identity and topology remain in kfnetlist types through hierarchy traversal,
array expansion, model selection, probe insertion, mode expansion, and backend
analysis. SAX can keep separate instance-to-model bindings and runtime model
settings because these describe simulation, not a netlist. Generic topology
mutation and validation should live in kfnetlist; SAX-specific probe policy and
numerical wiring remain in SAX.

Implemented on the kfnetlist `sax` branch: `Netlist.prune_unconnected()`,
`Netlist.expand_arrays()`, and explicit child references in plain extracted
netlists. SAX calls the generic methods before applying its simulation-specific
port and probe policies.

SAX-owned `Instance`, `Instances`, `Net`, `Nets`, `Ports`, `Connections`, and
placement aliases/validators are removed. `sax.Net` is a direct re-export of
`kfnetlist.Net`; callers use kfnetlist types. The
backend registry receives kfnetlist topology plus model bindings, then returns
solver-specific analyzed data.

## Verification

- Search all `src/sax` and exports for SAX-owned netlist types, validators,
  dictionary instances/nets/ports, and legacy input paths; none remain.
- Hierarchical simulation, model substitution, arrays, probes, multimode,
  KLU/FG/additive, and asymmetric numerical cases work with kfnetlist objects.
- JIT and real-objective gradients remain valid for supported paths.
- Run focused checks first, then a broad audit; report old-contract failures
  honestly without restoring adapters to make tests green.

Tests for deleted SAX adapter, PIC/YAML loaders, and SAX-owned transforms are
removed. Direct-object tests cover hierarchy, reference identity, serialization,
array expansion, numerical backends, and probes; kfnetlist's own Rust/Python
suites cover generic transforms.

The full SAX suite passes with optional layout dependencies and all four test
notebooks. Public circuit notebooks use kfnetlist objects; the migrated
examples were executed separately. The locked non-notebook suite passes with
kfnetlist `sax` commit `3dc1d37f`. Details are in
[`specs/verification.md`](../verification.md).
