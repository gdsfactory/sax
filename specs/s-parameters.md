# S-parameters, modes, and names

## Representations and direction

For incident amplitudes `a` and outgoing amplitudes `b`, dense matrices use
`b = S @ a`: **rows are outputs, columns are inputs**. Dictionary keys instead
read **`(input_port, output_port)`**. This distinction matters for nonreciprocal
models; symmetric fixtures cannot detect a transposition error.

| Representation | Structure | Numerical axes |
| --- | --- | --- |
| `SDict` | `{(input, output): value}` | Each value has scalar or broadcast-compatible batch shape |
| `SCoo` | `(Si, Sj, Sx, port_map)` | `Si` output rows, `Sj` input columns; `Sx.shape == (*batch, nnz)` |
| `SDense` | `(S, port_map)` | `S.shape == (*batch, n, n)` |

`port_map` maps names to integer indices. Missing dictionary entries represent
zero coupling when materialized as dense data; they are not necessarily present
as explicit keys. Dictionary-to-COO conversion discovers ports by first occurrence,
not natural sorting. `get_ports` returns naturally sorted names, so its ordering
must not be mistaken for the numerical matrix's index order.

`sax.sdict`, `sax.scoo`, and `sax.sdense` accept evaluated data or wrap a model to
convert its returned data. Same-format conversions may return the original object;
these APIs do not promise defensive copies. Dense-to-dictionary/COO conversion
includes zeros. Dictionary values are broadcast before stacking onto the final COO
axis. Empty dictionaries are not a generally supported conversion fixture.

Dense data can carry a port map selecting a subset of matrix indices;
dense-to-COO consolidates that selection. Do not generalize this into support for
arbitrary sparse maps: COO-to-dense sizes the result from `len(port_map)`.

### Edge semantics

- Repeated COO coordinates **sum** in both dense and dictionary conversions.
  Round trips preserve numerical values, not the original duplicate storage layout.
  `test_sparse_duplicates.py` covers batches, cancellation, JIT, and gradients.
- `reciprocal` copies entries without conjugation. For conflicting directions,
  the first supplied direction defines both values. It preserves the input object
  and supports traced values without data-dependent comparisons.
  `test_api_contracts.py` checks conflicts, diagonal entries, JIT, and gradients.
- `get_ports` and `get_port_combinations` reject unevaluated callables.
- `block_diag` requires square last-two axes and identical batch shapes.

Evidence: [`s.py`](../src/sax/s.py), especially `_sdense_to_sdict`,
`_sdict_to_scoo`, `_scoo_to_sdense`, and `reciprocal`.
Tests: [`test_smatrix_convention.py`](../src/tests/test_smatrix_convention.py),
including `test_sdense_convention`, `test_scoo_convention`, and conversion chains.
Duplicate-coordinate consistency is additionally covered by pytest regression tests.

## Modes

Multimode ports are named `port@mode`; defaults are `TE` and `TM`.
`multimode` replicates single-mode behavior within each mode, without adding
cross-mode coupling. Actual mode conversion requires an explicitly multimode model.
`singlemode` selects entries where both ports have the requested mode and removes
suffixes. Already single-/multimode inputs can pass through unchanged.

Dense single-mode extraction consolidates the matrix. COO single-mode extraction
only filters/renames the port map: it does not compact indices or values. Convert
with care, especially when selecting a non-first mode.

Custom `modes` are honored for all three representations and model wrappers;
`test_custom_modes.py` covers replication, extraction, and absent cross-mode coupling.
`get_modes` returns unique mode names in natural sorted order across all formats,
verified by `test_api_contracts.py`.

Evidence: [`multimode.py`](../src/sax/multimode.py), `get_modes` in `s.py`.
Tests: [`02_multimode.ipynb`](../src/tests/nbs/02_multimode.ipynb) covers default-mode
replication and selected conversion cases; custom modes additionally have pytest coverage.

## Naming and validation

`PortNamer(n_inputs, n_outputs)` uses `inout` by default, or `optical` (`o1`, ...).
In `inout` indexing, inputs ascend and outputs descend: a 2x2 device's indices are
`in0, in1, out1, out0`. Negative/out-of-range indexing raises `IndexError`.
The global naming strategy affects new `PortNamer` objects; explicit strategy
arguments override it. Not all built-in models use this helper. Configure naming
before constructing or tracing models, rather than relying on JIT cache invalidation.

`sax.into[T](value)` validates/coerces through SAX validators or Pydantic adapters;
`sax.try_into[T](value)` returns `None` on caught conversion failures. These are not
the same as the lightweight `sax.sdict/scoo/sdense` format dispatchers. Validation
at one entry point does not imply every operation validates all invariants.
Native Python target types delegate to the installed Pydantic adapter; supported
coercions can vary across supported Pydantic releases. `test_native_type_coercion.py`
checks that delegation, concrete return values (including zero), and error translation.

Evidence: [`ports.py`](../src/sax/ports.py),
[`saxtypes/into.py`](../src/sax/saxtypes/into.py), and singlemode/multimode/core type
validators. Tests: [`00_typing.ipynb`](../src/tests/nbs/00_typing.ipynb).
