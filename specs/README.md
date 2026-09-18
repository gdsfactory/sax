# SAX development specs

A compact behavioral baseline for SAX **0.18.2**, initially inspected at
`9ebd77d0844b48dd1268066372db929946069f34`. These are living development
contracts, not generated API documentation or a proposal to rewrite SAX.

## Start here

| Spec | Scope |
| --- | --- |
| [Architecture](architecture.md) | Purpose, subsystem boundaries, execution lifecycle |
| [S-parameters](s-parameters.md) | Representations, indexing, modes, naming, validation |
| [Circuits](circuits.md) | Netlists, hierarchy, settings, arrays, probes |
| [Backends](backends.md) | Solver semantics, restrictions, JAX boundary |
| [Models](models.md) | Callable contract, optical and RF conventions |
| [Data workflows](data-workflows.md) | Parsers, interpolation, fitting, utilities |
| [Verification](verification.md) | Checks, evidence, coverage limits |
| [Open questions](open-questions.md) | Observed discrepancies, not approved fixes |

`src/sax/__init__.py` and module exports remain the public symbol inventory;
`docs/` remains curated user-facing documentation. These specs cover subsystem
contracts and important edge cases, not every parameter or model equation.
Do not publish the whole development spec tree through Zensical by default.
Explain user-relevant contracts in the appropriate user docs when they change.

## How to read the evidence

- **Baseline behavior** describes the inspected implementation. Preserve it unless
  the requested change intentionally revises it; it is not proof of design intent.
- **Tests** name existing verification surfaces. Only the runs recorded in
  [verification](verification.md) have been executed for this baseline. A linked
  test file does not establish every claim in a spec.
- **Open questions** describe uncertain intent, limitations, or contradictions.
  They are not requirements to preserve a bug, nor permission to fix it silently.
- **Proposed behavior** must be labeled as such until the corresponding change is
  implemented and verified. Never rewrite the baseline to hide a failing test.

Links point to source files; symbol and test names identify the relevant evidence
without fragile line numbers. Keep evidence near the contract it supports.

## Low-friction spec-driven flow

1. Read only the relevant specs, implementation, and tests. State the intended
   behavior and how to check it before changing runtime code.
2. For a small, clear change, a short plan in the conversation is enough. The
   user's request authorizes work within that scope; no approval-token ceremony.
3. For a cross-cutting, breaking, ambiguous, or multi-session change, create
   `specs/changes/<slug>.md` with **intent, scope/non-goals, proposed contract,
   verification, and unresolved decisions**. Use plain Markdown; no numbering,
   branch convention, or mandatory template. Ask only material questions.
4. Implement in verifiable slices. Update the affected baseline specs and tests
   in the same change. Mechanical edits with no contract impact need no new spec.
5. Report checks as **passed / failed / not run**, with reasons for gaps. Update
   the baseline only to reflect established behavior, not unfinished proposals.
   Mark a completed change note as implemented and link its resulting contracts;
   keep it only when its rationale is useful.

Use the local [spec-change](../.agents/skills/spec-change/SKILL.md) skill for changes
and [spec-check](../.agents/skills/spec-check/SKILL.md) for conformance reviews or
refreshing these specs. There is no additional CLI, external framework, or required
spec lifecycle database. The user can opt out of this flow.
