---
name: spec-change
description: Make behavior changes using this repository's lightweight spec-driven flow. Use when adding features, fixing behavioral bugs, or changing contracts; also use when drafting or revising a change proposal. Keeps specs, implementation, and verification aligned without mandatory approval files or branch management.
---

# Spec-driven change

## Context

From the repository root, read `AGENTS.md` and `specs/README.md`, then only the
relevant subsystem specs and source/tests. For SAX's check commands and numerical
caveats, consult `specs/verification.md`. Respect a user request to bypass this flow.

Paths above are repository-root paths, not relative to this skill directory.
The index is also available at [specs/README.md](../../../specs/README.md).

## Workflow

1. **Establish the change.** Summarize current behavior, requested behavior, affected
   contracts, and one or more concrete acceptance checks. Inspect the implementation
   rather than asking the user questions the repo can answer. Distinguish intended
   changes from baseline discrepancies and uncertain intent.
2. **Choose the lightest record.** A clear, local change needs only a short plan in
   chat. For breaking/cross-cutting work, unresolved design choices, or work spanning
   sessions, write `specs/changes/<slug>.md`: intent, scope/non-goals, proposed
   contract, verification, unresolved decisions. Create that directory only if needed.
   No required numbering, status tokens, or standalone plan for mechanical edits.
3. **Resolve material uncertainty.** The user's clear request authorizes its scope.
   Ask only when alternatives materially change public behavior, compatibility,
   safety, or effort. Do not treat an observed bug as an approved requirement, and
   do not implement proposals when the user requested only analysis/specification.
4. **Implement and verify in slices.** Add/update relevant tests, implement the
   agreed behavior, and run targeted checks before broader regressions. Use explicit
   tolerances and asymmetric fixtures for numerical direction. Run JIT/gradient checks
   only where relevant. Never weaken acceptance criteria to conceal a failure.
5. **Reconcile the specs.** Update affected baseline contracts, evidence links, open
   questions, and user-facing explanations in the same change. Keep unimplemented
   behavior clearly proposed. Mark a completed change note as implemented; retain it
   only if useful rationale remains. Do not replace historical baseline run results
   with unqualified claims that everything is verified.
6. **Close with evidence.** Report changed contracts, files, commands/results, and
   any failed or unrun checks. If blocked, state the blocker and next input needed.

## Guardrails

- No automatic branches, commits, stashes, dependency updates, or destructive setup.
  Preserve unrelated working-tree changes; ask if edits overlap unsafely.
- No mandatory pause after every tiny step. Pause for decisions, surprising failures,
  or user-requested review boundaries, not arbitrary line counts.
- Update existing specs instead of producing duplicate contracts. A change that
  leaves observable behavior intact may need no spec edit; say so briefly.
- Source, tests, and prose can each be wrong. Explain disagreements before deciding
  which artifact to change. `specs/open-questions.md` is not an automatic fix list.
