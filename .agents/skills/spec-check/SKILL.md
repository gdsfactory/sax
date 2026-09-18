---
name: spec-check
description: Review or refresh repository specs against implementation and tests. Use for conformance audits, spec drift checks after changes, documenting existing subsystems, or updating evidence. Read-only by default; edit specs when the user requests a refresh, without silently changing runtime behavior.
---

# Check and refresh specs

Read repository-root `AGENTS.md`, `specs/README.md`, and the relevant subsystem
specs. See [the spec index](../../../specs/README.md) and
[verification guidance](../../../specs/verification.md). Do not load every spec
for a local review.

1. **Bound the review.** Use the requested subsystem or diff. If scope is clear,
   proceed without a questionnaire; if a whole-repo audit is requested, inventory
   public modules and prioritize observable contracts over private implementation.
2. **Trace each important claim.** Inspect its source symbol, tests, and any example
   needed to resolve meaning. Test presence is not test execution, and a passing
   file does not verify every nearby claim. Check inputs, outputs, errors, units,
   shape/direction, compatibility, and applicable numerical assumptions.
3. **Classify disagreements.** Report stale spec, implementation mismatch, missing
   verification, or unresolved intent. Give a path/symbol, impact, and smallest
   useful next check. Do not automatically declare prose or code authoritative.
4. **Verify proportionally.** Run focused, non-destructive checks when practical.
   Record command, environment assumptions, result, and limits. Distinguish static
   observations from runtime reproductions and untested hypotheses. See the
   verification guide before running environment setup or notebook commands.
5. **Review mode:** return prioritized findings first, followed by coverage limits.
   Do not edit specs or runtime code merely because a mismatch was discovered.
6. **Refresh mode (requested by user):** update the existing spec's established
   behavior and evidence; put contradictions in `specs/open-questions.md`. Label
   future behavior as proposed. Do not fix runtime code unless separately in scope.
   Check relative links, the index, and `git diff --check` before summarizing edits.

A useful refresh is concise: contract, evidence, verification limits, open questions.
Avoid duplicating full signatures/API docs, asserting physical properties for models
that intentionally violate them, or adding a bureaucracy of requirement IDs and
status files. If evidence is unavailable, state what remains unknown and what would
establish it; do not invent a guarantee.
