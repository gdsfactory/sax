# Working on SAX

**Use spec-driven flow by default unless prompted otherwise.** Keep it lightweight:
read the relevant contract, state the intended change and verification, implement,
then update specs/tests together. A small clear change needs only a short plan in
chat, not an approval ceremony or a new spec file.

## work.md

Treat `work.md` as a living plan that I can later give you as an execution goal.
Organize it into ordered stages with a clear outcome, task checkboxes, and
verification criteria. Distinguish agreed decisions from open questions, and
preserve relevant findings when reorganizing it. Editing the plan does not by
itself authorize implementation.

While executing the goal, keep `work.md` current with progress, decisions,
verification results, and remaining work. Check off tasks only when their stated
outcome is achieved; record failed or unrun checks explicitly. After completing
each stage, commit the updated `work.md` together with that stage's related
changes. This is authorization for those stage commits; exclude unrelated work.

## Specs and skills

- Start at [`specs/README.md`](specs/README.md); load only relevant subsystem specs.
- Baseline descriptions are evidence-backed observations, not proof that every quirk
  is intentional. [`specs/open-questions.md`](specs/open-questions.md) is not permission
  to silently fix unrelated behavior.
- Use [spec-change](.agents/skills/spec-change/SKILL.md) for behavioral changes or
  proposals, and [spec-check](.agents/skills/spec-check/SKILL.md) for reviews/refreshes.
  In Pi: `/skill:spec-change <task>` and `/skill:spec-check <scope>`.
- Record a proposal in `specs/changes/<slug>.md` only for consequential, ambiguous,
  or multi-session work. Keep proposed behavior distinct from the implemented baseline.
- Keep `specs/` out of the user-facing documentation build by default. Update `docs/`
  deliberately when users need an explanation of a changed contract.

## Code and verification

- Implementation: `src/sax/`; tests: `src/tests/`; notebook tests: `src/tests/nbs/`.
  Public exports live in `src/sax/__init__.py` and submodule exports.
- Read [`specs/verification.md`](specs/verification.md) for commands and known setup
  limitations. Prefer focused pytest checks first, then relevant regression checks.
- Normal targeted command:
  `uv run --locked --no-default-groups --group test pytest <test-path> -q`.
  The tracked lockfile now passes `uv lock --check`. Do not silently regenerate it
  during unrelated work. Use `just smoke` for fast checks without synchronization;
  report use of an existing environment and any verification limits.
- `just test` includes notebook execution and user-kernel setup. Inspect `justfile`
  first: `just dev` clears `.venv`; `just clean` is destructive. Do not run them as
  routine inspection commands. Lint/type tooling is in `.pre-commit-config.yaml`
  and `pyproject.toml`; avoid auto-fixing unrelated files.
- Preserve S-dictionary `(input, output)` versus dense/COO `(output, input)` indexing,
  documented units, and static topology during numerical evaluation. Use asymmetric
  tests; check broadcasting, JIT, and real-objective gradients when relevant.
- Do not assume every backend/model is reciprocal, passive, energy-conserving,
  differentiable, or shape-equivalent. Specify the applicable domain and tolerances.
- Report checks as passed, failed, or not run. Never describe skipped checks as passes.

## Scope and working tree

Preserve unrelated changes. Do not create/switch branches, commit, stash, change
runtime behavior during spec-only work, or update dependencies without appropriate
user authorization. Ask about material ambiguity, not facts discoverable in code.
