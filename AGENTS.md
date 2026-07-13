# Agent Instructions

## Required project context

- Read `CLAUDE.md` before changing code. It is the source of truth for project commands, implementation rules, and current phase follow-ups.
- Read `CONTEXT.md` before reasoning about the domain. Use its canonical ObjectCount terminology in code, tests, documentation, and status reports.
- If these documents conflict, stop and report the conflict instead of choosing silently.

## Working agreement

- Preserve user changes and keep work within the requested scope.
- Do not commit, push, or perform destructive Git operations unless the task explicitly authorizes that action.
- Verify changes with the narrowest relevant command from `CLAUDE.md` and report the command and result.
- Report completion as: summary, changed files, verification, risks or unresolved items.
