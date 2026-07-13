# Codex and OpenCode Defaults Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Codex the Orca default and branch-name generator, while configuring OpenCode as a safe local worker that shares repository context with Codex.

**Architecture:** Orca owns agent selection, OpenCode owns the local worker role and permissions, and the repository owns project knowledge. User-global settings remain outside Git; the repository adds only a compact `AGENTS.md` entry point.

**Tech Stack:** Orca 1.4.137 settings UI, Codex CLI 0.144.1, OpenCode 1.17.18, LM Studio OpenAI-compatible endpoint, Markdown, JSON, PowerShell.

## Global Constraints

- Orca's default TUI agent and Source Control AI `branchName` agent must both be `codex`.
- Keep OpenCode on `lmstudio/google/gemma-4-12b` at `http://127.0.0.1:1234/v1`; do not add a cloud fallback.
- Project knowledge belongs in repository files; global OpenCode memory contains only stable worker behavior.
- OpenCode must deny push, hard reset, clean, and forced branch deletion.
- OpenCode must not commit unless the caller explicitly authorizes it.
- Do not edit Orca's live `orca-data.json` directly; change Orca settings through the app UI.
- LLM wiki configuration is a separate follow-up and is not implemented by this plan.

## File Map

- Create `AGENTS.md`: shared project instruction entry point for Codex and OpenCode.
- Create `C:\Users\mmdal\.config\opencode\AGENTS.md`: stable global behavior for every OpenCode session.
- Create `C:\Users\mmdal\.config\opencode\agents\codex-worker.md`: primary local worker role, output contract, and permission boundaries.
- Modify `C:\Users\mmdal\.config\opencode\opencode.json`: select `codex-worker` while preserving the LM Studio provider and model limits.
- Change through Orca UI: `defaultTuiAgent` and Source Control AI `actions.branchName.agentId`.
- Verify only, never edit directly: `C:\Users\mmdal\AppData\Roaming\orca\profiles\local-default\orca-data.json`.

---

### Task 1: Repository Knowledge Entry Point

**Files:**
- Create: `AGENTS.md`
- Reference: `CLAUDE.md`
- Reference: `CONTEXT.md`

**Interfaces:**
- Consumes: existing project commands and rules in `CLAUDE.md`; domain terminology and current context in `CONTEXT.md`.
- Produces: one shared instruction entry point automatically discovered by Codex and OpenCode.

- [ ] **Step 1: Run the entry-point contract and verify it fails**

Run:

```powershell
if (-not (Test-Path -LiteralPath 'AGENTS.md')) { throw 'AGENTS.md is missing' }
```

Expected: command fails with `AGENTS.md is missing`.

- [ ] **Step 2: Create the minimal shared entry point**

Create `AGENTS.md` with exactly this content:

```markdown
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
```

- [ ] **Step 3: Verify the entry point references both sources and has no placeholders**

Run:

```powershell
$text = Get-Content -Raw -LiteralPath 'AGENTS.md'
if ($text -notmatch [regex]::Escape('`CLAUDE.md`')) { throw 'CLAUDE.md reference missing' }
if ($text -notmatch [regex]::Escape('`CONTEXT.md`')) { throw 'CONTEXT.md reference missing' }
if ($text -match '(?i)TBD|TODO|FIXME') { throw 'Placeholder found' }
if ($text -match '(?m)[ \t]+$') { throw 'Trailing whitespace found' }
```

Expected: exit code `0` and no assertion errors.

- [ ] **Step 4: Commit the repository instruction file**

Run:

```powershell
git add -- AGENTS.md
git commit -m "docs: add shared agent instructions"
```

Expected: one commit containing only `AGENTS.md`.

### Task 2: OpenCode Codex Worker

**Files:**
- Create: `C:\Users\mmdal\.config\opencode\AGENTS.md`
- Create: `C:\Users\mmdal\.config\opencode\agents\codex-worker.md`
- Modify: `C:\Users\mmdal\.config\opencode\opencode.json`

**Interfaces:**
- Consumes: OpenCode's global rule discovery, primary-agent selection, LM Studio provider, and repository `AGENTS.md`.
- Produces: primary agent `codex-worker` with model `lmstudio/google/gemma-4-12b` and a deterministic handoff report.

- [ ] **Step 1: Run the resolved-config contract and verify it fails**

Run:

```powershell
$resolved = opencode debug config | ConvertFrom-Json -Depth 100
if ($resolved.default_agent -ne 'codex-worker') { throw "Expected codex-worker, got '$($resolved.default_agent)'" }
```

Expected: command fails because `default_agent` is currently unset.

- [ ] **Step 2: Create the global stable behavior file**

Create `C:\Users\mmdal\.config\opencode\AGENTS.md` with exactly this content:

```markdown
# Global OpenCode Working Agreement

- Treat repository files as the source of project truth; do not store project facts in global memory.
- Read the nearest repository `AGENTS.md` before acting. If it references context files, read them before changing code.
- Work only within the caller's requested scope and preserve existing user changes.
- Prefer evidence from files and command output over guesses.
- Never push or run destructive Git commands.
- Return decisions or missing authority as blockers to the caller instead of inventing an answer.
```

- [ ] **Step 3: Create the primary worker agent**

Create `C:\Users\mmdal\.config\opencode\agents\codex-worker.md` with exactly this content:

```markdown
---
description: Local implementation and investigation worker coordinated by Codex
mode: primary
model: lmstudio/google/gemma-4-12b
permission:
  edit: allow
  bash:
    "*": allow
    "git push*": deny
    "git reset --hard*": deny
    "git clean*": deny
    "git branch -D*": deny
    "git commit*": ask
  external_directory: ask
  task: deny
  question: deny
---

Execute the caller's bounded task using repository evidence.

Before acting:
1. Read the repository `AGENTS.md` and every project context file it requires.
2. Inspect relevant files and current Git status.
3. Keep unrelated user changes untouched.

When finished, return these headings:

## Summary
## Changed files
## Commands and verification
## Risks or blockers

Do not claim success without command output that supports it. Do not commit unless the task explicitly authorizes a commit. Never push or use destructive Git commands.
```

- [ ] **Step 4: Preserve the provider and select the worker in `opencode.json`**

Replace `C:\Users\mmdal\.config\opencode\opencode.json` with:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "model": "lmstudio/google/gemma-4-12b",
  "default_agent": "codex-worker",
  "provider": {
    "lmstudio": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "LM Studio (local)",
      "options": {
        "baseURL": "http://127.0.0.1:1234/v1"
      },
      "models": {
        "google/gemma-4-12b": {
          "name": "Gemma 4 12B",
          "limit": {
            "context": 32768,
            "output": 8192
          }
        }
      }
    }
  }
}
```

- [ ] **Step 5: Validate the merged OpenCode configuration and permissions**

Run:

```powershell
$resolved = opencode debug config | ConvertFrom-Json -Depth 100
if ($resolved.default_agent -ne 'codex-worker') { throw 'default_agent mismatch' }
if ($resolved.model -ne 'lmstudio/google/gemma-4-12b') { throw 'model changed unexpectedly' }
$agent = opencode debug agent codex-worker | ConvertFrom-Json -Depth 100
if ($agent.model.providerID -ne 'lmstudio') { throw 'worker provider missing' }
if ($agent.model.modelID -ne 'google/gemma-4-12b') { throw 'worker model missing' }
$denials = @($agent.permission | Where-Object { $_.permission -eq 'bash' -and $_.action -eq 'deny' } | ForEach-Object pattern)
foreach ($pattern in @('git push*','git reset --hard*','git clean*','git branch -D*')) {
  if ($pattern -notin $denials) { throw "denial missing: $pattern" }
}
if (-not ($agent.permission | Where-Object { $_.permission -eq 'bash' -and $_.pattern -eq 'git commit*' -and $_.action -eq 'ask' })) { throw 'commit ask missing' }
"resolved.default_agent=$($resolved.default_agent)"
"resolved.model=$($resolved.model)"
"agent.model=$($agent.model.providerID)/$($agent.model.modelID)"
"agent.git_denials=$($denials -join ', ')"
$agentListMatch = opencode agent list | Select-String -Pattern 'codex-worker'
if (-not $agentListMatch) { throw 'codex-worker missing from agent list' }
$agentListMatch
```

Expected: exit code `0`; resolved agent is `codex-worker`, the global and worker models resolve to `lmstudio/google/gemma-4-12b`, all four destructive Git patterns are denied, commits require approval, and the agent appears in the list.

Do not make a Git commit: these are user-global personal settings outside the repository.

### Task 3: Orca Codex Defaults

**Files:**
- Change through Orca Settings UI: global default TUI agent.
- Change through Orca Settings UI: Source Control AI `branchName` recipe agent.
- Verify: `C:\Users\mmdal\AppData\Roaming\orca\profiles\local-default\orca-data.json`

**Interfaces:**
- Consumes: installed Orca settings UI and supported Source Control AI agent recipe schema.
- Produces: `settings.defaultTuiAgent = "codex"` and `settings.sourceControlAi.actions.branchName.agentId = "codex"`.

- [ ] **Step 1: Run the Orca settings contract and verify it fails**

Run:

```powershell
$path = "$env:APPDATA\orca\profiles\local-default\orca-data.json"
$data = Get-Content -Raw -LiteralPath $path | ConvertFrom-Json -Depth 100
if ($data.settings.defaultTuiAgent -ne 'codex') { throw "Default TUI agent is '$($data.settings.defaultTuiAgent)'" }
if ($data.settings.sourceControlAi.actions.branchName.agentId -ne 'codex') { throw 'Branch-name agent is not explicit Codex' }
```

Expected: command fails; current default is `claude` and the branch-name recipe has no `agentId`.

- [ ] **Step 2: Change the global default agent through Orca UI**

Open Orca Settings, search for `default agent`, and select `Codex` as the default TUI agent. Leave the Codex command override and existing permission arguments unchanged.

Expected UI state: default agent selector displays `Codex`.

- [ ] **Step 3: Change the branch-name recipe through Orca UI**

Open Orca Settings → Git → Source Control AI. In the `Branch name` / `Auto-rename branch & worktree` recipe, select `Codex` as the action agent. Keep the command template exactly `{basePrompt}` and keep automatic rename enabled.

Expected UI state: Branch name recipe displays `Codex`; auto-rename remains on.

- [ ] **Step 4: Verify persisted settings without modifying the state file**

Run:

```powershell
$path = "$env:APPDATA\orca\profiles\local-default\orca-data.json"
$data = Get-Content -Raw -LiteralPath $path | ConvertFrom-Json -Depth 100
[pscustomobject]@{
  DefaultTuiAgent = $data.settings.defaultTuiAgent
  BranchNameAgent = $data.settings.sourceControlAi.actions.branchName.agentId
  AutoRename = $data.settings.autoRenameBranchFromWork
} | Format-List
if ($data.settings.defaultTuiAgent -ne 'codex') { throw 'Default TUI agent mismatch' }
if ($data.settings.sourceControlAi.actions.branchName.agentId -ne 'codex') { throw 'Branch-name agent mismatch' }
if ($data.settings.autoRenameBranchFromWork -ne $true) { throw 'Auto-rename was disabled' }
```

Expected: `DefaultTuiAgent: codex`, `BranchNameAgent: codex`, `AutoRename: True`, exit code `0`.

Do not make a Git commit: these values belong to Orca's user profile.

### Task 4: End-to-End Worker Verification

**Files:**
- Verify: `AGENTS.md`
- Verify: `C:\Users\mmdal\.config\opencode\AGENTS.md`
- Verify: `C:\Users\mmdal\.config\opencode\agents\codex-worker.md`
- Verify: `C:\Users\mmdal\.config\opencode\opencode.json`

**Interfaces:**
- Consumes: the completed repository, OpenCode, and Orca configuration tasks.
- Produces: evidence that OpenCode reads shared project context and returns the handoff contract without editing files.

- [ ] **Step 1: Run the bounded read-only probe and assert its contract**

Run:

```powershell
$before = git status --porcelain=v1
$output = opencode run --agent codex-worker --format default "Read AGENTS.md and its required context files. Do not edit anything. Report the current project phase using exactly the four required headings."
$probeExit = $LASTEXITCODE
Write-Output "probeExit=$probeExit"
if ($probeExit -ne 0) { throw "OpenCode probe exited $probeExit" }
$output
foreach ($heading in @('## Summary', '## Changed files', '## Commands and verification', '## Risks or blockers')) {
  if (($output -join "`n") -notmatch [regex]::Escape($heading)) { throw "Missing heading: $heading" }
  Write-Output "heading[$heading]=PASS"
}
$after = git status --porcelain=v1
Write-Output 'beforeStatus:'
if ($before) { $before } else { Write-Output '<clean>' }
Write-Output 'afterStatus:'
if ($after) { $after } else { Write-Output '<clean>' }
if (($after -join "`n") -ne ($before -join "`n")) { throw 'Read-only probe changed the worktree' }
Write-Output 'worktreeEquality=PASS'
```

Expected: exit code `0`; output includes all four headings and the probe makes no worktree changes.

- [ ] **Step 2: Run final configuration checks**

Run:

```powershell
opencode --version
codex --version
$resolved = opencode debug config | ConvertFrom-Json -Depth 100
$orca = Get-Content -Raw -LiteralPath "$env:APPDATA\orca\profiles\local-default\orca-data.json" | ConvertFrom-Json -Depth 100
if ($resolved.default_agent -ne 'codex-worker') { throw 'OpenCode default agent failed final check' }
if ($orca.settings.defaultTuiAgent -ne 'codex') { throw 'Orca default failed final check' }
if ($orca.settings.sourceControlAi.actions.branchName.agentId -ne 'codex') { throw 'Orca branch-name agent failed final check' }
git status --short
```

Expected: OpenCode `1.17.18`, Codex CLI `0.144.1`, all assertions pass, and Git status contains no unexpected files.

No additional commit is needed unless implementation changed the plan or design documentation.
