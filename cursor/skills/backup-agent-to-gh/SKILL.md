---
name: backup-agent-to-gh
description: Back up an extended debug session to a GitHub issue by collecting host, Triton, PyTorch, ROCm, date, and work-summary details, then appending or amending an issue comment. Use only when manually invoked with /backup-agent-to-gh.
disable-model-invocation: true
---

# Backup Agent To GitHub

## Purpose

Use this skill during extended debug sessions to save the current agent state and environment context to a GitHub issue.

## Required Prompts

When invoked, prompt the user for:

1. The GitHub issue number.
2. Whether to append a new comment or amend a specific existing comment.
3. If amending, the specific comment URL or comment database ID to update.

If the current repository cannot be inferred, also ask for the GitHub repository in `owner/repo` format.

## Metadata To Collect

Automatically collect:

- Host machine name.
- Triton version.
- PyTorch commit hash.
- ROCm version.
- Current date.
- Summary of the current work.

Use best-effort collection. If a value cannot be determined, write `unknown` and include the failed command or reason briefly.

## Collection Commands

Run these from the environment where the work happened.

Host machine:

```bash
hostname -f 2>/dev/null || hostname
```

Current date:

```bash
date -Is
```

Triton version:

```bash
python3 - <<'PY'
try:
    import triton
    print(getattr(triton, "__version__", "unknown"))
except Exception as exc:
    print(f"unknown ({exc})")
PY
```

PyTorch commit hash:

```bash
python3 - <<'PY'
try:
    import torch
    print(getattr(torch.version, "git_version", None) or "unknown")
except Exception as exc:
    print(f"unknown ({exc})")
PY
```

ROCm version:

```bash
if [ -f /opt/rocm/.info/version ]; then
  cat /opt/rocm/.info/version
elif command -v hipcc >/dev/null 2>&1; then
  hipcc --version | head -n 5
elif command -v rocminfo >/dev/null 2>&1; then
  rocminfo | rg -m 5 'Runtime Version|ROCk module version|HSA Runtime'
else
  echo unknown
fi
```

## Summary Requirements

Summarize the current work from the active conversation and available context.

Include:

- Original goal or bug being investigated.
- Important commands run or files changed.
- Current status.
- Known failures, blockers, or next steps.

Keep the summary concise but useful for resuming later.

## Comment Template

Use this markdown body:

```markdown
## Agent Debug Backup

**Date:** <date>
**Host:** <host>
**Triton:** <triton-version>
**PyTorch commit:** <pytorch-commit>
**ROCm:** <rocm-version>

### Current Work
<summary>

### Status
<current-status>

### Next Steps
<next-steps>
```

## GitHub Workflow

Use `gh` for all GitHub issue operations.

Infer the repository when possible:

```bash
gh repo view --json nameWithOwner --jq .nameWithOwner
```

Append a new comment:

```bash
gh issue comment <issue-number> --repo <owner/repo> --body-file <backup-file>
```

List existing issue comments when the user wants to amend but has not provided a comment ID:

```bash
gh api repos/<owner>/<repo>/issues/<issue-number>/comments --jq '.[] | {id, user: .user.login, created_at, updated_at, body: (.body[0:120])}'
```

Amend an existing comment:

```bash
gh api \
  --method PATCH \
  repos/<owner>/<repo>/issues/comments/<comment-id> \
  -f body="$(cat <backup-file>)"
```

## Safety Rules

- Do not post to GitHub until the user has provided the issue number and chosen append or amend.
- If amending, do not guess which comment to update. Ask for the exact comment ID or show available comments and ask the user to choose.
- Do not include secrets, tokens, private keys, or environment variable dumps.
- If `gh` is not authenticated, stop and ask the user to authenticate before posting.
- After posting or amending, report the GitHub issue URL and whether a new comment was added or an existing comment was updated.
