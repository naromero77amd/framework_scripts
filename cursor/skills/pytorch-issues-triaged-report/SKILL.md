---
name: pytorch-issues-triaged-report
description: Generate a compact "Issues Triaged" report for pytorch/pytorch using GitHub CLI activity. Use when the user asks for a weekly or two-week PyTorch issue-triage summary, issues they closed as not planned, or issues on which they commented.
---

# PyTorch Issues Triaged Report

Generate a deduplicated Markdown report from the authenticated user's GitHub
activity.

## Default scope

- Repository: `pytorch/pytorch`
- Window: rolling 14 days
- Identity: the account authenticated in `gh`
- Completed contributors: `k-artem`, `fjankovi`, and `jeffdaily`
- Additional activity contributors: `giuseppegrossi`
- Content: GitHub issues only, never pull requests

Honor user-provided overrides for the repository, account, number of days, or
contributor lists.

## Inclusion rules

Include an issue when it matches one of these rules:

1. **Not planned**: the selected account performed the close action and the
   issue's state reason is `not_planned` during the requested window.
2. **Completed**: the issue was closed as `completed` during the requested
   window and a configured completed contributor authored it, commented on it,
   or performed the close action.
3. **Other issues triaged**: the selected account created an issue comment
   during the requested window. Annotate every issue the selected account
   commented on with that account's display name and GitHub login, regardless
   of which section contains it.
4. **Additional contributor activity**: a configured activity contributor
   directly acted on an issue during the requested window. Annotate that issue
   with the contributor's display name and GitHub login.

Do not include:

- Issues merely authored, assigned, mentioned, subscribed to, or updated by the
  selected account.
- Completed issues connected to a person only through assignment, mention, or
  subscription.
- Passive mentions and subscriptions for additional activity contributors.
- Pull requests or pull-request review comments.

List each issue once. Section precedence is **Not Planned**, **Completed**, then
**Other Issues Triaged**. Do not append the word `commented`; the selected
account's name and login indicate its comment activity.

## Generate the report

Resolve the script path relative to this `SKILL.md`, then run it:

```bash
python3 <skill-directory>/scripts/report.py
```

Available overrides:

```bash
python3 <skill-directory>/scripts/report.py --days 7
python3 <skill-directory>/scripts/report.py --repo owner/repository
python3 <skill-directory>/scripts/report.py --login github-user
python3 <skill-directory>/scripts/report.py --completed-by user1,user2
python3 <skill-directory>/scripts/report.py --activity-by user3,user4
```

The script uses `gh api`, so verify that `gh auth status` succeeds. It also
checks whether the GitHub events feed covers the requested period. If coverage
is insufficient, do not present a partial report as complete; report the
coverage limitation and investigate candidate issues through their timelines.

## Output format

Keep the report small:

```markdown
## Issues Triaged

**Person Name (`github-user`) · owner/repository · last 14 days**

### Not Planned

- [#123](https://github.com/owner/repository/issues/123) — Short issue title — Person Name (`github-user`)

### Completed

- [#456](https://github.com/owner/repository/issues/456) — Short issue title — Person Name (`github-user`)

### Other Issues Triaged

- [#789](https://github.com/owner/repository/issues/789) — Short issue title — Person Name (`github-user`)
```

Sort each section by the most recent qualifying activity. Use the issue's exact
title unless a shorter title is needed for readability. Include each completed
contributor's display name and GitHub login.
