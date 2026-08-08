#!/usr/bin/env python3
"""Generate a compact GitHub issues-triaged report."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from typing import Any


def gh_json(*args: str) -> Any:
    result = subprocess.run(
        ["gh", *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        message = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(message or f"gh exited with status {result.returncode}")
    return json.loads(result.stdout)


def parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def load_events(login: str, cutoff: datetime) -> list[dict[str, Any]]:
    pages = gh_json(
        "api",
        "--paginate",
        "--slurp",
        f"/users/{login}/events?per_page=100",
    )
    events = [event for page in pages for event in page]

    if events:
        oldest = min(parse_timestamp(event["created_at"]) for event in events)
        feed_looks_capped = len(events) >= 300 or (
            len(pages) >= 3 and len(pages[-1]) == 100
        )
        if feed_looks_capped and oldest > cutoff:
            raise RuntimeError(
                f"GitHub's events feed for {login} does not reach the requested "
                f"cutoff ({cutoff.isoformat()}); refusing to emit a partial report"
            )
    return events


def remember_issue(
    collection: dict[str, dict[str, Any]],
    issue: dict[str, Any],
    activity_at: datetime,
) -> dict[str, Any]:
    url = issue["html_url"]
    entry = collection.setdefault(
        url,
        {
            "number": issue["number"],
            "title": " ".join(issue["title"].split()),
            "url": url,
            "reasons": set(),
            "contributors": set(),
            "activity_contributors": set(),
            "latest": activity_at,
        },
    )
    entry["latest"] = max(entry["latest"], activity_at)
    return entry


def format_people(logins: list[str], names: dict[str, str]) -> str:
    labels: list[str] = []
    seen: set[str] = set()
    for login in logins:
        key = login.lower()
        if key in seen:
            continue
        seen.add(key)
        labels.append(f"{names[login]} (`{login}`)")
    return ", ".join(labels)


def completed_candidates(
    repo: str,
    login: str,
    cutoff: datetime,
    now: datetime,
) -> list[dict[str, Any]]:
    start = cutoff.strftime("%Y-%m-%dT%H:%M:%SZ")
    end = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    candidates: dict[str, dict[str, Any]] = {}

    for relationship in ("author", "commenter"):
        query = (
            f"repo:{repo} is:issue is:closed reason:completed "
            f"closed:{start}..{end} {relationship}:{login}"
        )
        pages = gh_json(
            "api",
            "--method",
            "GET",
            "--paginate",
            "--slurp",
            "search/issues",
            "-f",
            f"q={query}",
            "-f",
            "per_page=100",
        )
        items = [item for page in pages for item in page.get("items", [])]
        if any(page.get("incomplete_results") for page in pages):
            raise RuntimeError(f"GitHub returned incomplete search results for {login}")
        total_count = pages[0].get("total_count", 0) if pages else 0
        if total_count > len(items):
            raise RuntimeError(
                f"GitHub search returned only {len(items)} of {total_count} "
                f"completed issues for {login}"
            )
        candidates.update({item["html_url"]: item for item in items})

    return list(candidates.values())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="pytorch/pytorch")
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--login", help="GitHub login; defaults to the gh user")
    parser.add_argument(
        "--completed-by",
        default="k-artem,fjankovi,jeffdaily",
        help="Comma-separated GitHub users whose completed issues are included",
    )
    parser.add_argument(
        "--activity-by",
        default="giuseppegrossi",
        help="Comma-separated GitHub users whose issue activity is included",
    )
    parser.add_argument(
        "--now",
        help="UTC ISO-8601 end time, primarily for reproducible reports",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.days < 1:
        raise RuntimeError("--days must be at least 1")

    user_endpoint = f"users/{args.login}" if args.login else "user"
    user = gh_json("api", user_endpoint)
    login = user["login"]
    display_name = user.get("name") or login

    now = (
        parse_timestamp(args.now)
        if args.now
        else datetime.now(timezone.utc)
    )
    cutoff = now - timedelta(days=args.days)

    events = load_events(login, cutoff)
    triaged: dict[str, dict[str, Any]] = {}
    for event in events:
        created_at = parse_timestamp(event["created_at"])
        if not cutoff <= created_at <= now:
            continue
        if event.get("repo", {}).get("name") != args.repo:
            continue
        if event.get("actor", {}).get("login", "").lower() != login.lower():
            continue

        payload = event.get("payload", {})
        issue = payload.get("issue", {})
        if not issue or "pull_request" in issue:
            continue

        reason: str | None = None
        if (
            event.get("type") == "IssuesEvent"
            and payload.get("action") == "closed"
            and issue.get("state_reason") == "not_planned"
        ):
            reason = "not planned"
        elif (
            event.get("type") == "IssueCommentEvent"
            and payload.get("action") == "created"
        ):
            reason = "commented"

        if reason is None:
            continue

        entry = remember_issue(triaged, issue, created_at)
        entry["reasons"].add(reason)

    requested_contributors = [
        contributor.strip()
        for contributor in args.completed_by.split(",")
        if contributor.strip()
    ]
    contributor_logins: list[str] = []
    contributor_names: dict[str, str] = {}
    completed: dict[str, dict[str, Any]] = {}
    for contributor in requested_contributors:
        profile = gh_json("api", f"users/{contributor}")
        canonical_login = profile["login"]
        contributor_logins.append(canonical_login)
        contributor_names[canonical_login] = profile.get("name") or canonical_login

        for issue in completed_candidates(args.repo, canonical_login, cutoff, now):
            entry = remember_issue(
                completed,
                issue,
                parse_timestamp(issue["closed_at"]),
            )
            entry["contributors"].add(canonical_login)

        for event in load_events(canonical_login, cutoff):
            created_at = parse_timestamp(event["created_at"])
            payload = event.get("payload", {})
            issue = payload.get("issue", {})
            if (
                cutoff <= created_at <= now
                and event.get("repo", {}).get("name") == args.repo
                and event.get("type") == "IssuesEvent"
                and payload.get("action") == "closed"
                and issue.get("state_reason") == "completed"
                and "pull_request" not in issue
            ):
                entry = remember_issue(completed, issue, created_at)
                entry["contributors"].add(canonical_login)

    requested_activity_contributors = [
        contributor.strip()
        for contributor in args.activity_by.split(",")
        if contributor.strip()
    ]
    activity_logins: list[str] = []
    activity_names: dict[str, str] = {}
    for contributor in requested_activity_contributors:
        profile = gh_json("api", f"users/{contributor}")
        canonical_login = profile["login"]
        activity_logins.append(canonical_login)
        activity_names[canonical_login] = profile.get("name") or canonical_login

        for event in load_events(canonical_login, cutoff):
            created_at = parse_timestamp(event["created_at"])
            payload = event.get("payload", {})
            issue = payload.get("issue", {})
            if (
                cutoff <= created_at <= now
                and event.get("repo", {}).get("name") == args.repo
                and event.get("type") in {"IssuesEvent", "IssueCommentEvent"}
                and issue
                and "pull_request" not in issue
            ):
                entry = remember_issue(triaged, issue, created_at)
                entry["reasons"].add("additional activity")
                entry["activity_contributors"].add(canonical_login)

    not_planned = [
        issue for issue in triaged.values() if "not planned" in issue["reasons"]
    ]
    other_triaged = [
        issue
        for issue in triaged.values()
        if {"commented", "additional activity"} & issue["reasons"]
        and "not planned" not in issue["reasons"]
        and issue["url"] not in completed
    ]
    person_names = {
        login: display_name,
        **contributor_names,
        **activity_names,
    }

    print("## Issues Triaged")
    print()
    print(
        f"**{display_name} (`{login}`) · {args.repo} · "
        f"{cutoff.date().isoformat()}–{now.date().isoformat()} UTC**"
    )
    print()

    if not not_planned and not completed and not other_triaged:
        print("- No matching issues found.")
        return 0

    if not_planned:
        print("### Not Planned")
        print()
        for issue in sorted(
            not_planned,
            key=lambda item: item["latest"],
            reverse=True,
        ):
            people_logins = []
            if "commented" in issue["reasons"]:
                people_logins.append(login)
            people_logins.extend(
                contributor
                for contributor in activity_logins
                if contributor in issue["activity_contributors"]
            )
            people = format_people(people_logins, person_names)
            suffix = f" — {people}" if people else ""
            print(
                f"- [#{issue['number']}]({issue['url']}) — "
                f"{issue['title']}{suffix}"
            )
        print()

    if completed:
        print("### Completed")
        print()
        for issue in sorted(
            completed.values(),
            key=lambda item: item["latest"],
            reverse=True,
        ):
            people_logins = [
                contributor
                for contributor in contributor_logins
                if contributor in issue["contributors"]
            ]
            triaged_issue = triaged.get(issue["url"], {})
            if "commented" in triaged_issue.get("reasons", set()):
                people_logins.append(login)
            activity_people = triaged_issue.get(
                "activity_contributors",
                set(),
            )
            people_logins.extend(
                contributor
                for contributor in activity_logins
                if contributor in activity_people
                and contributor not in issue["contributors"]
            )
            people = format_people(people_logins, person_names)
            print(
                f"- [#{issue['number']}]({issue['url']}) — "
                f"{issue['title']} — {people}"
            )
        print()

    if other_triaged:
        print("### Other Issues Triaged")
        print()
        for issue in sorted(
            other_triaged,
            key=lambda item: item["latest"],
            reverse=True,
        ):
            people_logins = []
            if "commented" in issue["reasons"]:
                people_logins.append(login)
            people_logins.extend(
                contributor
                for contributor in activity_logins
                if contributor in issue["activity_contributors"]
            )
            people = format_people(people_logins, person_names)
            suffix = f" — {people}" if people else ""
            print(
                f"- [#{issue['number']}]({issue['url']}) — "
                f"{issue['title']}{suffix}"
            )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, json.JSONDecodeError, KeyError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
