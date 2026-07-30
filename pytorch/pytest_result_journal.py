#!/usr/bin/env python3
"""Crash-safe pytest result journal used by framework_scripts/run_tests.py."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path


JOURNAL_ENV = "FRAMEWORK_SCRIPTS_RESULT_JOURNAL"

STATE_PASSED = "passed"
STATE_SKIPPED = "skipped"
STATE_XFAILED = "xfailed"
STATE_ERROR = "error"
STATE_FAILED = "failed"

_node_states: dict[str, dict[str, object]] = {}


def _journal_path() -> Path | None:
    value = os.environ.get(JOURNAL_ENV, "").strip()
    return Path(value) if value else None


def _append_event(event: dict[str, object]) -> None:
    path = _journal_path()
    if path is None:
        return
    payload = (
        json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        os.write(fd, payload)
    finally:
        os.close(fd)


def _is_xfail(report) -> bool:
    return bool(getattr(report, "wasxfail", False))


def _new_node_state() -> dict[str, object]:
    return {
        "state": STATE_PASSED,
        "duration": 0.0,
    }


def pytest_runtest_logstart(nodeid, location) -> None:
    _node_states[nodeid] = _new_node_state()
    _append_event(
        {
            "event": "start",
            "nodeid": nodeid,
            "timestamp": time.time(),
        }
    )


def pytest_runtest_logreport(report) -> None:
    nodeid = report.nodeid
    node_state = _node_states.setdefault(nodeid, _new_node_state())
    node_state["duration"] = float(node_state["duration"]) + float(
        getattr(report, "duration", 0.0) or 0.0
    )

    outcome = report.outcome
    when = report.when
    is_xfail = _is_xfail(report)

    if when == "setup":
        if outcome == "failed":
            node_state["state"] = STATE_ERROR
        elif outcome == "skipped":
            node_state["state"] = (
                STATE_XFAILED if is_xfail else STATE_SKIPPED
            )
        return

    if when == "call":
        if outcome == "passed":
            node_state["state"] = STATE_FAILED if is_xfail else STATE_PASSED
        elif outcome == "skipped":
            node_state["state"] = (
                STATE_XFAILED if is_xfail else STATE_SKIPPED
            )
        elif outcome in {"failed", "rerun"}:
            node_state["state"] = STATE_FAILED
        return

    if when == "teardown" and outcome == "failed":
        node_state["state"] = STATE_ERROR


def pytest_runtest_logfinish(nodeid, location) -> None:
    node_state = _node_states.pop(nodeid, _new_node_state())
    _append_event(
        {
            "event": "finish",
            "nodeid": nodeid,
            "state": node_state["state"],
            "duration": float(node_state["duration"]),
            "timestamp": time.time(),
        }
    )
