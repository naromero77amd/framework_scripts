#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


PYTORCH_SCRIPTS_DIR = Path(__file__).resolve().parent


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run_tests = _load_module(
    "framework_run_tests",
    PYTORCH_SCRIPTS_DIR / "run_tests.py",
)
result_journal = _load_module(
    "framework_pytest_result_journal",
    PYTORCH_SCRIPTS_DIR / "pytest_result_journal.py",
)


def test_parse_result_journal_preserves_completed_and_active_nodes(tmp_path):
    journal = tmp_path / "results.jsonl"
    events = [
        {"event": "start", "nodeid": "test_sample.py::test_pass"},
        {
            "event": "finish",
            "nodeid": "test_sample.py::test_pass",
            "state": "passed",
            "duration": 1.25,
        },
        {"event": "start", "nodeid": "test_sample.py::test_crash"},
    ]
    journal.write_text(
        "\n".join(json.dumps(event) for event in events)
        + "\n"
        + '{"event":"finish"',
        encoding="utf-8",
    )

    results, active_node = run_tests._parse_result_journal(journal)

    assert results == [
        {
            "name": "test_sample.py::test_pass",
            "success": True,
            "time": 1.25,
            "timed_out": False,
            "state": "passed",
        }
    ]
    assert active_node == "test_sample.py::test_crash"


def test_merge_partial_results_uses_node_order_and_latest_result():
    nodes = [
        "test_sample.py::test_a",
        "test_sample.py::test_b",
    ]
    older = [
        {
            "name": nodes[1],
            "state": "failed",
            "success": False,
            "time": 1.0,
        }
    ]
    newer = [
        {
            "name": nodes[0],
            "state": "passed",
            "success": True,
            "time": 2.0,
        },
        {
            "name": nodes[1],
            "state": "passed",
            "success": True,
            "time": 3.0,
        },
    ]

    merged = run_tests._merge_partial_results(nodes, older, newer)

    assert [result["name"] for result in merged] == nodes
    assert [result["state"] for result in merged] == ["passed", "passed"]
    assert merged[1]["time"] == 3.0


@pytest.mark.skipif(os.name != "posix", reason="SIGSEGV integration is POSIX-only")
def test_result_journal_survives_pytest_sigsegv(tmp_path):
    test_file = tmp_path / "test_forced_sigsegv.py"
    test_file.write_text(
        """
import os
import signal


def test_01_before_crash():
    assert True


def test_02_forced_sigsegv():
    os.kill(os.getpid(), signal.SIGSEGV)


def test_03_after_crash():
    raise AssertionError("must not run")
""".lstrip(),
        encoding="utf-8",
    )
    journal = tmp_path / "results.jsonl"
    env = os.environ.copy()
    env[result_journal.JOURNAL_ENV] = str(journal)
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    env.pop("PYTEST_ADDOPTS", None)
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(PYTORCH_SCRIPTS_DIR) + (
        os.pathsep + existing_pythonpath if existing_pythonpath else ""
    )

    def disable_core_dump():
        import resource

        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "pytest_result_journal",
            "-x",
            "-q",
            str(test_file),
        ],
        cwd=tmp_path,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=30,
        preexec_fn=disable_core_dump,
    )

    assert completed.returncode < 0
    assert -completed.returncode == 11

    results, active_node = run_tests._parse_result_journal(journal)
    result_by_name = {result["name"]: result for result in results}
    before_node = f"{test_file.name}::test_01_before_crash"
    crash_node = f"{test_file.name}::test_02_forced_sigsegv"
    after_node = f"{test_file.name}::test_03_after_crash"

    assert result_by_name[before_node]["state"] == "passed"
    assert crash_node not in result_by_name
    assert after_node not in result_by_name
    assert active_node == crash_node


def test_result_journal_keeps_final_rerun_outcome(tmp_path):
    pytest.importorskip("pytest_rerunfailures")
    test_file = tmp_path / "test_flaky.py"
    test_file.write_text(
        """
import os
from pathlib import Path


def test_passes_on_rerun():
    marker = Path(os.environ["FRAMEWORK_SCRIPTS_FLAKY_MARKER"])
    if not marker.exists():
        marker.touch()
        raise AssertionError("first attempt fails")
""".lstrip(),
        encoding="utf-8",
    )
    journal = tmp_path / "results.jsonl"
    marker = tmp_path / "flaky.marker"
    env = os.environ.copy()
    env[result_journal.JOURNAL_ENV] = str(journal)
    env["FRAMEWORK_SCRIPTS_FLAKY_MARKER"] = str(marker)
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    env.pop("PYTEST_ADDOPTS", None)
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(PYTORCH_SCRIPTS_DIR) + (
        os.pathsep + existing_pythonpath if existing_pythonpath else ""
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "pytest_result_journal",
            "-p",
            "pytest_rerunfailures",
            "--reruns",
            "1",
            "-q",
            str(test_file),
        ],
        cwd=tmp_path,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stdout
    results, active_node = run_tests._parse_result_journal(journal)
    assert active_node is None
    assert results[-1]["name"] == f"{test_file.name}::test_passes_on_rerun"
    assert results[-1]["state"] == "passed"


@pytest.mark.skipif(os.name != "posix", reason="SIGSEGV integration is POSIX-only")
def test_file_batch_recovers_completed_nodes_from_sigsegv_journal(
    tmp_path, monkeypatch
):
    test_file = tmp_path / "test_batch_sigsegv.py"
    test_file.write_text(
        """
import os
import signal


def test_01_before_crash():
    assert True


def test_02_forced_sigsegv():
    os.kill(os.getpid(), signal.SIGSEGV)


def test_03_after_crash():
    raise AssertionError("must not run")
""".lstrip(),
        encoding="utf-8",
    )
    (tmp_path / "conftest.py").write_text(
        """
def pytest_addoption(parser):
    parser.addoption("--sc", action="store", default=None)
""".lstrip(),
        encoding="utf-8",
    )
    node_ids = [
        f"{test_file.name}::test_01_before_crash",
        f"{test_file.name}::test_02_forced_sigsegv",
        f"{test_file.name}::test_03_after_crash",
    ]
    monkeypatch.setattr(
        run_tests,
        "_build_test_env",
        lambda: os.environ.copy(),
    )
    args = SimpleNamespace(
        per_test_timeout=10,
        per_file_timeout=30,
        retry_attempts=0,
        pytorch_path=str(tmp_path),
    )
    log_path = tmp_path / "batch.log"

    import resource

    previous_core_limit = resource.getrlimit(resource.RLIMIT_CORE)
    resource.setrlimit(
        resource.RLIMIT_CORE,
        (0, previous_core_limit[1]),
    )
    try:
        with log_path.open("w+", encoding="utf-8") as log_file:
            results, reason, _, problem_node = run_tests._run_file_batch(
                test_file.name,
                node_ids,
                args,
                log_file,
            )
    finally:
        resource.setrlimit(resource.RLIMIT_CORE, previous_core_limit)

    result_by_name = {result["name"]: result for result in results}
    assert reason == "crash"
    assert problem_node == node_ids[1]
    assert result_by_name[node_ids[0]]["state"] == "passed"
    assert node_ids[1] not in result_by_name
    assert node_ids[2] not in result_by_name
