"""Run the smallest documented regression group for the current change set."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

GROUPS: dict[str, tuple[str, ...]] = {
    "agent": (
        "tests/test_agent.py",
        "tests/test_conversation.py",
        "tests/test_agent_guidance.py",
        "tests/test_tool_contracts.py",
        "tests/test_desktop_chat.py",
        "tests/test_sets_and_streaming.py",
    ),
    "frontend": (
        "tests/test_console_and_doctor.py",
        "tests/test_desktop_manual_model.py",
        "tests/test_desktop_locate.py",
        "tests/test_desktop_scene.py",
        "tests/test_desktop_window.py",
        "tests/test_engineering_visual_acceptance.py",
        "tests/test_result_inspector.py",
    ),
    "solver": (
        "tests/test_benchmark.py",
        "tests/test_advanced_beam.py",
        "tests/test_buckling.py",
        "tests/test_deflection.py",
        "tests/test_kernel.py",
        "tests/test_modal.py",
        "tests/test_span_loads.py",
        "tests/test_solid_joint.py",
        "tests/test_convergence.py",
        "tests/test_solid_cache.py",
        "tests/test_solid_task.py",
    ),
    "docs": (
        "tests/test_docs_current.py",
        "tests/test_packaging.py",
        "tests/test_build_exe.py",
    ),
}


def _unique_tests(groups: list[str]) -> list[str]:
    return list(dict.fromkeys(test for group in groups for test in GROUPS[group]))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a documented incremental regression group; never runs full pytest.")
    parser.add_argument(
        "groups", nargs="*", choices=sorted(GROUPS),
        help="One or more groups: agent, frontend, solver, docs.")
    parser.add_argument(
        "--candidate", action="store_true",
        help="Run the union of all current candidate groups (still not the full suite).")
    parser.add_argument(
        "--list", action="store_true", help="Print the selected tests without running them.")
    args = parser.parse_args()

    selected = list(GROUPS) if args.candidate else list(args.groups)
    if not selected:
        parser.error("choose at least one group, or pass --candidate")
    tests = _unique_tests(selected)
    print(f"incremental groups: {', '.join(selected)}", flush=True)
    print(f"test files: {len(tests)}", flush=True)
    if args.list:
        print("\n".join(tests))
        return 0

    command = [sys.executable, "-m", "pytest", "-q", *tests]
    return subprocess.call(command, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
