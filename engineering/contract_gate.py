#!/usr/bin/env python3
"""Independent acceptance gate for small, spec-driven agent work.

No LLM, API, paid service, source modification, deployment or implicit retries.
The maker edits on a separate review branch; this verifier runs actual tests,
checks machine-readable invariants and writes a fail-closed result for feedback.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
import sys


HEADINGS = (
    "Problem statement", "Non-goals", "Assumptions",
    "Reference implementation", "Architecture", "Test plan",
)
SCHEMA = "closedloop.acceptance-contract.v1"


def inside(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("Require nonempty relative paths")
    target = (root / relative).resolve()
    if target != root and root not in target.parents:
        raise ValueError("Path escapes project checkout: " + relative)
    return target


def validate(root: Path, contract: dict) -> tuple[str, list]:
    if contract.get("schema") != SCHEMA:
        raise ValueError("Invalid acceptance contract schema")
    identifier = contract.get("id")
    if not isinstance(identifier, str) or not identifier or "/" in identifier:
        raise ValueError("Contract must have a safe ID")
    spec_path = inside(root, contract.get("spec", ""))
    source = spec_path.read_text(encoding="utf-8")
    if any(f"## {name}" not in source for name in HEADINGS):
        raise ValueError("Spec must contain all six required sections")
    if "Given " not in source or "When " not in source or "Then " not in source:
        raise ValueError("Spec test plan must name an observable Given/When/Then")
    checks = contract.get("checks")
    if not isinstance(checks, list) or not checks or len(checks) > 8:
        raise ValueError("Require 1..8 checks")
    observed = set()
    for item in checks:
        if not isinstance(item, dict):
            raise ValueError("Every acceptance check must be an object")
        name = item.get("name")
        if not isinstance(name, str) or not name or name in observed:
            raise ValueError("Missing or duplicate check name")
        observed.add(name)
        kind = item.get("kind")
        if kind == "command":
            argv = item.get("argv")
            if not isinstance(argv, list) or not 2 <= len(argv) <= 12 or not all(
                isinstance(s, str) and s for s in argv
            ):
                raise ValueError("Command must be fixed argv")
            if argv[0] not in ("python", "python3", "node"):
                raise ValueError("Command executable not on allowlist")
            # No shells, scripts from /tmp, package downloads, eval flags,
            # or arbitrary commands. A spec gate is not a remote-code runner.
            if any(x in argv for x in ("-c", "-e", "--eval", "pip", "npm", "npx")) or (argv[0] == "node" and "-p" in argv):
                raise ValueError("Interactive evaluation/download commands prohibited")
            for part in argv[1:]:
                if part.startswith("/") or part == ".." or "../" in part or part.startswith("../"):
                    raise ValueError("Absolute or parent path in command")
            if argv[1] == "-m" and (len(argv) < 3 or argv[2] not in ("unittest", "py_compile")):
                raise ValueError("Only unittest and py_compile modules allowed")
            timeout = item.get("timeout_seconds", 60)
            if not isinstance(timeout, int) or not 1 <= timeout <= 120:
                raise ValueError("Individual command timeout must be 1..120 seconds")
        elif kind == "json_assert":
            inside(root, item.get("path", ""))
            expected = item.get("equals")
            if not isinstance(expected, dict) or not expected:
                raise ValueError("JSON assertions need explicit expected fields")
        elif kind == "contains":
            inside(root, item.get("path", ""))
            if not isinstance(item.get("needle"), str) or not item["needle"]:
                raise ValueError("Contains assertions require an exact snippet")
        else:
            raise ValueError("Unknown acceptance check kind: " + str(kind))
    return hashlib.sha256(source.encode("utf-8")).hexdigest(), checks


def read_member(obj: dict, dotted: str):
    cur = obj
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError("Missing contract field: " + dotted)
        cur = cur[part]
    return cur


def check(root: Path, item: dict) -> dict:
    result = {"name": item["name"], "kind": item["kind"], "passed": False}
    try:
        if item["kind"] == "command":
            run = subprocess.run(
                item["argv"], cwd=root, capture_output=True, text=True,
                timeout=item.get("timeout_seconds", 60), check=False
            )
            result.update(exit_code=run.returncode,
                          stdout_tail=run.stdout[-1400:],
                          stderr_tail=run.stderr[-1400:])
            result["passed"] = run.returncode == 0
        elif item["kind"] == "json_assert":
            file = inside(root, item["path"])
            data = json.loads(file.read_text("utf-8"))
            got = {k: read_member(data, k) for k in item["equals"]}
            result["observed"] = got
            result["passed"] = got == item["equals"]
        else:
            file = inside(root, item["path"])
            result["passed"] = item["needle"] in file.read_text("utf-8")
    except (OSError, ValueError, TypeError, KeyError, subprocess.TimeoutExpired) as err:
        result["error"] = type(err).__name__ + ": " + str(err)[:450]
    return result


def evaluate(root: Path, contract_path: Path) -> dict:
    root = root.resolve()
    source = contract_path.read_bytes()
    contract = json.loads(source)
    spec_digest, checks = validate(root, contract)
    outcomes = []
    for item in checks:
        outcomes.append(check(root, item))
        if not outcomes[-1]["passed"]:
            # Stop on the FIRST failed invariant. The next agent turn should
            # consume this evidence and propose a change, not spin repeats.
            break
    passed = len(outcomes) == len(checks) and all(x["passed"] for x in outcomes)
    return {
        "schema": "closedloop.acceptance-result.v1",
        "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "contract_id": contract["id"],
        "contract_sha256": hashlib.sha256(source).hexdigest(),
        "spec_sha256": spec_digest,
        "status": "VERIFIED_SPEC_CHECKS" if passed else "REJECTED_BY_INDEPENDENT_CHECK",
        "accepted": passed,
        "checks_total": len(checks),
        "checks_executed": len(outcomes),
        "results": outcomes,
        "next_action": (
            "No change necessary for this contract; still not a domain-level verifier."
            if passed else
            "Read the first failed check, change only a review branch, rerun gate; "
            "do not relabel a failed candidate successful."
        ),
        "interpretation": (
            "These checks verify only specified local invariants. They DO NOT certify "
            "a new Kaggle score, safe live food payments, or production readiness."
        ),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--contract", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    root = args.root.resolve()
    path = args.contract if args.contract.is_absolute() else root / args.contract
    try:
        summary = evaluate(root, path)
        code = 0 if summary["accepted"] else 1
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        summary = {
            "schema": "closedloop.acceptance-result.v1",
            "status": "INVALID_SPEC_OR_INFRASTRUCTURE",
            "accepted": False,
            "error": type(exc).__name__ + ": " + str(exc)[:600],
            "interpretation": "Contract was not executed. Never claim success.",
        }
        code = 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", "utf-8")
    print(json.dumps({k: summary.get(k) for k in
                      ("status", "accepted", "checks_executed", "checks_total", "error")}), flush=True)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
