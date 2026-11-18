#!/usr/bin/env python3
"""Evidence-first cross-project learning snapshot from Closed Loop historical runs.

Standard library only. DOES NOT train model weights, assert causation, or validate
quantum circuits. It makes source-anchored observations reusable across agents.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


METHOD_PATHS = (
    "closedloop/rotation.py",
    "closedloop/zipper.py",
    "closedloop/menu.py",
    "closedloop/engine.py",
    "closedloop/lab.py",
    "autorun/nightly.py",
    "autorun/weekly.py",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def survivor_key(value: dict) -> str:
    """Stable observational identity, never equate a lens with its score."""
    return " | ".join(str(value.get(k, "?")) for k in ("name", "lens", "eq"))


def summarize_run(file: Path, repo: Path) -> dict:
    data = json.loads(file.read_text("utf-8"))
    items = data.get("survivors", [])
    if not isinstance(items, list):
        raise ValueError("survivors must be a list: " + str(file))
    survivors = sorted({survivor_key(x) for x in items if isinstance(x, dict)})
    log = data.get("log", [])
    if not isinstance(log, list):
        raise ValueError("log must be a list: " + str(file))
    return {
        "date": file.stem,
        "source": file.relative_to(repo).as_posix(),
        "sha256": sha(file),
        "ran_at": data.get("ran_at"),
        "feeds_ok": sorted(data.get("feeds_ok") or []),
        "feeds_skipped": data.get("feeds_skipped") or [],
        "survivor_keys": survivors,
        "distinct_survivors": len(survivors),
        "luck_pile_count": len(data.get("luck_pile") or []),
        "tested_steps": len(log),
        "fake_arm_passed_steps": sum(
            bool(x.get("fake_arm_passed")) for x in log if isinstance(x, dict)
        ),
    }


def inventory_csv(repo: Path) -> list[dict]:
    out = []
    for directory in ("autorun/data", "demo/modeling_fingerprint/snapshot", "samples"):
        folder = repo / directory
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("*.csv")):
            with path.open(newline="", encoding="utf-8-sig") as stream:
                reader = csv.reader(stream)
                header = next(reader, [])
                rows = sum(1 for _ in reader)
            out.append({
                "source": path.relative_to(repo).as_posix(),
                "sha256": sha(path),
                "rows": rows,
                "columns": header[:30],
                "columns_truncated": len(header) > 30,
            })
    return out


def build(repo: Path) -> dict:
    repo = repo.resolve()
    hist = repo / "autorun/results/nightly"
    if not hist.is_dir():
        raise FileNotFoundError("Original Closed Loop nightly history absent: " + str(hist))
    methods = []
    for relative in METHOD_PATHS:
        path = repo / relative
        if not path.is_file():
            raise FileNotFoundError("Required original method absent: " + relative)
        methods.append({"source": relative, "sha256": sha(path)})
    days = [summarize_run(p, repo) for p in sorted(hist.glob("????-??-??.json"))]
    if not days:
        raise ValueError("No historical nightly runs available")
    for i, day in enumerate(days):
        prior = set(days[i - 1]["survivor_keys"]) if i else set()
        current = set(day["survivor_keys"])
        day["new_vs_previous"] = sorted(current - prior) if i else []
        day["lost_vs_previous"] = sorted(prior - current) if i else []
        day["comparison_available"] = i > 0
    streaks: dict[str, int] = {}
    for day in days:
        current = set(day["survivor_keys"])
        streaks = {k: streaks.get(k, 0) + 1 for k in current}
    latest = days[-1]
    # This count tracks consecutive daily FILES, not independent experimental proof.
    latest_streaks = sorted(
        [{"key": k, "consecutive_recorded_days": v} for k, v in streaks.items()],
        key=lambda x: (-x["consecutive_recorded_days"], x["key"]),
    )
    inventory = inventory_csv(repo)
    return {
        "schema": "closedloop.evidence.v1",
        "source_repository": "zachfeemster-create/Closed-loop",
        "interpretation": "Historical observational evidence; no causality, new circuit, verifier or customer safety claim.",
        "methods": methods,
        "datasets": inventory,
        "nightly_history": days,
        "latest": {
            "date": latest["date"], "source": latest["source"], "sha256": latest["sha256"],
            "feeds_ok": latest["feeds_ok"], "feeds_skipped": latest["feeds_skipped"],
            "survivors": latest["distinct_survivors"],
            "luck_pile_count": latest["luck_pile_count"],
            "new_vs_previous": latest["new_vs_previous"],
            "lost_vs_previous": latest["lost_vs_previous"],
            "streaks": latest_streaks,
            "fake_arm_passed_steps": latest["fake_arm_passed_steps"],
        },
        "historical_run_count": len(days),
        "dataset_count": len(inventory),
    }


def context(manifest: dict) -> str:
    latest = manifest["latest"]
    lines = [
        "# Shared Closed Loop evidence (read-only)",
        "SOURCE: https://github.com/zachfeemster-create/Closed-loop",
        "SCHEMA: closedloop.evidence.v1",
        "RESEARCH STATUS: observational history; no new verified quantum score or production food guarantee.",
        "METHODS (inspect the original implementations, not merely these labels):",
    ]
    lines.extend("- " + x["source"] + " sha256=" + x["sha256"][:16] for x in manifest["methods"])
    lines += [
        f"Historical nightly runs: {manifest['historical_run_count']}.",
        f"Inventory: {manifest['dataset_count']} real CSV datasets/snapshots, each with SHA-256 and row count.",
        f"Most recent source run: {latest['source']} sha256={latest['sha256'][:16]}.",
        f"Latest survivors (distinct name+lens+EQ): {latest['survivors']}; luck-pile entries: {latest['luck_pile_count']}.",
        f"Latest feeds okay: {', '.join(latest['feeds_ok']) or 'none'}.",
        f"Latest feeds skipped: {', '.join(latest['feeds_skipped']) or 'none'}.",
        f"Latest fake-arm passed steps: {latest['fake_arm_passed_steps']} (inspect why before interpreting).",
    ]
    if len(manifest["nightly_history"]) > 1:
        lines.extend([
            "New survivor keys vs previous *recorded file*:",
            *("- " + s for s in latest["new_vs_previous"][:8]),
            "Lost survivor keys vs previous *recorded file*:",
            *("- " + s for s in latest["lost_vs_previous"][:8]),
        ])
    lines.append("Consecutive recorded-day survivors (not independent validation):")
    lines.extend(
        f"- {x['key']}: {x['consecutive_recorded_days']}"
        for x in latest["streaks"][:8]
    )
    lines += [
        "INFERENCE GUARDRAILS:",
        "- Do not treat surviving matched fakes as causation, circuit equivalence, or real-world forecast.",
        "- Compare flat vs curve, real vs momentum-matched fake, distinct vs raw hits, and held-out data.",
        "- Missing feeds are missing, not negative evidence; do not silently fill them.",
        "- For Quantum Arcane, pair these methods with ORIGINAL circuit/verifier evidence.",
        "- For Smoke Signal, transfer the test methodology, not astrophysics/economics conclusions.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    manifest = build(args.root)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "evidence.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", "utf-8"
    )
    (args.out / "context.md").write_text(context(manifest), "utf-8")
    print(json.dumps({
        "schema": manifest["schema"],
        "nightlies": manifest["historical_run_count"],
        "datasets": manifest["dataset_count"],
        "latest": manifest["latest"]["date"],
        "context": str(args.out / "context.md"),
    }))


if __name__ == "__main__":
    main()
