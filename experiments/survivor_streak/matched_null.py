"""Global survivor-streak falsification using degree-matched history shuffles.

Every original day and each distinct survivor's total appearance count stays
fixed in the null. No original files modified. This is an exploratory check,
NOT a quantum verification or a proof of predictive power.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import random
from pathlib import Path


def read_history(repo: Path):
    files = sorted((repo / "autorun/results/nightly").glob("????-??-??.json"))
    if len(files) < 3:
        raise ValueError("Require at least three real nightly files")
    dates, rows, sources = [], [], []
    for file in files:
        obj = json.loads(file.read_text("utf-8"))
        if obj.get("kind") != "nightly" or not isinstance(obj.get("survivors"), list):
            raise ValueError("Missing complete nightly source: " + str(file))
        keys = []
        for s in obj["survivors"]:
            if not isinstance(s, dict) or not all(s.get(x) for x in ("name", "lens", "eq")):
                raise ValueError("Bad survivor identity: " + str(file))
            keys.append(" | ".join(s[x] for x in ("name", "lens", "eq")))
        dates.append(dt.date.fromisoformat(file.stem).isoformat())
        rows.append(set(keys))
        sources.append({"path": file.relative_to(repo).as_posix(),
                        "sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
                        "skipped_feeds": obj.get("feeds_skipped", []),
                        "raw": len(keys), "distinct": len(rows[-1])})
    return dates, rows, sources


def counts(rows):
    out = {}
    for items in rows:
        for k in items:
            out[k] = out.get(k, 0) + 1
    return out


def longest(dates, rows):
    run, best, winners, previous = {}, 0, set(), None
    for d, keys in zip(dates, rows):
        day = dt.date.fromisoformat(d)
        consecutive = previous is not None and (day - previous).days == 1
        run = {k: (run.get(k, 0) if consecutive else 0) + 1 for k in keys}
        for k, length in run.items():
            if length > best:
                best, winners = length, {k}
            elif length == best:
                winners.add(k)
        previous = day
    return best, sorted(winners)


def shuffle(rows, rng, attempts):
    """Two-day two-key swaps preserve every day and key degree exactly."""
    candidate = [set(row) for row in rows]
    swaps = 0
    for _ in range(attempts):
        a, b = rng.sample(range(len(rows)), 2)
        left, right = candidate[a] - candidate[b], candidate[b] - candidate[a]
        if not left or not right:
            continue
        ka, kb = rng.choice(sorted(left)), rng.choice(sorted(right))
        candidate[a].remove(ka)
        candidate[b].remove(kb)
        candidate[a].add(kb)
        candidate[b].add(ka)
        swaps += 1
    return candidate, swaps


def adjacent_overlap(dates, rows):
    """How often does the same key persist across neighboring calendar days?"""
    total = 0
    for i in range(1, len(dates)):
        if (dt.date.fromisoformat(dates[i]) - dt.date.fromisoformat(dates[i - 1])).days == 1:
            total += len(rows[i] & rows[i - 1])
    return total


def intermittent_streak(dates, rows, eligible):
    """Exclude forced/near-forced always-present keys from streak analysis."""
    return longest(dates, [r & eligible for r in rows])


def evaluate(dates, rows, sources, fakes=200, swaps=900, seed=20261118):
    if fakes < 20 or swaps < 1 or len(dates) != len(rows) or len(rows) != len(sources):
        raise ValueError("Need source-backed dates and at least 20 matched fakes")
    original_degrees, original_counts = [len(x) for x in rows], counts(rows)
    best, winners = longest(dates, rows)
    # An always-present key trivially has a 21-day streak in EVERY null.
    # Analyse only genuinely intermittent keys as an independent, labeled
    # secondary statistic; never report the degenerate global max as a hit.
    eligible = {k for k, v in original_counts.items() if 2 <= v <= len(dates) - 2}
    inter_best, inter_winners = intermittent_streak(dates, rows, eligible)
    real_overlap = adjacent_overlap(dates, rows)
    rng, null, accepted = random.Random(seed), [], []
    intermittent_null, overlap_null = [], []
    for _ in range(fakes):
        fake, n = shuffle(rows, rng, swaps)
        if [len(x) for x in fake] != original_degrees or counts(fake) != original_counts:
            raise AssertionError("Null failed to preserve original degrees")
        null.append(longest(dates, fake)[0])
        intermittent_null.append(intermittent_streak(dates, fake, eligible)[0])
        overlap_null.append(adjacent_overlap(dates, fake))
        accepted.append(n)
    if min(accepted) < 1:
        raise ValueError("Fake history could not be shuffled: not an informative null")
    sorted_null = sorted(null)
    sorted_intermittent = sorted(intermittent_null)
    sorted_overlap = sorted(overlap_null)
    return {
        "schema": "closedloop.matched-survivor-null.v1",
        "status": "EXPLORATORY_MATCHED_FAKE_TEST_NOT_CAUSAL_PROOF",
        "dates": dates, "sources": sources, "unique_survivor_keys": len(original_counts),
        "original_day_survivor_counts": original_degrees,
        "observed_global_max_consecutive_days": best, "observed_best_keys": winners,
        "nontrivial_streak": {
            "key_inclusion_rule": "2 <= total day appearances <= (number of days - 2)",
            "eligible_keys": len(eligible),
            "observed_max": inter_best,
            "observed_best_keys": inter_winners,
            "null_max_values": intermittent_null,
            "null_median": sorted_intermittent[fakes // 2],
            "p_ge_observed_plus_one": (
                (1 + sum(v >= inter_best for v in intermittent_null)) / (fakes + 1)
                if eligible else None
            ),
        },
        "adjacent_calendar_day_overlap": {
            "statistic": "sum across consecutive calendar dates of survivor intersection count",
            "observed": real_overlap,
            "null_values": overlap_null,
            "null_median": sorted_overlap[fakes // 2],
            "p_ge_observed_plus_one": (
                1 + sum(v >= real_overlap for v in overlap_null)
            ) / (fakes + 1),
        },
        "null_control": {
            "method": "two-day/two-key degree-preserving double-edge swaps",
            "invariants": ["original per-day counts", "original per-key appearances"],
            "multiple_scan_correction": "global maximum over all keys and all windows",
            "n_fakes": fakes, "swaps_per_fake": swaps, "seed": seed,
            "actual_swaps_min": min(accepted),
            "null_global_max_streaks": null,
            "null_median": sorted_null[fakes // 2],
            "null_nearest_rank_95pct": sorted_null[(95 * fakes + 99) // 100 - 1],
            "p_ge_observed_plus_one": (1 + sum(n >= best for n in null)) / (fakes + 1),
        },
        "cautions": [
            "Daily missing feeds can create apparent streaks and are shown per file",
            "Degree matching assumes history is exchangeable except for day and key frequency",
            "Do not extrapolate to VQCC scores, causal dynamics, or app production",
            "This does not re-run original matched momentum controls",
        ],
    }


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--repo", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    a.add_argument("--fakes", type=int, default=200)
    a.add_argument("--swaps", type=int, default=900)
    a.add_argument("--seed", type=int, default=20261118)
    x = a.parse_args()
    dates, rows, sources = read_history(x.repo.resolve())
    result = evaluate(dates, rows, sources, x.fakes, x.swaps, x.seed)
    x.out.parent.mkdir(parents=True, exist_ok=True)
    x.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", "utf-8")
    print(json.dumps({"days": len(dates), "max_streak": result["observed_global_max_consecutive_days"],
                      "null_p": result["null_control"]["p_ge_observed_plus_one"],
                      "output": str(x.out)}))


if __name__ == "__main__":
    main()
