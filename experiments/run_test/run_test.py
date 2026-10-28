"""Run any Closed Loop bench test from the command line, no app needed.

    python experiments/run_test/run_test.py turnback my_data.csv
    python experiments/run_test/run_test.py orbit my_data.csv --groups --fakes 300
    python experiments/run_test/run_test.py leaderboard scores.csv
    python experiments/run_test/run_test.py three_point --values 0.51099895 105.6583755 1776.86
    python experiments/run_test/run_test.py dose_response doses.csv --ceiling 100

CSV layout for the series tests: one column per series, time down the rows (a first date/year column is skipped).
--groups takes out each group's own shared path, where a group is the part of a column name before '|' or ':'.
Prints the verdict and writes the full result to <test>_result.json next to your data.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from closedloop import lab as LB  # noqa: E402

PANEL = list(LB.PANEL_TESTS)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("test", choices=PANEL + ["leaderboard", "three_point", "dose_response"])
    ap.add_argument("csv", nargs="?", help="data file (not needed for three_point)")
    ap.add_argument("--fakes", type=int, default=200)
    ap.add_argument("--bar", type=float, default=95.0, help="percentile bar: 95, 99 or 99.9")
    ap.add_argument("--seed", type=int, default=20261028)
    ap.add_argument("--groups", action="store_true", help="take out each group's own shared path")
    ap.add_argument("--log", action="store_true", help="take logs of every series first")
    ap.add_argument("--values", type=float, nargs=3, help="three_point: the three values")
    ap.add_argument("--ceiling", type=float, default=100.0, help="dose_response: the ceiling")
    a = ap.parse_args()
    kw = dict(seed=a.seed, bar_pct=a.bar)

    if a.test == "three_point":
        if not a.values:
            ap.error("three_point needs --values v1 v2 v3")
        res = LB.three_point(a.values, **kw)
        out_dir = os.getcwd()
    else:
        if not a.csv:
            ap.error(f"{a.test} needs a CSV file")
        df = pd.read_csv(a.csv)
        out_dir = os.path.dirname(os.path.abspath(a.csv))
        if a.test == "leaderboard":
            res = LB.leaderboard(pd.to_numeric(df.iloc[:, -1], errors="coerce").dropna().values,
                                 n_fakes=max(a.fakes * 50, 10000), **kw)
        elif a.test == "dose_response":
            d = df.apply(pd.to_numeric, errors="coerce").dropna()
            res = LB.dose_response(d.iloc[:, 0].values, d.iloc[:, 1].values, ceiling=a.ceiling, **kw)
        else:
            num = df.apply(pd.to_numeric, errors="coerce")
            first = str(df.columns[0]).lower()
            if first in ("year", "date", "time", "row", "t", "period", "week", "month") or not pd.api.types.is_numeric_dtype(df.iloc[:, 0]):
                num = num.iloc[:, 1:]
            num = num.dropna(axis=1, how="all").dropna()
            if a.log:
                num = np.log(num.loc[:, (num > 0).all()])
            X = num.values.astype(float)
            names = list(map(str, num.columns))
            grp = LB.group_labels(names) if a.groups else None
            if a.groups and grp is None:
                print("note: no groups found in the column names; using all series")
            if a.test == "real_vs_fakes":
                res = LB.real_vs_fakes(X, **kw)
            elif a.test == "leftovers":
                res = LB.leftovers(X, names=names, n_slides=max(a.fakes, 300), groups=grp, remove_shared=a.groups, **kw)
            elif a.test == "step_flip":
                res = LB.step_flip(X, n_slides=a.fakes, groups=grp, **kw)
            elif a.test in ("orbit", "interlock"):
                res = LB.PANEL_TESTS[a.test](X, n_fakes=a.fakes, groups=grp, **kw)
            else:
                res = LB.PANEL_TESTS[a.test](X, n_fakes=a.fakes, **kw)

    print(f"\n{res['question']}\n  {'PASS' if res['passed'] else 'NO PASS'} - {res['verdict']}\n")
    for r in res["rows"]:
        bits = [f"{r['stat']}: {r.get('real')}"]
        if r.get("bar") is not None:
            bits.append(f"bar {r['bar']}")
        if r.get("percentile") is not None:
            bits.append(f"{r['percentile']}th pct")
        if r.get("note"):
            bits.append(r["note"])
        print("  - " + " | ".join(map(str, bits)))
    for n in res.get("notes", []):
        print(f"  * {n}")
    path = os.path.join(out_dir, f"{a.test}_result.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=2, default=str)
    print(f"\nFull result: {path}")


if __name__ == "__main__":
    main()
