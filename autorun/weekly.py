"""Weekly run: take the week's feeds and run them against each other as one batch, then give the week's
luck pile its own stricter pass. Run by .github/workflows/closed-loop-weekly.yml. No AI, no tokens.

Cross-feed tests use a rolling 120-day window of daily feeds (a week alone is only 7 daily points) and
flag what's new this week. Shared-clock links (weekday rhythm) are measured and removed first and reported
separately as 'lens of time', so 'both dip on Sundays' never counts as one feed driving another.

    python autorun/weekly.py
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from closedloop import engine as E        # noqa: E402
from closedloop import rotation as R      # noqa: E402
from nightly import RES, load_all_series, load_json, save_json   # noqa: E402

WINDOW_DAYS = 120
DAILY_PREFIXES = ("WSPR", "FEC", "FRED DGS10", "FRED T10Y2Y", "FRED DCOILWTICO", "FRED VIXCLS")


def weekday_share(s: pd.Series) -> float:
    """How much of a daily series' wobble is just the day of the week (lens of time)."""
    y = np.log1p(s.clip(lower=0)) if (s >= 0).all() else s
    y = y - y.rolling(7, center=True, min_periods=4).mean()
    y = y.dropna()
    if len(y) < 28 or y.var() == 0:
        return float("nan")
    m = y.groupby(y.index.dayofweek).transform("mean")
    return float(m.var() / y.var())


def build_panel(series: dict, end: pd.Timestamp):
    start = end - pd.Timedelta(days=WINDOW_DAYS)
    cols, time_lens = {}, {}
    for k, s in series.items():
        if not k.startswith(DAILY_PREFIXES) or not isinstance(s.index, pd.DatetimeIndex):
            continue
        s = s[(s.index > start) & (s.index <= end)]
        if len(s) < 40:
            continue
        time_lens[k] = round(weekday_share(s), 4)
        cols[k] = s
    if len(cols) < 3:
        return None, time_lens
    df = pd.DataFrame(cols)
    df = df[df.index.dayofweek < 5]                       # business days: every feed has them
    df = df.ffill(limit=3).dropna(axis=1, thresh=int(0.9 * len(df))).dropna()
    for c in df.columns:                                  # positive counts/amounts -> log scale
        if (df[c] >= 0).all():
            df[c] = np.log1p(df[c])
    dow = df.index.dayofweek
    df = df - df.groupby(dow).transform("mean") + df.mean()   # remove the shared weekday clock
    return df, time_lens


def main():
    now = datetime.now(timezone.utc)
    series = load_all_series()
    end = pd.Timestamp(now.date()) - pd.Timedelta(days=1)
    out = dict(kind="weekly", ran_at=now.isoformat(timespec="seconds"), window_days=WINDOW_DAYS, headlines=[],
               cross_feed={}, lens_of_time={}, luck_pass=[], week_survivors={}, feeds_ok=[], feeds_skipped=[])

    panel, tl = build_panel(series, end)
    out["lens_of_time"] = dict(sorted(tl.items(), key=lambda kv: -(kv[1] if kv[1] == kv[1] else -1)))
    strong_clock = [k for k, v in tl.items() if v == v and v > 0.2]
    if strong_clock:
        out["headlines"].append("Lens of time: a weekday rhythm carries over 20% of the wobble in "
                                + ", ".join(strong_clock) + ". Logged as a shared clock, removed before the cross-feed tests.")
    if panel is not None:
        names = list(panel.columns)
        X = panel.values
        out["cross_feed"]["series"] = names
        out["cross_feed"]["rows"] = int(len(panel))
        out["cross_feed"]["new_this_week"] = int((panel.index > end - pd.Timedelta(days=7)).sum())
        for label, fn in [("lead", lambda: E.lead_test(X, names, n_fakes=300, use_log=False, detection=False)),
                          ("bunching", lambda: E.bunching_test(X, n_fakes=200, use_log=False)),
                          ("drift", lambda: E.drift_test(X, n_fakes=150, use_log=False))]:
            try:
                r = fn()
                fa = (r.get("fake_arm") or {}).get("passed")
                verdict = "frame trip (doesn't count)" if r["passed"] and fa else ("PASS" if r["passed"] else "no")
                out["cross_feed"][label] = dict(question=r["question"], stat=r["stat_name"], real=r["real"],
                                                bar=r["bar"], percentile=r["percentile"], verdict=verdict,
                                                details=r.get("details", {}), notes=r.get("notes", []))
                if verdict == "PASS":
                    extra = ""
                    if label == "lead":
                        d = r["details"]; extra = f": {d['strongest']} {d['strongest_direction']} the others"
                    if label == "bunching":
                        extra = f": busiest days {', '.join(str(panel.index[min(i, len(panel) - 1)].date()) for i in r['details']['busiest_rows'])}"
                    out["headlines"].append(f"Cross-feed {label} PASSED{extra}. Check it against the calendar before trusting it.")
            except Exception as ex:
                out["cross_feed"][label] = dict(error=str(ex)[:200])
        out["feeds_ok"] = names
    else:
        out["headlines"].append("Not enough daily feeds lined up for the cross-feed tests this week.")

    # the week's luck pile, all together, at a stricter bar
    week = []
    for i in range(7):
        d = load_json(os.path.join(RES, "nightly", (now - timedelta(days=i)).strftime("%Y-%m-%d") + ".json"), None)
        if d:
            week.append(d)
            for s in d.get("survivors", []):
                k = f"{s['name']} | {s['lens']} | {s['eq']}"
                out["week_survivors"][k] = out["week_survivors"].get(k, 0) + 1
    piles = {}
    for d in week:
        for s in d.get("luck_pile", []):
            k = (s["name"], s["lens"], s["eq"])
            piles.setdefault(k, []).append(s.get("why", ""))
    rng = np.random.default_rng(int(now.strftime("%Y%m%d")))
    curve = R.ZACH_CURVE
    for (name, lens, eqn), whys in piles.items():
        x = series.get(name)
        if x is None or len(x) < 24:
            continue
        r = R.measure(x.values, curve if eqn == "curve" else R.FLAT, lens, 1000, rng, bar_pct=99.9)
        out["luck_pass"].append(dict(name=name, lens=lens, eq=eqn, nights_in_pile=len(whys), why=sorted(set(whys)),
                                     strict_percentile=r["percentile"], strict_pass=r["passed"] and not r["fake_arm_passed"]))
    if piles:
        by_lens = pd.Series([k[1] for k in piles]).value_counts().to_dict()
        out["headlines"].append(f"Luck pile this week: {len(piles)} items, by lens {by_lens}. "
                                f"{sum(p['strict_pass'] for p in out['luck_pass'])} cleared the strict 99.9 bar (Max's jinxes still broke them, so they stay in the pile; strong-but-fragile usually means one big event).")
    for k, n in sorted(out["week_survivors"].items(), key=lambda kv: -kv[1]):
        if n >= 5:
            out["headlines"].append(f"Held {n} of 7 nights: {k}. Ready to lock as a lead and test on the next weeks.")
    save_json(os.path.join(RES, "weekly", now.strftime("%G-W%V") + ".json"), out)
    save_json(os.path.join(RES, "latest_weekly.json"), out)
    print("\n".join(out["headlines"]) or "no headlines")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
