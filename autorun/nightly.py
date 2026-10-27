"""Nightly run: pull every feed, rotate every series through the fixed lenses (flat + EQ curve) against its own
fakes, let Max watch, and save the results. Run by .github/workflows/closed-loop-nightly.yml. No AI, no tokens.

    python autorun/nightly.py
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from datetime import datetime, timezone

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import feeds as FD                       # noqa: E402
from closedloop import rotation as R     # noqa: E402

DATA = os.path.join(HERE, "data")
RES = os.path.join(HERE, "results")
STATE = os.path.join(HERE, "state.json")
CONFIG = os.path.join(HERE, "config.json")

SERIES_FEEDS = {"fred": FD.fred, "worldbank": FD.worldbank, "wspr": FD.wspr, "fec": FD.fec}


def load_json(p, default):
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return default


def save_json(p, obj):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        json.dump(obj, f, indent=1, default=str)


def save_feed_csv(name, series: dict):
    os.makedirs(DATA, exist_ok=True)
    df = pd.concat({k: v for k, v in series.items() if not k.startswith("_")}, axis=1, sort=True)
    df.index.name = "date"
    df.to_csv(os.path.join(DATA, f"{name}.csv"), float_format="%.6g")


def load_all_series() -> dict:
    """Every saved feed series, by name (used by the weekly job too)."""
    out = {}
    if not os.path.isdir(DATA):
        return out
    for fn in sorted(os.listdir(DATA)):
        if fn.endswith(".csv"):
            df = pd.read_csv(os.path.join(DATA, fn), index_col=0)
            try:
                df.index = pd.to_datetime(df.index)
            except Exception:
                pass
            for c in df.columns:
                out[c] = df[c].dropna().astype(float)
    return out


def main():
    now = datetime.now(timezone.utc)
    cfg = load_json(CONFIG, {})
    state = load_json(STATE, {})
    curve = {int(k): float(v) for k, v in cfg.get("eq_curve", {}).items()} or R.ZACH_CURVE
    n_fakes = int(cfg.get("n_fakes", 500))
    bar_pct = float(cfg.get("bar_pct", 99.0))
    seed = int(now.strftime("%Y%m%d"))

    ok, skipped, notes, headlines = [], [], [], []
    series = {}
    for name, fn in SERIES_FEEDS.items():
        try:
            got = fn()
            note = got.pop("_note", None)
            if note:
                notes.append(note)
            got = {k: v for k, v in got.items() if len(v.dropna()) >= 24}
            if not got:
                raise RuntimeError("returned no usable series")
            save_feed_csv(name, got)
            series.update({k: v.dropna() for k, v in got.items()})
            ok.append(name)
        except Exception as ex:
            skipped.append(f"{name} ({str(ex)[:120]})")

    # watchers
    try:
        items = FD.arxiv_juno()
        seen = set(state.get("arxiv_seen", []))
        new = [i for i in items if i["id"] not in seen]
        if seen and new:
            for i in new:
                headlines.append(f"New JUNO paper on arXiv: {i['title']} ({i['id']}, {i['published']}). Check it against the locked Δm²21 bet.")
        state["arxiv_seen"] = sorted(seen | {i["id"] for i in items})[-200:]
        ok.append("arxiv-juno")
    except Exception as ex:
        skipped.append(f"arxiv-juno ({str(ex)[:120]})")
    try:
        w = FD.who_infant_watch()
        if state.get("who_newest") and w["newest_year"] != state["who_newest"]:
            headlines.append(f"WHO infant mortality now runs to {w['newest_year']} (was {state['who_newest']}): "
                             "time to rerun the modeling fingerprint on fresh years.")
        state["who_newest"] = w["newest_year"]
        ok.append("who-watch")
    except Exception as ex:
        skipped.append(f"who-watch ({str(ex)[:120]})")

    # fresh data since last night
    rows_before = state.get("rows", {})
    fresh = {k: int(len(v)) - int(rows_before.get(k, len(v))) for k, v in series.items()}
    state["rows"] = {k: int(len(v)) for k, v in series.items()}

    res = R.rotate({k: v.values for k, v in series.items()}, curve=curve, n_fakes=n_fakes, bar_pct=bar_pct,
                   seed=seed, max_on=True) if series else None

    # survivors that keep surviving on fresh data climb toward 'lead'
    streaks = state.get("streaks", {})
    today_keys = set()
    if res:
        for s in res["survivors"]:
            k = f"{s['name']} | {s['lens']} | {s['eq']}"
            today_keys.add(k)
            streaks[k] = streaks.get(k, 0) + 1
            headlines.append(f"Survivor: {k} ({s['standing']}), held {streaks[k]} night(s) running.")
        for s in res["luck_pile"]:
            headlines.append(f"Luck pile: {s['name']} | {s['lens']} | {s['eq']}: {s['why']}.")
    for k in list(streaks):
        if k not in today_keys:
            headlines.append(f"Dropped: {k} stopped surviving after {streaks[k]} night(s).")
            del streaks[k]
    state["streaks"] = streaks
    if res and not res["survivors"] and not res["luck_pile"]:
        headlines.append("Nothing passed tonight. Every series looked like its own fakes.")

    out = dict(kind="nightly", ran_at=now.isoformat(timespec="seconds"), feeds_ok=ok, feeds_skipped=skipped,
               notes=notes, headlines=headlines, fresh_rows=fresh,
               survivors=res["survivors"] if res else [], luck_pile=res["luck_pile"] if res else [],
               containment=res["containment"] if res else {}, max_notes=(res["max_notes"] if res else [])[-80:],
               settings=res["settings"] if res else {}, key=res["key"] if res else {},
               log=res["log"] if res else [])
    save_json(os.path.join(RES, "nightly", now.strftime("%Y-%m-%d") + ".json"), out)
    slim = {k: v for k, v in out.items() if k != "log"}
    save_json(os.path.join(RES, "latest_nightly.json"), slim)
    save_json(STATE, state)
    print("\n".join(headlines) or "no headlines")
    print("ok:", ok, "skipped:", skipped)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
