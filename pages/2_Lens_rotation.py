"""Lens rotation: fixed lenses, the data rotates through them, an EQ pass rides along, Max watches the tunnels."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from closedloop import rotation as R      # noqa: E402
from closedloop import runlog as RL       # noqa: E402
from closedloop import sources as S       # noqa: E402
from closedloop import thumbs as T        # noqa: E402
from closedloop import eqcurve as EQ      # noqa: E402
from closedloop import bandview as BV     # noqa: E402
from closedloop import globe as G         # noqa: E402
from closedloop import embed as EM        # noqa: E402
import importlib, inspect      # noqa: E401,E402
if "slide" not in inspect.signature(EQ.eq_curve).parameters:     # the server may still hold the old drag code in memory
    EQ = importlib.reload(EQ)

st.set_page_config(page_title="Lens rotation · Closed Loop", page_icon="🔁", layout="centered")
ss = st.session_state
AUTORUN = os.path.join(ROOT, "autorun", "results")

st.title("Lens rotation")
st.markdown(
    "The **lenses stay fixed** and **every dataset rotates through every lens**, then comes back carrying "
    "all of its results. Datasets are **blinded** (D01, D02…) while they're measured. An **EQ pass** rides "
    "along: flat first as the control, then your curve, and the **fakes get the exact same EQ** so the knobs "
    "can't invent a pattern. **Max in the back** watches each step as it's built and, when something passes "
    "or stumbles, calls a jinx and actually runs it.")

tab_run, tab_auto = st.tabs(["Run it", "Nightly & weekly"])

# ---------------------------------------------------------------- sidebar settings
with st.sidebar:
    st.header("Rotation settings")
    n_fakes = st.select_slider("Fakes per step", [200, 300, 500, 1000], 500)
    bar_pct = st.select_slider("Bar (how far past the fakes)", [95.0, 99.0, 99.9], 99.0)
    seed = st.number_input("Seed", 0, 99_999_999, 20261026, step=1)
    max_on = st.toggle("Max in the back", value=True)
    st.caption("Fixed lenses: " + " · ".join(R.LENSES))

# ---------------------------------------------------------------- run tab
with tab_run:
    src = st.radio("Data", ["Demo (noise + one planted cycle)", "Upload a CSV", "FRED series IDs"], horizontal=True)
    series: dict[str, np.ndarray] = {}
    if src.startswith("Demo"):
        rng = np.random.default_rng(7)
        n = 240
        for i in range(9):
            series[f"noise {i + 1}"] = np.cumsum(rng.normal(0.05, 1, n)) * 0.3 + rng.normal(0, 1, n)
        t = np.arange(n)
        series["planted cycle"] = np.cumsum(rng.normal(0.05, 1, n)) * 0.3 + rng.normal(0, 1, n) + 2.0 * np.sin(2 * np.pi * t / 19)
        st.caption("Nine series of plain noise and one with a cycle hidden in it. The blind labels hide which is which.")
    elif src.startswith("Upload"):
        # No browser-side type filter: Android pickers grey out CSVs they haven't tagged as CSV.
        f = st.file_uploader("CSV: one column per series (a date column is fine, it's skipped)", type=None)
        if f is not None and not f.name.lower().endswith((".csv", ".tsv", ".txt")):
            st.error("Pick a CSV, TSV or TXT file.")
            st.stop()
        if f is not None:
            df = S.read_csv_bytes(f.getvalue())
            num = df.select_dtypes("number")
            for c in num.columns:
                v = num[c].dropna().values.astype(float)
                if len(v) >= 24:
                    series[str(c)] = v
            st.caption(f"{len(series)} usable series (24+ numeric points each).")
    else:
        ids = st.text_input("FRED IDs, comma-separated", "UNRATE, INDPRO, CPIAUCSL, HOUST, PAYEMS")
        if ids.strip():
            try:
                df = S.fetch_fred(ids, rename=False)
                for c in df.columns[1:]:
                    series[c] = df[c].values.astype(float)
                st.caption(f"{len(series)} series, {len(df)} rows lined up on shared dates.")
            except Exception as ex:
                st.error(str(ex))

    # ------------------------------------------------ EQ with the amalgamation handles
    st.subheader("EQ pass")
    preset = st.radio("Curve", ["Flat only", "Zach's curve", "Phone curve", "Custom"], index=1, horizontal=True)
    if "eq_custom" not in ss:
        ss["eq_custom"] = [R.PHONE_CURVE[f] for f in R.EQ_BANDS]
    if preset == "Custom":
        ss["eq_custom"] = EQ.eq_curve(ss["eq_custom"], key="eq_drag_custom", slide=True, shelves=True)
        curve = EQ.as_curve(ss["eq_custom"])
        if st.button("Start custom from the phone curve"):
            ss["eq_custom"] = [R.PHONE_CURVE[f] for f in R.EQ_BANDS]
            st.rerun()
    else:
        base = {"Flat only": R.FLAT, "Zach's curve": R.ZACH_CURVE, "Phone curve": R.PHONE_CURVE}[preset]
        EQ.eq_curve(R.curve_at(base) if preset == "Zach's curve" else [base[f] for f in R.EQ_BANDS],
                    key=f"eq_show_{preset}", disabled=True)
        curve = base
    st.caption("Bands are read as if each series were audio at 44.1 kHz, so a 2 kHz cut always hits the same "
               "relative slice (about 9% of the way to the top) whatever the time scale. Short series have few "
               "slices, so the low bands may not touch anything. Zach's curve keeps its original handles "
               "(it's drawn here at the ten dots). In Custom, drag a sigil up or down for gain, left or right to move it, twist it to shape it, and tap shape to switch a band between bell and shelf. "
               "Each quarter turn doubles (clockwise) or halves (counter-clockwise) how wide it reaches.")

    # ------------------------------------------------ band spiral (look only, nothing here is scored)
    with st.expander("Band spiral: each band spiraling across the dB plane", expanded=False):
        if series:
            pick = st.selectbox("Series", list(series), key="spiral_pick")
            mode = st.radio("Radius", ["Time (staircase out)", "dB level (orbit)"], horizontal=True, key="spiral_mode")
            st.altair_chart(BV.parametric_chart(series[pick], curve), width="stretch")
            bands = R.band_split(series[pick])
            st.altair_chart(BV.spiral_chart(bands, by_time=mode.startswith("Time")), width="stretch")
            lock = R.octave_lock(bands)
            if lock:
                st.caption("Octave interlock (1 = teeth locked like a zipper, 0 = no relation): "
                           + " · ".join(f"{k} **{v:.2f}**" for k, v in lock.items()))
            st.caption("For looking, not scoring. A band only counts if it passes in the rotation against the fakes.")
        else:
            st.caption("Load some data first.")

    # ------------------------------------------------ plasma globe (bands as cones, pins where a band beats its fakes)
    st.subheader("Plasma globe")
    if series:
        g1, g2 = st.columns(2)
        gpick = g1.selectbox("Series in the globe", list(series), key="globe_pick")
        zpick = g2.selectbox("Zipper with", ["(none)"] + [s for s in series if s != gpick], key="globe_zip")
        knobs = json.dumps(R.curve_json(curve), sort_keys=True)
        seen = ss.setdefault("globe_seen", {}).setdefault(gpick, set())
        seen.add(knobs)
        payload, dings = G.band_check(series[gpick], curve, looks=len(seen))
        payload["name"] = gpick
        other, cross = None, None
        if zpick != "(none)":
            seen_b = ss["globe_seen"].setdefault(zpick, set())
            seen_b.add(knobs)
            other, dings_b = G.band_check(series[zpick], curve, looks=len(seen_b))
            other["name"] = zpick
            dings = dings + [dict(d, who=zpick) for d in dings_b]
            cross = G.cross_teeth(series[gpick], series[zpick], curve)
        EM.show(G.render(payload, other, cross), height=470)
        st.caption(
            "The ring around the middle is your EQ: the flat line is the circle, ups push out, downs pull in. "
            "The glowing shell is the data, each band at its own spot around the circle, rippling with that band's "
            "energy over time. Pick a second series and its rings zip in between the first one's; teeth light up "
            "where the same band locks across the two (green in step, yellow a quarter turn, pink opposite). "
            "Drag to spin, tap a pin to see why it dinged. "
            f"**Looks on this series: {len(seen)}.** Every new knob setting is another look, so the bar gets stricter "
            "each time. A pin is a flag, not a finding, until it holds on frozen settings, fresh fakes, Max, and new data.")
        if dings:
            st.markdown("**Pins:** " + " · ".join(
                f"{d.get('who', gpick)}: {d['hz'] // 1000 if d['hz'] >= 1000 else d['hz']}{'k' if d['hz'] >= 1000 else ' Hz'} "
                f"(rows {d['rows'][0]}–{d['rows'][1]}, {d['pct']}th pct)" for d in dings))
    else:
        st.caption("Load some data first.")

    if st.button("Rotate", type="primary", disabled=len(series) == 0):
        prog = st.progress(0.0, text="Building tunnels…")
        live = st.empty()

        def on_step(i, total, e):
            prog.progress(i / total, text=f"Step {i}/{total}: {e['dataset']} · {e['lens']} · {e['eq']}")
            if i % 4 == 0 or i == total:
                live.caption(f"Last step: {e['dataset']} {e['lens']} ({e['eq']}) → "
                             f"{'PASS' if e.get('passed') else 'no'} at the {e.get('percentile', float('nan')):.1f}th percentile")

        res = R.rotate(series, curve=None if preset == "Flat only" else curve, n_fakes=int(n_fakes),
                       bar_pct=float(bar_pct), seed=int(seed), max_on=max_on, on_step=on_step)
        prog.empty(); live.empty()
        ss["rot"] = res

    res = ss.get("rot")
    if res and "run_id" not in res:      # stamp once per run, so redraws don't mint new IDs and re-save it
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        res["logged_at"] = now
        res["run_id"] = "LR-" + hashlib.sha256(json.dumps([res["settings"], res["log"], now], sort_keys=True,
                                                          default=str).encode()).hexdigest()[:8].upper()
    if res:
        key = res["key"]
        s1, s2, s3 = st.columns(3)
        s1.metric("Steps", len(res["log"]))
        s2.metric("Survivors", len(res["survivors"]))
        s3.metric("Luck pile", len(res["luck_pile"]))

        st.markdown("#### Max in the back")
        if not res["max_notes"]:
            st.caption("Max had nothing to say. Nothing passed, nothing stumbled.")
        for m in res["max_notes"][-40:]:
            st.markdown(f"**Step {m['step']}** · _{m['line']}_  \n{m['note']}" if m["line"] else f"**Step {m['step']}** · {m['note']}")

        st.markdown("#### Survivors (passed, and Max couldn't break them)")
        if res["survivors"]:
            st.dataframe(pd.DataFrame(res["survivors"]), hide_index=True)
        else:
            st.caption("None this round.")
        st.markdown("#### Luck pile (kept, not thrown away)")
        if res["luck_pile"]:
            st.dataframe(pd.DataFrame(res["luck_pile"]), hide_index=True)
        else:
            st.caption("Empty this round.")

        with st.expander("Containment (every lens and EQ state per dataset)"):
            st.dataframe(pd.DataFrame(res["containment"]).T, width="stretch")
            st.caption(f"'After correction' uses the stricter bar {res['settings']['corrected_bar_pct']}th percentile, "
                       "because each dataset gets several looks.")
        with st.expander("Step log (the tunnels, step by step)"):
            lg = pd.DataFrame(res["log"]).drop(columns=["max"], errors="ignore")
            lg["bar"] = lg["bar"].map(lambda b: f"{b[0]:.4g} to {b[1]:.4g}" if isinstance(b, (list, tuple)) else (f"{b:.4g}" if isinstance(b, float) else str(b)))
            st.dataframe(lg, hide_index=True)
        with st.expander("Unblind (which label was which)"):
            st.table(pd.DataFrame({"label": list(key), "dataset": list(key.values())}).set_index("label"))

        rid = res["run_id"]
        record = json.dumps(dict(run_id=rid, logged_at=res["logged_at"],
                                 **{k: v for k, v in res.items() if k not in ("run_id", "logged_at")}),
                            indent=2, default=str)
        cfg = RL.config(st.secrets)
        if not ss.get(f"saved_{rid}") and cfg and RL.put_file(
                cfg, f"rotation_runs/{rid}.json", record.encode(), f"Lens rotation {rid}"):
            ss[f"saved_{rid}"] = True
        if ss.get(f"saved_{rid}"):
            st.caption(f"Run **{rid}** saved to the shared log ✓")
        st.download_button("Download this run (.json)", record, file_name=f"{rid}.json")

# ---------------------------------------------------------------- nightly / weekly tab
with tab_auto:
    st.markdown("The repo runs this on its own: **every night** each feed is rotated against its own fakes; "
                "**every week** the whole week's feeds are run against each other. No AI, no tokens.")
    for label, fname in [("Latest nightly", "latest_nightly.json"), ("Latest weekly", "latest_weekly.json")]:
        p = os.path.join(AUTORUN, fname)
        st.markdown(f"#### {label}")
        if not os.path.exists(p):
            st.caption("Nothing yet. The first run lands after the scheduled job fires.")
            continue
        d = json.load(open(p))
        st.caption(f"Ran {d.get('ran_at', '?')} · feeds: {', '.join(d.get('feeds_ok', [])) or 'none'}"
                   + (f" · skipped: {', '.join(d.get('feeds_skipped', []))}" if d.get("feeds_skipped") else ""))
        for m in d.get("headlines", []):
            st.markdown(f"- {m}")
        for m in d.get("max_notes", [])[-15:]:
            st.markdown(f"**{m.get('kind', '')}** · _{m.get('line', '')}_  \n{m.get('note', '')}")
        with st.expander("Full record"):
            st.json(d, expanded=False)
