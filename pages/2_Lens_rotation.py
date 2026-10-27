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
        f = st.file_uploader("CSV: one column per series (a date column is fine, it's skipped)", type=["csv", "tsv", "txt"])
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
    preset = st.radio("Curve", ["Flat only", "Zach's curve", "Custom"], index=1, horizontal=True)
    base = R.FLAT if preset == "Flat only" else R.ZACH_CURVE
    keys = [f"eqband{i}" for i in range(len(R.EQ_BANDS))]
    st.markdown(T.slider_css(keys), unsafe_allow_html=True)
    curve = {}
    for i, fhz in enumerate(R.EQ_BANDS):
        with st.container(key=keys[i]):
            label = f"{fhz / 1000:g} kHz" if fhz >= 1000 else f"{fhz} Hz"
            curve[fhz] = st.slider(label, -6.0, 6.0, float(base[fhz]), 0.05, format="%+.2f dB",
                                   disabled=(preset != "Custom"), key=f"eqv_{preset}_{fhz}")
    st.caption("Bands are read as if each series were audio at 44.1 kHz, so a 2 kHz cut always hits the same "
               "relative slice (about 9% of the way to the top) whatever the time scale. Short series have few "
               "slices, so the low bands may not touch anything.")

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
        st.dataframe(pd.DataFrame(res["survivors"]) if res["survivors"] else pd.DataFrame({"": ["none this round"]}),
                     hide_index=True)
        st.markdown("#### Luck pile (kept, not thrown away)")
        st.dataframe(pd.DataFrame(res["luck_pile"]) if res["luck_pile"] else pd.DataFrame({"": ["empty"]}),
                     hide_index=True)

        with st.expander("Containment (every lens and EQ state per dataset)"):
            st.dataframe(pd.DataFrame(res["containment"]).T, use_container_width=True)
            st.caption(f"'After correction' uses the stricter bar {res['settings']['corrected_bar_pct']}th percentile, "
                       "because each dataset gets several looks.")
        with st.expander("Step log (the tunnels, step by step)"):
            lg = pd.DataFrame(res["log"]).drop(columns=["max"], errors="ignore")
            lg["bar"] = lg["bar"].map(lambda b: f"{b[0]:.4g} to {b[1]:.4g}" if isinstance(b, (list, tuple)) else (f"{b:.4g}" if isinstance(b, float) else str(b)))
            st.dataframe(lg, hide_index=True)
        with st.expander("Unblind (which label was which)"):
            st.table(pd.DataFrame({"label": list(key), "dataset": list(key.values())}).set_index("label"))

        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        rid = "LR-" + hashlib.sha256(json.dumps([res["settings"], now], sort_keys=True, default=str).encode()).hexdigest()[:8].upper()
        record = json.dumps(dict(run_id=rid, logged_at=now, **res), indent=2, default=str)
        cfg = RL.config(st.secrets)
        if not ss.get(f"saved_{rid[:6]}_{len(res['log'])}") and cfg and RL.put_file(
                cfg, f"rotation_runs/{rid}.json", record.encode(), f"Lens rotation {rid}"):
            ss[f"saved_{rid[:6]}_{len(res['log'])}"] = True
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
