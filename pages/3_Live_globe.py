"""Live globe: every logged run on one globe, with plain-words reasons for every pass, break and miss."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from closedloop import loglobe as LG     # noqa: E402
from closedloop import rotation as R     # noqa: E402
from closedloop import runlog as RL      # noqa: E402
from closedloop import sources as S      # noqa: E402
from closedloop import embed as EM       # noqa: E402

st.set_page_config(page_title="Live globe · Closed Loop", page_icon="🌐", layout="centered")
ss = st.session_state
st.title("Live globe")
st.markdown("Every logged run on one globe: the **nightly** and **weekly** jobs, the **shared log** of rotations people "
            "ran in the app, runs **you run right here**, and any run file you **upload**. Each light is one dataset in "
            "one run. Tap it to see, in plain words, why it passed, broke, or missed.")

cfg = RL.config(st.secrets)


@st.cache_data(ttl=300, show_spinner=False)
def shared_runs(_cfg_key: str, limit: int = 40) -> list[dict]:
    names = [n for n in RL.list_files(cfg, "rotation_runs") if n.endswith(".json")][-limit:]
    out = []
    for n in names:
        rec = RL.get_json(cfg, f"rotation_runs/{n}")
        r = LG.normalize(rec, "shared log", n[:-5]) if rec else None
        if r:
            out.append(r)
    return out


runs, weeks = LG.local_runs(ROOT)
if cfg:
    runs += shared_runs(hashlib.sha256(cfg[1].encode()).hexdigest()[:8])
if ss.get("rot"):
    r = LG.normalize(dict(ss["rot"], logged_at=datetime.now(timezone.utc).isoformat()), "this session (rotation page)", "session")
    if r:
        runs.append(r)
runs += ss.get("live_runs", [])

with st.expander("Add runs: upload saved run files, or run a fresh pass now"):
    ups = st.file_uploader("Run files (.json from 'Download this run')", type=["json"], accept_multiple_files=True)
    for f in ups or []:
        try:
            r = LG.normalize(json.loads(f.getvalue()), "uploaded", f.name.rsplit(".", 1)[0])
            if r:
                runs.append(r)
            else:
                st.warning(f"{f.name}: no step log in it, so there's nothing to put on the globe.")
        except Exception as ex:
            st.warning(f"{f.name}: couldn't read it ({ex}).")
    ids = st.text_input("FRED series for a fresh pass", "UNRATE, INDPRO, CPIAUCSL, HOUST, PAYEMS")
    if st.button("Run a fresh pass now", type="primary"):
        with st.spinner("Pulling the data and rotating it through the lenses…"):
            try:
                df = S.fetch_fred(ids, rename=False)
                series = {c: df[c].dropna().values.astype(float) for c in df.columns[1:]}
                res = R.rotate(series, curve=R.ZACH_CURVE, n_fakes=200, seed=int(datetime.now().strftime("%Y%m%d")))
                now = datetime.now(timezone.utc).isoformat(timespec="seconds")
                rid = "LR-" + hashlib.sha256(json.dumps([res["settings"], now], sort_keys=True, default=str).encode()).hexdigest()[:8].upper()
                rec = dict(run_id=rid, logged_at=now, **res)
                ss.setdefault("live_runs", []).append(LG.normalize(rec, "run here", rid))
                if cfg and RL.put_file(cfg, f"rotation_runs/{rid}.json", json.dumps(rec, indent=2, default=str).encode(),
                                       f"Lens rotation {rid}"):
                    st.success(f"Run {rid} saved to the shared log.")
                st.rerun()
            except Exception as ex:
                st.error(f"The fresh pass fell over: {ex}")

if not runs:
    st.info("No runs to show yet. The nightly job hasn't saved anything in this repo yet: turn its schedule on "
            "(see the autorun README), run a rotation on the Lens rotation page, upload a run file, or tap "
            "**Run a fresh pass now** above.")
else:
    sources = sorted({r["source"] for r in runs})
    pick = st.multiselect("Show", sources, default=sources)
    shown = [r for r in runs if r["source"] in pick]
    if shown:
        p = LG.payload(shown)
        EM.show(LG.render(p), height=480)

        from closedloop import luckgraph as LK
        LK.show(shown)

        st.subheader("Fails and misses, explained")
        labels = [f"{r['when']:%Y-%m-%d %H:%M} · {r['source']} · {r['id']}" for r in sorted(shown, key=lambda r: r["when"], reverse=True)]
        which = st.selectbox("Run", labels)
        run = sorted(shown, key=lambda r: r["when"], reverse=True)[labels.index(which)]
        rows = [dict(dataset=run["key"].get(e["dataset"], e["dataset"]), lens=e.get("lens"), eq=e.get("eq"),
                     result=LG.outcome(e), why=LG.explain(e, run["bar_pct"])) for e in run["log"]]
        df = pd.DataFrame(rows)
        only = st.radio("Which", ["Everything", "Only passes", "Only misses"], horizontal=True)
        if only == "Only passes":
            df = df[df.result.isin(["survivor", "luck", "frame"])]
        elif only == "Only misses":
            df = df[df.result.isin(["miss", "near", "error"])]
        st.dataframe(df, hide_index=True, width="stretch")
        tally = pd.Series([r["result"] for r in rows]).value_counts()
        st.caption("This run: " + " · ".join(f"{k} {v}" for k, v in tally.items()) +
                   ". A miss means the series looked like its own momentum-matched fakes; that's an answer, not a failure.")
        st.download_button("Save these runs (.json)", json.dumps(
            [dict(r, when=r["when"].isoformat()) for r in shown], indent=1, default=str), file_name="closed-loop-runs.json")

if weeks:
    st.subheader("Weekly passes")
    for w in reversed(weeks[-4:]):
        st.markdown(f"**{str(w.get('ran_at', '?'))[:10]}**")
        for h in w.get("headlines", []) or ["Nothing stood out that week."]:
            st.markdown(f"- {h}")

with st.expander("Put this on your phone"):
    st.markdown("On Android: open this site in **Chrome**, tap **⋮** (top right), then **Add to Home screen** "
                "(or **Install app**). It opens full-screen from its own icon like any other app, and all your "
                "data inputs (file, Google Sheets, paste, link) work the same on your work phone.")
