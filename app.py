"""Closed Loop — a careful way for anybody to test a hunch against data.

Method by Zach Feemster ("Vex"), CC BY 4.0.
Run locally:  streamlit run app.py
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from closedloop import engine as E

try:
    import base64 as _b64
    from closedloop.logo import LOGO_WEBP_B64
    LOGO = Image.open(io.BytesIO(_b64.b64decode(LOGO_WEBP_B64)))
except Exception:
    LOGO = "assets/closed_loop_coin_256.png" if os.path.exists("assets/closed_loop_coin_256.png") else None

st.set_page_config(page_title="Closed Loop", page_icon=(LOGO if LOGO is not None else "🔁"), layout="centered")

SAMPLES = {
    "US gas + crude + CPI (weekly)": "samples/us_gas_crude_weekly.csv",
    "US regional gas (weekly)": "samples/us_regional_gas_weekly.csv",
    "Country GDP (yearly)": "samples/country_gdp_yearly.csv",
}
TEST_LABELS = {
    "cycle": "Hidden cycle (one series)",
    "asymmetry": "Faster up than down (driver → response)",
    "lead": "Who moves first (many series)",
    "drift": "Shifting links (many series)",
    "bunching": "Leftover turns bunching (many series)",
}

ss = st.session_state
ss.setdefault("locked", False)
ss.setdefault("plan", None)
ss.setdefault("tree", [])          # list of nodes
ss.setdefault("ran", False)


# ---------------------------------------------------------------- helpers

def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def plan_hash(plan: dict) -> str:
    return sha(json.dumps(plan, sort_keys=True, default=str).encode())[:16]


def parse_windows(text: str, index) -> tuple[np.ndarray, list]:
    """Lines like '2008-07-01 to 2009-06-30' or '2008 to 2009'. Returns mask over index."""
    mask = np.zeros(len(index), bool)
    parsed = []
    for line in (text or "").splitlines():
        parts = re.split(r"\s+(?:to|-|–)\s+", line.strip())
        if len(parts) != 2:
            continue
        a, b = parts
        try:
            if isinstance(index, pd.DatetimeIndex):
                a_, b_ = pd.to_datetime(a), pd.to_datetime(b)
                if len(b) == 4:
                    b_ = b_ + pd.offsets.YearEnd(0)
            else:
                a_, b_ = float(a), float(b)
            mask |= (index >= a_) & (index <= b_)
            parsed.append(f"{a} to {b}")
        except Exception:
            continue
    return mask, parsed


def prepare(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    d = df.copy()
    if cfg.get("deflator"):
        d = d[[c for c in cfg["series"] if c != cfg["deflator"]] + [cfg["deflator"]]].dropna()
        for c in cfg["series"]:
            if c != cfg["deflator"]:
                d[c] = d[c] / d[cfg["deflator"]]
        d = d.drop(columns=[cfg["deflator"]])
    cols = [c for c in cfg["series"] if c != cfg.get("deflator")]
    return d[cols].dropna()


def run_spec(spec: dict, data: pd.DataFrame, mask_all: np.ndarray, seed: int) -> dict:
    p = dict(spec["params"])
    mod = spec.get("modifiers", {})
    d = data
    m = mask_all
    if mod.get("rows") == "later_half":
        h = len(d) // 2
        d, m = d.iloc[h:], m[h:]
    if mod.get("rows") == "earlier_half":
        h = len(d) // 2
        d, m = d.iloc[:h], m[:h]
    mask = m if mod.get("drop_events") else None
    bar = mod.get("bar_pct", 99.0)
    nf = spec["n_fakes"]
    use_log = spec["use_log"]
    t = spec["type"]
    if t == "cycle":
        y = d[p["target"]].values.astype(float)
        drv = d[p["drivers"]].values.astype(float) if p["drivers"] else None
        if use_log:
            y = np.log(y)
            drv = None if drv is None else np.log(drv)
        return E.cycle_test(y, drivers=drv, season=int(p["season"]), band=(p["pmin"], p["pmax"] or None),
                            n_fakes=nf, seed=seed, mask=mask, bar_pct=bar)
    if t == "asymmetry":
        return E.asymmetry_test(d[p["response"]].values, d[p["driver"]].values, lags=int(p["lags"]),
                                early=int(p["early"]), direction=p["direction"], n_fakes=min(nf, 500),
                                seed=seed, mask=mask, bar_pct=bar, use_log=use_log)
    cols = list(p["series"])
    if mod.get("half_series"):
        rng = np.random.default_rng(seed)
        cols = list(rng.choice(cols, max(3, len(cols) // 2), replace=False))
    X = d[cols].values.astype(float)
    if t == "lead":
        return E.lead_test(X, names=cols, n_fakes=nf, seed=seed, mask=mask, bar_pct=bar, use_log=use_log)
    if t == "drift":
        return E.drift_test(X, n_fakes=min(nf, 300), seed=seed, use_log=use_log, bar_pct=bar)
    if t == "bunching":
        return E.bunching_test(X, n_fakes=min(nf, 300), seed=seed, use_log=use_log, bar_pct=bar)
    raise ValueError(t)


def ladder(node: dict) -> str:
    r = node["result"]
    mod = node["spec"].get("modifiers", {})
    if r.get("fake_arm") and r["fake_arm"]["passed"]:
        return "⚠️ the lens draws this (fake arm passed) — doesn't count"
    if r["passed"]:
        if mod.get("rows") in ("later_half", "earlier_half"):
            return "🟢 lead — held on a slice it wasn't checked on (a finding needs someone else to reproduce it)"
        if mod.get("bar_pct", 99) > 99 or mod.get("half_series") or mod.get("drop_events"):
            return "🟢 sturdy survivor — holds under a stricter check"
        return "🟢 survivor — beats the fakes"
    det = r.get("detection")
    if det:
        return f"⚪ luck pile — {det['pile']}"
    return "⚪ luck pile — not seen (no detection limit for this test yet)"


NEXT_ON_PASS = [
    ("later_half", "Replicate on the later half", "Run the same locked test on the second half of the rows only."),
    ("earlier_half", "Replicate on the earlier half", "Same test, first half of the rows only."),
    ("strict", "Stricter bar (99.9%)", "Same data, much harder pass bar."),
    ("drop_events", "Drop the flagged event windows", "Zero out the windows you flagged (wars, crashes…) and rerun."),
    ("half_series", "Half the series at random", "Same test on a random half of the series."),
]


def child_spec(parent: dict, action: str) -> dict:
    s = json.loads(json.dumps(parent["spec"]))
    mod = s.setdefault("modifiers", {})
    if action in ("later_half", "earlier_half"):
        mod["rows"] = action
    elif action == "strict":
        mod["bar_pct"] = 99.9
    elif action == "drop_events":
        mod["drop_events"] = True
    elif action == "half_series":
        mod["half_series"] = True
    return s


def add_node(spec: dict, parent_id, data, mask, base_seed):
    nid = len(ss.tree)
    locked_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lock = dict(spec=spec, parent=parent_id, data_hash=ss.data_hash, locked_at=locked_at, seed=base_seed + nid)
    lock_id = plan_hash(lock)
    result = run_spec(spec, data, mask, base_seed + nid)
    ss.tree.append(dict(id=nid, parent=parent_id, spec=spec, lock_id=lock_id, locked_at=locked_at,
                        seed=base_seed + nid, result=result))
    return nid


def label_for(node):
    s = node["spec"]
    mods = s.get("modifiers", {})
    extra = []
    if mods.get("rows"):
        extra.append(mods["rows"].replace("_", " "))
    if mods.get("bar_pct", 99) > 99:
        extra.append("99.9% bar")
    if mods.get("drop_events"):
        extra.append("events dropped")
    if mods.get("half_series"):
        extra.append("half the series")
    return TEST_LABELS[s["type"]] + (f" · {', '.join(extra)}" if extra else "")


def fmt_bar(b):
    return f"{b[0]} / {b[1]}" if isinstance(b, list) else f"{b}"


def show_result(node, data, mask, base_seed):
    r = node["result"]
    ok, fa = r["passed"], r.get("fake_arm")
    with st.container(border=True):
        st.markdown(f"**{label_for(node)}**  \n<small>Lock `{node['lock_id']}` · seed {node['seed']} · {node['locked_at']} UTC</small>",
                    unsafe_allow_html=True)
        st.caption(r["question"])
        c1, c2, c3 = st.columns(3)
        c1.metric("Real", r["real"])
        c2.metric("Pass bar", fmt_bar(r["bar"]))
        c3.metric("Percentile of fakes", f"{r['percentile']:.1f}")
        (st.success if ok else st.error)(("PASS — " if ok else "MISS — ") + r["stat_name"])
        if fa:
            if fa["passed"]:
                st.warning(f"Fake arm passed too ({fa['real']}) — this test is drawing its own shape here.")
            else:
                st.caption(f"Fake arm (no real pattern) stayed quiet: {fa['real']}")
        st.markdown(f"**Ladder:** {ladder(node)}")
        det = r.get("detection")
        if det:
            lim = det["limit"]
            st.info(f"**Detection limit:** {'nothing bigger than ' + str(lim) if lim is not None else 'above the largest planted size'} "
                    f"({det['unit']}). Pile: **{det['pile']}**." +
                    (" Too small for this data — finer data (daily instead of weekly, cities instead of the nation) could go below the floor."
                     if det["pile"] == "below the floor" else ""))
        for note in r.get("notes", []):
            st.caption("• " + note)
        with st.expander("Details"):
            st.json(r["details"])
            if det:
                st.write("Catch rate by planted size:", det["catch_rate"])
        # next-step cards
        if ok and not (fa and fa["passed"]):
            st.markdown("**Next steps**")
            moves = [m for m in NEXT_ON_PASS
                     if not (m[0] == "drop_events" and not ss.event_windows)
                     and not (m[0] == "half_series" and node["spec"]["type"] in ("cycle", "asymmetry"))]
            cols = st.columns(2)
            for i, (key, title, helptext) in enumerate(moves):
                if cols[i % 2].button(title, key=f"n{node['id']}_{key}", help=helptext, use_container_width=True):
                    with st.spinner("Locking and running…"):
                        add_node(child_spec(node, key), node["id"], data, mask, base_seed)
                    st.rerun()
        elif not ok:
            st.caption("Next step for a miss: if it's 'below the floor', look for finer data and bring it back as a new run. "
                       "If it's 'nothing there', the test could have seen it and didn't — log it and move on.")


def render_tree():
    kids = {}
    for n in ss.tree:
        kids.setdefault(n["parent"], []).append(n)

    def walk(pid, depth):
        for n in kids.get(pid, []):
            r = n["result"]
            chip = "⚠️" if r.get("fake_arm") and r["fake_arm"]["passed"] else ("✅" if r["passed"] else "❌")
            st.markdown(f"{'&nbsp;' * 6 * depth}{'↳ ' if depth else ''}{chip} **{label_for(n)}** — "
                        f"{r['real']} vs {fmt_bar(r['bar'])} · {ladder(n).split(' — ')[0]}", unsafe_allow_html=True)
            walk(n["id"], depth + 1)

    walk(None, 0)


def report_md() -> str:
    p = ss.plan
    lines = ["# Closed Loop report", "", f"Method by Zach Feemster (\"Vex\"), CC BY 4.0.", "",
             f"**Keystone:** {p['intake']['keystone']}", "", f"Plan lock `{p['plan_id']}` at {p['locked_at']} UTC · data hash `{p['data_hash'][:16]}`", ""]
    for k, v in p["intake"].items():
        if k != "keystone" and v:
            lines.append(f"- **{k.replace('_', ' ')}:** {v}")
    lines += ["", "## Runs"]
    for n in ss.tree:
        r = n["result"]
        lines += ["", f"### {label_for(n)} (lock `{n['lock_id']}`, parent {n['parent']})",
                  f"- {r['stat_name']}: **{r['real']}** vs bar {fmt_bar(r['bar'])} (percentile {r['percentile']:.1f}) → **{'PASS' if r['passed'] else 'MISS'}**",
                  f"- Fake arm: {r['fake_arm']}" if r.get("fake_arm") else "- Fake arm: n/a",
                  f"- Ladder: {ladder(n)}"]
        if r.get("detection"):
            lines.append(f"- Detection limit: {r['detection']['limit']} ({r['detection']['unit']}); pile: {r['detection']['pile']}")
        lines.append(f"- Details: `{json.dumps(r['details'], default=str)}`")
    return "\n".join(lines)


# ---------------------------------------------------------------- UI

_l, _r = st.columns([1, 4], vertical_alignment="center")
if LOGO is not None:
    _l.image(LOGO, width=96)
_r.title("Closed Loop")
st.caption("Test a hunch against data without fooling yourself. Lock the plan first, beat matched fakes, "
           "keep what misses, and know how big a miss could be. Method by Zach Feemster (\"Vex\"), CC BY 4.0.")

locked = ss.locked

# ---- 1. Data
st.header("1 · Data")
SAMPLES = {k: v for k, v in SAMPLES.items() if os.path.exists(v)}
src = st.radio("Source", ["Upload a CSV"] + (["Use a sample"] if SAMPLES else []), horizontal=True, disabled=locked, key="src")
raw = None
if src == "Upload a CSV":
    up = st.file_uploader("CSV file", type=["csv"], disabled=locked, key="upl")
    if up is not None:
        raw = up.getvalue()
else:
    choice = st.selectbox("Sample", list(SAMPLES), disabled=locked, key="sample")
    with open(SAMPLES[choice], "rb") as f:
        raw = f.read()

if raw is None:
    st.info("Upload a CSV with one column for time and one or more number columns — or pick a sample.")
    st.stop()

df0 = pd.read_csv(io.BytesIO(raw))
ss.data_hash = sha(raw)
if ss.get("_last_hash") != ss.data_hash and not locked:   # new file: reset column-dependent choices
    for k in ("wt", "lt", "lid", "lv", "series", "defl", "cy_t", "cy_d", "as_d", "as_r", "log"):
        ss.pop(k, None)
    ss["_last_hash"] = ss.data_hash
fmt = st.radio("Layout", ["Wide (one column per series)", "Long (time, series name, value)"], horizontal=True,
               disabled=locked, key="fmt")
if fmt.startswith("Long"):
    c1, c2, c3 = st.columns(3)
    tcol = c1.selectbox("Time column", df0.columns, disabled=locked, key="lt")
    idcol = c2.selectbox("Series name column", df0.columns, index=min(1, len(df0.columns) - 1), disabled=locked, key="lid")
    vcol = c3.selectbox("Value column", df0.columns, index=min(2, len(df0.columns) - 1), disabled=locked, key="lv")
    df0 = df0.pivot_table(index=tcol, columns=idcol, values=vcol).reset_index()
else:
    tcol = st.selectbox("Time column", df0.columns, disabled=locked, key="wt")

df = df0.copy()
try:
    if not np.issubdtype(df[tcol].dtype, np.number) or df[tcol].max() > 3000:
        df[tcol] = pd.to_datetime(df[tcol])
except Exception:
    pass
df = df.set_index(tcol).sort_index()
num = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
series = st.multiselect("Series to use", num, default=num if len(num) <= 50 else num[:50], disabled=locked, key="series")
deflator = st.selectbox("Take out inflation or scale — divide every series by (optional)", ["(none)"] + num,
                        disabled=locked, key="defl")
deflator = None if deflator == "(none)" else deflator
if not series:
    st.stop()
cfg = dict(series=series + ([deflator] if deflator and deflator not in series else []), deflator=deflator)
data = prepare(df, cfg)
can_log = bool((data > 0).all().all())
use_log = st.checkbox("Work in logs (percent changes)", value=can_log, disabled=locked or not can_log, key="log")
for w in E.data_warnings(data, list(data.columns)):
    st.warning(w)
st.caption(f"{len(data)} rows × {data.shape[1]} series · {data.index.min()} → {data.index.max()} · data hash `{ss.data_hash[:12]}`")
st.line_chart((data / data.iloc[0]).iloc[:, :12], height=220)

# ---- 2. Question
st.header("2 · The question")
st.caption("Answer in your own words. The question shapes everything — it's the keystone.")
intake = dict(
    what_the_data_is=st.text_area("What's the data? What does one number mean, who recorded it?", key="q1", disabled=locked),
    looking_for=st.text_area("What are you looking for?", key="q2", disabled=locked),
    why=st.text_area("Why — what made you suspect it?", key="q3", disabled=locked),
    counts_as_found=st.text_area("What would count as found? What would convince you it's not there?", key="q4", disabled=locked),
    known_events=st.text_area("Big known events in the window (wars, crashes, rule changes)?", key="q5", disabled=locked),
)
win_text = st.text_area("Flag event windows, one per line (e.g. 2008-07-01 to 2009-06-30, or 2020 to 2021)",
                        key="wins", disabled=locked, height=90)
mask_all, ss.event_windows = parse_windows(win_text, data.index)
if ss.event_windows:
    st.caption(f"{int(mask_all.sum())} rows flagged in {len(ss.event_windows)} window(s).")
intake["keystone"] = st.text_area("Restate the question in one or two sentences (the keystone)", key="ks", disabled=locked)

# ---- 3. Plan
st.header("3 · Plan and lock")
shape = st.radio("Shape of the question", ["One series", "A driver and a response", "Many series together"],
                 horizontal=False, disabled=locked, key="shape")
cols = list(data.columns)
specs = []
nf = st.select_slider("Fakes per test (more = slower, steadier)", [100, 200, 300, 500], value=200, disabled=locked, key="nf")
if shape == "One series":
    if st.checkbox(TEST_LABELS["cycle"], value=True, disabled=locked, key="t_cycle"):
        c1, c2 = st.columns(2)
        target = c1.selectbox("Series", cols, disabled=locked, key="cy_t")
        drivers = c2.multiselect("Take out these drivers first", [c for c in cols if c != target], disabled=locked, key="cy_d")
        c3, c4, c5 = st.columns(3)
        season = c3.number_input("Season length in rows (52 weekly, 12 monthly, 0 none)", 0, 400, 0, disabled=locked, key="cy_s")
        pmin = c4.number_input("Shortest cycle (rows)", 2.0, 10000.0, 2.0, disabled=locked, key="cy_a")
        pmax = c5.number_input("Longest cycle (rows, 0 = a third of the data)", 0.0, 100000.0, 0.0, disabled=locked, key="cy_b")
        specs.append(dict(type="cycle", params=dict(target=target, drivers=drivers, season=season, pmin=pmin, pmax=pmax)))
elif shape == "A driver and a response":
    if st.checkbox(TEST_LABELS["asymmetry"], value=True, disabled=locked, key="t_asym") and len(cols) >= 2:
        c1, c2 = st.columns(2)
        driver = c1.selectbox("Driver", cols, disabled=locked, key="as_d")
        response = c2.selectbox("Response", [c for c in cols if c != driver], disabled=locked, key="as_r")
        c3, c4, c5 = st.columns(3)
        direction = c3.selectbox("Direction you expect", ["up", "down", "either"],
                                 format_func={"up": "faster up", "down": "faster down", "either": "either way"}.get,
                                 disabled=locked, key="as_dir")
        lags = c4.number_input("Lags to model", 2, 60, 12, disabled=locked, key="as_l")
        early = c5.number_input("Early window (steps)", 1, 30, 4, disabled=locked, key="as_e")
        specs.append(dict(type="asymmetry", params=dict(driver=driver, response=response, direction=direction,
                                                        lags=lags, early=early)))
else:
    if len(cols) < 3:
        st.warning("Pick at least 3 series for these tests.")
    else:
        for t in ("lead", "drift", "bunching"):
            if st.checkbox(TEST_LABELS[t], value=(t == "lead"), disabled=locked, key=f"t_{t}"):
                specs.append(dict(type=t, params=dict(series=cols)))
for s in specs:
    s.update(n_fakes=nf, use_log=use_log, modifiers={})

c1, c2 = st.columns(2)
if not locked:
    ready = bool(specs) and bool(intake["keystone"].strip())
    if not intake["keystone"].strip():
        st.caption("Write the keystone before locking.")
    if c1.button("🔒 Lock the plan", type="primary", disabled=not ready, use_container_width=True):
        now = datetime.now(timezone.utc)
        plan = dict(intake=intake, event_windows=ss.event_windows, specs=specs, data_hash=ss.data_hash,
                    seed=int(now.timestamp()), locked_at=now.isoformat(timespec="seconds"),
                    data_config=dict(series=series, deflator=deflator, use_log=use_log, rows=len(data)))
        plan["plan_id"] = plan_hash(plan)
        ss.plan, ss.locked, ss.tree, ss.ran = plan, True, [], False
        st.rerun()
else:
    p = ss.plan
    st.success(f"Locked · plan `{p['plan_id']}` · {p['locked_at']} UTC · seed {p['seed']}. "
               "Nothing above changes now — any change starts a new plan.")
    c1.download_button("Download locked plan (.json)", json.dumps(p, indent=2, default=str),
                       file_name=f"closed_loop_plan_{p['plan_id']}.json", use_container_width=True)
    if c2.button("Unlock (start a new plan)", use_container_width=True):
        ss.locked, ss.plan, ss.tree, ss.ran = False, None, [], False
        st.rerun()

# ---- 4. Run
if ss.locked:
    st.header("4 · Run")
    p = ss.plan
    if not ss.ran:
        if st.button("▶️ Run the locked plan", type="primary", use_container_width=True):
            with st.spinner("Running against the fakes…"):
                for s in p["specs"]:
                    add_node(s, None, data, mask_all, p["seed"])
            ss.ran = True
            st.rerun()
    for n in ss.tree:
        show_result(n, data, mask_all, p["seed"])

    # ---- 5. Tree
    if ss.tree:
        st.header("5 · Results tree")
        render_tree()
        res = dict(plan=p, runs=ss.tree)
        c1, c2 = st.columns(2)
        c1.download_button("Results (.json)", json.dumps(res, indent=2, default=str),
                           file_name=f"closed_loop_results_{p['plan_id']}.json", use_container_width=True)
        c2.download_button("Report (.md)", report_md(), file_name=f"closed_loop_report_{p['plan_id']}.md",
                           use_container_width=True)
        st.caption("A result climbs from survivor → sturdy survivor → lead on its own runs here. "
                   "It only becomes a finding when someone else reproduces it — send them the locked plan file.")

with st.expander("How this works"):
    st.markdown(
        "- **Lock first.** The question, tests and pass bars are frozen with a timestamp and a fingerprint before anything runs.\n"
        "- **Beat the fakes.** Each test builds fake data that copies the real data's momentum and size but has no pattern. "
        "A result counts only if the real data beats 99% of them.\n"
        "- **Fake arm.** Each test also runs once on a pure fake. If the fake passes, the test is drawing its own shape.\n"
        "- **Detection limit.** On a miss, known effects are planted at growing sizes to find the smallest one the test catches. "
        "A miss then reads 'nothing bigger than X' — either *nothing there* or *below the floor*.\n"
        "- **Next steps.** Passes offer standard follow-ups as buttons. Your own leaps — new questions — start a new plan.")
