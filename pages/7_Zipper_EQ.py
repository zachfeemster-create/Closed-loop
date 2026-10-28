"""Zipper EQ: shape the EQ by hand (up/down, left/right, wider/narrower, bell or shelf), slide it through time,
and catch the moments two series zip together, with a snapshot of the EQ at each zip."""
from __future__ import annotations

import itertools
import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from closedloop import eqcurve as EQ      # noqa: E402
from closedloop import rotation as R      # noqa: E402
from closedloop import sources as S       # noqa: E402
from closedloop import zipper as Z        # noqa: E402
import importlib, inspect      # noqa: E401,E402
if "slide" not in inspect.signature(EQ.eq_curve).parameters:     # the server may still hold the old drag code in memory
    EQ = importlib.reload(EQ)

st.set_page_config(page_title="Zipper EQ · Closed Loop", page_icon="🎚️", layout="centered")
ss = st.session_state
st.title("Zipper EQ")
st.markdown(
    "Shape the EQ like Parametric EQ 2: drag a sigil **up or down** for how much that band counts, **left or right** "
    "to move it to another cycle length, **twist** it (or the ¼ turn buttons) to make it wider or narrower, and switch "
    "it between a **bell** and a **shelf**. The finder only listens to the parts you **boost**. It slides a window "
    "through time, catches the moments two series **zip** together in that part of the EQ, checks each one against "
    "fakes, and saves a **snapshot** of what the EQ looked like at the zip.")

# ---------------------------------------------------------------- data
src = st.radio("Data", ["Demo (two zipped series + noise)", "World Bank", "Upload a CSV"], horizontal=True)
series: dict[str, np.ndarray] = {}
labels = None
if src.startswith("Demo"):
    rng = np.random.default_rng(12)
    n = 63
    t = np.arange(n)
    plant = np.where((t >= 20) & (t < 36), np.sin(2 * np.pi * t / 4), 0)
    series["zipped A"] = rng.normal(size=n) + 2.0 * plant
    series["zipped B"] = rng.normal(size=n) + 2.0 * plant
    for i in range(3):
        series[f"noise {i + 1}"] = rng.normal(size=n)
    labels = list(range(1960, 1960 + n))
    st.caption("Made up, 63 'years'. **zipped A** and **zipped B** share a 4-year cycle only in years 1980–1995. "
               "The noise series share nothing. The default EQ (one bell boosted at a 4-row cycle) should find "
               "A–B and point at those years. (This example was picked to show a clean result: with 10 pairs and a 5% bar, "
               "a stray false alarm shows up in about one run in three, and that's the bar working as it should.)")
elif src == "World Bank":
    c1, c2 = st.columns([2, 1])
    ind = c1.text_input("Indicator code", "NY.GDP.MKTP.CD", help="GDP in current US$. Others: SP.DYN.LE00.IN (life expectancy), FP.CPI.TOTL.ZG (inflation).")
    years = c2.slider("Years", 1960, 2024, (1961, 2022))
    ctry = st.text_input("Countries (ISO codes, blank = all with full data)", "USA, CAN, AUS, BRA, MEX, ZAF, CHN, GBR, DEU, JPN")
    if st.button("Load World Bank data"):
        try:
            ss["zip_wb"] = S.fetch_worldbank(ind, years[0], years[1], ctry)
        except Exception as ex:
            st.error(str(ex))
    if "zip_wb" in ss:
        df = ss["zip_wb"]
        labels = df["year"].tolist()
        for c in df.columns[1:]:
            series[str(c)] = df[c].values.astype(float)
else:
    f = st.file_uploader("CSV: one column per series (a date column is fine, it's used for labels)", type=None)
    if f is not None:
        df = S.read_csv_bytes(f.getvalue())
        first = df.columns[0]
        if not pd.api.types.is_numeric_dtype(df[first]):
            labels = df[first].astype(str).tolist()
        for c in df.select_dtypes("number").columns:
            v = df[c].values.astype(float)
            if np.isfinite(v).all() and len(v) >= 24:
                series[str(c)] = v
        st.caption(f"{len(series)} usable series (24+ rows, no gaps).")

if series:
    how = st.radio("Look at", ["growth (change from row to row)", "the values as they are"], horizontal=True,
                   index=0 if src != "Demo (two zipped series + noise)" else 1)
    if how.startswith("growth"):
        new = {}
        for k, v in series.items():
            g = np.diff(np.log(v)) if (v > 0).all() else np.diff(v)
            new[k] = (g - g.mean()) / (g.std() + 1e-12)
        series = new
        labels = labels[1:] if labels else None
n = len(next(iter(series.values()))) if series else 0
labels = labels or list(range(n))

# ---------------------------------------------------------------- the EQ
st.subheader("Shape the EQ")
DEFAULT = [[0.0, R.EQ_WIDTH_OCT, float(i), 0] for i in range(len(R.EQ_BANDS))]
DEFAULT[8] = [6.0, 0.5, float(np.log2((2 * R.REF_NYQUIST / 4) / R.EQ_BANDS[0])), 0]    # one bell on a 4-row cycle
if "zip_eq" not in ss:
    ss["zip_eq"] = DEFAULT
ss["zip_eq"] = EQ.eq_curve(ss["zip_eq"], key="zip_eq_drag", lo=-6, hi=6, slide=True, shelves=True, unit="row")
if st.button("Reset to one bell on a 4-row cycle"):
    ss["zip_eq"] = DEFAULT
    st.rerun()
bl = Z.bands(ss["zip_eq"])
on = Z.boosted(bl)
if on:
    st.caption("Listening to: " + " · ".join(
        f"{Z.KINDS[b['kind']]} at a {Z.cycle_of(b['hz']):.1f}-row cycle (+{b['gain']:.1f} dB, {b['width']:.2f} oct)" for b in on)
        + "".join(f" · cutting with a {Z.KINDS[b['kind']]} at a {Z.cycle_of(b['hz']):.1f}-row cycle" for b in Z.carving(bl))
        + ". Cycle length = 44,100 ÷ the band's Hz, in rows (rows are years for yearly data). High/low passes and "
          "notches cut parts away; they don't add anything to listen to, so boost something too.")
else:
    st.warning("Boost at least one band (drag a sigil above the line). The finder only listens to boosted parts.")

# ---------------------------------------------------------------- run
if series and on and n >= 24:
    names = list(series)
    pick = st.multiselect("Series to check against each other", names, default=names[:12])
    W_auto = Z.auto_window(bl, n)
    W = st.slider("Window (rows)", 12, max(13, n - 5), min(W_auto, max(13, n - 5)),
                  help=f"Auto: {W_auto} rows (4 cycles of the slowest boosted bell, 5x a low shelf's cycle).")
    if len(pick) >= 2 and st.button("Find zippers", type="primary"):
        res = []
        bar = st.progress(0.0)
        pairs = list(itertools.combinations(pick, 2))
        for j, (a, b) in enumerate(pairs):
            out = Z.pair(series[a], series[b], bl, W=W)
            res.append((a, b, out))
            bar.progress((j + 1) / len(pairs))
        bar.empty()
        ss["zip_res"] = dict(res=res, W=W, labels=labels, eq=ss["zip_eq"])

if "zip_res" in ss and series:
    Rz = ss["zip_res"]; res, W, lab = Rz["res"], Rz["W"], Rz["labels"]
    sig = [(a, b, o) for a, b, o in res if o["p"] < 0.05]
    st.subheader("Zippers")
    st.markdown(f"**{len(sig)} of {len(res)} pairs** zip beyond their fakes. About **{0.05 * len(res):.1f}** would by luck. "
                f"Fakes: {res[0][2]['fake_kind']} ({'shelf boosted' if res[0][2]['fake_kind'] == 'phase-scrambled' else 'bells only'}).")
    tab = pd.DataFrame([dict(pair=f"{a} × {b}", p=round(o["p"], 3), zipped_windows=len(o["zip_idx"]),
                             strongest=f"{lab[int(o['r'].argmax())]}–{lab[min(len(lab) - 1, int(o['r'].argmax()) + W - 1)]}",
                             lock=round(float(o["r"].max()), 2)) for a, b, o in res]).sort_values("p")
    st.dataframe(tab, hide_index=True, width='stretch')
    if sig:
        when = Counter()
        for _, _, o in sig:
            for i in o["zip_idx"]:
                when[lab[min(len(lab) - 1, int(i) + W // 2)]] += 1
        st.markdown("**When the zippers sit** (middle of each zipped window, all zipped pairs):")
        st.bar_chart(pd.Series(when).sort_index())
    choice = st.selectbox("Look at one pair", [f"{a} × {b}" for a, b, _ in sorted(res, key=lambda t: t[2]["p"])])
    a, b, o = next(t for t in res if f"{t[0]} × {t[1]}" == choice)
    mid = [lab[min(len(lab) - 1, i + W // 2)] for i in range(len(o["r"]))]
    st.line_chart(pd.DataFrame({"lock (this pair)": o["r"], "fake bar (95% of fakes' best)": o["threshold"]}, index=mid))
    st.caption(f"p = {o['p']:.3f}: the share of {o['n_fakes']} fakes whose best moment beat this pair's best moment.")
    i0 = int(o["r"].argmax())
    snap = Z.snapshot(series[a], series[b], i0, W)
    allsnap = [Z.snapshot(series[a], series[b], i, W) for i in range(len(o["r"]))]
    base = np.mean([[(s["a"][k] + s["b"][k]) / 2 for k in range(len(R.EQ_BANDS))] for s in allsnap], 0)
    band_lab = [f"{f // 1000}k ({Z.cycle_of(f):.1f})" if f >= 1000 else f"{f} ({Z.cycle_of(f):.0f})" for f in R.EQ_BANDS]
    st.markdown(f"**EQ snapshot at the strongest moment** ({lab[i0]}–{lab[min(len(lab) - 1, i0 + W - 1)]}): share of each "
                "series' movement in each band, next to that pair's usual shape. Band labels: Hz (cycle in rows).")
    st.bar_chart(pd.DataFrame({a: snap["a"], b: snap["b"], "usual (all windows)": base}, index=band_lab))
    run = dict(eq=Rz["eq"], window=W, pairs=[dict(a=a_, b=b_, p=o_["p"], zipped_windows=[lab[int(i)] for i in o_["zip_idx"]],
                                               snapshot=Z.snapshot(series[a_], series[b_], int(o_["r"].argmax()), W))
                                          for a_, b_, o_ in res])
    st.download_button("Download this run (EQ shape + every snapshot)", json.dumps(run, default=float, indent=1),
                       "zipper_run.json", "application/json")
    st.caption("Honest limits: the finder misses weak, short locks a lot of the time (on planted data it caught a 16-row "
               "4-row cycle about 1 time in 3), but when it fires it fires in the right place, and false alarms stay "
               "near 5%. Big shared shocks (2008, 2020) zip everything at once, so check results with those years cut.")
