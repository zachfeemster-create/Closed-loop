"""Run a Test: every test from the run logs, as a menu. Pick data, pick a test, lock the plan, run. No AI."""
from __future__ import annotations

import hashlib
import io
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from closedloop import events as EV       # noqa: E402
from closedloop import lab as LB          # noqa: E402
from closedloop import pulse as PL        # noqa: E402
from closedloop import runlog as RL       # noqa: E402
from closedloop import sources as S       # noqa: E402

st.set_page_config(page_title="Run a Test · Closed Loop", page_icon="🧪", layout="centered")
ss = st.session_state
cfg = RL.config(st.secrets)

# ---------------------------------------------------------------- the menu
# kind: panel = many series over time; scores = one list; triplet = three values; dose = two columns
TESTS = {
    "real_vs_fakes": dict(
        label="Real vs fakes", kind="panel", star=True,
        what="Can a standard six-measure toolkit tell your series apart from matched fakes (noise, random walks, "
             "spiky walks, shuffled steps)? Passes only if it beats both hard ones: walks and shuffled steps.",
        record="World Bank: beat noise (~0.27), near chance against walks. Fewer than 10 series get cut into windows."),
    "leftovers": dict(
        label="Who moves with whom (leftovers)", kind="panel", star=False,
        what="Which pairs of series move together step by step (same step, or one leading the other by a step) "
             "beyond what independent look-alike fakes give? False-discovery corrected. Optionally take out a shared path first.",
        record="World Bank leftovers tied through the economy (inflation-trade, GDP-unemployment)."),
    "turnback": dict(
        label="Turn-back beyond momentum", kind="panel", star=True,
        what="Do the series reverse mid-curve (step against both level and bend) more often than their own "
             "momentum explains? Fakes copy each series' drift, step size and step memory.",
        record="Survivor: countries and baby names passed, US states and rivers didn't."),
    "orbit": dict(
        label="Orbit steadiness and spiral", kind="panel", star=True,
        what="Each series' remainder drawn as a point going around (level, step). Do the orbits turn more "
             "regularly than momentum-matched fakes? Does the orbit spiral out or in?",
        record="Survivor: steadiness passed on both World Bank sets; spiral passed on one of two."),
    "interlock": dict(
        label="Interlock and zipper offset", kind="panel", star=True,
        what="Do the orbits of different series lock together, and is the lock a quarter turn apart (the zipper)? "
             "Fakes are independent series run through the same steps.",
        record="Survivor: interlock passed on both World Bank sets; zipper offset missed (in step or opposite instead)."),
    "step_flip": dict(
        label="Step flip and parked receivers (the Y)", kind="panel", star=True,
        what="Each series' usual corner is its blue. When a series sits in its blue, do other series land one step "
             "later on the flip corners, and were they already parked there? Control: each series' own corner "
             "sequence slid in time.",
        record="Lead on countries and calm-year US states; missed on weather, baby names, rivers. Landing corners are rare by construction."),
    "vortex": dict(
        label="Vortex short stays", kind="panel", star=False,
        what="Are visits to the rare corners shorter than visits elsewhere, more than random walks would give?",
        record="Countries and states passed, names and rivers didn't; read a pass as 'turns back quickly'."),
    "pulse": dict(
        label="Pulse sweep (cycles and what you could have heard)", kind="panel", star=False,
        what="Reads one series for a cycle, then plants waves of known period and height into its fakes to show the "
             "smallest wave each lens can hear. Name a period in advance (seasons, orbits, elections) to make it sharper. "
             "A miss then reads 'nothing taller than X at period P'.",
        record="New (Oct 29): weekly gas 1990-2026 has a yearly cycle (100th pct); Nasdaq has no 4-year cycle (27th pct)."),
    "leaderboard": dict(
        label="Leaderboard gaps", kind="scores", star=False,
        what="A top list (high scores, rankings, records). Do the gaps between neighbor ranks bunch, or keep a "
             "rhythm, beyond what a smooth tail gives? Has to clear two kinds of fakes.",
        record="Donkey Kong top 50: bunching leaned high but stayed inside the fakes."),
    "three_point": dict(
        label="Three-point phase (Koide / Brannen)", kind="triplet", star=True,
        what="Three positive values. Solves the amplitude and phase of the three-point cycle, then counts how many "
             "random triplets land as close to the target amplitude AND to a simple phase.",
        record="Charged leptons: root two and 2/9, 0 of 500,000 fakes (a rediscovery). Quarks and neutrinos miss."),
    "panic_before_drops": dict(
        label="Panic before drops", kind="event", star=False,
        what="Finds every big market drop, then asks whether a signal (news panic, fear gauge) was already unusual "
             "in the months BEFORE, beyond what the same windows at random times give. Also shows during and after.",
        record="New (Oct 28): Nasdaq 1985-2026, 51 drops of 10%+. Newspaper uncertainty counts: nothing before, big "
               "spike after. VIX: higher before (98th pct), partly drops landing in already-shaky markets."),
    "dose_response": dict(
        label="Dose-response leftover", kind="dose", star=False,
        what="Fits a Hill curve to dose and response pairs and reads how much of the ceiling the curve never "
             "reaches, with a bootstrap range. Name suspected values and it says which the range rules out.",
        record="Dopamine run: leftover about 23% (range too wide to tell 23 from 17 or 19)."),
}

st.title("Run a Test")
st.markdown(
    "Every test from the run logs, rebuilt so anybody can run it. **Pick a test, bring data, lock the plan, run.** "
    "Each test builds matched fakes and only passes what beats them, under a rule fixed before the run. "
    "Runs in your browser session on the app's own math: free, no AI, no account.")

with st.sidebar:
    st.header("Settings")
    bar_pct = st.select_slider("Bar (how far past the fakes)", [95.0, 99.0, 99.9], 95.0)
    n_fakes = st.select_slider("Fakes", [100, 200, 300, 500, 1000], 200)
    seed = st.number_input("Seed", 0, 99_999_999, 20261028, step=1)
    st.caption("More fakes = steadier verdicts, slower runs. The bar is part of the locked plan.")

keys = list(TESTS)
pick = st.selectbox("Test", keys, format_func=lambda k: ("★ " if TESTS[k]["star"] else "") + TESTS[k]["label"])
T = TESTS[pick]
st.markdown(T["what"])
st.caption("Where it's stood in our runs: " + T["record"] + "  ·  ★ = held up as a survivor or better")
with st.expander("Looking for another test?"):
    st.markdown("- **Who answers whom** (two sides: ads, prices, teams) and **budget across fronts** live on the Game theory page.\n"
                "- **Raw vs modeled** (does a smoothed dataset show something the raw counts don't) lives on the Modeling fingerprint page.\n"
                "- **Hidden cycle, faster up than down, who moves first, shifting links, leftover turns bunching** are on the main page.\n"
                "- **Lens rotation** runs fixed lenses with Max in the back.")
    try:
        st.page_link("pages/4_Game_theory.py", label="Game theory", icon="♟️")
        st.page_link("pages/1_Modeling_fingerprint.py", label="Modeling fingerprint", icon="🔬")
        st.page_link("app.py", label="Main page", icon="🔁")
        st.page_link("pages/2_Lens_rotation.py", label="Lens rotation", icon="🔁")
    except Exception:
        pass


# ---------------------------------------------------------------- data
@st.cache_data(ttl=6 * 3600, show_spinner="Fetching from FRED…")
def _fred(ids):
    return S.fetch_fred(ids, rename=False)


@st.cache_data(ttl=24 * 3600, show_spinner="Fetching from the World Bank…")
def _wb(ind, a, b):
    return S.fetch_worldbank(ind, a, b)


WB_TEN = {"NY.GDP.PCAP.CD": "GDP per person", "SP.POP.TOTL": "Population", "SP.DYN.LE00.IN": "Life expectancy",
          "SP.DYN.TFRT.IN": "Fertility rate", "IT.NET.USER.ZS": "Internet users", "SP.URB.TOTL.IN.ZS": "Urban share",
          "FP.CPI.TOTL.ZG": "Inflation", "SL.UEM.TOTL.ZS": "Unemployment", "SP.DYN.IMRT.IN": "Infant mortality",
          "NE.TRD.GNFS.ZS": "Trade share"}
WB_LOG = {"NY.GDP.PCAP.CD", "SP.POP.TOTL"}


@st.cache_data(ttl=24 * 3600, show_spinner="Fetching from the World Bank…")
def _wb_many(codes, a, b, every):
    parts = []
    for c in codes:
        d = S.fetch_worldbank(c, a, b)
        d = d[(d["year"] - a) % every == 0].set_index("year")
        if c in WB_LOG:
            d = np.log(d.where(d > 0))
        d.columns = [f"{c}|{k}" for k in d.columns]
        parts.append(d)
    return pd.concat(parts, axis=1).dropna(axis=1).reset_index()


@st.cache_data(ttl=6 * 3600, show_spinner="Fetching…")
def _link(url):
    return S.fetch_link(url)


@st.cache_data(show_spinner="Reading the file…", max_entries=3)
def _read_csv(raw):
    return S.read_csv_bytes(raw)


@st.cache_data(show_spinner="Reading the file…", max_entries=3)
def _read_xlsx(raw, sheet):
    return S.read_excel(raw, sheet)


@st.cache_data(show_spinner=False)
def _sample(name):
    return S.load_sample(name)


def _demo_cycles(seed_=7):
    rng = np.random.default_rng(seed_)
    t = np.arange(80)
    out = {"row": t}
    for i in range(10):
        out[f"cycle_{i + 1}"] = np.sin(2 * np.pi * t / 16 + rng.uniform(0, 6.28)) * 2 + 0.03 * t + rng.normal(0, 0.6, 80)
    for i in range(4):
        out[f"walk_{i + 1}"] = np.cumsum(rng.normal(0, 1, 80))
    return pd.DataFrame(out)


data_desc = ""
payload = None      # what goes into the test
MAX_ROWS = 5000     # beyond this, long files keep every k-th row by default

if T["kind"] == "panel":
    st.subheader("1. Data")
    src = st.radio("Where from", ["Demo", "Upload", "Paste", "Google Sheet or link", "FRED", "World Bank"],
                   horizontal=True)
    df = None
    try:
        if src == "Demo":
            dpick = st.selectbox("Demo set", ["Planted patterns (weekly)", "Ten cycles and four walks"] + list(S.SAMPLES)[1:])
            if dpick.startswith("Planted"):
                df = S.demo_planted()
                st.caption(S.SAMPLE_NOTES[list(S.SAMPLES)[0]])
            elif dpick.startswith("Ten cycles"):
                df = _demo_cycles()
                st.caption("Ten series share a 16-row cycle at random offsets; four are plain walks.")
            else:
                df = _sample(dpick)
            data_desc = f"demo: {dpick}"
        elif src == "Upload":
            f = st.file_uploader("CSV or Excel: one column per series, time down the rows", type=None)
            if f is not None:
                raw = f.getvalue()
                if f.name.lower().endswith((".xlsx", ".xls")):
                    sheets = S.excel_sheets(raw)
                    sh = st.selectbox("Sheet", sheets) if len(sheets) > 1 else sheets[0]
                    df = _read_xlsx(raw, sh)
                else:
                    df = _read_csv(raw)
                data_desc = f"upload: {f.name}"
        elif src == "Paste":
            txt = st.text_area("Paste cells copied from a spreadsheet (header row first)", height=160)
            if txt.strip():
                df = S.read_pasted(txt)
                data_desc = "pasted cells"
        elif src == "Google Sheet or link":
            url = st.text_input("Google Sheets share link, or a link to a CSV / Excel / JSON file")
            if url.strip():
                df = _link(url.strip())
                data_desc = f"link: {url.strip()[:120]}"
        elif src == "FRED":
            preset = st.selectbox("Preset", ["(type your own)"] + list(S.FRED_PRESETS))
            ids = st.text_input("FRED series IDs, comma-separated",
                                S.FRED_PRESETS.get(preset, "UNRATE, INDPRO, CPIAUCSL, HOUST, PAYEMS"))
            if ids.strip():
                df = _fred(ids)
                data_desc = f"FRED: {ids}"
        else:
            many = st.toggle("Several indicators at once (the reference-run layout)", value=False)
            if many:
                codes = st.multiselect("Indicators", list(WB_TEN), default=list(WB_TEN), format_func=lambda c: WB_TEN[c])
                a, b = st.slider("Years", 1960, 2024, (2000, 2022))
                every = st.select_slider("Every", [1, 2, 5], 2, format_func=lambda k: f"{k} yr")
                if codes and st.toggle("Fetch", value=False, help="Ten pulls; takes a little while the first time."):
                    df = _wb_many(tuple(codes), a, b, every)
                    data_desc = f"World Bank {len(codes)} indicators {a}-{b} every {every} yr"
                    st.caption("Columns are named indicator|country, so 'Within groups' takes out each indicator's own path.")
            else:
                wpick = st.selectbox("Indicator", list(S.WB_PRESETS) + ["(type a code)"])
                code = st.text_input("Indicator code", "SP.DYN.IMRT.IN") if wpick.startswith("(") else S.WB_PRESETS[wpick]
                a, b = st.slider("Years", 1960, 2024, (2000, 2022))
                if code:
                    df = _wb(code, a, b)
                    data_desc = f"World Bank {code} {a}-{b}"
    except Exception as ex:
        st.error(str(ex))
        df = None

    if df is not None and len(df):
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        first = df.columns[0]
        if first in num_cols and str(first).lower() in ("year", "row", "t", "time", "date", "period", "week", "month"):
            num_cols.remove(first)          # a numeric time column is the clock, not a series
        if len(num_cols) > 60 and not st.toggle(f"Pick series by hand (otherwise all {len(num_cols)} go in)", value=False):
            cols = num_cols
        else:
            cols = st.multiselect("Series to include", num_cols, default=num_cols)
        c1, c2 = st.columns(2)
        use_log = c1.toggle("Use logs (for prices, counts, sizes)", value=False)
        n = len(df)
        r0, r1 = c2.slider("Rows", 0, n, (0, n)) if n > 2 else (0, n)
        sub = df.iloc[r0:r1][cols].apply(pd.to_numeric, errors="coerce").dropna()
        if use_log:
            bad = (sub <= 0).any()
            if bad.any():
                st.warning("Logs skip columns with zero or negative values: " + ", ".join(map(str, bad[bad].index[:8])))
                sub = sub.loc[:, ~bad]
            sub = np.log(sub)
        avg_note = ""
        if len(sub) > MAX_ROWS:
            k = int(np.ceil(len(sub) / MAX_ROWS))
            if st.toggle(f"Long file: keep every {k}th row so it runs on the free server", value=True,
                         help=f"Over {MAX_ROWS:,} rows the tests get slow on the free server. Keeping every {k}th row "
                              "(not averaging, which smooths the steps and fools the fakes) keeps the data honest. "
                              "Turn it off to run every row; big files may be slow."):
                sub = sub.iloc[::k]
                avg_note = f" · every {k}th row kept"
        st.caption(f"{sub.shape[1]} series × {sub.shape[0]} complete rows go in.")
        if sub.shape[1] and sub.shape[0]:
            view = ((sub - sub.mean()) / sub.std().replace(0, 1)).iloc[:, :12]
            st.line_chart(view.iloc[::max(1, len(view) // 800)], height=180)   # thin it so phones don't stall
        payload = sub
        data_desc += f" · {sub.shape[1]} series × {sub.shape[0]} rows" + (" · logs" if use_log else "") + avg_note

elif T["kind"] == "scores":
    st.subheader("1. Scores")
    txt = st.text_area("One score per line (or comma-separated), any order",
                       "1272700\n1260700\n1247700\n1244400\n1218000\n1206800\n1190400\n1179200\n1178800\n1170500\n"
                       "1160000\n1150000\n1145000\n1141800\n1130000", height=180)
    vals = pd.to_numeric(pd.Series(txt.replace(",", "\n").split()), errors="coerce").dropna().values
    st.caption(f"{len(vals)} scores read (example list is made up; paste your own).")
    payload = vals
    data_desc = f"{len(vals)} scores"

elif T["kind"] == "triplet":
    st.subheader("1. Three values")
    c = st.columns(3)
    v1 = c[0].number_input("First", value=0.51099895, format="%.8g")
    v2 = c[1].number_input("Second", value=105.6583755, format="%.10g")
    v3 = c[2].number_input("Third", value=1776.86, format="%.8g")
    st.caption("Defaults are the electron, muon and tau masses in MeV. Order matters only for which one is n = 0.")
    c = st.columns(2)
    tA = c[0].number_input("Target amplitude", value=float(np.sqrt(2)), format="%.8f")
    dmode = c[1].radio("Phase target", ["Any simple fraction (q ≤ 9)", "One phase I name"], horizontal=False)
    tD = st.number_input("Phase (radians)", value=2 / 9, format="%.8f") if dmode.startswith("One") else None
    c = st.columns(2)
    lo = c[0].number_input("Fakes from (smallest)", value=0.5, format="%.6g")
    hi = c[1].number_input("Fakes up to (largest)", value=200000.0, format="%.6g")
    payload = dict(values=[v1, v2, v3], target_A=tA, target_delta=tD, lo=lo, hi=hi)
    data_desc = f"triplet {v1:g}, {v2:g}, {v3:g}"

elif T["kind"] == "event":
    st.subheader("1. Market and signal (from FRED)")
    MKTS = {"Nasdaq Composite (since 1971)": "NASDAQCOM", "S&P 500 (last 10 years)": "SP500",
            "Dow Jones (last 10 years)": "DJIA"}
    SIGS = {"News: stock-market uncertainty (daily since 1985)": "WLEMUINDXD",
            "News: economic policy uncertainty (daily since 1985)": "USEPUINDXD",
            "VIX fear gauge (daily since 1990, from option prices)": "VIXCLS",
            "Consumer sentiment (monthly)": "UMCSENT"}
    c = st.columns(2)
    mk = c[0].selectbox("Market", list(MKTS) + ["Other FRED ID"])
    mk_id = c[0].text_input("Market FRED ID", "NASDAQCOM") if mk == "Other FRED ID" else MKTS[mk]
    sg = c[1].selectbox("Signal", list(SIGS) + ["Other FRED ID"])
    sg_id = c[1].text_input("Signal FRED ID", "WLEMUINDXD") if sg == "Other FRED ID" else SIGS[sg]
    c = st.columns(2)
    drop_pct = c[0].select_slider("Drop that counts", [5.0, 7.5, 10.0, 15.0, 20.0], 10.0, format_func=lambda v: f"{v:g}%")
    lead = c[1].select_slider("Lead-up window", [21, 42, 63, 126], 63,
                              format_func=lambda v: f"{v} trading days (~{round(v * 1.45)} calendar)")
    st.caption("Drops = the market falls that much within 60 trading days. The signal is read against its own previous "
               "year, so a slow climb over decades doesn't count as panic.")
    payload = dict(market=mk_id.strip().upper(), signal=sg_id.strip().upper(), drop_pct=drop_pct, lead=lead)
    data_desc = f"FRED {payload['market']} vs {payload['signal']}, drops {drop_pct:g}%, lead {lead} days"

else:
    st.subheader("1. Dose and response")
    txt = st.text_area("Two columns: dose, response (one pair per line)", "5, 12\n10, 40\n20, 54\n40, 72\n60, 74", height=160)
    try:
        d = pd.read_csv(io.StringIO(txt), header=None, sep=None, engine="python").apply(pd.to_numeric, errors="coerce").dropna()
        doses, resp = d.iloc[:, 0].values, d.iloc[:, 1].values
    except Exception:
        doses, resp = np.array([]), np.array([])
    c = st.columns(2)
    ceiling = c[0].number_input("Ceiling (100 for percent)", value=100.0)
    tg = c[1].text_input("Suspected leftover values (optional)", "23, 17, 19")
    targets = [float(x) for x in pd.to_numeric(pd.Series(tg.replace(",", " ").split()), errors="coerce").dropna()]
    st.caption(f"{len(doses)} pairs read. Defaults: oral methylphenidate dose vs dopamine-transporter occupancy (Volkow 1998).")
    payload = dict(doses=doses.tolist(), responses=resp.tolist(), ceiling=ceiling, targets=targets)
    data_desc = f"{len(doses)} dose/response pairs"

# ---------------------------------------------------------------- test-specific knobs
extra = {}
use_groups = False
if pick in ("leftovers", "orbit", "interlock", "step_flip"):
    opts = ["Across all series", "Within groups"] + (["Off"] if pick in ("orbit", "step_flip", "leftovers") else [])
    base_mode = st.radio("Shared baseline to take out first", opts, horizontal=True,
                         index=opts.index("Off") if pick == "leftovers" else 0,
                         help="Within groups = each group's own average path, where a group is the part of the series "
                              "name before '|' or ':' (like an indicator). That's how the World Bank runs were done.")
    if pick in ("orbit", "step_flip", "leftovers"):
        extra["remove_shared"] = base_mode != "Off"
    if pick == "leftovers" and base_mode != "Off":
        st.caption("Only for many series of the same kind (one indicator across countries). On mixed series, taking out "
                   "the shared path ties every leftover to the busiest one and fakes a crowd of links.")
    if base_mode == "Within groups":
        if isinstance(payload, pd.DataFrame) and LB.group_labels(payload.columns) is not None:
            use_groups = True
            extra["baseline"] = "within groups"
        else:
            st.warning("No groups in these series names (nothing before '|' or ':'), so it'll use all series.")
if pick == "step_flip":
    extra["funnel"] = st.toggle("Funnel only (ignore points near the middle)", value=True)
if pick == "pulse" and isinstance(payload, pd.DataFrame) and payload.shape[1]:
    c = st.columns(2)
    extra["column"] = c[0].selectbox("Series to read", list(map(str, payload.columns)))
    named = c[1].number_input("Named period in rows (0 = none, scan everything)", min_value=0.0, value=0.0, step=1.0,
                              help="Pick it from a reason before looking: 52 for yearly in weekly data, 12 in monthly, "
                                   "4 in quarterly. Naming it after a peek turns it into a scan you didn't pay for.")
    extra["named_period"] = float(named) if named >= 3 else None
    extra["shape"] = st.radio("Planted wave shape", ["steady", "burst"], horizontal=True,
                              help="steady runs the whole length; burst is a wave packet over about a fifth of it.")

# ---------------------------------------------------------------- lock
st.subheader("2. Lock the plan")


def _fingerprint(p):
    if isinstance(p, pd.DataFrame):
        return hashlib.sha256(pd.util.hash_pandas_object(p, index=False).values.tobytes()
                              + "|".join(map(str, p.columns)).encode()).hexdigest()[:16]
    return hashlib.sha256(json.dumps(p, default=str, sort_keys=True).encode()).hexdigest()[:16]


ready = payload is not None and (len(payload) > 0 if not isinstance(payload, dict) else True)
plan = dict(test=pick, test_label=T["label"], bar_pct=bar_pct, fakes=int(n_fakes), seed=int(seed), settings=extra,
            data=data_desc, data_fingerprint=_fingerprint(payload) if ready else None,
            rule="pass = real beats the fakes past the bar, as defined in the test; decided before running")
plan_id = hashlib.sha256(json.dumps(plan, sort_keys=True, default=str).encode()).hexdigest()
locked = ss.get("lab_lock") == plan_id
c1, c2 = st.columns([1, 2])
if c1.button("Lock", disabled=not ready or locked):
    ss["lab_lock"] = plan_id
    ss["lab_lock_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    st.rerun()
if locked:
    c2.success(f"Locked · plan {plan_id[:8].upper()} · {ss.get('lab_lock_at', '')}")
else:
    c2.caption("Lock first. Changing anything after locking unlocks it, so the rule can't move after you peek.")
with st.expander("The plan"):
    st.json(plan)

# ---------------------------------------------------------------- run
st.subheader("3. Run")


def _run():
    kw = dict(seed=int(seed), bar_pct=float(bar_pct))
    if T["kind"] == "panel":
        X = payload.values.astype(float)
        names = list(map(str, payload.columns))
        grp = LB.group_labels(names) if use_groups else None
        opt = {k: v for k, v in extra.items() if k in ("remove_shared", "funnel")}
        if pick == "pulse":
            return PL.pulse_test(payload[extra["column"]].values, named_period=extra.get("named_period"),
                                 shape=extra.get("shape", "steady"), bar_pct=float(bar_pct), seed=int(seed),
                                 name=extra["column"])
        if pick == "real_vs_fakes":
            return LB.real_vs_fakes(X, **kw)
        if pick == "leftovers":
            return LB.leftovers(X, names=names, n_slides=max(int(n_fakes), 300), groups=grp, **kw, **opt)
        if pick == "step_flip":
            return LB.step_flip(X, n_slides=int(n_fakes), groups=grp, **kw, **opt)
        if pick == "orbit":
            return LB.orbit(X, n_fakes=int(n_fakes), groups=grp, **kw, **opt)
        if pick == "interlock":
            return LB.interlock(X, n_fakes=int(n_fakes), groups=grp, **kw)
        return LB.PANEL_TESTS[pick](X, n_fakes=int(n_fakes), **kw)
    if pick == "leaderboard":
        return LB.leaderboard(payload, n_fakes=max(int(n_fakes) * 50, 10000), **kw)
    if pick == "three_point":
        return LB.three_point(**payload, n_fakes=500_000, **kw)
    if pick == "panic_before_drops":
        df = _fred(f"{payload['market']}, {payload['signal']}")
        return EV.lead_up(df[payload["market"]].values, df[payload["signal"]].values, dates=df["date"].values,
                          names=(payload["market"], payload["signal"]), drop_pct=payload["drop_pct"],
                          before=payload["lead"], n_fakes=max(int(n_fakes) * 5, 1000), **kw)
    return LB.dose_response(payload["doses"], payload["responses"], ceiling=payload["ceiling"],
                            targets=payload["targets"], n_boot=max(int(n_fakes) * 5, 1000), **kw)


if st.button("Run", type="primary", disabled=not locked):
    with st.spinner("Building fakes and scoring…"):
        try:
            res = _run()
            now = datetime.now(timezone.utc).isoformat(timespec="seconds")
            rid = "RT-" + hashlib.sha256((plan_id + now).encode()).hexdigest()[:8].upper()
            ss["lab_res"] = dict(run_id=rid, ran_at=now, plan_id=plan_id[:8].upper(), plan=plan, result=res)
        except Exception as ex:
            st.error(f"Couldn't run: {ex}")

out = ss.get("lab_res")
if out and out["plan"]["test"] == pick:
    res = out["result"]
    (st.success if res.get("passed") else st.error)(f"**{res.get('verdict', '')}**")
    st.caption(res.get("question", ""))
    tbl = pd.DataFrame(res["rows"])
    keep = [c for c in ["stat", "real", "fakes_mid", "bar", "percentile", "z", "p", "q", "passed", "note"] if c in tbl]
    tbl = tbl[keep]
    if "bar" in tbl:
        tbl["bar"] = tbl["bar"].map(lambda b: f"{b[0]:.4g} to {b[1]:.4g}" if isinstance(b, (list, tuple))
                                    else ("" if b is None or (isinstance(b, float) and np.isnan(b)) else f"{b:.4g}"))
    st.dataframe(tbl, hide_index=True, width="stretch")
    for n_ in res.get("notes", []):
        st.markdown(f"- {n_}")
    if res.get("floor_table"):
        st.markdown("**Smallest wave each lens heard 80% of the time** (in typical steps; columns are periods in rows)")
        st.dataframe(pd.DataFrame(res["floor_table"]), hide_index=True, width="stretch")
    if res.get("events"):
        with st.expander(f"The drops ({len(res['events'])} shown) and the signal before and during each"):
            st.dataframe(pd.DataFrame(res["events"]), hide_index=True, width="stretch")
            st.caption("before_z / during_z: how far the signal sat above (+) or below (−) its own previous year, "
                       "in standard deviations.")
    st.caption("A pass is a pattern that beat these fakes, not a finding. The ladder: pattern → survivor → sturdy "
               "survivor (holds under another measure) → lead (holds on fresh data) → finding (someone else reproduces it).")
    with st.expander("Details"):
        st.json(res.get("details", {}))

    rid = out["run_id"]
    record = json.dumps(out, indent=2, default=str)
    if cfg and not ss.get(f"lab_saved_{rid}") and RL.put_file(cfg, f"test_runs/{rid}.json", record.encode(),
                                                              f"Run a Test {rid}: {T['label']}"):
        ss[f"lab_saved_{rid}"] = True
    st.caption(f"Run **{rid}** · plan {out['plan_id']}" + (" · saved to the shared log ✓" if ss.get(f"lab_saved_{rid}") else ""))
    st.download_button("Download this run (.json)", record, file_name=f"{rid}.json")
