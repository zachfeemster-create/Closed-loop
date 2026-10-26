"""Modeling fingerprint test (Closed Loop Runs 009-010): run the locked test, or a variation, and log the result."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pycountry
import streamlit as st
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO = os.path.join(ROOT, "demo", "modeling_fingerprint")
SNAP = os.path.join(DEMO, "snapshot")
sys.path.insert(0, DEMO)
sys.path.insert(0, ROOT)
import fingerprint as F                      # noqa: E402
from closedloop import runlog as RL          # noqa: E402

st.set_page_config(page_title="Modeling fingerprint · Closed Loop", page_icon="🔁", layout="centered")

MEASURES = {
    "Infant deaths (Run 009)": dict(raw="raw_infant_deaths.csv", igme="igme_infant_deaths.csv", run="009",
                                    locked=dict(eA=0.054, pA=0.0017)),
    "Under-5 deaths (Run 010)": dict(raw="raw_under5_deaths.csv", igme="igme_under5_deaths.csv", run="010",
                                     locked=dict(eA=0.032, pA=0.0052)),
}
MANUAL = {"United Kingdom": "GBR", "Republic of Korea": "KOR", "Republic of Moldova": "MDA", "Russian Federation": "RUS",
          "USA": "USA", "United States of America": "USA", "Czech Republic": "CZE", "Czechia": "CZE", "Venezuela": "VEN",
          "Iran (Islamic Rep of)": "IRN", "TFYR Macedonia": "MKD", "North Macedonia": "MKD", "Hong Kong SAR": "HKG",
          "Kyrgyzstan": "KGZ", "Turkey": "TUR", "Türkiye": "TUR", "Netherlands": "NLD", "Brunei Darussalam": "BRN",
          "Saint Vincent and Grenadines": "VCT", "Bahamas": "BHS", "Moldova": "MDA", "Reunion": "REU", "Rodrigues": None,
          "Virgin Islands (USA)": "VIR", "Occupied Palestinian Territory": "PSE"}


def iso(name):
    if name in MANUAL:
        return MANUAL[name]
    try:
        return pycountry.countries.lookup(name).alpha_3
    except LookupError:
        try:
            return pycountry.countries.search_fuzzy(name)[0].alpha_3
        except LookupError:
            return None


@st.cache_data(show_spinner=False)
def load_pairs(measure: str, min_deaths: int):
    m = MEASURES[measure]
    raw = pd.read_csv(os.path.join(SNAP, m["raw"]), index_col="who_code")
    cc = raw.pop("name"); raw.columns = raw.columns.astype(int)
    ig = pd.read_csv(os.path.join(SNAP, m["igme"]), index_col="REF_AREA"); ig.columns = ig.columns.astype(int)
    out = []
    for code, r in raw.iterrows():
        if r.notna().sum() != 20 or code not in cc.index:
            continue
        i3 = iso(cc[code])
        if i3 is None or i3 not in ig.index or ig.loc[i3].notna().sum() != 20:
            continue
        if r.mean() < min_deaths:
            continue
        out.append((cc[code], i3, r.values.astype(float), ig.loc[i3].values.astype(float)))
    return out


@st.cache_resource(show_spinner=False)
def scorer(scorer_seed: int, state: int, leave_out: tuple):
    w = pd.read_csv(os.path.join(SNAP, "who_modeled_series.csv"))
    if leave_out:
        w = w[~w.iso.isin(set(leave_out))]
    X = [r.astype(float) for r in w.iloc[:, 2:].values]
    fk = F.Fakes(scorer_seed); vec = lambda f: f["theta"] + f["alpha"]; R = []; Fk = {k: [] for k in "ABCD"}
    for x in X:
        R.append(vec(F.feats(x)))
        for k in "ABCD":
            Fk[k].append(vec(F.feats(getattr(fk, k)(x))))
        fk.A(x); fk.B(x); fk.C(x)
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    Xtr = np.array(R + Fk["A"] + Fk["B"] + Fk["C"] + Fk["D"]); y = np.r_[np.ones(len(R)), np.zeros(4 * len(R))]
    sc = StandardScaler().fit(Xtr); clf = LogisticRegression(max_iter=3000, random_state=state).fit(sc.transform(Xtr), y)
    return (lambda series: clf.predict_proba(sc.transform(np.array([vec(F.feats(s)) for s in series])))[:, 1]), len(X)


def run_test(pairs, score, fake_seed: int, draws: int):
    Rw = [p[2] for p in pairs]; Sm = [p[3] for p in pairs]
    fk = F.Fakes(fake_seed)
    d = score(Sm) - score(Rw)
    gap = lambda f: score([f(x) for x in Sm]) - score([f(x) for x in Rw])
    dA = np.mean([gap(fk.A) for _ in range(draws)], axis=0); dB = np.mean([gap(fk.B) for _ in range(draws)], axis=0)
    eA, eB = d - dA, d - dB
    pA = stats.wilcoxon(eA, alternative="greater").pvalue; pB = stats.wilcoxon(eB, alternative="greater").pvalue
    pF = stats.wilcoxon(gap(fk.A) - dA).pvalue
    ok = pA < 0.01 and np.median(eA) > 0 and pB < 0.01 and np.median(eB) > 0 and pF > 0.05
    verdict = "PASS" if ok else ("VOID (frame check tripped)" if pF <= 0.05 else "FAIL")
    res = dict(n=len(pairs), gap=float(np.median(d)), gap_pos=int((d > 0).sum()), momA=float(np.median(dA)),
               eA=float(np.median(eA)), eA_pos=int((eA > 0).sum()), pA=float(pA),
               eB=float(np.median(eB)), eB_pos=int((eB > 0).sum()), pB=float(pB), pF=float(pF), verdict=verdict)
    per = pd.DataFrame(dict(country=[p[0] for p in pairs], gap=d, excess_beyond_momentum=eA)).sort_values("excess_beyond_momentum")
    return res, per


# ------------------------------------------------------------------ page
st.title("Modeling fingerprint")
st.markdown(
    "Official numbers are often **smoothed by an agency's model** before you see them. This test asks whether "
    "the Closed Loop's quarter-turn lenses can tell the **smoothed** version (UN IGME) from the **raw registered "
    "counts** (WHO Mortality Database) of the same deaths, country by country, **beyond plain momentum**. "
    "Plans were locked before each run: [paper](https://doi.org/10.5281/zenodo.23638245) · "
    "[code](https://github.com/zachfeemster-create/Closed-loop/tree/main/demo/modeling_fingerprint).")

measure = st.radio("Measure", list(MEASURES), horizontal=True)
mode = st.radio("What to run", ["Locked test (exact rerun)", "Variation (change a setting)"], horizontal=True)
locked = mode.startswith("Locked")

if locked:
    fake_seed, scorer_seed, state, draws, leave, min_deaths = 69, 66, 67, 50, False, 0
    st.caption("Uses the exact settings of the locked run. A match confirms the result reproduces on this machine.")
else:
    st.caption("Variations are exploratory: logged as peripherals, they never change a locked verdict.")
    c1, c2, c3 = st.columns(3)
    fake_seed = c1.number_input("Fake seed", 0, 10_000_000, 69, step=1)
    scorer_seed = c2.number_input("Scorer fake seed", 0, 10_000_000, 66, step=1)
    state = c3.number_input("Scorer random state", 0, 10_000_000, 67, step=1)
    c4, c5 = st.columns(2)
    draws = c4.slider("Fake draws per country", 10, 100, 50, step=10)
    min_deaths = c5.select_slider("Drop countries averaging fewer raw deaths per year than", [0, 10, 50, 100, 500, 1000], 0)
    leave = st.checkbox("Leave the test countries out of the scorer's training data", value=False,
                        help="Checks that the scorer isn't recognizing a country's 'house style' from the WHO training series.")

if st.button("Run the test", type="primary"):
    pairs = load_pairs(measure, int(min_deaths))
    if len(pairs) < 10:
        st.error(f"Only {len(pairs)} countries left after that filter, too few for the test.")
        st.stop()
    leave_out = tuple(sorted({p[1] for p in pairs})) if leave else ()
    with st.spinner("Training the scorer (first run takes about a minute; later runs reuse it) …"):
        score, n_train = scorer(int(scorer_seed), int(state), leave_out)
    with st.spinner(f"Scoring {len(pairs)} countries against {draws} fake draws each …"):
        res, per = run_test(pairs, score, int(fake_seed), int(draws))

    ok = res["verdict"] == "PASS"
    (st.success if ok else st.warning)(f"**{res['verdict']}**" + (" - the fingerprint sees modeling beyond momentum" if ok else ""))
    st.table(pd.DataFrame({
        "": ["Smoothed scores higher than raw", "Gap from momentum alone", "Excess beyond momentum (fake A)",
             "Excess beyond smoothing (fake B)", "Frame check (fake vs fake)"],
        "Median": [f"{res['gap']:.3f}", f"{res['momA']:.3f}", f"{res['eA']:.3f}", f"{res['eB']:.3f}", ""],
        "Countries": [f"{res['gap_pos']}/{res['n']}", "", f"{res['eA_pos']}/{res['n']}", f"{res['eB_pos']}/{res['n']}", ""],
        "p": ["", "", f"{res['pA']:.2g}", f"{res['pB']:.2g}", f"{res['pF']:.2f}"],
        "Needed": ["", "", "p < 0.01", "p < 0.01", "p > 0.05"],
    }).set_index(""))
    st.caption(f"Scorer trained on {n_train} WHO modeled series" + (" (test countries left out)" if leave else "") + ".")

    reproduced = None
    if locked:
        L = MEASURES[measure]["locked"]
        reproduced = abs(res["eA"] - L["eA"]) < 0.0005 and abs(res["pA"] - L["pA"]) < 0.00005
        if reproduced:
            st.info(f"✓ Reproduced: identical to locked Run {MEASURES[measure]['run']} (excess {L['eA']}, p {L['pA']}).")
        else:
            st.error(f"Didn't match locked Run {MEASURES[measure]['run']} (expected excess {L['eA']}, p {L['pA']}). "
                     "Please report this - it's exactly what reproduction is for.")

    with st.expander("Per-country results"):
        st.bar_chart(per.set_index("country")["excess_beyond_momentum"])
        st.dataframe(per.round(3), hide_index=True)

    settings = dict(measure=measure, mode="locked" if locked else "variation", fake_seed=int(fake_seed),
                    scorer_seed=int(scorer_seed), scorer_state=int(state), fake_draws=int(draws),
                    leave_test_countries_out=bool(leave), min_raw_deaths=int(min_deaths))
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rid = "FP-" + hashlib.sha256(json.dumps([settings, now], sort_keys=True).encode()).hexdigest()[:8].upper()
    record = dict(run_id=rid, logged_at=now, test=f"modeling fingerprint (Run {MEASURES[measure]['run']} family)",
                  settings=settings, result=res, reproduced_locked=reproduced,
                  versions=dict(numpy=np.__version__, pandas=pd.__version__, python=sys.version.split()[0]))
    cfg = RL.config(st.secrets)
    if cfg and RL.put_file(cfg, f"fingerprint_runs/{rid}.json", json.dumps(record, indent=2).encode(), f"Fingerprint run {rid}: {res['verdict']}"):
        st.caption(f"Run **{rid}** saved to the shared log ✓")
    else:
        st.caption(f"Run **{rid}** (shared log not connected; download the record below to share it).")
    st.download_button("Download this run's record (.json)", json.dumps(record, indent=2), file_name=f"{rid}.json")
