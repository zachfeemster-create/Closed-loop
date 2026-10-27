"""Game theory: who answers whom over time, and how two sides split a budget across fronts (Colonel Blotto)."""
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
from closedloop import games as GT        # noqa: E402
from closedloop import runlog as RL       # noqa: E402
from closedloop import sources as S       # noqa: E402

st.set_page_config(page_title="Game theory · Closed Loop", page_icon="♟️", layout="centered")
ss = st.session_state
cfg = RL.config(st.secrets)

st.title("Game theory")
st.markdown(
    "For **contests**: two sides choosing moves against each other. **Who answers whom** reads the strategy each "
    "side plays over time (match, escalate, back off, or ignore). **Budget across fronts** checks how two sides "
    "split their money against **Colonel Blotto equilibrium** (Roberson 2006). Every result has to beat fakes, "
    "same as the rest of the Closed Loop.")
st.caption("Only for real contests. Fed two series that aren't playing each other, the math will still fit "
           "something; the fakes and the fake arm are what catch that.")

with st.sidebar:
    st.header("Settings")
    n_fakes = st.select_slider("Fakes", [200, 300, 500, 1000], 500)
    bar_pct = st.select_slider("Bar (how far past the fakes)", [95.0, 99.0, 99.9], 99.0)
    seed = st.number_input("Seed", 0, 99_999_999, 20261027, step=1)


# ---------------------------------------------------------------- shared helpers

def _load(kind: str, key: str):
    """Returns a DataFrame from upload or paste, or None."""
    if kind == "Upload a CSV":
        f = st.file_uploader("CSV file", type=None, key=f"{key}_up")
        if f is None:
            return None
        if not f.name.lower().endswith((".csv", ".tsv", ".txt")):
            st.error("Pick a CSV, TSV or TXT file.")
            st.stop()
        return S.read_csv_bytes(f.getvalue())
    txt = st.text_area("Paste cells here (include the header row)", key=f"{key}_paste", height=150)
    return S.read_pasted(txt) if txt.strip() else None


def _stamp(res: dict, prefix: str, data_hash: str, settings: dict) -> dict:
    """One ID per run, so redraws don't re-save it."""
    if "run_id" not in res:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        res["logged_at"] = now
        res["data_hash"] = data_hash
        res["settings"] = settings
        res["run_id"] = prefix + hashlib.sha256(json.dumps([settings, data_hash, now], sort_keys=True,
                                                           default=str).encode()).hexdigest()[:8].upper()
    return res


def _save_and_download(res: dict, label: str):
    rid = res["run_id"]
    record = json.dumps(dict(run_id=rid, logged_at=res["logged_at"],
                             **{k: v for k, v in res.items() if k not in ("run_id", "logged_at")}),
                        indent=2, default=str)
    if not ss.get(f"saved_{rid}") and cfg and RL.put_file(cfg, f"game_runs/{rid}.json", record.encode(),
                                                          f"{label} {rid}"):
        ss[f"saved_{rid}"] = True
    st.caption(f"Run **{rid}**" + (" · saved to the shared log ✓" if ss.get(f"saved_{rid}") else ""))
    st.download_button("Download this run (.json)", record, file_name=f"{rid}.json", key=f"dl_{rid}")


def _rows_table(rows, extra=()):
    df = pd.DataFrame(rows)
    df["bar"] = df["bar"].map(lambda b: f"{b[0]:.3g} to {b[1]:.3g}")
    df["passed"] = df["passed"].map(lambda p: "PASS" if p else "")
    cols = ["stat", "real", *extra, "bar", "percentile", "passed"]
    st.dataframe(df[[c for c in cols if c in df.columns]], hide_index=True, width="stretch")


def _hash(df: pd.DataFrame) -> str:
    return hashlib.sha256(df.to_csv(index=False).encode()).hexdigest()


tab_resp, tab_blotto = st.tabs(["Who answers whom", "Budget across fronts (Blotto)"])

# ---------------------------------------------------------------- 1. response lens
with tab_resp:
    st.markdown("Two sides measured over the same time steps (weekly spending, prices, moves). The lens asks how "
                "each side's next change follows the other side's last change, with its own momentum held fixed. "
                "Fakes swap the side being answered for one with the **same rhythm and shocks** but no link.")
    src = st.radio("Data", ["Demo (made up)", "Upload a CSV", "Paste from a spreadsheet", "FRED series IDs"],
                   horizontal=True, key="r_src")
    df, A, B, names = None, None, None, ("A", "B")
    try:
        if src.startswith("Demo"):
            rng = np.random.default_rng(11)
            n = 160
            dA = rng.normal(0, 1, n)
            dB = rng.normal(0, 1, n)
            for t in range(1, n):
                dB[t] += 0.9 * max(dA[t - 1], 0) + 0.2 * dB[t - 1]       # B answers A's surges, not its pullbacks
            df = pd.DataFrame({"week": np.arange(n), "Side A": 100 * np.exp(np.cumsum(dA) * 0.03),
                               "Side B": 100 * np.exp(np.cumsum(dB) * 0.03)})
            st.caption("Made-up pair: Side B answers Side A's surges the next week, and ignores A's pullbacks. "
                       "Side A ignores B. The lens should find exactly that.")
            A, B, names = df["Side A"].values, df["Side B"].values, ("Side A", "Side B")
        elif src.startswith("FRED"):
            ids = st.text_input("Two FRED IDs, comma-separated", "GASREGW, DCOILWTICO", key="r_fred")
            if ids.strip():
                fd = S.fetch_fred(ids, rename=False)
                if fd.shape[1] < 3:
                    st.warning("Give two series IDs.")
                else:
                    df = fd
                    A, B, names = fd.iloc[:, 1].values, fd.iloc[:, 2].values, (fd.columns[1], fd.columns[2])
        else:
            raw = _load(src, "r")
            if raw is not None:
                lay = st.radio("Layout", ["Wide (time, side A, side B columns)", "Long (time, side name, value)"],
                               horizontal=True, key="r_lay")
                cols = list(raw.columns)
                if lay.startswith("Long"):
                    c1, c2, c3 = st.columns(3)
                    tcol = c1.selectbox("Time", cols, key="r_t")
                    scol = c2.selectbox("Side name", cols, index=min(1, len(cols) - 1), key="r_s")
                    vcol = c3.selectbox("Value", cols, index=min(2, len(cols) - 1), key="r_v")
                    how = st.radio("Combine repeats in a time step by", ["sum", "mean"], horizontal=True, key="r_how")
                    wide = raw.pivot_table(index=tcol, columns=scol, values=vcol, aggfunc=how).sort_index()
                    sides = [str(c) for c in wide.columns]
                    wide.columns = sides
                    if len(sides) < 2:
                        st.warning("Need at least two sides in the side-name column.")
                        st.stop()
                    c1, c2 = st.columns(2)
                    sa = c1.selectbox("Side A", sides, key="r_la")
                    sb = c2.selectbox("Side B", [s for s in sides if s != sa], key="r_lb")
                    pair = wide[[sa, sb]].fillna(0.0) if how == "sum" else wide[[sa, sb]].dropna()
                else:
                    c1, c2, c3 = st.columns(3)
                    tcol = c1.selectbox("Time", cols, key="r_wt")
                    num = [c for c in cols if c != tcol and pd.api.types.is_numeric_dtype(raw[c])]
                    if len(num) < 2:
                        st.warning("Need two number columns.")
                        st.stop()
                    sa = c2.selectbox("Side A", num, key="r_wa")
                    sb = c3.selectbox("Side B", [c for c in num if c != sa], key="r_wb")
                    if raw[tcol].duplicated().any():
                        st.warning(f"**{tcol}** repeats, so this looks like long layout. Switch Layout to Long.")
                        st.stop()
                    pair = raw.set_index(tcol).sort_index()[[sa, sb]].dropna()
                df = pair.reset_index()
                A, B, names = pair[sa].values.astype(float), pair[sb].values.astype(float), (str(sa), str(sb))
    except Exception as ex:
        st.error(f"Couldn't read that: {ex}")
        st.stop()

    if A is not None:
        c1, c2 = st.columns(2)
        lags = c1.number_input("Answer window (steps back)", 1, 8, 1, key="r_lags",
                               help="1 = answers the other side's last step only. Larger windows add up the answer "
                                    "over that many steps.")
        can_log = bool(np.all(A > 0) and np.all(B > 0))
        use_log = c2.checkbox("Work in percent changes", value=can_log, disabled=not can_log, key="r_log")
        st.caption(f"{len(A)} time steps · {names[0]} vs {names[1]}")
        if len(A) < 26 + lags:
            st.warning(f"Needs at least {26 + lags} time steps; this has {len(A)}.")
        elif st.button("Run the response lens", type="primary", key="r_go"):
            with st.spinner("Breaking the pairing against fakes…"):
                res = GT.response_test(A, B, names=names, lags=int(lags), n_fakes=int(n_fakes), seed=int(seed),
                                       use_log=use_log, bar_pct=float(bar_pct))
            ss["gt_resp"] = res
            ss["gt_resp_hash"] = _hash(pd.DataFrame({"A": A, "B": B}))

    res = ss.get("gt_resp")
    if res:
        res = _stamp(res, "GR-", ss.get("gt_resp_hash", ""),
                     dict(lens="response", n_fakes=int(n_fakes), bar_pct=float(bar_pct), seed=int(seed),
                          lags=res["details"]["lags"], logged=res["details"]["logged"]))
        st.subheader("What each side plays")
        for k, v in res["reads"].items():
            st.markdown(f"- **{k}:** {v}")
        _rows_table(res["rows"])
        st.caption(f"Each row must clear the stricter bar {res['details']['corrected_bar_pct']}th percentile, "
                   "because the pair gets five looks. Answer sizes are how much of the other side's move comes back.")
        fa = res["fake_arm"]
        (st.warning if fa["passed"] else st.caption)(
            "Fake arm passed: two unrelated sides with these rhythms fooled the lens." if fa["passed"] else
            f"Fake arm (two unrelated sides with these rhythms) stayed quiet: {fa['real']}")
        det = res.get("detection")
        if det:
            st.info(f"**Detection limit:** answers smaller than about {det['limit']} ({det['unit']}) would slip past "
                    f"at this length. Pile: **{det['pile']}**." if det["limit"] is not None else
                    "**Detection limit:** even the largest planted answer wasn't caught reliably at this length.")
        for note in res.get("notes", []):
            st.caption("• " + note)
        _save_and_download(res, "Response lens")

# ---------------------------------------------------------------- 2. Colonel Blotto
with tab_blotto:
    st.markdown("One row per **front** (a race, a market, a district) with each side's spend there. Include fronts a "
                "side skipped (spend 0). In equilibrium the stronger side spreads money evenly-at-random up to a cap "
                "and contests nearly every front; the weaker side gives up a share of fronts and competes on the rest; "
                "and neither side follows the other front by front.")
    bsrc = st.radio("Data", ["Demo (made up)", "Upload a CSV", "Paste from a spreadsheet"], horizontal=True,
                    key="b_src")
    fronts = None
    try:
        if bsrc.startswith("Demo"):
            rng = np.random.default_rng(5)
            n = 140
            close = rng.beta(2, 5, n)
            v = close ** 3
            dem = 6e7 * v / v.sum() * rng.lognormal(0, 0.5, n)
            rep = 3.5e7 * v / v.sum() * rng.lognormal(0, 0.5, n)
            rep[close < 0.2] = 0
            fronts = pd.DataFrame({"front": [f"Race {i + 1}" for i in range(n)], "Side D": dem.round(0),
                                   "Side R": rep.round(0), "closeness": close.round(3)})
            st.caption("Made-up fronts: both sides pour money into close races, and the smaller side skips safe "
                       "seats. Neither side reacts to the other front by front. The value check should say both "
                       "chase the prize, not each other.")
            a_col, b_col, val_col, grp_col = "Side D", "Side R", "closeness", None
        else:
            fronts = _load(bsrc, "b")
            if fronts is not None:
                cols = list(fronts.columns)
                num = [c for c in cols if pd.api.types.is_numeric_dtype(fronts[c])]
                if len(num) < 2:
                    st.warning("Need two number columns (each side's spend).")
                    st.stop()
                c1, c2 = st.columns(2)
                a_col = c1.selectbox("Side A spend", num, key="b_a")
                b_col = c2.selectbox("Side B spend", [c for c in num if c != a_col], key="b_b")
                c3, c4 = st.columns(2)
                val_col = c3.selectbox("Front value (optional, e.g. closeness)", ["(none)"] + num, key="b_v")
                grp_col = c4.selectbox("Group (optional, e.g. cycle)", ["(none)"] + cols, key="b_g")
                val_col = None if val_col == "(none)" else val_col
                grp_col = None if grp_col == "(none)" else grp_col
    except Exception as ex:
        st.error(f"Couldn't read that: {ex}")
        st.stop()

    if fronts is not None:
        work = fronts
        if grp_col:
            groups = sorted(fronts[grp_col].dropna().astype(str).unique())
            pick = st.selectbox("Which group", groups, key="b_pick")
            work = fronts[fronts[grp_col].astype(str) == pick]
            st.caption("Each group you run is another look; a pattern that holds across groups counts for much more "
                       "than one that shows up once.")
        work = work[[c for c in [a_col, b_col, val_col] if c]].dropna()
        st.caption(f"{len(work)} fronts · {a_col} total {work[a_col].sum():,.0f} · {b_col} total {work[b_col].sum():,.0f}")
        if st.button("Run the Blotto check", type="primary", key="b_go"):
            try:
                with st.spinner("Drawing equilibrium play to compare against…"):
                    bres = GT.blotto_test(work[a_col].values, work[b_col].values, names=(str(a_col), str(b_col)),
                                          value=(work[val_col].values if val_col else None),
                                          n_fakes=int(n_fakes), seed=int(seed), bar_pct=float(bar_pct))
                ss["gt_blotto"] = bres
                ss["gt_blotto_hash"] = _hash(work)
            except ValueError as ex:
                st.error(str(ex))

    bres = ss.get("gt_blotto")
    if bres:
        bres = _stamp(bres, "GB-", ss.get("gt_blotto_hash", ""),
                      dict(lens="blotto", n_fakes=int(n_fakes), bar_pct=float(bar_pct), seed=int(seed)))
        d = bres["details"]
        st.subheader("How each side departs from equilibrium")
        st.caption(f"Strong side **{bres['strong']}**, weak side **{bres['weak']}** · budget ratio "
                   f"{d['budget_ratio']:.2f} · equilibrium cap per front {d['equilibrium_cap_per_front']:,.0f}")
        for line in bres["reads"]:
            st.markdown(f"- {line}")
        _rows_table(bres["rows"], extra=("equilibrium",))
        st.caption(f"A PASS means real play is outside what equilibrium play produces, at the stricter bar "
                   f"{d['corrected_bar_pct']}th percentile (six looks). Fronts counted empty: {d['empty_means']}.")
        vc = bres.get("value_check")
        if vc:
            st.markdown("**Chasing each other, or just the prize?**")
            _rows_table([vc])
            st.caption("How closely each side's money follows the front's value (rank match): "
                       + " · ".join(f"{k} {v:+.2f}" for k, v in vc["value_track"].items()))
        st.caption("Fake arm (one equilibrium draw run through the same check) "
                   + ("stood out." if bres["fake_arm"]["passed"] else "stayed quiet."))
        for note in bres.get("notes", []):
            st.caption("• " + note)
        _save_and_download(bres, "Blotto check")

with st.expander("How this works"):
    st.markdown(
        "- **Who answers whom.** Each side's next change is fit on the other side's last changes, holding its own last "
        "change fixed. The answer size says the strategy: about 1 is match move for move (tit-for-tat), above 1.25 "
        "escalates, between 0 and 0.75 answers partly, below 0 backs off. Rises and drops are also split, to catch a "
        "side that only answers surges (a ratchet).\n"
        "- **Fakes for answering.** The side being answered is swapped for a fake with its own rhythm, driven by its own "
        "shocks reshuffled, so bursts and fat tails survive but any link to the other side is gone.\n"
        "- **Colonel Blotto.** Two sides, fixed budgets, many fronts; the bigger spend on a front takes it. With equal-"
        "value fronts the equilibrium is known (Roberson 2006). Fakes are drawn from that equilibrium with the same "
        "budgets, so a PASS says how a side plays differently from equilibrium.\n"
        "- **Prize or each other.** With a front value given, each side's trend against value is stripped out, and the "
        "check asks whether what's left still lines up. If it does, they're reacting to each other, not just to the "
        "prize.")
