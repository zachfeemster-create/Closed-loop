"""Rebuild the FEC outside-spending tables used by the Game theory page.

    python experiments/fec/build_fec.py [cache_dir]

Writes samples/fec_fronts_2016_2024.csv (one row per race per cycle) and
samples/fec_weekly_2016_2024.csv (national weekly D-side / R-side spend, long layout).

Source: FEC's processed committee-to-candidate file (pas2), independent expenditures only
(24E support, 24A oppose), general election, House + Senate, memo lines dropped, latest filing per
(committee, transaction), joined to the FEC candidate master for party, office and district.
Side: D-side = supports a Democrat or opposes a Republican; R-side the reverse. Dates are payment dates.

Why not FEC's independent_expenditure_YYYY.csv: it is missing nearly all Senate Leadership Fund
spending (2022: $3.2M listed vs $246M reported) and the party/candidate ID on most Congressional
Leadership Fund rows, which undercounts the R side by roughly half.

Race value columns:
  closeness        1 - |D - R| / (D + R), two-party general vote, FEC compiled results (2016-2022)
  fc_margin_close  minus the gap between the top two in FiveThirtyEight's projected vote share,
                   averaged Aug 1 to Election Day (House 2018-2022, Senate 2016-2022)
  fc_prob_close    1 - (gap between the top two win probabilities), same window
"""
from __future__ import annotations

import io
import os
import re
import sys
import urllib.request
import zipfile

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, ".fec_cache")
OUT = os.path.join(ROOT, "samples")
ELECTION = {2016: "2016-11-08", 2018: "2018-11-06", 2020: "2020-11-03", 2022: "2022-11-08", 2024: "2024-11-05"}
FEC = "https://www.fec.gov/files/bulk-downloads"
F38 = "https://raw.githubusercontent.com/fivethirtyeight/checking-our-work-data/master"
PAS = ("CMTE_ID AMNDT_IND RPT_TP TRANSACTION_PGI IMAGE_NUM TRANSACTION_TP ENTITY_TP NAME CITY STATE ZIP_CODE "
       "EMPLOYER OCCUPATION TRANSACTION_DT TRANSACTION_AMT OTHER_ID CAND_ID TRAN_ID FILE_NUM MEMO_CD MEMO_TEXT SUB_ID").split()
CN = ("CAND_ID CAND_NAME CAND_PTY_AFFILIATION CAND_ELECTION_YR CAND_OFFICE_ST CAND_OFFICE CAND_OFFICE_DISTRICT "
      "CAND_ICI CAND_STATUS CAND_PCC CAND_ST1 CAND_ST2 CAND_CITY CAND_ST CAND_ZIP").split()


def fetch(url: str, path: str) -> str:
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print("downloading", url, file=sys.stderr)
        req = urllib.request.Request(url, headers={"User-Agent": "ClosedLoop/1.0"})
        with urllib.request.urlopen(req, timeout=600) as r, open(path, "wb") as f:
            f.write(r.read())
    return path


def unzip_one(zpath: str, member_suffix: str) -> str:
    out = zpath[:-4] + "_" + member_suffix
    if not os.path.exists(out):
        with zipfile.ZipFile(zpath) as z:
            name = next(n for n in z.namelist() if n.endswith(member_suffix))
            with z.open(name) as src, open(out, "wb") as dst:
                dst.write(src.read())
    return out


def load_ie(y: int) -> pd.DataFrame:
    yy = str(y)[2:]
    pas = unzip_one(fetch(f"{FEC}/{y}/pas2{yy}.zip", f"{CACHE}/pas2{yy}.zip"), "itpas2.txt")
    cnf = unzip_one(fetch(f"{FEC}/{y}/cn{yy}.zip", f"{CACHE}/cn{yy}.zip"), "cn.txt")
    p = pd.read_csv(pas, sep="|", header=None, names=PAS, dtype=str, low_memory=False)
    p = p[p.TRANSACTION_TP.isin(["24E", "24A"]) & p.MEMO_CD.isna()]
    p = p[p.TRANSACTION_PGI.fillna("").str.startswith("G")].copy()
    p["fn"] = pd.to_numeric(p.FILE_NUM, errors="coerce")
    p = p.sort_values("fn").drop_duplicates(["CMTE_ID", "TRAN_ID"], keep="last")
    p["amt"] = pd.to_numeric(p.TRANSACTION_AMT, errors="coerce")
    p = p[p.amt > 0]
    p["date"] = pd.to_datetime(p.TRANSACTION_DT, format="%m%d%Y", errors="coerce")
    p = p[(p.date >= pd.Timestamp(f"{y}-01-01")) & (p.date <= pd.Timestamp(ELECTION[y]))]
    cn = pd.read_csv(cnf, sep="|", header=None, names=CN, dtype=str, low_memory=False)
    cn = cn.drop_duplicates("CAND_ID")[["CAND_ID", "CAND_PTY_AFFILIATION", "CAND_OFFICE", "CAND_OFFICE_ST",
                                         "CAND_OFFICE_DISTRICT"]]
    p = p.merge(cn, on="CAND_ID", how="left")
    p = p[p.CAND_OFFICE.isin(["H", "S"])]
    pty = p.CAND_PTY_AFFILIATION.fillna("").str.upper()
    party = np.where(pty.isin(["DEM", "DFL"]), "D", np.where(pty == "REP", "R", ""))
    sup = (p.TRANSACTION_TP == "24E").values
    p["side"] = np.where(((party == "D") & sup) | ((party == "R") & ~sup), "D-side",
                         np.where(((party == "R") & sup) | ((party == "D") & ~sup), "R-side", ""))
    p = p[p.side != ""].copy()
    dist = pd.to_numeric(p.CAND_OFFICE_DISTRICT, errors="coerce").fillna(0).astype(int)
    p["race"] = np.where(p.CAND_OFFICE == "S", p.CAND_OFFICE_ST + "-S",
                         p.CAND_OFFICE_ST + "-" + dist.map(lambda v: f"{v:02d}"))
    return p


def results_closeness(y: int) -> dict:
    if y >= 2024:
        return {}
    x = pd.ExcelFile(fetch(f"https://www.fec.gov/resources/cms-content/documents/federalelections{y}.xlsx",
                           f"{CACHE}/federalelections{y}.xlsx"))
    out = {}
    for s in x.sheet_names:
        if "Results" not in s or not ("House" in s or "Senate" in s):
            continue
        r = pd.read_excel(x, s)
        r.columns = [str(c).strip() for c in r.columns]
        dcol = "D" if "D" in r.columns else "DISTRICT"
        comb = [c for c in r.columns if c.startswith("COMBINED GE PARTY TOTALS")]
        r = r[r["STATE ABBREVIATION"].notna() & r["PARTY"].notna()].copy()
        r["v"] = pd.to_numeric(r["GENERAL VOTES"], errors="coerce")
        if comb:
            r["v"] = pd.to_numeric(r[comb[0]], errors="coerce").fillna(r["v"])
        pt = r["PARTY"].astype(str).str.upper().str.strip()
        r["pp"] = np.where(pt.str.match(r"^(D|DEM|DFL)\b"), "D", np.where(pt.str.match(r"^(R|REP)\b"), "R", ""))
        r = r[(r.pp != "") & r.v.notna()]
        senate = "Senate" in s
        for (st_, dd), g in r.groupby(["STATE ABBREVIATION", dcol]):
            if senate:
                key = f"{st_}-S"
            else:
                m = re.match(r"^(\d+)", str(dd).strip())
                if not m:
                    continue
                key = f"{st_}-{int(m.group(1)):02d}"
            D = g.loc[g.pp == "D", "v"].max()
            R = g.loc[g.pp == "R", "v"].max()
            D, R = (0 if pd.isna(D) else D), (0 if pd.isna(R) else R)
            c = 1 - abs(D - R) / (D + R) if (D + R) > 0 else 0.0
            out[key] = max(out.get(key, 0.0), c)      # two Senate races in one state: keep the closer one
    return out


def forecasts() -> pd.DataFrame:
    """FiveThirtyEight forecasts averaged Aug 1 to Election Day, one row per (cycle, race)."""
    parts = []
    for fname, office in (("us_house_elections.csv", "H"), ("us_senate_elections.csv", "S")):
        d = pd.read_csv(fetch(f"{F38}/{fname}", f"{CACHE}/{fname}"), low_memory=False)
        d = d[d.year.isin([2016, 2018, 2020, 2022])].copy()
        d["forecast_date"] = pd.to_datetime(d.forecast_date)
        d = d[d.forecast_date >= pd.to_datetime(d.year.astype(str) + "-08-01")]
        pref = {"deluxe": 0, "classic": 1, "lite": 2}
        d["pr"] = d.forecast_type.map(pref).fillna(3)
        d = d[d.pr == d.year.map(d.groupby("year").pr.min())]
        if office == "S":
            d["race"] = d.state + "-S"
            sp = d.special.astype(object).where(d.special.notna(), "").astype(str)
            d["rk"] = d.race + "-" + sp
        else:
            d["race"] = d.state + "-" + pd.to_numeric(d.district, errors="coerce").fillna(0).astype(int).map(
                lambda v: f"{v:02d}")
            d["rk"] = d.race
        rows = []
        for (y, rk, _), g in d.groupby(["year", "rk", "forecast_date"]):
            v = g.projected_voteshare.sort_values(ascending=False).values
            pw = g.probwin.sort_values(ascending=False).values
            if len(v) < 2:
                v, pw = np.r_[v, 0], np.r_[pw, 0]
            rows.append((y, g.race.iat[0], rk, -(v[0] - v[1]) / 100, 1 - (pw[0] - pw[1])))
        r = pd.DataFrame(rows, columns=["cycle", "race", "rk", "fc_margin_close", "fc_prob_close"])
        r = r.groupby(["cycle", "race", "rk"])[["fc_margin_close", "fc_prob_close"]].mean().reset_index()
        parts.append(r.groupby(["cycle", "race"])[["fc_margin_close", "fc_prob_close"]].max().reset_index())
    return pd.concat(parts)


def main():
    races, weeks = [], []
    for y in ELECTION:
        d = load_ie(y)
        t = d.pivot_table(index="race", columns="side", values="amt", aggfunc="sum").fillna(0.0)
        t = t[t.sum(axis=1) >= 10_000]
        clo = results_closeness(y)
        for race, row in t.iterrows():
            races.append(dict(cycle=y, race=race, office="Senate" if race.endswith("-S") else "House",
                              d_side=round(float(row.get("D-side", 0.0))), r_side=round(float(row.get("R-side", 0.0))),
                              closeness=(round(clo[race], 4) if race in clo else np.nan)))
        w = d.groupby([pd.Grouper(key="date", freq="W-SUN"), "side"]).amt.sum().unstack().fillna(0.0)
        w = w.reindex(pd.date_range(f"{y}-01-01", ELECTION[y], freq="W-SUN")).fillna(0.0)
        for wk, row in w.iterrows():
            for s in ("D-side", "R-side"):
                weeks.append(dict(cycle=y, week=wk.date().isoformat(), side=s, spend=round(float(row.get(s, 0.0)))))
        print(y, "races", len(t), "D-side $M", round(t.get("D-side", pd.Series(0.0)).sum() / 1e6, 1),
              "R-side $M", round(t.get("R-side", pd.Series(0.0)).sum() / 1e6, 1), file=sys.stderr)
    fronts = pd.DataFrame(races).merge(forecasts(), on=["cycle", "race"], how="left")
    for c in ("fc_margin_close", "fc_prob_close"):
        fronts[c] = fronts[c].round(4)
    os.makedirs(OUT, exist_ok=True)
    fronts.to_csv(os.path.join(OUT, "fec_fronts_2016_2024.csv"), index=False)
    pd.DataFrame(weeks).to_csv(os.path.join(OUT, "fec_weekly_2016_2024.csv"), index=False)
    print("wrote", os.path.join(OUT, "fec_fronts_2016_2024.csv"), "and fec_weekly_2016_2024.csv", file=sys.stderr)


if __name__ == "__main__":
    main()
