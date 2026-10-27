"""Public feeds the nightly job pulls on its own. Each returns {series_name: pandas Series indexed by date}
(or a watcher dict) and never raises: a feed that fails is reported as skipped, the rest still run.

Feeds that need a free key read it from the environment (GitHub repo secrets):
    FEC_API_KEY  - api.open.fec.gov (get one at api.data.gov/signup)
"""
from __future__ import annotations

import io
import json
import os
import sys
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta, timezone

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from closedloop import sources as S      # noqa: E402

UA = {"User-Agent": "ClosedLoop-autorun/1.0 (github.com/zachfeemster-create/Closed-loop)"}


def _get(url, timeout=90):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
        return r.read()


# ---------------------------------------------------------------- FRED (turn-backs, bunching)

FRED_MONTHLY = ["UNRATE", "INDPRO", "CPIAUCSL", "PAYEMS", "HOUST"]
FRED_DAILY = ["DGS10", "T10Y2Y", "DCOILWTICO", "VIXCLS"]


def fred():
    out = {}
    for c in FRED_MONTHLY + FRED_DAILY:
        df = S.fetch_fred(c, rename=False)
        s = df.set_index("date")[c].astype(float)
        if c in FRED_DAILY:
            s = s[s.index >= pd.Timestamp(date.today() - timedelta(days=3 * 365))]
        else:
            s = s[s.index >= pd.Timestamp("1990-01-01")]
        out[f"FRED {c}"] = s
    return out


# ---------------------------------------------------------------- World Bank (drift, bunching)

WB_COUNTRIES = "USA,CHN,JPN,DEU,GBR,FRA,IND,BRA,CAN,KOR,MEX,AUS,ZAF,TUR,IDN"


def worldbank():
    w = S.fetch_worldbank("NY.GDP.PCAP.KD.ZG", 1971, date.today().year - 2, WB_COUNTRIES, True)
    w = w.set_index("year")
    return {f"WB growth {c}": w[c].astype(float) for c in w.columns}


# ---------------------------------------------------------------- WSPR (daily spots per band)

WSPR_BANDS = {1: "160m", 3: "80m", 5: "60m", 7: "40m", 10: "30m", 14: "20m", 18: "17m", 21: "15m", 24: "12m", 28: "10m"}


def wspr(days: int = 120):
    q = ("SELECT toDate(time) AS d, band, count() AS n FROM wspr.rx "
         f"WHERE time > now() - INTERVAL {int(days)} DAY AND band IN ({','.join(map(str, WSPR_BANDS))}) "
         "GROUP BY d, band ORDER BY d, band FORMAT CSVWithNames")
    df = pd.read_csv(io.BytesIO(_get("https://db1.wspr.live/?query=" + urllib.parse.quote(q))))
    df["d"] = pd.to_datetime(df["d"])
    today = pd.Timestamp(datetime.now(timezone.utc).date())
    df = df[df["d"] < today]                                   # today is still filling in
    p = df.pivot_table(index="d", columns="band", values="n").sort_index()
    return {f"WSPR {WSPR_BANDS[b]}": p[b].astype(float) for b in p.columns if b in WSPR_BANDS}


# ---------------------------------------------------------------- FEC outside spending (answer-back, duels)

def fec(days: int = 60, max_pages: int = 250):
    key = os.environ.get("FEC_API_KEY", "").strip()
    if not key:
        raise RuntimeError("no FEC_API_KEY secret set")
    since = (date.today() - timedelta(days=days)).isoformat()
    base = ("https://api.open.fec.gov/v1/schedules/schedule_e/?per_page=100&sort=-expenditure_date"
            f"&min_date={since}&api_key={urllib.parse.quote(key)}")
    rows, last, pages, partial = [], None, 0, False
    while True:
        url = base + ("" if last is None else "&" + urllib.parse.urlencode(last))
        d = json.loads(_get(url))
        res = d.get("results", [])
        for r in res:
            rows.append((r.get("expenditure_date") or r.get("dissemination_date"), r.get("expenditure_amount"),
                         r.get("support_oppose_indicator"), (r.get("candidate_party") or "")[:3].upper()))
        pages += 1
        li = d.get("pagination", {}).get("last_indexes")
        if not res or not li:
            break
        if pages >= max_pages:
            partial = True
            break
        last = li
    df = pd.DataFrame(rows, columns=["date", "amt", "so", "party"]).dropna(subset=["date", "amt"])
    df["date"] = pd.to_datetime(df["date"].str[:10], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df[df["party"].isin(["REP", "DEM"]) & df["so"].isin(["S", "O"])]
    idx = pd.date_range(since, date.today() - timedelta(days=1), freq="D")
    out = {}
    for party in ("REP", "DEM"):
        for so, word in (("S", "for"), ("O", "against")):
            s = df[(df.party == party) & (df.so == so)].groupby("date")["amt"].sum().reindex(idx, fill_value=0.0)
            out[f"FEC spending {word} {party}"] = s.astype(float)
    if partial:
        out["_note"] = f"FEC pull capped at {max_pages} pages; newest days are complete, oldest may be short"
    return out


# ---------------------------------------------------------------- watchers (no series, just 'did anything new land?')

def arxiv_juno(n: int = 15):
    q = "search_query=" + urllib.parse.quote('abs:JUNO AND abs:neutrino') + f"&sortBy=submittedDate&sortOrder=descending&max_results={n}"
    import xml.etree.ElementTree as ET
    txt = _get("https://export.arxiv.org/api/query?" + q).decode("utf-8", "replace")
    ns = {"a": "http://www.w3.org/2005/Atom"}
    items = []
    for e in ET.fromstring(txt).findall("a:entry", ns):
        items.append(dict(id=e.findtext("a:id", "", ns).split("/abs/")[-1],
                          title=" ".join(e.findtext("a:title", "", ns).split()),
                          published=e.findtext("a:published", "", ns)[:10]))
    return items


def who_infant_watch():
    """WHO infant mortality (modeled): reports the newest year and how many country rows exist."""
    d = json.loads(_get("https://ghoapi.azureedge.net/api/MDG_0000000001?$select=TimeDim&$orderby=TimeDim%20desc&$top=1"))
    newest = d["value"][0]["TimeDim"] if d.get("value") else None
    return dict(newest_year=newest)
