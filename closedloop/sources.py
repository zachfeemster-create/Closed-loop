"""Six ways to get data into the Closed Loop — no CSV needed.

1. Upload a file (CSV or Excel)   2. Google Sheets link   3. Paste from a spreadsheet
4. Link to a data file            5. Public data (FRED, World Bank)   6. Samples and a planted demo

Every function returns a pandas DataFrame whose first column is time (date or year).
"""
from __future__ import annotations

import io
import json
import os
import re
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

UA = {"User-Agent": "ClosedLoop/1.0 (+https://github.com/zachfeemster-create/Closed-loop)"}


def _get(url: str, timeout: int = 45) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


# ---------------------------------------------------------------- 1. files

def read_csv_bytes(raw: bytes) -> pd.DataFrame:
    """CSV, TSV or semicolon files — the separator is detected."""
    text = raw.decode("utf-8-sig", errors="replace")
    return read_pasted(text)


def excel_sheets(raw: bytes) -> list[str]:
    return list(pd.ExcelFile(io.BytesIO(raw)).sheet_names)


def read_excel(raw: bytes, sheet=0) -> pd.DataFrame:
    df = pd.read_excel(io.BytesIO(raw), sheet_name=sheet)
    return _tidy(df)


# ---------------------------------------------------------------- 2. Google Sheets

def google_csv_url(url: str) -> str:
    """Turn any Google Sheets share/edit link into its CSV download link."""
    m = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", url)
    if not m:
        raise ValueError("That doesn't look like a Google Sheets link.")
    gid = re.search(r"[#&?]gid=(\d+)", url)
    out = f"https://docs.google.com/spreadsheets/d/{m.group(1)}/export?format=csv"
    return out + (f"&gid={gid.group(1)}" if gid else "")


def fetch_google_sheet(url: str) -> pd.DataFrame:
    csv_url = google_csv_url(url)
    try:
        raw = _get(csv_url)
    except Exception as e:
        raise ValueError("Couldn't open the sheet. In Google Sheets, tap Share → General access → "
                         "'Anyone with the link' (Viewer), then paste the link again.") from e
    if raw[:15].lower().startswith(b"<!doctype html") or b"<html" in raw[:200].lower():
        raise ValueError("Google asked for a sign-in. Set the sheet to 'Anyone with the link can view' and try again.")
    return read_csv_bytes(raw)


# ---------------------------------------------------------------- 3. paste

def read_pasted(text: str) -> pd.DataFrame:
    """Cells copied from Excel, Google Sheets, Numbers or a web table (tabs, commas or semicolons)."""
    text = text.strip("\n")
    if not text.strip():
        raise ValueError("Nothing pasted yet.")
    first = text.splitlines()[0]
    sep = "\t" if "\t" in first else (";" if first.count(";") > first.count(",") else ",")
    euro = sep == ";"                      # semicolon files usually use a decimal comma (1,5)
    df = pd.read_csv(io.StringIO(text), sep=sep, engine="python", dtype=str if euro else None)
    return _tidy(df, decimal_comma=euro)


# ---------------------------------------------------------------- 4. link

def fetch_link(url: str) -> pd.DataFrame:
    url = url.strip()
    if "docs.google.com/spreadsheets" in url:
        return fetch_google_sheet(url)
    if "github.com" in url and "/blob/" in url:              # a GitHub page → the raw file
        url = url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
    raw = _get(url)
    path = urllib.parse.urlparse(url).path.lower()
    if path.endswith((".xlsx", ".xls")) or raw[:2] == b"PK":
        return read_excel(raw)
    if path.endswith(".json"):
        return _tidy(pd.json_normalize(json.loads(raw)))
    return read_csv_bytes(raw)


# ---------------------------------------------------------------- 5. public data

FRED_PRESETS = {
    "US gas, crude oil and CPI (weekly)": "GASREGW, WCOILWTICO, CPIAUCSL",
    "Regional gas prices (weekly)": "GASREGECW, GASREGMWW, GASREGGCW, GASREGRMW, GASREGWCW",
    "Unemployment, inflation, interest rate (monthly)": "UNRATE, CPIAUCSL, FEDFUNDS",
    "Housing starts, mortgage rate, home prices (monthly)": "HOUST, MORTGAGE30US, CSUSHPINSA",
    "Retail sales, consumer sentiment (monthly)": "RSAFS, UMCSENT",
}
FRED_NAMES = {"GASREGW": "gas_retail", "WCOILWTICO": "wti_crude", "CPIAUCSL": "cpi",
              "GASREGECW": "east_coast", "GASREGMWW": "midwest", "GASREGGCW": "gulf_coast",
              "GASREGRMW": "rocky_mountain", "GASREGWCW": "west_coast"}


def fetch_fred(ids: str, rename: bool = True) -> pd.DataFrame:
    """FRED series by ID (comma-separated), no account needed. Mixed frequencies are lined up on
    the first series' dates, carrying slower series (like monthly CPI) forward."""
    codes = [c.strip().upper() for c in re.split(r"[,\s]+", ids) if c.strip()]
    if not codes:
        raise ValueError("Type at least one FRED series ID, e.g. UNRATE.")
    frames = []
    for c in codes:
        try:
            raw = _get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={urllib.parse.quote(c)}")
        except Exception as e:
            raise ValueError(f"FRED didn't return series '{c}'. Check the ID on fred.stlouisfed.org.") from e
        s = pd.read_csv(io.BytesIO(raw), na_values=".")
        s.columns = ["date", c]
        s["date"] = pd.to_datetime(s["date"])
        frames.append(s.set_index("date")[c].dropna())
    base = frames[0].index
    df = pd.concat(frames, axis=1).sort_index().ffill(limit=62).reindex(base)
    df = df.dropna(how="any")
    if rename:
        df = df.rename(columns={k: v for k, v in FRED_NAMES.items() if k in df.columns})
    return df.rename_axis("date").reset_index()


WB_PRESETS = {
    "GDP (current US$)": "NY.GDP.MKTP.CD",
    "Population": "SP.POP.TOTL",
    "Life expectancy": "SP.DYN.LE00.IN",
    "CO2 emissions per person": "EN.GHG.CO2.PC.CE.AR5",
    "Urban population (%)": "SP.URB.TOTL.IN.ZS",
    "Exports (% of GDP)": "NE.EXP.GNFS.ZS",
}


def _wb_countries() -> set:
    raw = _get("https://api.worldbank.org/v2/country?format=json&per_page=400")
    return {c["id"] for c in json.loads(raw)[1] if c["region"]["id"] != "NA"}


def fetch_worldbank(indicator: str, start: int = 1990, end: int = 2022, countries: str = "",
                    complete_only: bool = True) -> pd.DataFrame:
    """One World Bank indicator for many countries (group averages like 'World' or 'OECD' removed)."""
    ind = indicator.strip()
    raw = _get(f"https://api.worldbank.org/v2/country/all/indicator/{urllib.parse.quote(ind)}"
               f"?format=json&per_page=20000&date={int(start)}:{int(end)}", timeout=90)
    js = json.loads(raw)
    if not isinstance(js, list) or len(js) < 2 or not js[1]:
        raise ValueError(f"The World Bank returned nothing for '{ind}'. Check the indicator code.")
    real = _wb_countries()
    rows = [(r["countryiso3code"], r["country"]["value"], int(r["date"]), r["value"]) for r in js[1]
            if r["countryiso3code"] in real and r["value"] is not None]
    df = pd.DataFrame(rows, columns=["code", "name", "year", "value"])
    if countries.strip():
        want = {c.strip().upper() for c in re.split(r"[,\s]+", countries) if c.strip()}
        df = df[df.code.isin(want) | df.name.str.upper().isin(want)]
    w = df.pivot_table(index="year", columns="name", values="value").sort_index()
    if complete_only:
        w = w.loc[:, w.notna().all()]
    if w.shape[1] == 0:
        raise ValueError("No country has every year in that range. Shorten the years or untick 'complete only'.")
    return w.rename_axis("year").reset_index()


# ---------------------------------------------------------------- 6. samples and demo

def demo_planted(seed: int = 7) -> pd.DataFrame:
    """Made-up weekly data with patterns planted on purpose, so first-timers can watch the method find them:
    - response rises faster than it falls when driver moves
    - leader moves first; follower_1-4 copy its moves a week later
    - cycle_series carries a cycle every 104 rows (about two years) under its noise
    - noise_1 and noise_2 have nothing in them (the method should find nothing there)."""
    rng = np.random.default_rng(seed)
    n = 600
    dates = pd.date_range("2014-01-06", periods=n, freq="W-MON")
    d_drv = rng.normal(0, 0.02, n)
    up, dn = np.clip(d_drv, 0, None), np.clip(d_drv, None, 0)
    lag = lambda x, k: np.concatenate([np.zeros(k), x[:-k]]) if k else x
    d_resp = 0.55 * up + 0.25 * lag(up, 1) + 0.25 * dn + 0.15 * lag(dn, 1) + 0.15 * lag(dn, 2) + rng.normal(0, 0.006, n)
    d_lead = rng.normal(0, 0.02, n)
    out = {"date": dates, "driver": 50 * np.exp(np.cumsum(d_drv)), "response": 3 * np.exp(np.cumsum(d_resp)),
           "leader": 100 * np.exp(np.cumsum(d_lead))}
    for i in range(1, 5):
        d_f = 0.6 * lag(d_lead, 1) + rng.normal(0, 0.012, n)
        out[f"follower_{i}"] = (80 + 10 * i) * np.exp(np.cumsum(d_f))
    t = np.arange(n)
    ar = np.zeros(n)
    e = rng.normal(0, 0.01, n)
    for k in range(1, n):
        ar[k] = 0.5 * ar[k - 1] + e[k]
    out["cycle_series"] = 20 * np.exp(0.04 * np.sin(2 * np.pi * t / 104) + ar)
    out["noise_1"] = 10 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    out["noise_2"] = 10 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame(out)


def _gdp_github() -> pd.DataFrame:
    raw = _get("https://raw.githubusercontent.com/datasets/gdp/main/data/gdp.csv")
    d = pd.read_csv(io.BytesIO(raw))
    real = _wb_countries()
    d = d[d["Country Code"].isin(real)]
    w = d.pivot_table(index="Year", columns="Country Name", values="Value").loc[1970:2023]
    w = w.loc[:, w.notna().all() & (w > 0).all()]
    return w.rename_axis("year").reset_index()


SAMPLES = {
    "Demo — made-up data with planted patterns": ("demo", None),
    "US gas, crude oil and CPI (weekly)": ("local_or", ("samples/us_gas_crude_weekly.csv", "fred:GASREGW, WCOILWTICO, CPIAUCSL")),
    "US regional gas prices (weekly)": ("local_or", ("samples/us_regional_gas_weekly.csv",
                                                     "fred:GASREGECW, GASREGMWW, GASREGGCW, GASREGRMW, GASREGWCW")),
    "Country GDP (yearly)": ("local_or", ("samples/country_gdp_yearly.csv", "gdp")),
}
SAMPLE_NOTES = {
    "Demo — made-up data with planted patterns":
        "Planted on purpose: **response** rises faster than it falls with **driver**; **leader** moves a week before "
        "**follower_1–4**; **cycle_series** has a cycle every 104 rows. **noise_1** and **noise_2** contain nothing.",
}


def load_sample(name: str) -> pd.DataFrame:
    kind, arg = SAMPLES[name]
    if kind == "demo":
        return demo_planted()
    local, remote = arg
    if os.path.exists(local):
        return pd.read_csv(local)
    if remote.startswith("fred:"):
        return fetch_fred(remote[5:])
    return _gdp_github()


# ---------------------------------------------------------------- helpers

def _tidy(df: pd.DataFrame, decimal_comma: bool = False) -> pd.DataFrame:
    """Drop empty rows/columns and turn number-looking text ('1,234', '$5.20', '12%') into numbers."""
    df = df.dropna(how="all").dropna(axis=1, how="all")
    df.columns = [str(c).strip() for c in df.columns]
    for c in df.columns[1:]:
        if not pd.api.types.is_numeric_dtype(df[c]):
            txt = df[c].astype(str)
            txt = (txt.str.replace(r"[.$%£€\s]", "", regex=True).str.replace(",", ".") if decimal_comma
                   else txt.str.replace(r"[,$%£€\s]", "", regex=True))
            cleaned = pd.to_numeric(txt, errors="coerce")
            if cleaned.notna().mean() >= 0.8:
                df[c] = cleaned
    return df.reset_index(drop=True)
