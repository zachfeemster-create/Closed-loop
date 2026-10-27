"""Race-level counter-spending check: does one side's surge in a race draw the other side's money the next week?

    python experiments/fec/lead_check.py [cache_dir]

Races with more than $1M from each side, 2016-2024. Weekly log spend per side, race level and calendar week
removed, own last week held fixed. Fakes shuffle the answered side's weeks within each race. The gap
(D->R minus R->D) gets a 95% range from resampling races. Dates: FEC dissemination date where the plain
independent-expenditure file has it for that transaction, else payment date.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_fec as B  # noqa: E402


def dissem_dates(y):
    path = B.fetch(f"{B.FEC}/{y}/independent_expenditure_{y}.csv", f"{B.CACHE}/independent_expenditure_{y}.csv")
    d = pd.read_csv(path, dtype=str, low_memory=False, usecols=["spe_id", "tran_id", "dissem_dt", "file_num"])
    d["fn"] = pd.to_numeric(d.file_num, errors="coerce")
    d = d.sort_values("fn").drop_duplicates(["spe_id", "tran_id"], keep="last")
    d["dis"] = pd.to_datetime(d.dissem_dt, format="%d-%b-%y", errors="coerce")
    return d.dropna(subset=["dis"]).set_index(["spe_id", "tran_id"]).dis


def panel(y):
    d = B.load_ie(y)
    dis = dissem_dates(y).reindex(pd.MultiIndex.from_arrays([d.CMTE_ID, d.TRAN_ID]))
    dis = pd.Series(dis.values, index=d.index)
    ok = dis.notna() & (dis >= pd.Timestamp(f"{y}-01-01")) & (dis <= pd.Timestamp(B.ELECTION[y]))
    d["when"] = dis.where(ok, d.date)
    d["wk"] = d.when.dt.to_period("W-SUN").dt.end_time.dt.normalize()
    weeks = pd.date_range(f"{y}-01-01", B.ELECTION[y], freq="W-SUN")
    tot = d.pivot_table(index="race", columns="side", values="amt", aggfunc="sum").fillna(0.0)
    keep = tot[(tot["D-side"] > 1e6) & (tot["R-side"] > 1e6)].index
    t = d[d.race.isin(keep)].pivot_table(index=["race", "wk"], columns="side", values="amt", aggfunc="sum").fillna(0.0)
    rows = []
    for r in keep:
        g = t.loc[r].reindex(weeks).fillna(0.0)
        rows.append(pd.DataFrame({"race": f"{y}-{r}", "cyc": y, "k": np.arange(len(weeks)),
                                  "D": np.log1p(g["D-side"].values), "R": np.log1p(g["R-side"].values)}))
    return pd.concat(rows)


def prep(P):
    P = P.copy()
    for c in ("D", "R"):
        P[c] = P[c] - P.groupby("race")[c].transform("mean")
        P[c] = P[c] - P.groupby(["cyc", "k"])[c].transform("mean")
    P = P.sort_values(["race", "k"])
    for c in ("D", "R"):
        P[c + "1"] = P.groupby("race")[c].shift(1)
    return P.dropna()


def beta(y, x, own):
    return np.linalg.lstsq(np.column_stack([np.ones(len(y)), x, own]), y, rcond=None)[0][1]


def main(n_fakes=1000, n_boot=500, seed=1):
    rng = np.random.default_rng(seed)
    P = prep(pd.concat([panel(y) for y in B.ELECTION]))
    d2r = beta(P.R.values, P.D1.values, P.R1.values)
    r2d = beta(P.D.values, P.R1.values, P.D1.values)
    idx = P.groupby("race").indices
    D1, R1 = P.D1.values, P.R1.values
    fd, fr = np.empty(n_fakes), np.empty(n_fakes)
    for i in range(n_fakes):
        sD, sR = D1.copy(), R1.copy()
        for ix in idx.values():
            sD[ix] = D1[rng.permutation(ix)]
            sR[ix] = R1[rng.permutation(ix)]
        fd[i] = beta(P.R.values, sD, P.R1.values)
        fr[i] = beta(P.D.values, sR, P.D1.values)
    races = np.array(list(idx))
    gaps = []
    for _ in range(n_boot):
        S = P.iloc[np.concatenate([idx[r] for r in rng.choice(races, len(races))])]
        gaps.append(beta(S.R.values, S.D1.values, S.R1.values) - beta(S.D.values, S.R1.values, S.D1.values))
    lo, hi = np.percentile(gaps, [2.5, 97.5])
    print(f"{len(races)} races with >$1M per side, 2016-2024")
    print(f"D-side surge -> R-side next week: {d2r:+.3f} (beats {np.mean(fd < d2r):.1%} of within-race shuffles)")
    print(f"R-side surge -> D-side next week: {r2d:+.3f} (beats {np.mean(fr < r2d):.1%} of within-race shuffles)")
    print(f"gap (D->R minus R->D): {d2r - r2d:+.3f}, 95% range from resampling races [{lo:+.3f}, {hi:+.3f}]")


if __name__ == "__main__":
    main()
