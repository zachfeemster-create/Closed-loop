"""Freeze the exact inputs the demo uses into ./snapshot (small CSVs), from a full ./data download."""
import glob, os, numpy as np, pandas as pd
import fingerprint as F
os.makedirs("snapshot", exist_ok=True)
X, labels = F.load_who("data", with_labels=True)
pd.DataFrame([[a, b] + list(x) for (a, b), x in zip(labels, X)], columns=["indicator", "iso"] + list(range(2000, 2020))).to_csv("snapshot/who_modeled_series.csv", index=False)
M = pd.concat([pd.read_csv(f, low_memory=False, dtype={"Cause": str, "List": str, "Admin1": str, "SubDiv": str}) for f in sorted(glob.glob("data/Morticd10_part*"))])
M = M[M.Admin1.isna() & M.SubDiv.isna() & M.Year.between(2000, 2019) & M.Cause.isin(["AAA", "1000"])]
raw = M.groupby(["Country", "Year"]).Deaths2.sum(min_count=1).unstack()
raw = raw[raw.notna().sum(axis=1) == 20].astype("int64")
cc = pd.read_csv("data/country_codes").set_index("country")["name"]
raw.insert(0, "name", [cc.get(c, "") for c in raw.index]); raw.index.name = "who_code"; raw.to_csv("snapshot/raw_infant_deaths.csv")
ig = pd.read_csv("data/igme_infant_deaths.csv"); ig = ig[ig.REF_AREA.str.fullmatch(r"[A-Z]{3}")]
ig.pivot_table(index="REF_AREA", columns="TIME_PERIOD", values="OBS_VALUE").to_csv("snapshot/igme_infant_deaths.csv")
for f in sorted(os.listdir("snapshot")): print(f, os.path.getsize("snapshot/" + f))
