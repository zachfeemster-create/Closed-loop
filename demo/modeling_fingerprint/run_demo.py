"""Closed Loop demo: can the lenses tell an agency's smoothed numbers from the raw counts, beyond momentum?
Locked plan: closed-loop-log runs/2026-10-25_run009_beyond-momentum_PLAN.md (reproduced here as Run 009)."""
import glob, numpy as np, pandas as pd, pycountry
from scipy import stats
from fingerprint import load_who, train_scorer, Fakes
D = "data"

print("1) Training the fingerprint scorer on WHO modeled estimates ...")
score = train_scorer(load_who(D))

print("2) Pairing raw registered infant deaths (WHO) with UN IGME smoothed estimates ...")
M = pd.concat([pd.read_csv(f, low_memory=False, dtype={"Cause": str, "List": str, "Admin1": str, "SubDiv": str})
               for f in sorted(glob.glob(f"{D}/Morticd10_part*"))])
M = M[M.Admin1.isna() & M.SubDiv.isna() & M.Year.between(2000, 2019) & M.Cause.isin(["AAA", "1000"])]
raw = M.groupby(["Country", "Year"]).Deaths2.sum(min_count=1).unstack()
cc = pd.read_csv(f"{D}/country_codes").set_index("country")["name"]
MANUAL = {"United Kingdom": "GBR", "Republic of Korea": "KOR", "Republic of Moldova": "MDA", "Russian Federation": "RUS",
          "USA": "USA", "United States of America": "USA", "Czech Republic": "CZE", "Czechia": "CZE", "Venezuela": "VEN",
          "Iran (Islamic Rep of)": "IRN", "TFYR Macedonia": "MKD", "North Macedonia": "MKD", "Hong Kong SAR": "HKG",
          "Kyrgyzstan": "KGZ", "Turkey": "TUR", "Türkiye": "TUR", "Netherlands": "NLD", "Brunei Darussalam": "BRN",
          "Saint Vincent and Grenadines": "VCT", "Bahamas": "BHS", "Moldova": "MDA", "Reunion": "REU", "Rodrigues": None,
          "Virgin Islands (USA)": "VIR", "Occupied Palestinian Territory": "PSE"}
def iso(name):
    if name in MANUAL: return MANUAL[name]
    try: return pycountry.countries.lookup(name).alpha_3
    except LookupError:
        try: return pycountry.countries.search_fuzzy(name)[0].alpha_3
        except LookupError: return None
ig = pd.read_csv(f"{D}/igme_infant_deaths.csv"); ig = ig[ig.REF_AREA.str.fullmatch(r"[A-Z]{3}")]
ig = ig.pivot_table(index="REF_AREA", columns="TIME_PERIOD", values="OBS_VALUE")
pairs = []
for code, r in raw.iterrows():
    if r.notna().sum() != 20 or code not in cc.index: continue
    i3 = iso(cc[code])
    if i3 is None or i3 not in ig.index or ig.loc[i3].notna().sum() != 20: continue
    pairs.append((cc[code], r.values.astype(float), ig.loc[i3].values.astype(float)))
Rw = [p[1] for p in pairs]; Sm = [p[2] for p in pairs]; n = len(pairs)
print(f"   {n} countries paired")

print("3) Real gap vs momentum-only gap (50 fake draws each) ...")
fk = Fakes(69)
d = score(Sm) - score(Rw)
gap = lambda f: score([f(x) for x in Sm]) - score([f(x) for x in Rw])
dA = np.mean([gap(fk.A) for _ in range(50)], axis=0); dB = np.mean([gap(fk.B) for _ in range(50)], axis=0)
eA, eB = d - dA, d - dB
pA = stats.wilcoxon(eA, alternative="greater").pvalue; pB = stats.wilcoxon(eB, alternative="greater").pvalue
pF = stats.wilcoxon(gap(fk.A) - dA).pvalue

print(f"""
   smoothed scores higher than raw in {(d > 0).sum()}/{n} countries (median gap {np.median(d):.3f})
   gap from momentum alone:      {np.median(dA):.3f}
   EXCESS beyond momentum (A):   {np.median(eA):.3f}  in {(eA > 0).sum()}/{n} countries, p = {pA:.2g}   (pass: p < 0.01)
   EXCESS beyond smoothing (B):  {np.median(eB):.3f}  in {(eB > 0).sum()}/{n} countries, p = {pB:.2g}   (pass: p < 0.01)
   frame check (fake vs fake):   p = {pF:.2f}   (clean: p > 0.05)
""")
ok = pA < 0.01 and np.median(eA) > 0 and pB < 0.01 and np.median(eB) > 0 and pF > 0.05
print("VERDICT:", "PASS - the fingerprint sees modeling beyond momentum" if ok else ("VOID (frame check tripped)" if pF <= 0.05 else "FAIL"))
