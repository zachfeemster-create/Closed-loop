"""Download the public data the demo needs into ./data (about 60 MB, one time)."""
import os, zipfile, urllib.request
D = "data"; os.makedirs(D, exist_ok=True)
WHO_GHO = ["NCD_BMI_30A", "NCD_BMI_18A", "NCD_BMI_PLUS2C", "NCD_HYP_PREVALENCE_A",
           "NUTRITION_ANAEMIA_CHILDREN_PREV", "WHOSIS_000001", "SA_0000001688"]
MDB = "https://cdn.who.int/media/docs/default-source/world-health-data-platform/mortality-raw-data/"
MDB_FILES = ["mort_country_codes"] + [f"morticd10_part{i}" for i in range(1, 7)]
IGME = ("https://sdmx.data.unicef.org/ws/public/sdmxapi/rest/data/UNICEF,CME,1.0/.CME_TMY0._T._T."
        "?format=csv&startPeriod=2000&endPeriod=2019")

def get(url, path, headers=None):
    if os.path.exists(path): print("have", path); return
    print("downloading", path)
    req = urllib.request.Request(url, headers=headers or {"User-Agent": "closed-loop-demo"})
    with urllib.request.urlopen(req, timeout=600) as r, open(path, "wb") as f: f.write(r.read())

for c in WHO_GHO:                       # WHO modeled estimates (training data for the scorer)
    get(f"https://ghoapi.azureedge.net/api/{c}", f"{D}/{c}.json")
for f in MDB_FILES:                     # WHO Mortality Database, raw registered deaths
    z = f"{D}/{f}.zip"; get(MDB + f + ".zip", z)
    with zipfile.ZipFile(z) as zz: zz.extractall(D)
get(IGME, f"{D}/igme_infant_deaths.csv",  # UN IGME smoothed estimates of the same deaths
    {"Accept": "application/vnd.sdmx.data+csv;version=1.0.0", "User-Agent": "closed-loop-demo"})
print("done")
