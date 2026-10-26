import requests, pandas as pd, sys
inds = {"NY.GDP.MKTP.KD.ZG":"gdp_growth","SP.DYN.LE00.IN":"life_exp","SP.DYN.IMRT.IN":"infant_mort",
"FP.CPI.TOTL.ZG":"inflation","NE.TRD.GNFS.ZS":"trade","SL.UEM.TOTL.ZS":"unemployment","IT.NET.USER.ZS":"internet",
"SP.URB.TOTL.IN.ZS":"urban","SP.POP.GROW":"pop_growth","SP.DYN.TFRT.IN":"fertility"}
rows=[]
for code,name in inds.items():
    r=requests.get(f"https://api.worldbank.org/v2/country/all/indicator/{code}",params={"format":"json","date":"2000:2022","per_page":20000},timeout=60).json()
    for d in r[1]:
        if d["value"] is not None and len(d["countryiso3code"])==3:
            rows.append((name,d["countryiso3code"],int(d["date"]),d["value"]))
    print(name,len(r[1]),file=sys.stderr)
c=requests.get("https://api.worldbank.org/v2/country",params={"format":"json","per_page":400},timeout=60).json()[1]
agg={x["id"] for x in c if x["region"]["value"]=="Aggregates"}
df=pd.DataFrame(rows,columns=["ind","iso","year","v"]); df=df[~df.iso.isin(agg)]
df.to_csv("wb_raw.csv",index=False); print(df.shape)
