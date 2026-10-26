exec(open("board.py").read().split("# chance band")[0])
raw=meta.ind.isin(["gdp_growth","inflation","trade"]).values; r2=np.r_[raw,raw]
print("\nmeasured-ish only (GDP growth, inflation, trade):", raw.sum(),"series")
print(pd.DataFrame({k:{c:round(sep(mat(k,c)[r2],y[r2],groups[r2]),3) for c in CH} for k in ["walk","shuffled"]}))
names=["energy","spin","turn-back","burst"]
print("\ntheta channel, one lens feature at a time (vs shuffled):")
for j,n in enumerate(names):
    F=mat("shuffled","theta")[:,[j]]; print(f"  {n:10s} {sep(F):.3f}", " real med %.3f fake med %.3f"%(np.median(F[:len(X)]),np.median(F[len(X):])))
