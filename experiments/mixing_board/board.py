import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
rng=np.random.default_rng(60)
df=pd.read_csv("wb_raw.csv")
P=df.pivot_table(index=["ind","iso"],columns="year",values="v")
P=P.dropna(); X=P.values; meta=P.index.to_frame(index=False)
N=X.shape[1]; print("series",len(X),"length",N)
BANDS={"delta":(1,2),"theta":(3,4),"alpha":(5,7),"beta":(8,11)}  # cycles per 23 yrs, same octave ladder as brain bands
def detrend(x):
    t=np.arange(len(x)); return x-np.polyval(np.polyfit(t,x,1),t)
def split(x):
    F=np.fft.rfft(x); out={}
    for b,(lo,hi) in BANDS.items():
        G=np.zeros_like(F); G[lo:hi+1]=F[lo:hi+1]; out[b]=np.fft.irfft(G,len(x))
    return out
def lens(c,total):
    # rebuilt quarter turn: point -> (u=level, v=step), theta angle, dtheta advance, Q=u^2+v^2
    u=c[1:]; v=np.diff(c); s=np.std(c)+1e-12; u=u/s; v=v/s
    th=np.unwrap(np.arctan2(v,u)); dth=np.diff(th); Q=u**2+v**2
    return [np.var(c)/(total+1e-12), np.mean(np.abs(dth)), np.mean(dth<0), (np.mean((Q-Q.mean())**4)/(Q.var()**2+1e-12))]
def feats(x):
    d=detrend(np.asarray(x,float)); tot=np.var(d); ch=split(d)
    return {"dry":lens(d,tot), **{b:lens(ch[b],tot) for b in BANDS}}
def fake(x,kind):
    st=np.diff(x); sd=st.std()+1e-12; n=len(x)
    if kind=="noise": return rng.normal(0,x.std()+1e-12,n)
    if kind=="walk": return np.cumsum(np.r_[0,rng.normal(st.mean(),sd,n-1)])
    if kind=="spiky": t=rng.standard_t(2.5,n-1); return np.cumsum(np.r_[0,st.mean()+sd*t/t.std()])
    if kind=="shuffled": return np.cumsum(np.r_[x[0],rng.permutation(st)])
KINDS=["noise","walk","spiky","shuffled"]
CH=["dry"]+list(BANDS)
real=[feats(x) for x in X]
fk={k:[feats(fake(x,k)) for x in X] for k in KINDS}
groups=np.r_[np.arange(len(X)),np.arange(len(X))]
y=np.r_[np.ones(len(X)),np.zeros(len(X))]
def sep(F,yy=y,gg=groups):
    p=np.zeros(len(yy))
    for tr,te in GroupKFold(5).split(F,yy,gg):
        sc=StandardScaler().fit(F[tr]); m=LogisticRegression(max_iter=2000).fit(sc.transform(F[tr]),yy[tr]); p[te]=m.predict_proba(sc.transform(F[te]))[:,1]
    return abs(roc_auc_score(yy,p)-0.5)*2
def mat(kind,c): return np.array([r[c] for r in real]+[r[c] for r in fk[kind]])
# chance band: label shuffle within real/fake pairs
null=[]
F=mat("walk","theta")
for i in range(40):
    flip=rng.random(len(X))<0.5; yy=y.copy(); yy[:len(X)][flip]=0; yy[len(X):][flip]=1
    null.append(sep(F,yy))
chance=np.percentile(null,95); print(f"chance band (95th pct) {chance:.3f}")
res=pd.DataFrame({k:{c:sep(mat(k,c)) for c in CH} for k in KINDS}); print("\nseparation per channel\n",res.round(3))
# wet/dry: set mix levels on half the countries, test the mix on the other half
iso=meta.iso.values; u=np.unique(iso); A=set(rng.choice(u,len(u)//2,replace=False))
inA=np.array([i in A for i in iso]); inA2=np.r_[inA,inA]
out=[]
for k in KINDS:
    lv={}
    for c in BANDS:
        Fc=mat(k,c); s=sep(Fc[inA2],y[inA2],groups[inA2])
        lv[c]=max(0.1,(s-chance)/(1-chance))  # floor 0.1 = tambourine
    Fb={c:mat(k,c)[~inA2] for c in CH}; yb=y[~inA2]; gb=groups[~inA2]
    def mix(levels):
        cols=[StandardScaler().fit_transform(Fb[c])*levels.get(c,0) for c in levels]; return np.hstack(cols)
    def sep_ridge(F):
        p=np.zeros(len(yb))
        for tr,te in GroupKFold(5).split(F,yb,gb):
            m=LogisticRegression(C=0.05,max_iter=3000).fit(F[tr],yb[tr]); p[te]=m.predict_proba(F[te])[:,1]
        return abs(roc_auc_score(yb,p)-0.5)*2
    dry=sep_ridge(mix({"dry":1})); flat=sep_ridge(mix({"dry":1,**{c:1 for c in BANDS}})); wet=sep_ridge(mix({"dry":1,**lv}))
    out.append(dict(fake=k,**{f"lvl_{c}":round(lv[c],2) for c in BANDS},dry_only=round(dry,3),all_equal=round(flat,3),wet_dry_mix=round(wet,3)))
print("\nwet/dry mix (levels set on half A, scored on half B)\n",pd.DataFrame(out).to_string(index=False))
res.to_csv("separation.csv"); pd.DataFrame(out).to_csv("mix.csv",index=False)
# which indicators lean on theta
thF=np.array([r["theta"][0] for r in real]); meta["theta_share"]=thF
print("\ntheta energy share by indicator (real median)\n",meta.groupby("ind").theta_share.median().round(3).sort_values())
print("fake theta share medians:",{k:round(float(np.median([r['theta'][0] for r in fk[k]])),3) for k in KINDS})
