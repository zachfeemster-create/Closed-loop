"""Lens features, fakes and the frozen 'modeling fingerprint' scorer (Closed Loop, Runs 006-009)."""
import json, numpy as np, pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

BANDS = {"delta": (1, 2), "theta": (3, 4), "alpha": (5, 7), "beta": (8, 10)}   # FFT bins in a 20-year window

def detrend(x):
    t = np.arange(len(x)); return x - np.polyval(np.polyfit(t, x, 1), t)

def split(x):
    F = np.fft.rfft(x); out = {}
    for b, (lo, hi) in BANDS.items():
        G = np.zeros_like(F); G[lo:hi + 1] = F[lo:hi + 1]; out[b] = np.fft.irfft(G, len(x))
    return out

def lens(c, total):
    """Quarter-turn lens: each point -> (level, step); read its orbit."""
    u = c[1:]; v = np.diff(c); s = np.std(c) + 1e-12; u = u / s; v = v / s
    th = np.unwrap(np.arctan2(v, u)); dth = np.diff(th); Q = u**2 + v**2
    return [np.var(c) / (total + 1e-12),                                   # energy share
            np.mean(np.abs(dth)),                                          # spin
            np.mean(dth < 0),                                              # turn-back
            np.mean((Q - Q.mean())**4) / (Q.var()**2 + 1e-12)]             # burst

def feats(x):
    d = detrend(np.asarray(x, float)); tot = np.var(d); ch = split(d)
    return {b: lens(ch[b], tot) for b in BANDS}

def lag1(x):
    s = np.diff(x); s = s - s.mean(); return (s[:-1] @ s[1:]) / (s @ s + 1e-12)

class Fakes:
    """Momentum-matched fakes. A = AR(1) walk, B = smoothed walk, C = interpolated walk, D = smoothed shuffle."""
    SIG = (0, 0.5, 0.75, 1, 1.5, 2, 2.5, 3)
    def __init__(self, seed): self.rng = np.random.default_rng(seed)
    def walk(self, x):
        st = np.diff(x); return np.cumsum(np.r_[x[0], self.rng.normal(st.mean(), st.std() + 1e-12, len(x) - 1)])
    def A(self, x):
        st = np.diff(x); phi = float(np.clip(lag1(x), -0.95, 0.95)); e = self.rng.normal(0, 1, len(st))
        s = np.zeros(len(st)); s[0] = e[0]
        for i in range(1, len(st)): s[i] = phi * s[i - 1] + np.sqrt(1 - phi**2) * e[i]
        return np.cumsum(np.r_[x[0], st.mean() + (st.std() + 1e-12) * s])
    def _gsm(self, b, t):
        c = [b if s == 0 else gaussian_filter1d(b, s, mode="reflect") for s in self.SIG]
        return c[int(np.argmin([abs(lag1(z) - t) for z in c]))]
    def B(self, x): return self._gsm(self.walk(x), lag1(x))
    def C(self, x):
        b = self.walk(x); t = lag1(x); N = len(b); cands = []
        for k in (1, 2, 3, 4, 5, 6):
            a = sorted(set([0, N - 1] + list(range(self.rng.integers(0, k), N, k)))); cands.append(np.interp(np.arange(N), a, b[a]))
        return cands[int(np.argmin([abs(lag1(z) - t) for z in cands]))]
    def D(self, x): return self._gsm(np.cumsum(np.r_[x[0], self.rng.permutation(np.diff(x))]), lag1(x))

WHO_GHO = ["NCD_BMI_30A", "NCD_BMI_18A", "NCD_BMI_PLUS2C", "NCD_HYP_PREVALENCE_A",
           "NUTRITION_ANAEMIA_CHILDREN_PREV", "WHOSIS_000001", "SA_0000001688"]

def load_who(D="data", with_labels=False):
    """WHO modeled estimates, both sexes / totals, 2000-2019, complete series only (Run 006 rules)."""
    TOT = {"SEVERITY_TOTAL", "PREGNANCYSTATUS_TOTAL"}; X = []; L = []
    for c in WHO_GHO:
        recs = []
        for d in json.load(open(f"{D}/{c}.json"))["value"]:
            if d.get("SpatialDimType") != "COUNTRY" or d.get("NumericValue") is None or not (2000 <= d["TimeDim"] <= 2019): continue
            dims = {d.get(k) for k in ("Dim1", "Dim2", "Dim3")} - {None}
            if not any(x == "SEX_BTSX" for x in dims) and any(str(x).startswith("SEX_") for x in dims): continue
            if c == "NCD_BMI_PLUS2C" and "AGEGROUP_YEARS05-19" not in dims: continue
            others = [x for x in dims if not str(x).startswith(("SEX_", "AGEGROUP_"))]
            if others and not any(x in TOT for x in others): continue
            recs.append((d["SpatialDim"], d["TimeDim"], d["NumericValue"]))
        P = pd.DataFrame(recs, columns=["iso", "year", "v"]).groupby(["iso", "year"]).v.mean().unstack()
        for iso, r in P.iterrows():
            if r.notna().sum() == 20 and r.std() > 0: X.append(r.values.astype(float)); L.append((c, iso))
    return (X, L) if with_labels else X

def load_who_snapshot(path="snapshot/who_modeled_series.csv"):
    """The same series, frozen from the 25 Oct 2026 download (same order)."""
    return [r.astype(float) for r in pd.read_csv(path).iloc[:, 2:].values]

def train_scorer(X):
    """Real WHO modeled curves (1) vs their fakes A-D (0), on theta+alpha lens features. Frozen after this."""
    fk = Fakes(66); vec = lambda f: f["theta"] + f["alpha"]; R = []; F = {k: [] for k in "ABCD"}
    for x in X:   # draw order kept identical to Run 006 (A, B, C, D, then three frame twins)
        R.append(vec(feats(x)))
        for k in "ABCD": F[k].append(vec(feats(getattr(fk, k)(x))))
        fk.A(x); fk.B(x); fk.C(x)
    Xtr = np.array(R + F["A"] + F["B"] + F["C"] + F["D"]); y = np.r_[np.ones(len(R)), np.zeros(4 * len(R))]
    sc = StandardScaler().fit(Xtr); clf = LogisticRegression(max_iter=3000, random_state=67).fit(sc.transform(Xtr), y)
    return lambda series: clf.predict_proba(sc.transform(np.array([vec(feats(s)) for s in series])))[:, 1]
