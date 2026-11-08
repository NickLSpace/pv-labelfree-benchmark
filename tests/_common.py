"""Synthetic smoke test: SPS vs Kalman on a fake fleet with known b1 trajectories."""
import sys, time, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from types import SimpleNamespace
import pv_kalman as kf, pv_sps, pv_estimator_comparison as cmpm, pv_clustering as clu

rng = np.random.default_rng(1)
idx = pd.date_range("2021-01-01", "2023-12-31 23:00", freq="h")
T = len(idx); n_pl = 10
hour = idx.hour.to_numpy(); doy = idx.dayofyear.to_numpy()
clear = np.clip(np.sin(np.pi * (hour - 6) / 12), 0, None) * (900 + 150 * np.cos(2 * np.pi * (doy - 172) / 365))
daycloud = np.repeat(rng.uniform(0.3, 1.0, T // 24 + 1), 24)[:T]
ghi = clear * daycloud
kt = pd.Series(np.where(clear > 0, ghi / np.maximum(clear, 1), np.nan), index=idx)
tcell = 25 + 0.03 * ghi + rng.normal(0, 1, T)
cols = [f"power(W)_P{i}" for i in range(n_pl)]
rated = {c: 10.0 for c in cols}
pexp = np.zeros((T, n_pl)); df = pd.DataFrame(index=idx)
yrs = (idx - idx[0]).days.to_numpy() / 365.25
truth = {}
for j, c in enumerate(cols):
    pexp[:, j] = 10_000 * ghi / 1000 * 0.8
    if j < 3:   b1 = 1 - 0.05 * yrs                       # gradual degradation
    elif j < 5: b1 = np.where(idx >= "2023-03-01", 0.85, 1.0)   # step loss
    else:       b1 = np.ones(T)
    dnoise = np.repeat(rng.normal(0, 0.03, T // 24 + 1), 24)[:T]   # within-day correlated
    P = b1 * pexp[:, j] * (1 + dnoise) + rng.normal(0, 150, T)
    P = np.clip(P, 0, None)
    if j == 9: P[idx < "2023-01-01"] = np.nan                      # new plant (cohort 3)
    df[c] = P; truth[short_ := c.replace("power(W)_", "")] = b1[-1]
phys = dict(p_expected=pexp, ghi=ghi)
valid = pd.DataFrame({c: (ghi > 200) & np.isfinite(df[c].to_numpy()) for c in cols}, index=idx)
cloud = (1 - kt.clip(0, 1.2)).fillna(1).to_numpy()
tcell2 = np.repeat(tcell[:, None], n_pl, 1)


# fake pipeline objects for the comparison
D = dict(df=df, phys=phys, pv_cols=cols, rated=rated, t_cell=tcell2, kt=kt)
regime = pd.Series(np.where(ghi > 20, 1 + (kt.fillna(0) < 0.6).astype(int), 0), index=idx)
C = dict(valid=valid, tech=SimpleNamespace(labels=pd.Series([0, 0, 0, 1, 1, 1, 2, 2, 2, 2], index=cols)),
         wr=SimpleNamespace(regime=regime))
runs = []
for s in (0, 1, 2):
    r = np.random.default_rng(s)
    plants = [c.replace("power(W)_", "") for c in cols]
    rank = pd.Series([2.5 if i < 5 else 0.8 for i in range(n_pl)], index=plants) + r.normal(0, .3, n_pl)
    exc = pd.Series([0.05 if i < 5 else 0.005 for i in range(n_pl)], index=plants) + r.uniform(0, .01, n_pl)
    ev = []
    for _ in range(120):
        p = plants[r.integers(n_pl)]
        st = pd.Timestamp("2023-01-05") + pd.Timedelta(hours=int(r.integers(0, 8000)))
        ev.append(dict(plant=p, start=st, end=st + pd.Timedelta(hours=int(r.integers(3, 40))),
                       peak_score=float(r.uniform(3, 9)), mean_physics_z=float(r.normal())))
    runs.append(dict(variant="multiview", seed=s, rank=rank, exceed_share=exc, events=pd.DataFrame(ev)))

