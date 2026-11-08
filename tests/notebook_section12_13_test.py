"""Executes notebook sections 10, 12, 13 and 14 (code cells) on the synthetic fixture (not HKUST data)."""
import os, sys, time, pickle
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, nbformat, numpy as np, pandas as pd
from IPython.display import display
from _semisynth_fixture import make
import pv_kalman as kf_mod
nb = nbformat.read(os.path.join(ROOT, "pv-stgnn-fuzzy-multi-full.ipynb"), as_version=4)
get = lambda tag: [c.source for c in nb.cells if c.cell_type == "code" and tag in c.metadata.get("tags", [])]
s10, s12, s13, s14 = get("semisynth-section"), get("fixes-section"), get("fcm-section"), get("value-section")
assert len(s10) == 6 and len(s12) == 6 and len(s13) == 5 and len(s14) == 5, (len(s10), len(s12), len(s13), len(s14))
# order: references stay the last cell; section 12 after 11, 13 after 12
tags = [tuple(c.metadata.get("tags", [])) for c in nb.cells]
first = lambda t: next(i for i, x in enumerate(tags) if t in x)
assert first("sensitivity-section") < first("fixes-section") < first("fcm-section") < first("value-section")
assert nb.cells[-1].source.startswith(("## Литература", "## References"))
D, C = make(n_pv=30, cohorts=True)
# small raw frame for the aggregation diagnostic: 15-min power on a 1-min grid with gaps
rng = np.random.default_rng(0)
idx = pd.date_range("2023-06-01", periods=6 * 24 * 60, freq="1min")
g = np.clip(np.sin(np.pi * (idx.hour + idx.minute / 60 - 6) / 12), 0, None) * 900
df_all = pd.DataFrame({"Irradiance (W/m2)": g}, index=idx)
for c in ("power(W)_A", "power(W)_B"):
    v = pd.Series(np.nan, index=idx); q = idx.minute % 15 == 0
    v[q] = g[q] * 5 * (1 + rng.normal(0, .02, q.sum()))
    drop = rng.random(len(v)) < 0.15
    v[drop & q] = np.nan
    v[(idx.day == 3) & (idx.hour >= 10) & (idx.hour < 13)] = np.nan       # fully missing daylight hours
    df_all[c] = v
ns = dict(D=D, C=C, A=None, train_seeds=[0], DEVICE="cpu", KALMAN_CFG=kf_mod.KalmanConfig(), plt=plt,
          display=display, os=os, pickle=pickle, pd=pd, np=np, df_all=df_all, pv2_cols_list=[])
for name, cells in (("10", s10), ("12", s12), ("13", s13), ("14", s14)):
    for i, src in enumerate(cells):
        src = src.replace("PLACEMENT_SEEDS = [0, 1, 2]", "PLACEMENT_SEEDS = [0, 1]")
        src = src.replace("epochs=15, train_stride=1", "epochs=5, train_stride=2")
        t0 = time.time(); exec(compile(src, f"cell{name}_{i}", "exec"), ns)
        print(f"--- section {name} cell {i} ok ({time.time() - t0:.1f}s)", flush=True)
fx_res, fcm_res, val_res = ns["fx_res"], ns["fcm_res"], ns["val_res"]
assert fx_res.meta["weights_unchanged"]
agg = ns["AGG_TABLE"]; print(agg.round(2).to_string(index=False))
assert agg.iloc[0]["zero hours, %"] > agg.iloc[2]["zero hours, %"]        # legacy zero fill creates zero hours
for k in ("operating_points", "equal_budget", "classification", "aggregation"):
    assert k in fx_res.tables and len(fx_res.tables[k]), k
for k in ("partitions", "equal_budget", "delta_recall", "operating_points"):
    assert k in fcm_res.tables and len(fcm_res.tables[k]), k
assert set(fcm_res.points["method"]) == {"kmeans", "fcm_hard", "fcm_soft"}
for p in ("results/fixes/fixes_tables.tex", "results/fixes/fixes_results.zip", "results/fcm/fcm_tables.tex",
          "results/fcm/fcm_results.zip", "results/figures/fig_fcm_curves.png", "results/figures/fig_fixes_tradeoff.png"):
    assert os.path.exists(p), p
assert val_res.meta["weights_unchanged"]
for k in ("equal_budget", "fusion_bootstrap", "alarms_by_stratum", "cold_start", "hard_scenarios", "sensor_vs_plant", "verdict"):
    assert k in val_res.tables and len(val_res.tables[k]), k
assert len(val_res.tables["verdict"]) >= 3 and val_res.tables["verdict"]["Met"].dtype == bool
# PR+ (main comparison STGNN vs PR+) in sections 12 and 14
assert "prp_rule" in set(fx_res.points["method"]) and "prp_rule" in set(fx_res.curves["method"])
assert fx_res.tables["equal_budget"]["Method"].str.contains("PR\\+").any()
assert {"prp", "fusion_prp"} <= set(val_res.points["method"])
fb = val_res.tables["fusion_bootstrap"]
assert (fb["Comparison"] == "STGNN-day − PR+").any() and (fb["Comparison"] == "Fusion+ − PR+").any()
assert val_res.tables["verdict"]["Condition"].str.startswith("1+.").any()
assert "Rate ratio STGNN-day / PR+" in val_res.tables["alarms_by_stratum"]
for p in ("results/value/value_tables.tex", "results/value/value_results.zip", "results/figures/fig_value_fusion.png",
          "results/figures/fig_value_strata.png", "results/figures/fig_value_hard.png"):
    assert os.path.exists(p), p
# all generated tables must be English
import re
cyr = re.compile("[А-Яа-яЁё]")
for res in (fx_res, fcm_res, val_res):
    for k, t in res.tables.items():
        txt = t.to_csv()
        assert not cyr.search(txt), (k, cyr.findall(txt)[:5])
print(fcm_res.tables["delta_recall"].round(3).to_string(index=False))
print(val_res.tables["verdict"].to_string(index=False))
print("OK")
