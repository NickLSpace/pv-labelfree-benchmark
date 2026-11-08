"""Full-length check of the frozen model on the 30-plant synthetic fleet WITH / WITHOUT the
real-world effects (pyranometer drift, P_exp model error, seasonal bias).

Executes the notebook code cells of sections 10, 12 and 14 (optionally 13) with the notebook
training settings (epochs=15, train_stride=1) and writes the tables to the current directory.
Not HKUST data.

    EFFECTS=all|none  SEEDS=0,1  PLACEMENTS=0,1,2  EPOCHS=15  STRIDE=1  SECTIONS=10,12,14
    cd /tmp/run_all && python /path/to/tests/realistic_fixture_check.py
"""
import os, sys, time, pickle, json
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, nbformat, numpy as np, pandas as pd
from IPython.display import display
from _semisynth_fixture import make, ALL_EFFECTS
import pv_kalman as kf_mod

EFF = os.environ.get("EFFECTS", "all")
effects = ALL_EFFECTS if EFF == "all" else () if EFF == "none" else tuple(EFF.split(","))
seeds = [int(x) for x in os.environ.get("SEEDS", "0,1").split(",")]
placements = [int(x) for x in os.environ.get("PLACEMENTS", "0,1,2").split(",")]
epochs, stride = int(os.environ.get("EPOCHS", "15")), int(os.environ.get("STRIDE", "1"))
sections = os.environ.get("SECTIONS", "10,12,14").split(",")
tag = {"10": "semisynth-section", "12": "fixes-section", "13": "fcm-section", "14": "value-section"}

nb = nbformat.read(os.path.join(ROOT, "pv-stgnn-fuzzy-multi-full.ipynb"), as_version=4)
get = lambda t: [c.source for c in nb.cells if c.cell_type == "code" and t in c.metadata.get("tags", [])]
D, C = make(n_pv=30, cohorts=True, effects=effects)
print("effects:", effects, "| plants:", len(D["pv_cols"]), "| cohorts:",
      D["ca"].cohorts["cohort"].value_counts().sort_index().to_dict(), flush=True)

# small raw frame for the aggregation diagnostic of section 12 (as in notebook_section12_13_test)
rng = np.random.default_rng(0)
idx = pd.date_range("2023-06-01", periods=6 * 24 * 60, freq="1min")
g = np.clip(np.sin(np.pi * (idx.hour + idx.minute / 60 - 6) / 12), 0, None) * 900
df_all = pd.DataFrame({"Irradiance (W/m2)": g}, index=idx)
for c in ("power(W)_A", "power(W)_B"):
    v = pd.Series(np.nan, index=idx); q = idx.minute % 15 == 0
    v[q] = g[q] * 5 * (1 + rng.normal(0, .02, q.sum())); v[(rng.random(len(v)) < 0.15) & q] = np.nan
    df_all[c] = v

ns = dict(D=D, C=C, A=None, train_seeds=seeds, DEVICE="cpu", KALMAN_CFG=kf_mod.KalmanConfig(), plt=plt,
          display=lambda *a, **k: None, os=os, pickle=pickle, pd=pd, np=np, df_all=df_all, pv2_cols_list=[])
t_all = time.time()
for sec in sections:
    for i, src in enumerate(get(tag[sec])):
        src = src.replace("PLACEMENT_SEEDS = [0, 1, 2]", f"PLACEMENT_SEEDS = {placements}")
        src = src.replace("epochs=15, train_stride=1", f"epochs={epochs}, train_stride={stride}")
        src = src.replace("MODEL_SEEDS = train_seeds[:2]", f"MODEL_SEEDS = {seeds}")
        t0 = time.time(); exec(compile(src, f"cell{sec}_{i}", "exec"), ns)
        print(f"--- section {sec} cell {i} ok ({time.time() - t0:.1f}s)", flush=True)
    plt.close("all")

os.makedirs("summary", exist_ok=True)
out = {}
for name in ("ss_res", "fx_res", "fcm_res", "val_res"):
    r = ns.get(name)
    if r is None:
        continue
    for k, t in r.tables.items():
        t.to_csv(f"summary/{name}_{k}.csv", index=False)
    out[name] = sorted(r.tables)
with open("summary/meta.json", "w") as f:
    json.dump(dict(effects=list(effects), seeds=seeds, placements=placements, epochs=epochs, stride=stride,
                   sections=sections, minutes=round((time.time() - t_all) / 60, 1), tables=out), f, indent=1)
print("done in %.1f min" % ((time.time() - t_all) / 60))
