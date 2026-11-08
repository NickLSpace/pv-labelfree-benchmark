"""Executes notebook sections 10 and 11 (code cells) on the synthetic fixture (not HKUST data)."""
import os, sys, time, pickle
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, nbformat, numpy as np
from IPython.display import display
from _semisynth_fixture import make
import pv_kalman as kf_mod
nb = nbformat.read(os.path.join(ROOT, "pv-stgnn-fuzzy-multi-full.ipynb"), as_version=4)
get = lambda tag: [c.source for c in nb.cells if c.cell_type == "code" and tag in c.metadata.get("tags", [])]
s10, s11 = get("semisynth-section"), get("sensitivity-section")
assert len(s10) == 6 and len(s11) == 5, (len(s10), len(s11))
D, C = make(n_pv=30, cohorts=True)
ns = dict(D=D, C=C, A=None, train_seeds=[0], DEVICE="cpu", KALMAN_CFG=kf_mod.KalmanConfig(), plt=plt,
          display=display, os=os, pickle=pickle)
for name, cells in (("10", s10), ("11", s11)):
    for i, src in enumerate(cells):
        src = src.replace("PLACEMENT_SEEDS = [0, 1, 2]", "PLACEMENT_SEEDS = [0, 1]")
        src = src.replace("epochs=15, train_stride=1", "epochs=5, train_stride=2")
        src = src.replace("qs=(0.95, 0.97, 0.98, 0.99, 0.995, 0.998, 0.999), min_lens=(1, 2, 3, 4, 6)",
                          "qs=(0.95, 0.99, 0.995, 0.999), min_lens=(1, 3)")
        t0 = time.time(); exec(compile(src, f"cell{name}_{i}", "exec"), ns); print(f"--- section {name} cell {i} ok ({time.time() - t0:.1f}s)")
sens = ns["sens"]
rep = sens.tables["reproduction"]
print(rep.round(3).to_string(index=False))
assert np.allclose(rep.iloc[:, 1].values, rep.iloc[:, 2].values, equal_nan=True)
assert sens.meta["weights_unchanged"] and bool(ns["ss_res"].tables["integrity"]["unchanged"].all())
assert os.path.exists("results/sensitivity/sensitivity_tables.tex") and os.path.exists("results/sensitivity/sensitivity_results.zip")
print(sens.tables["fa_matched"].round(3).to_string(index=False))
print("OK")
