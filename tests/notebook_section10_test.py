"""Executes the code cells of notebook section 10 on the synthetic fixture (not HKUST data)."""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, nbformat
from IPython.display import display
from _semisynth_fixture import make
import pv_clustering as clu, pv_kalman as kf_mod
nb = nbformat.read(os.path.join(ROOT, "pv-stgnn-fuzzy-multi-full.ipynb"), as_version=4)
cells = [c.source for c in nb.cells if c.cell_type == "code" and "semisynth-section" in c.metadata.get("tags", [])]
assert len(cells) == 6, len(cells)
D, C = make(n_pv=30, cohorts=True)
ns = dict(D=D, C=C, A=None, train_seeds=[0], DEVICE="cpu", KALMAN_CFG=kf_mod.KalmanConfig(), plt=plt, display=display)
fast = os.environ.get("FAST", "1") == "1"
for i, src in enumerate(cells):
    if fast:
        src = src.replace("PLACEMENT_SEEDS = [0, 1, 2]", "PLACEMENT_SEEDS = [0, 1]")
        src = src.replace("epochs=15, train_stride=1", "epochs=5, train_stride=2")
    t0 = time.time(); exec(compile(src, f"cell10_{i}", "exec"), ns); print(f"--- cell {i} ok ({time.time() - t0:.1f}s)")
r = ns["ss_res"]
assert bool(r.tables["integrity"]["unchanged"].all())
assert os.path.exists("results/semisynth/semisynth_tables.tex") and os.path.exists("results/semisynth/semisynth_results.zip")
print(sorted(r.tables)); print("OK")
