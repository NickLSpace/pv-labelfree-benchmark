"""Executes notebook section 16 (code cells) on the synthetic fixture after section 15:
(1) section 15 skipped -> section 16 skipped cleanly; (2) fixture inverter data + injected shading,
STGNN-fmm trained once (2 + 2 epochs) and frozen, run_modal with the new methods."""
import os, sys, time, shutil, tempfile
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, nbformat, numpy as np, pandas as pd
from IPython.display import display
import _semisynth_fixture as fxt
import pv_modal_data as md
nb = nbformat.read(os.path.join(ROOT, "pv-stgnn-fuzzy-multi-full.ipynb"), as_version=4)
c15 = [c.source for c in nb.cells if c.cell_type == "code" and "modal-section" in c.metadata.get("tags", [])]
c16 = [c.source for c in nb.cells if c.cell_type == "code" and "fmm-section" in c.metadata.get("tags", [])]
assert len(c16) == 7, len(c16)
tags = [tuple(c.metadata.get("tags", [])) for c in nb.cells]
first = lambda t: next(i for i, x in enumerate(tags) if t in x)
last = lambda t: max(i for i, x in enumerate(tags) if t in x)
assert last("modal-section") < first("fmm-section") and nb.cells[-1].source.startswith(("## Литература", "## References"))
assert "pv_stgnn_factorized.py" in nb.cells[21].source and "pv_shading.py" in nb.cells[21].source
work = tempfile.mkdtemp(); os.chdir(work)
D, C = fxt.make(cohorts=True); fxt.add_shading(D)

def run(ns, cells, tag):
    for i, src in enumerate(cells):
        src = src.replace("epochs=15, train_stride=1", "epochs=2, train_stride=8").replace("hidden=64", "hidden=32")
        src = src.replace("MODAL_PLACEMENTS = (0, 1, 2)", "MODAL_PLACEMENTS = (0,)")
        src = src.replace("pfm.FMMConfig(epochs=8, stride=3)", "pfm.FMMConfig(epochs=2, stride=12)")
        t0 = time.time(); exec(compile(src, f"cell{tag}_{i}", "exec"), ns)
        print(f"--- section {tag} cell {i} ok ({time.time() - t0:.1f}s)", flush=True)

import pv_semisynth_eval as sse
ss_models = sse.frozen_models(D, C, seeds=[0], cache_dir="results/cache", hidden=32, epochs=2, train_stride=8, device="cpu")
base = dict(D=D, C=C, train_seeds=[0], DEVICE="cpu", plt=plt, display=display, pd=pd, np=np, ss_models=ss_models)
# (1) no inverter folder: both sections skipped
ns = dict(base, path="/nonexistent", pv_path_1="/nonexistent"); run(ns, c15, 15); run(ns, c16, 16)
assert ns["MODAL_OK"] is False and ns["FMM_OK"] is False
# (2) fixture inverter data
inv = os.path.join(work, "inv"); os.makedirs(inv)
Mfx = fxt.make_modal(D)
md.load_inverters = lambda d, D_, *a, **k: Mfx
md.load_site_energy = lambda d, D_, *a, **k: {}
ns = dict(base, path=work, pv_path_1=work)
c15 = [s.replace("path + '/Dataset/Time series dataset/PV generation dataset/PV stations with panel level optimizer/Inverter level dataset'", repr(inv)) for s in c15]
run(ns, c15, 15)
fp15 = {v: {s: m.fingerprint() for s, m in mods.items()} for v, mods in ns["mm_models"].items()}
run(ns, c16, 16)
r = ns["fmm_res"]
assert ns["FMM_OK"] and {"mm_fact", "mm_fact_m", "prp_sk", "prp_skm", "mm_graph"} <= set(r.events["method"]), set(r.events["method"])
assert "new_obstruction" in set(r.events["scenario"])
assert "verdict" in r.tables and os.path.exists(ns["fmm_zip"])
assert {v: {s: m.fingerprint() for s, m in mods.items()} for v, mods in ns["mm_models"].items()} == fp15
# section 15 result object untouched by section 16
assert "mm_fact" not in set(ns["modal_res"].events["method"])
print("notebook section 16 OK")
