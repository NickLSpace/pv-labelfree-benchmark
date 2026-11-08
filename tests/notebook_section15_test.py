"""Executes notebook section 15 (code cells) on the synthetic fixture: (1) no inverter folder ->
the section is skipped cleanly; (2) inverter data present (fixture ModalData injected in place of the
HKUST loader), STGNN-mm trained once (2 epochs) and frozen, full stages 0-4."""
import os, sys, time, pickle, shutil, tempfile
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, nbformat, numpy as np, pandas as pd
from IPython.display import display
import _semisynth_fixture as fxt
import pv_modal_data as md
nb = nbformat.read(os.path.join(ROOT, "pv-stgnn-fuzzy-multi-full.ipynb"), as_version=4)
cells = [c.source for c in nb.cells if c.cell_type == "code" and "modal-section" in c.metadata.get("tags", [])]
assert len(cells) == 7, len(cells)
tags = [tuple(c.metadata.get("tags", [])) for c in nb.cells]
first = lambda t: next(i for i, x in enumerate(tags) if t in x)
assert first("value-section") < first("modal-section") and nb.cells[-1].source.startswith(("## Литература", "## References"))
assert "pv_modal_eval.py" in nb.cells[21].source
# sections 1-14 unchanged apart from the %run list
work = tempfile.mkdtemp(); os.chdir(work)
D, C = fxt.make(cohorts=True)

def run(ns):
    for i, src in enumerate(cells):
        src = src.replace("epochs=15, train_stride=1", "epochs=2, train_stride=8").replace("hidden=64", "hidden=32")
        src = src.replace("MODAL_PLACEMENTS = (0, 1, 2)", "MODAL_PLACEMENTS = (0,)")
        t0 = time.time(); exec(compile(src, f"cell15_{i}", "exec"), ns)
        print(f"--- section 15 cell {i} ok ({time.time() - t0:.1f}s)", flush=True)

import pv_semisynth_eval as sse
ss_models = sse.frozen_models(D, C, seeds=[0], cache_dir="results/cache", hidden=32, epochs=2, train_stride=8, device="cpu")
base = dict(D=D, C=C, train_seeds=[0], DEVICE="cpu", plt=plt, display=display, pd=pd, np=np, ss_models=ss_models)
# (1) missing folder
ns = dict(base, path="/nonexistent", pv_path_1="/nonexistent"); run(ns)
assert ns["MODAL_OK"] is False
# (2) fixture inverter data
inv = os.path.join(work, "inv"); os.makedirs(inv)
Mfx = fxt.make_modal(D)
md.load_inverters = lambda d, D_, *a, **k: Mfx
md.load_site_energy = lambda d, D_, *a, **k: {}
ns = dict(base, path=work, pv_path_1=work)
ns_src = [s.replace("path + '/Dataset/Time series dataset/PV generation dataset/PV stations with panel level optimizer/Inverter level dataset'", repr(inv)) for s in cells]
cells[:] = ns_src
run(ns)
r = ns["modal_res"]
assert ns["MODAL_OK"] and {"mm_early", "mm_graph", "stgnn", "prp_m"} <= set(r.events["method"]), set(r.events["method"])
assert "verdict" in r.tables and os.path.exists(ns["modal_zip"])
print("notebook section 15 OK")
