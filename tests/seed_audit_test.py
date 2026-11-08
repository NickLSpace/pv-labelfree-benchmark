"""Seed audit of all final tables (notebook sections 10-18, 20, 21) on the synthetic fixture.

Executes the notebook code cells of sections 10-18, 20 and 21 unchanged except for short training
(epochs / stride / hidden), two placements and fewer bootstrap replicates, with the notebook default
train_seeds = [0, 1, 42, 77, 4700].  Section 19 needs the real data and is not executed.
Checks, for every saved result:
  * meta seeds = all five seeds;
  * every model-based method has rows for all five seeds in the event / point tables;
  * the exported tables are computed from all five seeds: removing one seed from the result changes
    the method table of the protocol (the tables are not built from a subset of seeds).
Runs in four processes to stay within 8 GB of RAM (shared work dir SEED_AUDIT_WORK keeps the caches):
    SEED_AUDIT_MODE=A      sections 10-14, 18, 20
    SEED_AUDIT_MODE=B      section 15        (section 10: frozen models only)
    SEED_AUDIT_MODE=C      section 16        (section 15: data and frozen models only)
    SEED_AUDIT_MODE=audit  sections 17, 21 on the pickled results + checks
Writes tests/out_seed_audit/seed_audit.csv, seed_sensitivity.csv, seed_check.csv, modal_strengths.csv.
Note: the background table of P10 has rows only for seeds with background alarms, and the labels table
carries the placement seed; P10/P18 estimators (SPS / Kalman) and P20 (STGNN eta = mean over seeds) do not
have per-seed rows, so their method tables do not change when one seed is removed from the event frames.
"""
import os, sys, time, pickle, tempfile, copy, re, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, nbformat, numpy as np, pandas as pd
from IPython.display import display
import _semisynth_fixture as fxt
import pv_modal_data as md
import pv_kalman as kf_mod

SEEDS = [0, 1, 42, 77, 4700]
OUT = os.path.join(HERE, "out_seed_audit"); os.makedirs(OUT, exist_ok=True)
nb = nbformat.read(os.path.join(ROOT, "pv-stgnn-fuzzy-multi-full.ipynb"), as_version=4)
hdr = [i for i, c in enumerate(nb.cells) if c.cell_type == "markdown" and re.match(r"##\s+\d+\.", c.source)]
sec_of = {}
for i, c in enumerate(nb.cells):
    h = max([j for j in hdr if j <= i], default=None)
    if h is not None:
        sec_of[i] = int(re.match(r"##\s+(\d+)\.", nb.cells[h].source).group(1))
# modes (one process each, to stay within 8 GB):  A = sections 10-14, 18, 20;  B = 10 (models only), 15, 16;
# audit = sections 17, 21 on the pickled results of A and B, then the seed checks
MODE = os.environ.get("SEED_AUDIT_MODE", "A")
SECS = {"A": (10, 11, 12, 13, 14, 18, 20), "B": (10, 15), "C": (10, 15, 16), "audit": (17, 21)}[MODE]
cells = [(sec_of[i], i, c.source) for i, c in enumerate(nb.cells)
         if c.cell_type == "code" and sec_of.get(i) in SECS]
if MODE in ("B", "C"):       # section 10: only the parameters and the frozen models (cells 66, 68)
    cells = [x for x in cells if x[0] != 10 or "frozen_models" in x[2] or "MODEL_SEEDS =" in x[2]]
if MODE == "C":              # section 15: only parameters, inverter data and frozen models (no evaluation)
    cells = [x for x in cells if x[0] != 15 or "run_modal" not in x[2] and "modal_res" not in x[2]]
print("cells:", {s: sum(1 for x in cells if x[0] == s) for s in sorted({x[0] for x in cells})}, flush=True)

work = os.environ.get("SEED_AUDIT_WORK") or tempfile.mkdtemp(); os.makedirs(work, exist_ok=True); os.chdir(work)
D, C = fxt.make(n_pv=30, cohorts=True); fxt.add_shading(D)
inv = os.path.join(work, "inv"); os.makedirs(inv, exist_ok=True)
Mfx = fxt.make_modal(D)
md.load_inverters = lambda d, D_, *a, **k: Mfx
md.load_site_energy = lambda d, D_, *a, **k: {}
rng = np.random.default_rng(0)
idx = pd.date_range("2023-06-01", periods=6 * 24 * 60, freq="1min")
g = np.clip(np.sin(np.pi * (idx.hour + idx.minute / 60 - 6) / 12), 0, None) * 900
df_all = pd.DataFrame({"Irradiance (W/m2)": g}, index=idx)
for c in ("power(W)_A", "power(W)_B"):
    v = pd.Series(np.nan, index=idx); q = idx.minute % 15 == 0
    v[q] = g[q] * 5 * (1 + rng.normal(0, .02, q.sum())); df_all[c] = v
ns = dict(D=D, C=C, A=None, train_seeds=list(SEEDS), DEVICE="cpu", KALMAN_CFG=kf_mod.KalmanConfig(), plt=plt,
          display=display, os=os, pickle=pickle, pd=pd, np=np, df_all=df_all, pv2_cols_list=[],
          path=work, pv_path_1=work)
REPL = [("epochs=15, train_stride=1", "epochs=1, train_stride=8"), ("hidden=64", "hidden=32"),
        ("pfm.FMMConfig(epochs=8, stride=3)", "pfm.FMMConfig(epochs=1, stride=12)"),
        ("PLACEMENT_SEEDS = [0, 1, 2]", "PLACEMENT_SEEDS = [0, 1]"),
        ("MODAL_PLACEMENTS = (0, 1, 2)", "MODAL_PLACEMENTS = (0,)" if MODE in ("B", "C") else "MODAL_PLACEMENTS = (0, 1)"),
        ("CONFIG_PLACEMENTS = (0, 1, 2)", "CONFIG_PLACEMENTS = (0, 1)"),
        ("RULSS_PLACEMENTS = (0, 1, 2)", "RULSS_PLACEMENTS = (0, 1)"),
        ("n_boot=2000", "n_boot=200"), ("n_perm=2000", "n_perm=200"), ("n_boot=1000", "n_boot=200"),
        ("path + '/Dataset/Time series dataset/PV generation dataset/PV stations with panel level optimizer/Inverter level dataset'", repr(inv))]
T0 = time.time()
KEEP = ("ss_res", "sens", "fx_res", "fcm_res", "val_res", "modal_res", "fmm_res", "cres", "rulss_res")
if MODE == "audit":
    for part in ("A", "B", "C"):
        ns.update(pickle.load(open(os.path.join(OUT, f"part_{part}.pkl"), "rb")))
    ns["sse"] = __import__("pv_semisynth_eval")
for sec, i, src in cells:
    for a, b in REPL:
        src = src.replace(a, b)
    t0 = time.time()
    exec(compile(src, f"cell{i}_s{sec}", "exec"), ns)
    for v in ("MODEL_SEEDS", "MODAL_SEEDS"):
        if v in src and v + " =" in src:
            assert list(ns[v]) == SEEDS, (v, ns[v])
    print(f"--- section {sec} cell {i} ok ({time.time() - t0:.0f}s, total {time.time() - T0:.0f}s)", flush=True)

if MODE != "audit":
    keep = {k: ns[k] for k in KEEP if ns.get(k) is not None and (MODE == "A" or (MODE == "B" and k == "modal_res") or (MODE == "C" and k == "fmm_res"))}
    for k, v in keep.items():                 # drop large raw arrays that the tables do not use
        pass
    pickle.dump(keep, open(os.path.join(OUT, f"part_{MODE}.pkl"), "wb"))
    print("saved", sorted(keep), "in %.0fs" % (time.time() - T0)); sys.exit(0)

RES = {"P10/P11 ss_res": "ss_res", "P11 sens": "sens", "P12 fx_res": "fx_res", "P13 fcm_res": "fcm_res",
       "P14 val_res": "val_res", "P15 modal_res": "modal_res", "P16 fmm_res": "fmm_res", "P18 cres": "cres",
       "P20 rulss_res": "rulss_res"}
S = {str(s) for s in SEEDS}
NON_MODEL = re.compile(r"^(pr|prp|prp_|pr_|rule|rules|modal_rule|fuzzy|mamdani|sps|kalman|ols|theil|stl|yoy|pr_rule|prp_rule|prp_m|pr_m)", re.I)


def seed_cols(df):
    return [c for c in ("model_seed", "seed") if c in df.columns]


rows = []
for name, var in RES.items():
    r = ns.get(var)
    if r is None:
        rows.append({"Result": name, "Status": "missing"}); continue
    meta = getattr(r, "meta", {}) or {}
    ms = meta.get("model_seeds") or meta.get("seeds") or meta.get("stgnn_seeds") or []
    ms = {str(x) for x in ms} - {"-", "None", "nan"}
    frames = {k: v for k, v in vars(r).items() if isinstance(v, pd.DataFrame)}
    for k, v in getattr(r, "tables", {}).items():
        if isinstance(v, pd.DataFrame):
            frames[f"tables.{k}"] = v
    seen, short = {}, []
    for k, df in frames.items():
        for sc in seed_cols(df):
            mcol = "method" if "method" in df.columns else None
            if mcol is None:
                got = {str(x) for x in df[sc].unique()} - {"-", "None", "nan"}
                seen[k] = len(got)
                if got and got != S:
                    short.append(f"{k}: {sorted(got)}")
                continue
            for m, g in df.groupby(mcol):
                got = {str(x) for x in g[sc].unique()} - {"-", "None", "nan"}
                if not got:
                    continue
                seen[f"{k}[{m}]"] = len(got)
                if got != S:
                    short.append(f"{k}[{m}]: {sorted(got)}")
    rows.append({"Result": name, "Status": "ok" if (ms == S and not short) else "CHECK",
                 "meta seeds": len(ms), "frames with seeds": len(seen),
                 "min seeds per method": min(seen.values()) if seen else None, "short": "; ".join(short)[:400]})
A = pd.DataFrame(rows)
print(A.to_string(index=False))

# sensitivity of the method tables to one seed: drop seed 4700 from every event / point frame
import pv_metrics_summary as pms
kw = dict(alpha=0.05, n_boot=200, n_perm=200)
args = dict(ss_res="ss_res", fx_res="fx_res", fcm_res="fcm_res", val_res="val_res", modal_res="modal_res",
            fmm_res="fmm_res", cfg_res="cres", rul_res="rulss_res")
full = pms.method_tables(**{k: ns.get(v) for k, v in args.items()}, **kw)


def drop_seed(r, s="4700"):
    r2 = copy.copy(r)
    for k, v in list(vars(r).items()):
        if isinstance(v, pd.DataFrame):
            for sc in seed_cols(v):
                v = v[v[sc].astype(str) != s]
            setattr(r2, k, v)
    return r2


sens_rows = []
for key in full:
    red = {k: (drop_seed(ns[v]) if ns.get(v) is not None else None) for k, v in args.items()}
    t4 = pms.method_tables(**red, **kw).get(key)
    a, b = full[key].data, (t4.data if t4 is not None else None)
    num = [c for c in a.columns if pd.api.types.is_numeric_dtype(a[c])]
    changed = b is None or not np.allclose(a[num].to_numpy(float), b[num].to_numpy(float), equal_nan=True) \
        if b is not None and b.shape == a.shape else True
    sens_rows.append({"Method table": key, "changes without seed 4700": bool(changed)})
SENS = pd.DataFrame(sens_rows)
print(SENS.to_string(index=False))
A.to_csv(os.path.join(OUT, "seed_audit.csv"), index=False); SENS.to_csv(os.path.join(OUT, "seed_sensitivity.csv"), index=False)
for f in ("results/summary/seed_check.csv", "results/modal/modal_strengths.csv"):
    if os.path.exists(f):
        print(f); print(pd.read_csv(f).to_string(index=False))
        pd.read_csv(f).to_csv(os.path.join(OUT, os.path.basename(f)), index=False)
print("tables exported:", sorted(os.path.relpath(os.path.join(dp, f), work) for dp, _, fs in os.walk("results") for f in fs if f.endswith(".tex"))[:200])
print("work dir:", work)
print("seed_audit_test DONE in %.0fs" % (time.time() - T0))
