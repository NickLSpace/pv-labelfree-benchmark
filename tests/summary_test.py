"""Section 17 (summary table of classification metrics) on fixture results.

Uses the pickled results of tests/modal_test.py (out_modal) and tests/fmm_test.py (out_fmm) when they exist
and, optionally, a pickle with {ss_res, fx_res, fcm_res, val_res} from sections 10-14 (env SUMMARY_PKL).
Checks: missing sections are skipped, CIs contain the point estimate, paired p-values / Holm / deltas are
consistent and reproducible, protocol column filled, metrics in [0, 1], English only,
CSV / LaTeX written and (if pdflatex is available) the LaTeX compiles."""
import os, sys, pickle, re, shutil, subprocess, tempfile, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import numpy as np, pandas as pd
import pv_metrics_summary as pms


def _load(p):
    return pickle.load(open(p, "rb")) if p and os.path.exists(p) else None


R = _load(os.environ.get("SUMMARY_PKL", "")) or {}
modal = _load(os.path.join(HERE, "out_modal", "modal_result.pkl"))
fmm = _load(os.path.join(HERE, "out_fmm", "modal_result.pkl"))

# no results at all -> empty table, no crash
E = pms.summary_table(None, None, None, None, None, None)
assert E.empty and list(E.columns) == pms.COLUMNS, E.columns

S = pms.summary_table(R.get("ss_res"), R.get("fx_res"), R.get("fcm_res"), R.get("val_res"), modal, fmm)
if S.empty:
    print("no fixture results found (run tests/modal_test.py and tests/fmm_test.py first) - only the empty case checked")
    sys.exit(0)
assert S["Protocol"].isin(pms.PROTOCOLS).all()
for c in ("Recall", "Precision", "F1", "Macro-F1 (cause)", "Nuisance share", "Cause accuracy", "FPR (no injection)"):
    v = pd.to_numeric(S[c], errors="coerce").dropna()
    assert ((v >= 0) & (v <= 1)).all(), c
if modal is not None:
    assert (S["Protocol"] == "P15").any()
if fmm is not None:
    assert (S["Protocol"] == "P16").any() and S["Model"].str.contains("STGNN-fmm").any()
assert not S["Model"].str.contains(r"\(no ").any()            # ablations off by default
SA = pms.summary_table(R.get("ss_res"), R.get("fx_res"), R.get("fcm_res"), R.get("val_res"), modal, fmm, ablations=True)
assert len(SA) >= len(S)
T = pms.tables(S)
assert set(T) == {"summary", "protocols", "best"}
B = T["best"]; assert set(B["Protocol"]) <= set(S["Protocol"])

# ---- confidence intervals and paired comparisons (pv_metrics_stats)
import pv_metrics_stats as pst
assert pst.compute(E)["ci"].empty and pst.compute(E)["paired"].empty          # nothing -> empty, no crash
kw = dict(ss_res=R.get("ss_res"), fx_res=R.get("fx_res"), fcm_res=R.get("fcm_res"), val_res=R.get("val_res"),
          modal_res=modal, fmm_res=fmm)
ST = pst.compute(S, n_boot=500, n_perm=500, **kw)
CI = ST["ci"]; assert len(CI)
ok = CI["CI low"].notna()
assert (CI.loc[ok, "CI low"] <= CI.loc[ok, "Value"] + 1e-9).all() and (CI.loc[ok, "CI high"] >= CI.loc[ok, "Value"] - 1e-9).all()
rate = ok & ~CI["Metric"].str.startswith("MAE")
assert ((CI.loc[rate, "CI low"] >= 0) & (CI.loc[rate, "CI high"] <= 1)).all()
M = CI.merge(S.melt(id_vars=["Protocol", "Task", "Model"], var_name="Metric", value_name="S"),
             on=["Protocol", "Task", "Model", "Metric"])
assert np.allclose(M["Value"].astype(float), M["S"].astype(float), equal_nan=True), "CI point estimate != summary value"
P = ST["paired"]
if len(P):
    for c in ("p (permutation)", "p (Holm)"):
        v = P[c].dropna(); assert ((v > 0) & (v <= 1)).all(), c
    assert (P["p (Holm)"].fillna(1) >= P["p (permutation)"].fillna(1) - 1e-12).all()
    assert np.allclose(P["Delta (A - B)"], P["A"] - P["B"], equal_nan=True)
    assert P["Result"].isin(["A better", "B better", "no difference shown"]).all()
    if modal is not None:
        assert ((P["Protocol"] == "P15") & (P["Comparison (A vs B)"] == "PR+ & modal vs PR+")).any()
ST2 = pst.compute(S, n_boot=500, n_perm=500, **kw)                  # reproducible with the fixed seed
assert ST2["paired"].equals(ST["paired"]) and ST2["ci"].equals(ST["ci"])
T = pms.tables(S, ST)
assert {"summary_ci", "paired"} <= set(T) and "Value 95% CI" in T["best"]

cyr = re.compile("[А-Яа-яЁё]")
for k, t in T.items():
    assert not cyr.search(t.to_string()), k

out = tempfile.mkdtemp()
paths = pms.export_latex(S, out, ST)
for k in ("summary", "protocols", "best", "summary_ci", "paired", "all"):
    assert os.path.exists(paths[k]), k
    txt = open(paths[k], encoding="utf-8").read()
    assert not cyr.search(txt) and "<NA>" not in txt and "nan" not in txt.lower().split(), k
if shutil.which("pdflatex"):
    open(os.path.join(out, "doc.tex"), "w").write(
        "\\documentclass[10pt,a4paper]{article}\\usepackage[margin=2cm]{geometry}"
        "\\usepackage{booktabs,longtable,array,graphicx,pdflscape}\\begin{document}"
        "\\input{summary_metrics_all.tex}\\end{document}")
    r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "doc.tex"], cwd=out, capture_output=True, text=True,
                       timeout=120)
    assert r.returncode == 0 and "! " not in r.stdout, r.stdout[-2000:]
    print("LaTeX compiled:", os.path.join(out, "doc.pdf"))
print(S.groupby("Protocol").size().to_string())
print("summary_test OK")
