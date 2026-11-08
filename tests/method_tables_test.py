"""Section 21: per-protocol method tables (pv_metrics_summary.method_tables).

Uses the saved results of the other tests (no training):
  tests/results/semisynth/semisynth_result.pkl   (P10, semisynth_test.py)
  tests/results/fixes/fixes_result.pkl           (P12, notebook_section12_13_test.py)
  tests/results/fcm/fcm_result.pkl               (P13)
  tests/results/value/value_result.pkl           (P14)
  tests/out_modal/modal_result.pkl               (P15, modal_test.py)
  tests/out_fmm/modal_result.pkl                 (P16, fmm_test.py)
  tests/out_config/plain/config_result.pkl       (P18, config_test.py)
  $RULSS_PKL (optional)                          (P20, pickled result of pv_rul_semisynth.run_rul_semisynth)
Missing files are skipped.  Checks: bold = best of the column, underline only where p_Holm >= alpha,
English texts, CSV / LaTeX export and LaTeX compilation (if pdflatex is available).
"""
import os, sys, pickle, shutil, subprocess, tempfile, warnings

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
import pv_metrics_summary as pms

FILES = {"ss_res": "results/semisynth/semisynth_result.pkl", "fx_res": "results/fixes/fixes_result.pkl",
         "fcm_res": "results/fcm/fcm_result.pkl", "val_res": "results/value/value_result.pkl",
         "modal_res": "out_modal/modal_result.pkl", "fmm_res": "out_fmm/modal_result.pkl",
         "cfg_res": "out_config/plain/config_result.pkl"}


def load():
    R = {}
    for k, f in FILES.items():
        p = os.path.join(HERE, f)
        R[k] = pickle.load(open(p, "rb")) if os.path.exists(p) else None
    p = os.environ.get("RULSS_PKL")
    R["rul_res"] = pickle.load(open(p, "rb")) if p and os.path.exists(p) else None
    return R


def ascii_only(s):
    return all(ord(ch) < 128 or ch in "βΔη±≥≤×–—·°" for ch in str(s))


def check_table(k, t, alpha):
    assert len(t.data) > 0, k
    assert ascii_only(t.title) and ascii_only(t.note), k
    for c in t.data.columns:
        assert ascii_only(c), (k, c)
    for m in t.data["Method"]:
        assert ascii_only(m), (k, m)
    groups = t.data["Group"] if "Group" in t.data else None
    for c in t.metric_cols:
        rule = t.better.get(c)
        for g in ([None] if groups is None else list(dict.fromkeys(groups))):
            idx = t.data.index if g is None else t.data.index[groups == g]
            mk = t.marks.loc[idx, c]
            if rule is None:
                assert (mk == "").all(), (k, c)
                continue
            score = pms._score(t.data.loc[idx, c], rule)
            ok = score.dropna()
            if len(ok) < 2 or np.isclose(ok.max(), ok.min()):     # nothing to highlight
                assert (mk == "").all(), (k, c, g)
                continue
            top = np.isclose(score.fillna(-np.inf), ok.max(), rtol=0, atol=1e-12)
            assert (mk[top] == pms.BEST).all() and (mk[~top] != pms.BEST).all(), (k, c, g)
            if not t.tested.get(c, False):
                assert (mk != pms.TIE).all(), (k, c, "underline without test")
    P = t.pvals
    if len(P):
        assert P["p (Holm)"].between(0, 1).all() and (P["p (Holm)"] >= P["p"] - 1e-12).all(), k
        for _, r in P.iterrows():
            sel = t.data["Method"] == r["Method"]
            if groups is not None:
                sel &= groups == r["Group"]
            i = t.data.index[sel][0]
            assert t.marks.loc[i, r["Column"]] == (pms.TIE if r["p (Holm)"] >= alpha else ""), (k, r.to_dict())
    # every underline is backed by a test
    n_tie = int((t.marks == pms.TIE).values.sum())
    assert n_tie == int((P["p (Holm)"] >= alpha).sum()) if len(P) else n_tie == 0, k


def main():
    R = load()
    print("inputs:", ", ".join(k for k, v in R.items() if v is not None))
    T = pms.method_tables(**R, alpha=0.05, n_boot=200, n_perm=300)
    assert T, "no tables"
    print("tables:", ", ".join(T))
    for k, t in T.items():
        check_table(k, t, 0.05)
        n_b = int((t.marks == pms.BEST).values.sum()); n_t = int((t.marks == pms.TIE).values.sum())
        print(f"  {k:22s} rows={len(t.data):2d} metrics={len(t.metric_cols):2d} bold={n_b:3d} underline={n_t:3d} "
              f"tests={len(t.pvals)}")
    # reproducible
    T2 = pms.method_tables(**R, alpha=0.05, n_boot=200, n_perm=300)
    for k in T:
        assert T[k].marks.equals(T2[k].marks), k
    # empty input -> empty dict
    assert pms.method_tables() == {}
    out = tempfile.mkdtemp()
    F = pms.export_method_tables(T, out)
    for k in T:
        assert os.path.exists(os.path.join(out, f"methods_{k}.csv")), k
        assert os.path.exists(os.path.join(out, f"methods_{k}.tex")), k
    tex = open(F["all"]).read()
    assert "\\textbf{" in tex and ascii_only(tex.replace("$", ""))
    if shutil.which("pdflatex"):
        d = tempfile.mkdtemp()
        open(os.path.join(d, "doc.tex"), "w").write(
            "\\documentclass[10pt,twocolumn]{article}\\usepackage[margin=1.8cm]{geometry}"
            "\\usepackage{booktabs,array,graphicx,amsmath}\\begin{document}"
            f"\\input{{{F['all']}}}\\end{{document}}\n")
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "doc.tex"], cwd=d, capture_output=True,
                           text=True, stdin=subprocess.DEVNULL, timeout=180)
        assert r.returncode == 0 and os.path.exists(os.path.join(d, "doc.pdf")), r.stdout[-2000:]
        print("LaTeX compiled:", os.path.join(d, "doc.pdf"), "overfull boxes:", r.stdout.count("Overfull"))
    print("method_tables_test: OK")


if __name__ == "__main__":
    main()
