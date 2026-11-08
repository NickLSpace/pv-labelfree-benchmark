"""Summary of classification metrics over all models evaluated on (semi-)synthetic data.

Collects the already computed results of the notebook sections (no retraining, no new forward passes):

=====  ========  ==============================================================================
P10    sect. 10  frozen STGNN on semi-synthetic plant / fleet events (SPS and Kalman estimators)
P12    sect. 12  detector fixes "before / after" and simple baselines, equal false-alarm budget
P13    sect. 13  k-means vs fuzzy c-means (FCM) weather / technical clusters, equal budget
P14    sect. 14  STGNN vs PR rule and their fusion, equal alarm budget (conditions 2-4)
P15    sect. 15  modal scenarios (inverter data), equal clean alarm budget, cause classes
P16    sect. 16  as P15 + "new obstruction" scenario, shading mask, Kalman sd, STGNN-fmm
=====  ========  ==============================================================================

Each row: protocol, section, task, model, operating point, alarm budget, number of events and the
metrics that the protocol defines (missing metrics are empty, never imputed).  Metrics are comparable
WITHIN a protocol only: scenarios, event definitions and operating points differ between protocols
(see ``protocol_table``).  All texts are in English.

Usage (notebook, after the sections that were run)::

    import pv_metrics_summary as pms
    S = pms.summary_table(ss_res=ss_res, fx_res=fx_res, fcm_res=fcm_res, val_res=val_res,
                          modal_res=modal_res, fmm_res=fmm_res)
    pms.export_latex(S, "results/summary")

Per-protocol method tables (section 21; one table "methods x metrics" per protocol, best value in bold,
methods not significantly different from the best underlined: paired test on the same events, Holm over
the table)::

    T = pms.method_tables(ss_res=ss_res, fx_res=fx_res, fcm_res=fcm_res, val_res=val_res,
                          modal_res=modal_res, fmm_res=fmm_res, cfg_res=cres, rul_res=rulss_res)
    pms.show_method_tables(T); pms.export_method_tables(T, "results/summary")

Keys: P10_class, P10_estimators, P12_equal_budget, P12_operating_points, P13, P14, P15, P16,
P18_detectors, P18_estimators (P18 = sect. 18, configuration changes), P20 (sect. 20, RUL algorithms).
"""
from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np
import pandas as pd

COLUMNS = ["Protocol", "Section", "Task", "Model", "Operating point", "Alarm budget /1000 h", "Events",
           "Recall", "Precision", "F1", "Macro-F1 (cause)", "Nuisance share", "Cause accuracy",
           "FPR (no injection)", "Clean FA /1000 h", "Delay, h"]

def _scn_p10() -> str:
    import pv_semisynth_gen as ssg
    return "; ".join(v.title for v in ssg.SCENARIOS.values()) + " (durations: short 1-2 steps, day, week+)"


def _scn_p14() -> str:
    try:
        import pv_stgnn_value as sv
        return "As P10 + " + "; ".join(v["title"] for v in sv.HARD_SCENARIOS.values())
    except Exception:
        return "As P10 + hard scenarios"


def _scn_modal(extra: bool) -> str:
    import pv_semisynth_modal_gen as smg
    d = smg.ALL_SCENARIOS if extra else smg.MODAL_SCENARIOS
    return "; ".join(v["title"] + ("" if v["alarm"] else " (no alarm expected)") for v in d.values())


PROTOCOLS = {
    "P10": dict(section="10", data="Semi-synthetic: HKUST (or fixture) + injected plant / fleet events, 3 cohorts",
                scenarios=_scn_p10,
                operating_point="Threshold of the frozen STGNN from its own calibration (q = 0.995, POT)",
                metrics="Event recall; precision = detections overlapping an event / all detections near events; F1; "
                        "share of detected events with the correct fuzzy / crisp / k-means type; degradation: share "
                        "with evidence of a beta1 decrease (recall) and the same without injection (FPR)"),
    "P12": dict(section="12", data="As P10, same placements", scenarios=lambda: "As P10",
                operating_point="Equal false-alarm budget on control plants = rate of S0 at its operating point",
                metrics="Recall, precision, F1 at the budget (interpolated threshold curve); cause accuracy of detected events"),
    "P13": dict(section="13", data="As P10", scenarios=lambda: "As P10",
                operating_point="Equal false-alarm budget on control plants = k-means operating point",
                metrics="Recall, precision, F1 at the budget; cause accuracy of detected events (validation operating point)"),
    "P14": dict(section="14", data="As P10 + hard scenarios", scenarios=_scn_p14,
                operating_point="Equal alarm budget on clean data (all plants), reference budget of the section",
                metrics="Recall, precision, F1; FA on control plants"),
    "P15": dict(section="15", data="Semi-synthetic modal events on the inverter data (sites with inverter files)",
                scenarios=lambda: _scn_modal(False),
                operating_point="Equal clean alarm budget (events per 1000 scored hours, all plants), reference budget",
                metrics="Recall over fault events; nuisance share = alarms on no-alarm scenarios; macro-F1 over cause "
                        "classes (plant / sensor / grid / data); time to detection. Events include model seeds"),
    "P16": dict(section="16", data="As P15", scenarios=lambda: _scn_modal(True),
                operating_point="As P15", metrics="As P15"),
}


def protocol_table() -> pd.DataFrame:
    return pd.DataFrame([{"Protocol": k, "Section": v["section"], "Data": v["data"], "Scenarios": v["scenarios"](),
                          "Operating point": v["operating_point"], "Metrics": v["metrics"]} for k, v in PROTOCOLS.items()])


def _row(**kw) -> dict:
    r = {c: np.nan for c in COLUMNS}
    r.update({k: v for k, v in kw.items() if k in r})
    return r


def _f(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return np.nan
    return x if np.isfinite(x) else np.nan


def _pick(df: pd.DataFrame, *names):
    for n in names:
        if n in df.columns:
            return n
    return None


# ============================================================================ P10 (section 10)
def _p10(res) -> list:
    import pv_semisynth_gen as ssg
    import pv_semisynth_eval as sse
    E = res.events; rows = []
    if not len(E):
        return rows
    g = E[E["duration"] != "short"]
    tp, fp = g["n_tp_detections"].sum(), g["n_fp_near"].sum()
    rec = g["detected"].mean(); prec = tp / (tp + fp) if tp + fp > 0 else np.nan
    f1 = 2 * prec * rec / (prec + rec) if np.isfinite(prec) and prec + rec > 0 else np.nan
    fa = res.tables.get("false_alarms")
    clean = _f(fa["Alarms in clean data / 1000 h"].mean()) if fa is not None and len(fa) else np.nan
    seeds = res.meta.get("seeds") or sorted(E["model_seed"].unique())
    est = ", ".join(sse.EST_LABELS.get(e, e) for e in res.meta.get("estimators", []))
    rows.append(_row(Protocol="P10", Section="10", Task="Fault event detection (plant + fleet, day / week)",
                     Model="Frozen STGNN (multi-view, S0 detection)", **{"Operating point": "own calibration (q = 0.995)",
                     "Events": len(g), "Recall": rec, "Precision": prec, "F1": f1, "Clean FA /1000 h": clean,
                     "Delay, h": _f(g.loc[g["detected"], "delay_h"].median())}))
    # cause classification of detected events
    cls = res.classes
    det = cls[cls["kind"] == "detected"] if len(cls) else pd.DataFrame()
    if len(det):
        key = ["placement", "model_seed", "event_id"]
        m = det.merge(E[key + ["scenario"]], on=key)
        for col, lab in (("fuzzy", "Fuzzy classifier (Mamdani rules)"), ("crisp", "Crisp rules"), ("kmeans", "K-means anomaly clusters")):
            if col not in m or m[col].isna().all():
                continue
            ok = [r == ssg.SCENARIOS[s].expected_name for r, s in zip(m[col], m["scenario"])]
            v = m.assign(ok=ok).groupby(key)["ok"].mean() >= 0.5
            rows.append(_row(Protocol="P10", Section="10", Task="Cause classification of detected events",
                             Model=lab + " on frozen STGNN detections",
                             **{"Operating point": "own calibration (q = 0.995)", "Events": int(len(v)),
                                "Cause accuracy": float(v.mean())}))
        orc = cls[cls["kind"] == "oracle"]
        if len(orc):
            m = orc.merge(E[key + ["scenario"]], on=key)
            ok = [r == ssg.SCENARIOS[s].expected_name for r, s in zip(m["fuzzy"], m["scenario"])]
            v = m.assign(ok=ok).groupby(key)["ok"].mean() >= 0.5
            rows.append(_row(Protocol="P10", Section="10", Task="Cause classification (true event interval)",
                             Model="Fuzzy classifier (Mamdani rules), oracle interval",
                             **{"Operating point": "true event interval", "Events": int(len(v)), "Cause accuracy": float(v.mean())}))
    # degradation detection per estimator (SPS / Kalman)
    DG = res.degradation
    if len(DG) and "evidence_injected" in DG:
        for e, d in DG.groupby("estimator"):
            dn = lambda s: s.astype(str).str.contains("decrease").mean()
            rows.append(_row(Protocol="P10", Section="10", Task="Degradation detection (evidence of a beta1 decrease)",
                             Model=f"Frozen STGNN + {sse.EST_LABELS.get(e, e)} estimator",
                             **{"Operating point": "statistical evidence rule of the estimator", "Events": len(d),
                                "Recall": _f(dn(d["evidence_injected"])),
                                "FPR (no injection)": _f(dn(d["evidence_clean"])) if "evidence_clean" in d else np.nan}))
    return rows


# ============================================================================ P12-P14 (equal budget tables)
def _p12(res) -> list:
    rows = []; T = res.tables
    eb = T.get("equal_budget")
    if eb is not None and len(eb):
        bcol = _pick(eb, "FA budget (controls) /1000 h")
        for _, r in eb.iterrows():
            rows.append(_row(Protocol="P12", Section="12", Task="Fault event detection",
                             Model=r["Method"], **{"Operating point": "equal FA budget on control plants",
                             "Alarm budget /1000 h": _f(r[bcol]) if bcol else np.nan,
                             "Recall": _f(r.get(_pick(eb, "Recall", "recall"))),
                             "Precision": _f(r.get(_pick(eb, "Precision", "precision"))),
                             "F1": _f(r.get("F1")),
                             "Clean FA /1000 h": _f(r[bcol]) if bcol else np.nan}))
    n = len(res.events[(res.events["duration"] != "short")].drop_duplicates(["placement", "event_id"])) if len(res.events) else np.nan
    for r in rows:
        r["Events"] = n
    C = T.get("classification")
    if C is not None and len(C):
        a = C[C["Scenario"] == "All plant-level events"]
        if len(a):
            for c in [c for c in C.columns if c != "Scenario" and pd.api.types.is_float_dtype(C[c]) and c != "N"]:
                orc = c.lower().startswith("oracle")
                rows.append(_row(Protocol="P12", Section="12",
                                 Task="Cause classification (true event interval)" if orc else "Cause classification of detected events",
                                 Model=f"Fuzzy classifier: {c}", **{"Operating point": "true event interval" if orc else
                                                                    "validation operating point",
                                 "Cause accuracy": _f(a[c].iloc[0])}))
    return rows


def _p13(res) -> list:
    rows = []; T = res.tables
    eb = T.get("equal_budget")
    sr = T.get("scenario_recall")
    if eb is not None and len(eb):
        bcol = "FA budget (controls) /1000 h"
        ref = _f(sr[bcol].iloc[0]) if sr is not None and len(sr) and bcol in sr else np.nan
        sel = eb[np.isclose(eb[bcol].astype(float), ref)] if np.isfinite(ref) else eb
        if not len(sel):
            sel = eb
        for _, r in sel.iterrows():
            rows.append(_row(Protocol="P13", Section="13", Task="Fault event detection",
                             Model=f"STGNN + {r['Variant']} regimes / clusters",
                             **{"Operating point": "equal FA budget = k-means operating point",
                                "Alarm budget /1000 h": _f(r[bcol]), "Recall": _f(r["Recall"]),
                                "Precision": _f(r.get("Precision")), "F1": _f(r.get("F1")),
                                "Clean FA /1000 h": _f(r[bcol])}))
    n = len(res.events[(res.events["duration"] != "short")].drop_duplicates(["placement", "event_id"])) if len(res.events) and "duration" in res.events else np.nan
    for r in rows:
        r["Events"] = n
    C = T.get("classification")
    if C is not None and len(C):
        a = C[C["Scenario"] == "All plant-level events"]
        if len(a):
            for c in [c for c in C.columns if c != "Scenario" and pd.api.types.is_float_dtype(C[c]) and c != "N"]:
                rows.append(_row(Protocol="P13", Section="13", Task="Cause classification of detected events",
                                 Model=f"Fuzzy classifier, {c} clusters", **{"Operating point": "validation operating point",
                                 "Cause accuracy": _f(a[c].iloc[0])}))
    return rows


def _p14(res) -> list:
    rows = []; T = res.tables
    eb = T.get("equal_budget")
    if eb is None or not len(eb):
        return rows
    bcol = "Alarm budget (clean, all plants) /1000 h"
    ref = _f(res.meta.get("ref_budget"))
    sel = eb[np.isclose(eb[bcol].astype(float), ref)] if np.isfinite(ref) else eb
    if not len(sel):
        sel = eb
    E = res.events
    for _, r in sel.iterrows():
        rows.append(_row(Protocol="P14", Section="14", Task="Fault event detection",
                         Model=r["Method"], **{"Operating point": "equal alarm budget on clean data (all plants)",
                         "Alarm budget /1000 h": _f(r[bcol]), "Recall": _f(r["Recall"]),
                         "Precision": _f(r.get("Precision")), "F1": _f(r.get("F1")),
                         "Clean FA /1000 h": _f(r.get("FA on control plants /1000 h"))}))
    try:
        import pv_stgnn_value as sv
        Em = sv._main_events(E)
        n = Em.drop_duplicates([c for c in ("placement", "event_id") if c in Em]).shape[0]
    except Exception:
        n = np.nan
    for r in rows:
        r["Events"] = n
    return rows


# ============================================================================ P15 / P16 (modal)
def _modal(res, proto: str, ablations: bool) -> list:
    rows = []
    eb = res.tables.get("equal_budget")
    if eb is None or not len(eb):
        return rows
    ref = _f(res.meta.get("ref_budget", 1.0))
    sel = eb[np.isclose(eb["Budget /1000 h"].astype(float), ref)]
    sec = proto[1:]
    for _, r in sel.iterrows():
        abl = "(no " in str(r["Method"])
        if abl and not ablations:
            continue
        nf, nn = _f(r.get("n_fault")), _f(r.get("n_nonfault"))
        rows.append(_row(Protocol=proto, Section=sec,
                         Task="Modal fault detection + cause class" + (" (modality removed at test time)" if abl else ""),
                         Model=r["Method"], **{"Operating point": "equal clean alarm budget (all plants)",
                         "Alarm budget /1000 h": _f(r["Budget /1000 h"]),
                         "Events": (nf + nn) if np.isfinite(nf) and np.isfinite(nn) else nf,
                         "Recall": _f(r.get("recall")), "Macro-F1 (cause)": _f(r.get("macro_f1")),
                         "Nuisance share": _f(r.get("nuisance")), "Clean FA /1000 h": _f(r.get("Clean FA /1000 h")),
                         "Delay, h": _f(r.get("ttd_h"))}))
    return rows


# ============================================================================ public API
def summary_table(ss_res=None, fx_res=None, fcm_res=None, val_res=None, modal_res=None, fmm_res=None,
                  ablations: bool = False) -> pd.DataFrame:
    """Long table: one row per (protocol, task, model).  Results that are None are skipped."""
    rows = []
    for res, fn in ((ss_res, _p10), (fx_res, _p12), (fcm_res, _p13), (val_res, _p14)):
        if res is not None:
            rows += fn(res)
    if modal_res is not None:
        rows += _modal(modal_res, "P15", ablations)
    if fmm_res is not None:
        rows += _modal(fmm_res, "P16", ablations)
    S = pd.DataFrame(rows, columns=COLUMNS)
    if len(S):
        S["Events"] = pd.to_numeric(S["Events"], errors="coerce").round().astype("Int64")
    return S.reset_index(drop=True)


def best_by_protocol(S: pd.DataFrame) -> pd.DataFrame:
    """Best detector per protocol (max F1, else max recall at the budget; ties -> lower nuisance)."""
    out = []
    for p, g in S[S["Task"].str.contains("detection")].groupby("Protocol", sort=False):
        g = g[~g["Task"].str.contains("removed")]
        key = "F1" if g["F1"].notna().any() else ("Macro-F1 (cause)" if g["Macro-F1 (cause)"].notna().any() else "Recall")
        g = g.assign(_k=g[key].fillna(-1), _n=-g["Nuisance share"].fillna(0)).sort_values(["_k", "_n"], ascending=False)
        r = g.iloc[0]
        out.append({"Protocol": p, "Task": r["Task"], "Best model": r["Model"], "Criterion": key, "Value": r[key],
                    "Recall": r["Recall"]})
    return pd.DataFrame(out)


TITLES = {"summary": "Classification metrics of all models on (semi-)synthetic data; metrics are comparable within a protocol only",
          "protocols": "Evaluation protocols of the summary table",
          "best": "Best detector per protocol",
          "summary_ci": "Summary metrics with 95\\% confidence intervals (cluster bootstrap over injected events; "
                        "Wilson intervals for interpolated equal-budget points)",
          "paired": "Pre-specified paired comparisons on the same events: difference A - B, 95\\% paired bootstrap CI, "
                    "paired permutation p-value and Holm correction within each protocol"}


def best_with_ci(S: pd.DataFrame, CI: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """best_by_protocol + 95 % CI of the criterion and of recall (when a CI table is given)."""
    B = best_by_protocol(S)
    if CI is None or not len(CI) or not len(B):
        return B
    def ci(r, metric):
        c = CI[(CI["Protocol"] == r["Protocol"]) & (CI["Task"] == r["Task"]) & (CI["Model"] == r["Best model"])
               & (CI["Metric"] == metric)]
        if not len(c) or not np.isfinite(float(c["CI low"].iloc[0])):
            return ""
        return f"[{float(c['CI low'].iloc[0]):.3f}, {float(c['CI high'].iloc[0]):.3f}]"
    B.insert(B.columns.get_loc("Value") + 1, "Value 95% CI", [ci(r, r["Criterion"]) for _, r in B.iterrows()])
    B["Recall 95% CI"] = [ci(r, "Recall") for _, r in B.iterrows()]
    return B


def tables(S: pd.DataFrame, stats: Optional[Dict[str, pd.DataFrame]] = None) -> Dict[str, pd.DataFrame]:
    T = {"summary": S, "protocols": protocol_table(),
         "best": best_with_ci(S, stats.get("ci") if stats else None)}
    if stats:
        if len(stats.get("summary_ci", [])):
            T["summary_ci"] = stats["summary_ci"]
        if len(stats.get("paired", [])):
            T["paired"] = stats["paired"]
    return T


def show(S: pd.DataFrame, digits: int = 3, stats: Optional[Dict[str, pd.DataFrame]] = None):
    try:
        from IPython.display import display
    except ImportError:
        display = print
    for k, t in tables(S, stats).items():
        print(TITLES[k].replace("\\%", "%"))
        num = t.select_dtypes("number").columns
        display(t.style.format({c: f"{{:.{digits}f}}" for c in num if c not in ("Events",)}, na_rep="")
                if hasattr(t, "style") else t)


def _paired_tex(P: pd.DataFrame) -> pd.DataFrame:
    """Compact paired table for LaTeX: CI as text, p-values with a floor, empty columns removed."""
    f = lambda v: "--" if not np.isfinite(v) else f"{(0.0 if abs(v) < 5e-4 else v):.3f}"
    fp = lambda v: "--" if not np.isfinite(v) else ("< 0.001" if v < 0.001 else f"{v:.3f}")
    T = pd.DataFrame({"Protocol": P["Protocol"], "Comparison (A vs B)": P["Comparison (A vs B)"], "Metric": P["Metric"],
                      "Events": P["Events"], "A": P["A"].astype(float), "B": P["B"].astype(float),
                      "Delta (A - B)": P["Delta (A - B)"].astype(float),
                      "95% CI": [f"[{f(a)}, {f(b)}]" for a, b in zip(P["CI low"].astype(float), P["CI high"].astype(float))],
                      "p": [fp(v) for v in P["p (permutation)"].astype(float)],
                      "p (Holm)": [fp(v) for v in P["p (Holm)"].astype(float)],
                      "Only A / B hit": ["" if pd.isna(a) else f"{int(a)} / {int(b)}" for a, b in zip(P["Only A hit"], P["Only B hit"])],
                      "Clean FA A / B": P["Clean FA A / B"], "Result": P["Result"]})
    return T.drop(columns=[c for c in ("Only A / B hit", "Clean FA A / B") if (T[c] == "").all()])


def export_latex(S: pd.DataFrame, out_dir: str = "results/summary",
                 stats: Optional[Dict[str, pd.DataFrame]] = None) -> Dict[str, str]:
    """CSV + LaTeX (booktabs / longtable via pv_semisynth_eval.df_to_latex) for the three tables, plus
    one file ``summary_metrics_all.tex`` with all of them."""
    import pv_semisynth_eval as sse
    os.makedirs(out_dir, exist_ok=True)
    out, parts = {}, []
    for k, t in tables(S, stats).items():
        t.to_csv(os.path.join(out_dir, f"summary_{k}_raw.csv"), index=False) if k in ("summary_ci", "paired") else None
        t2 = _paired_tex(t) if k == "paired" else t.copy()
        if k == "summary_ci":
            t2 = t2.drop(columns=[c for c in t2.columns if (t2[c].astype(str) == "").all()])
            t2["CI method"] = t2["CI method"].map(lambda v: "B" if str(v).startswith("cluster") else
                                                  ("W" if str(v).startswith("Wilson") else "--"))
            t2 = t2.rename(columns={"CI method": "CI", "MAE Δβ1 (degradation), p.p. (95% CI)": "MAE Δβ1 degr., p.p. (95% CI)"})
            mcols = [c for c in t2.columns if c.endswith("(95% CI)")]
            for c in mcols:                     # value on the first line, CI on the second
                t2[c] = t2[c].map(lambda v: v.replace(" [", "\u2028[", 1) if isinstance(v, str) else v)
        if "Events" in t2:
            t2["Events"] = [v if isinstance(v, str) else (str(int(v)) if pd.notna(v) else "--") for v in t2["Events"]]
        for c in t2.columns:
            if not pd.api.types.is_numeric_dtype(t2[c]):
                t2[c] = t2[c].map(lambda v: v.replace(">", "\u2009gt\u2009").replace("<", "\u2009lt\u2009").replace("~", "≈")
                                  if isinstance(v, str) else v)
        if k == "summary":                 # operating point / section are in the protocol table
            t2 = t2.drop(columns=["Section", "Operating point"] +
                         [c for c in COLUMNS if c not in ("Protocol", "Model", "Task", "Section", "Operating point")
                          and t2[c].isna().all()])
        t2.replace("\u2028", " ", regex=True).to_csv(os.path.join(out_dir, f"summary_{k}.csv"), index=False)
        tex = sse.df_to_latex(t2, TITLES[k], f"tab:summary_{k}", digits=3,
                              long_rows=30 if k in ("summary", "summary_ci", "paired") else 100)
        if k == "summary":
            tex = tex.replace("p{2.5cm}", "p{3.6cm}", 2) if tex.count("p{2.5cm}") else tex
            tex = "\\begin{landscape}\n" + tex + "\n\\end{landscape}"   # 14 columns: landscape page
        if k == "summary_ci":
            import re
            al = "l>{\\raggedright\\arraybackslash}p{1.9cm}>{\\raggedright\\arraybackslash}p{2.5cm}r" + \
                 ">{\\raggedright\\arraybackslash}p{1.7cm}" * len(mcols) + "c"
            tex = re.sub(r"\\begin\{longtable\}\{[^\n]*\}\n", lambda m_: "\\begin{longtable}{" + al + "}\n", tex, count=1)
            tex = re.sub(r"\[([0-9.\-]+), ([0-9.\-]+)\]", r"[\1,~\2]", tex)   # keep the CI on one line
            tex = tex.replace("\u2028", "\\newline ").replace("\\setlength{\\tabcolsep}{2.5pt}", "\\setlength{\\tabcolsep}{2pt}", 1)
            tex = tex.replace("\\end{longtable}}", "\\multicolumn{" + str(len(t2.columns)) + "}{l}{CI: B = cluster bootstrap "
                              "over injected events; W = Wilson interval (interpolated equal-budget point or oracle "
                              "intervals); -- = not available.}\\\\\n\\end{longtable}}")
        if k in ("summary_ci", "paired"):
            if k == "paired":
                tex = tex.replace("p{2.5cm}", "p{4.2cm}", 1)
            tex = "\\begin{landscape}\n" + tex + "\n\\end{landscape}"
        if k == "protocols":
            tex = tex.replace("\\begin{tabular}{llllll}", "\\begin{tabular}{llP{2.4cm}P{4.0cm}P{2.8cm}P{4.6cm}}")
            tex = tex.replace("\\resizebox{\\textwidth}{!}{%", "{%").replace("\\small", "\\scriptsize\n\\setlength{\\tabcolsep}{3pt}")
        if k == "best":
            tex = tex.replace("\\begin{tabular}{llllrr}", "\\begin{tabular}{lP{4.2cm}P{5.0cm}P{2.2cm}rr}")
            tex = tex.replace("\\begin{tabular}{llllrlrl}", "\\begin{tabular}{lP{3.4cm}P{4.4cm}P{1.8cm}rlrl}")
        tex = tex.replace("P{", ">{\\raggedright\\arraybackslash}p{")
        tex = tex.replace("\u2009gt\u2009", "$>$").replace("\u2009lt\u2009", "$<$")
        p = os.path.join(out_dir, f"summary_{k}.tex")
        open(p, "w", encoding="utf-8").write(tex); out[k] = p; parts.append(tex)
    p = os.path.join(out_dir, "summary_metrics_all.tex")
    open(p, "w", encoding="utf-8").write("% Requires: booktabs, longtable, array, graphicx, pdflscape\n" + "\n\n".join(parts))
    out["all"] = p
    return out


# ============================================================================ method tables per protocol
# One table per protocol (or sub-protocol): methods x metrics on the SAME scenarios / events / budget.
#   bold       = best value of the column (within a row group, e.g. a scenario of P18 / P20);
#   underlined = not significantly different from the best: paired test on the same events
#                (paired permutation / sign-flip), Holm correction over all tests of the table, p_Holm >= alpha.
# Columns without event-level outcomes (interpolated equal-budget points, rates, medians, SD over seeds,
# coverage, bias, balanced accuracy) get the bold mark only; ``tested`` records which columns were tested.
# Tables are never mixed across protocols.  All texts are in English.
from dataclasses import dataclass, field  # noqa: E402
from typing import Callable, List  # noqa: E402

BEST, TIE = "best", "tie"


@dataclass
class MethodTable:
    protocol: str
    title: str
    data: pd.DataFrame                    # "Method" (+ optional "Group") + metric columns
    better: Dict[str, object]             # column -> "max" | "min" | "zero" | ("target", value)
    marks: pd.DataFrame = None            # same shape as data[metric columns]: "best" | "tie" | ""
    tested: Dict[str, bool] = field(default_factory=dict)
    note: str = ""
    pvals: pd.DataFrame = None            # tie tests: Group, Column, Best, Method, p, p (Holm)

    @property
    def metric_cols(self) -> List[str]:
        return [c for c in self.data.columns if c not in ("Method", "Group")]


def _score(v: pd.Series, how) -> pd.Series:
    """Higher score = better."""
    v = pd.to_numeric(v, errors="coerce").astype(float)
    if how == "max":
        return v
    if how == "min":
        return -v
    if how == "zero":
        return -v.abs()
    if isinstance(how, tuple) and how[0] == "target":
        return -(v - float(how[1])).abs()
    return pd.Series(np.nan, index=v.index)


def _finish(proto: str, title: str, data: pd.DataFrame, better: Dict[str, object],
            tests: Optional[Dict[str, Callable]] = None, alpha: float = 0.05, note: str = "") -> MethodTable:
    """Mark the best value per column (and group) and the methods not distinguishable from it."""
    tests = tests or {}
    data = data.reset_index(drop=True)
    cols = [c for c in data.columns if c not in ("Method", "Group")]
    marks = pd.DataFrame("", index=data.index, columns=cols)
    groups = data["Group"] if "Group" in data else pd.Series("", index=data.index)
    P = []
    for c in cols:
        how = better.get(c)
        if how is None:
            continue
        for g in groups.unique():
            ix = groups.index[groups == g]
            s = _score(data.loc[ix, c], how)
            ok = s.dropna()
            if len(ok) < 2 or np.isclose(ok.max(), ok.min()):
                continue
            top = ok.index[np.isclose(ok, ok.max(), rtol=0, atol=1e-12)]
            marks.loc[top, c] = BEST
            if c in tests:
                b = data.loc[top[0], "Method"]
                for i in ok.index.difference(top):
                    try:
                        p = tests[c](b, data.loc[i, "Method"], g)
                    except Exception:                       # no paired events -> not tested
                        p = np.nan
                    if p is not None and np.isfinite(p):
                        P.append(dict(Group=g, Column=c, Best=b, Method=data.loc[i, "Method"], p=float(p), _i=i))
    PV = pd.DataFrame(P, columns=["Group", "Column", "Best", "Method", "p", "_i"])
    if len(PV):
        import pv_metrics_stats as pst
        PV["p (Holm)"] = pst.holm(PV["p"]).to_numpy()
        for _, r in PV.iterrows():
            if r["p (Holm)"] >= alpha:
                marks.loc[int(r["_i"]), r["Column"]] = TIE
    tested = {c: c in tests for c in cols if better.get(c) is not None}
    return MethodTable(proto, title, data, better, marks, tested, note,
                       PV.drop(columns="_i") if len(PV) else PV.drop(columns="_i").assign(**{"p (Holm)": []}))


def _val(X: pd.DataFrame, metric: str) -> float:
    import pv_metrics_stats as pst
    if not len(X):
        return np.nan
    vf, cf = pst.METRICS[metric]
    return float(cf({k: v.sum() for k, v in vf(X).items()}))


def _ptest(X: Dict[str, pd.DataFrame], metric: str, n_boot: int, n_perm: int, key=None, sub=None):
    """Tie test between two methods on the same events (pv_metrics_stats.paired_test)."""
    import pv_metrics_stats as pst
    key = key or pst.DET_KEY

    def f(a, b, g):
        A, B = X[a], X[b]
        if sub is not None:
            A, B = sub(A, g), sub(B, g)
        if "model_seed" in key:
            A, B = pst._align_seeds(A, B)
        return pst.paired_test(A, B, key, metric, n_boot, n_perm).get("p", np.nan)
    return f


def _scn_title(s: str) -> str:
    try:
        import pv_semisynth_gen as ssg
        return ssg.SCENARIOS[s].title
    except Exception:
        return str(s)


# ---------------------------------------------------------------------------- P10
def _mt_p10(res, alpha, nb, npm) -> List[MethodTable]:
    import pv_metrics_stats as pst
    import pv_semisynth_gen as ssg
    import pv_semisynth_eval as sse
    out = []
    E, cls = res.events, res.classes
    key = pst.DET_KEY
    if len(cls) and len(E):
        c = cls[cls["kind"] == "detected"]
        m = c.merge(E[key + ["scenario"]].drop_duplicates(key), on=key)
        X, rows = {}, []
        for col, lab in pst.P10_CLS_LABEL.items():
            if col not in m or m[col].isna().all():
                continue
            ok = [r == ssg.SCENARIOS[s].expected_name for r, s in zip(m[col], m["scenario"])]
            X[lab] = (m.assign(ok=ok).groupby(key + ["scenario"])["ok"].mean() >= 0.5).reset_index()
        scn = [s for s in ssg.SCENARIOS if any(s in set(v["scenario"]) for v in X.values())]
        for lab, v in X.items():
            r = {"Method": lab}
            for s in scn:
                r[_scn_title(s)] = _val(v[v["scenario"] == s], "Cause accuracy")
            r["All detected events"] = _val(v, "Cause accuracy")
            rows.append(r)
        if rows:
            D = pd.DataFrame(rows)
            better = {k: "max" for k in D.columns if k != "Method"}
            tests = {_scn_title(s): _ptest(X, "Cause accuracy", nb, npm, sub=lambda Z, g, s=s: Z[Z["scenario"] == s])
                     for s in scn}
            tests["All detected events"] = _ptest(X, "Cause accuracy", nb, npm)
            out.append(_finish("P10", "P10: share of detected events with the correct fault type (frozen STGNN detections, "
                                      "majority class per event)", D, better, tests, alpha,
                               "Same detected events for all classifiers; columns = injected scenario."))
    F = pst._p10_frames(res)
    est = res.meta.get("estimators", [])
    tb = res.tables.get("estimators")
    if est:
        rows = []
        for e in est:
            lab = sse.EST_LABELS.get(e, e)
            r = {"Method": lab}
            r["MAE Δβ1, p.p."] = _val(F[f"b1_{e}"], "MAE Δβ1, p.p.") if f"b1_{e}" in F else np.nan
            sens = tb[tb["Estimator"] == lab]["Sensitivity Δb1 (quarters)"] if tb is not None and "Estimator" in tb else []
            r["Sensitivity to Δβ1 (est. / true)"] = float(sens.iloc[0]) if len(sens) else np.nan
            r["MAE Δβ1 under degradation, p.p."] = (_val(F[f"degmae_{e}"], "MAE Δβ1 (degradation), p.p.")
                                                    if f"degmae_{e}" in F else np.nan)
            r["Degradation detected (evidence share)"] = (_val(F[f"deg_{e}"], "Cause accuracy")
                                                          if f"deg_{e}" in F else np.nan)
            r["False evidence without injection"] = (_val(F[f"degfpr_{e}"], "FPR (no injection)")
                                                     if f"degfpr_{e}" in F else np.nan)
            rows.append(r)
        D = pd.DataFrame(rows)
        better = {"MAE Δβ1, p.p.": "min", "Sensitivity to Δβ1 (est. / true)": ("target", 1.0),
                  "MAE Δβ1 under degradation, p.p.": "min", "Degradation detected (evidence share)": "max",
                  "False evidence without injection": "min"}
        lab = {sse.EST_LABELS.get(e, e): e for e in est}

        def fr(prefix):
            return {l: F[f"{prefix}_{e}"] for l, e in lab.items() if f"{prefix}_{e}" in F}
        tests = {}
        if len(fr("b1")) == len(lab):
            tests["MAE Δβ1, p.p."] = _ptest(fr("b1"), "MAE Δβ1, p.p.", nb, npm, key=["placement", "event_id", "plant", "quarter"])
        if len(fr("degmae")) == len(lab):
            tests["MAE Δβ1 under degradation, p.p."] = _ptest(fr("degmae"), "MAE Δβ1 (degradation), p.p.", nb, npm)
        if len(fr("deg")) == len(lab):
            tests["Degradation detected (evidence share)"] = _ptest(fr("deg"), "Cause accuracy", nb, npm)
        if len(fr("degfpr")) == len(lab):
            tests["False evidence without injection"] = _ptest(fr("degfpr"), "FPR (no injection)", nb, npm)
        out.append(_finish("P10", "P10: degradation estimators (SPS vs Kalman filter) on the same injected events",
                           D, better, tests, alpha, "Sensitivity: median ratio estimated / injected Δβ1 (target 1)."))
    return out


# ---------------------------------------------------------------------------- P12
def _interp_curves(curves: pd.DataFrame, budget: float, cols: List[str]) -> pd.DataFrame:
    """As pv_fixes_eval.build_tables: per (method, seed) interpolation of the q-curve at the FA budget, mean over seeds."""
    rows = []
    for (m, ms), g in curves.assign(model_seed=curves["model_seed"].astype(str)).groupby(["method", "model_seed"], sort=False):
        g = g[np.isfinite(g["clean_fa"])].sort_values(["clean_fa", "recall"])
        fa = g["clean_fa"].to_numpy(float)
        if not len(fa):
            continue
        fa_u, iu = np.unique(fa, return_index=True)
        inside = len(fa_u) >= 2 and fa_u[0] <= budget <= fa_u[-1]
        r = {"method": m}
        for k in cols:
            if k not in g:
                continue
            v = g[k].to_numpy(float)[iu]; ok = np.isfinite(v)
            r[k] = (float(np.interp(budget, fa_u[ok], v[ok])) if inside and ok.sum() >= 2
                    else float(g.iloc[int(np.argmin(np.abs(fa - budget)))][k]))
        rows.append(r)
    return pd.DataFrame(rows).groupby("method", sort=False).mean(numeric_only=True) if rows else pd.DataFrame()


def _mt_p12(res, alpha, nb, npm) -> List[MethodTable]:
    import pv_fixes_eval as fx
    import pv_metrics_stats as pst
    out = []
    lab = dict(zip(res.points["method"], res.points["label"]))
    scn = [k for k in fx.SCN_ORDER if f"R_{k}" in res.curves]
    S0 = res.points[res.points["method"] == "S0"]
    if len(S0) and len(res.curves):
        budget = float(S0["clean_fa"].mean())
        I = _interp_curves(res.curves, budget, ["recall", "precision", "F1", "bg_all", "delay_h"] + [f"R_{k}" for k in scn])
        rows = []
        for m in [k for k in fx.ORDER if k in I.index]:
            r = {"Method": lab.get(m, m)}
            r.update({fx.SCN_TITLE[k]: I.loc[m, f"R_{k}"] for k in scn})
            r.update({"Recall (all)": I.loc[m, "recall"], "Precision": I.loc[m, "precision"], "F1": I.loc[m, "F1"],
                      "Test FA (all plants) /1000 h": I.loc[m, "bg_all"], "Delay, h (median)": I.loc[m, "delay_h"]})
            rows.append(r)
        D = pd.DataFrame(rows)
        better = {c: "max" for c in D.columns if c != "Method"}
        better.update({"Test FA (all plants) /1000 h": "min", "Delay, h (median)": "min"})
        out.append(_finish("P12", f"P12: methods at the equal false-alarm budget on control plants "
                                  f"({budget:.2f} /1000 h, S0 operating point)", D, better, None, alpha,
                           "Interpolated on the q-curves (no event-level outcomes): best values only, ties not tested. "
                           "S1-S3 have no q-curve and appear in the operating-point table."))
    E = pst._det_frame(res.events) if len(res.events) else pd.DataFrame()
    if len(E):
        op = res.tables.get("operating_points")
        X = {lab.get(m, m): E[E["method"] == m] for m in fx.ORDER if m in set(E["method"])}
        rows = []
        for l, Z in X.items():
            r = {"Method": l}
            r.update({fx.SCN_TITLE[k]: _val(Z[Z["scenario"] == k], "Recall") for k in scn})
            o = op[op["Method"] == l] if op is not None else pd.DataFrame()
            r.update({"Recall (all)": _val(Z, "Recall"), "Precision": _val(Z, "Precision"), "F1": _val(Z, "F1"),
                      "Test FA (controls) /1000 h": float(o["Test FA (controls) /1000 h"].iloc[0]) if len(o) else np.nan,
                      "Delay, h (median)": float(Z.loc[Z["detected"].astype(bool), "delay_h"].median())})
            rows.append(r)
        D = pd.DataFrame(rows)
        better = {c: "max" for c in D.columns if c != "Method"}
        better.update({"Test FA (controls) /1000 h": "min", "Delay, h (median)": "min"})
        tests = {fx.SCN_TITLE[k]: _ptest(X, "Recall", nb, npm, sub=lambda Z, g, k=k: Z[Z["scenario"] == k]) for k in scn}
        tests.update({"Recall (all)": _ptest(X, "Recall", nb, npm), "F1": _ptest(X, "F1", nb, npm),
                      "Precision": _ptest(X, "Precision", nb, npm)})
        out.append(_finish("P12", "P12: methods at their validation operating points (same events)", D, better, tests, alpha,
                           "Operating points differ in false-alarm rate: read recall together with the FA column."))
    return out


# ---------------------------------------------------------------------------- P13
def _mt_p13(res, alpha, nb, npm) -> List[MethodTable]:
    eb, sr = res.tables.get("equal_budget"), res.tables.get("scenario_recall")
    if eb is None or not len(eb):
        return []
    bcol = "FA budget (controls) /1000 h"
    ref = float(sr[bcol].iloc[0]) if sr is not None and len(sr) and bcol in sr else float(eb[bcol].iloc[0])
    sel = eb[np.isclose(eb[bcol].astype(float), ref)]
    sel = sel if len(sel) else eb[eb[bcol] == eb[bcol].iloc[0]]
    scn = [c for c in (sr.columns if sr is not None else []) if c not in ("Variant", bcol, "Recall (all)")]
    rows = []
    for _, r in sel.iterrows():
        d = {"Method": f"STGNN + {r['Variant']}"}
        s = sr[sr["Variant"] == r["Variant"]] if sr is not None else pd.DataFrame()
        d.update({c: float(s[c].iloc[0]) if len(s) else np.nan for c in scn})
        d.update({"Recall (all)": r["Recall"], "Precision": r.get("Precision"), "F1": r.get("F1"),
                  "Recall SD (seeds)": r.get("Recall SD (seeds)"), "Test FA (all plants) /1000 h": r.get("Test FA (all plants) /1000 h")})
        rows.append(d)
    D = pd.DataFrame(rows)
    better = {c: "max" for c in D.columns if c != "Method"}
    better.update({"Recall SD (seeds)": "min", "Test FA (all plants) /1000 h": "min"})
    return [_finish("P13", f"P13: k-means vs fuzzy c-means at the equal false-alarm budget ({ref:.2f} /1000 h, "
                           f"k-means operating point)", D, better, None, alpha,
                    "Interpolated equal-budget points: best values only, ties not tested.")]


# ---------------------------------------------------------------------------- P14
P14_TABLE_METHODS = ("pr", "prp", "stgnn", "stgnn_day", "fusion", "fusion_prp")


def _mt_p14(res, alpha, nb, npm) -> List[MethodTable]:
    import pv_stgnn_value as sv
    ref = float(res.meta.get("ref_budget"))
    E = res.events[np.isclose(res.events["budget"].astype(float), ref)]
    E = sv._main_events(E)
    E = E.assign(fault=True, tp=E["n_tp_detections"], fp=E["n_fp_near"])
    eb = res.tables["equal_budget"]
    eb = eb[np.isclose(eb["Alarm budget (clean, all plants) /1000 h"].astype(float), ref)]
    X = {sv.METHOD_LABEL[m]: E[E["method"] == m] for m in P14_TABLE_METHODS if m in set(E["method"])}
    scn = [s for s in E["scenario"].drop_duplicates()]
    rows = []
    for l, Z in X.items():
        o = eb[eb["Method"] == l]
        r = {"Method": l}
        r.update({_scn_title(s): _val(Z[Z["scenario"] == s], "Recall") for s in scn})
        r.update({"Recall (all)": float(o["Recall"].iloc[0]) if len(o) else _val(Z, "Recall"),
                  "Precision": float(o["Precision"].iloc[0]) if len(o) else _val(Z, "Precision"),
                  "F1": float(o["F1"].iloc[0]) if len(o) else _val(Z, "F1"),
                  "FA on control plants /1000 h": float(o["FA on control plants /1000 h"].iloc[0]) if len(o) else np.nan})
        rows.append(r)
    D = pd.DataFrame(rows)
    better = {c: "max" for c in D.columns if c != "Method"}
    better["FA on control plants /1000 h"] = "min"
    tests = {_scn_title(s): _ptest(X, "Recall", nb, npm, sub=lambda Z, g, s=s: Z[Z["scenario"] == s]) for s in scn}
    tests.update({"Recall (all)": _ptest(X, "Recall", nb, npm), "Precision": _ptest(X, "Precision", nb, npm),
                  "F1": _ptest(X, "F1", nb, npm)})
    return [_finish("P14", f"P14: PR, PR+, STGNN and fusions at the equal alarm budget ({ref:g} /1000 h, clean data, all plants)",
                    D, better, tests, alpha, "Same placements, model seeds and events for all methods.")]


# ---------------------------------------------------------------------------- P15 / P16
def _mt_modal(res, proto, alpha, nb, npm) -> List[MethodTable]:
    import pv_metrics_stats as pst
    import pv_modal_eval as pme
    ref = float(res.meta.get("ref_budget", 1.0))
    E = res.events
    have = [m for m in pme.METHODS if m in set(E["method"])]
    X = {pme.METHODS[m]["label"]: pst._modal_frame(E, m, ref) for m in have}
    X = {k: v for k, v in X.items() if len(v)}
    if not X:
        return []
    Z0 = next(iter(X.values()))
    fault_s = [s for s in Z0.loc[Z0["fault"], "scenario"].drop_duplicates()]
    nuis_s = [s for s in Z0.loc[~Z0["fault"], "scenario"].drop_duplicates()]
    t = lambda s: pme.SCN_SHORT.get(s, s)
    rows = []
    for l, Z in X.items():
        r = {"Method": l}
        r.update({f"Recall: {t(s)}": _val(Z[Z["scenario"] == s], "Recall") for s in fault_s})
        r["Recall (all faults)"] = _val(Z, "Recall")
        r.update({f"Nuisance: {t(s)}": _val(Z[Z["scenario"] == s], "Nuisance share") for s in nuis_s})
        r["Nuisance share (all)"] = _val(Z, "Nuisance share")
        r["Macro-F1 (cause)"] = _val(Z, "Macro-F1 (cause)")
        rows.append(r)
    D = pd.DataFrame(rows)
    better = {c: ("min" if c.startswith("Nuisance") else "max") for c in D.columns if c != "Method"}
    tests = {f"Recall: {t(s)}": _ptest(X, "Recall", nb, npm, sub=lambda Z, g, s=s: Z[Z["scenario"] == s]) for s in fault_s}
    tests.update({f"Nuisance: {t(s)}": _ptest(X, "Nuisance share", nb, npm, sub=lambda Z, g, s=s: Z[Z["scenario"] == s])
                  for s in nuis_s})
    tests.update({"Recall (all faults)": _ptest(X, "Recall", nb, npm), "Nuisance share (all)": _ptest(X, "Nuisance share", nb, npm),
                  "Macro-F1 (cause)": _ptest(X, "Macro-F1 (cause)", nb, npm)})
    sec = proto[1:]
    return [_finish(proto, f"{proto}: modal scenarios (section {sec}) at the equal clean alarm budget ({ref:g} /1000 h)",
                    D, better, tests, alpha, "Nuisance = alarms on non-fault scenarios (grid curtailment, meter fault)."
                    + (" Results of the metering / new-obstruction scenarios depend on the open generator issue."
                       if proto in ("P15", "P16") else ""))]


# ---------------------------------------------------------------------------- P18
def _mt_p18(res, alpha, nb, npm) -> List[MethodTable]:
    import pv_semisynth_config as psc
    out = []
    E = res.events
    ref = res.meta.get("ref_budget", 1.0)
    if len(E):
        X = E[E["scored"] & (E["budget"] == ref)]
        win = res.meta.get("windows", [[7, 30], [30, 90]])
        od = res.meta.get("onset_days", 7)
        cols = {f"Alarm in the first {od} d (recall)": ("detected", "max")}
        cols.update({f"Excess alarm share {a}-{b} d": (f"excess_{a}_{b}", "min") for a, b in win})
        rows, PE = [], {}
        for scn in psc.CONFIG_SCENARIOS:
            for m in [k for k in psc.METHOD_LABEL if k in set(X["method"])]:
                Y = X[(X["method"] == m) & (X["scenario"] == scn)]
                if not len(Y):
                    continue
                r = {"Group": scn, "Method": psc.METHOD_LABEL[m]}
                for c, (col, _) in cols.items():
                    v = psc._per_event(Y, col)
                    PE[(scn, psc.METHOD_LABEL[m], c)] = v
                    r[c] = float(np.nanmean(v)) if len(v) else np.nan
                rows.append(r)
        if rows:
            D = pd.DataFrame(rows)

            def mk(c):
                return lambda a, b, g: psc.paired_diff(PE[(g, a, c)], PE[(g, b, c)], nb, npm).get("p", np.nan)
            out.append(_finish("P18", f"P18: configuration changes, detectors at the equal alarm budget ({ref:g} /1000 h); "
                                      f"rows grouped by scenario", D, {c: h for c, (_, h) in cols.items()},
                               {c: mk(c) for c in cols}, alpha,
                               "module_add: an alarm in the onset window is the expected configuration notice; "
                               "excess = alarm share after the change minus the same window on clean data."))
    B1 = res.b1
    if len(B1):
        rows, PE = [], {}
        cq = {"MAE Δβ1, change quarter, p.p.": True, "MAE Δβ1, full quarters after, p.p.": False}
        for scn in psc.CONFIG_SCENARIOS:
            for e in [k for k in psc.EST_LABEL if k in set(B1["estimator"])]:
                Z = B1[(B1["estimator"] == e) & (B1["scenario"] == scn)]
                if not len(Z):
                    continue
                l = psc.EST_LABEL[e]
                r = {"Group": scn, "Method": l}
                for c, q in cq.items():
                    v = Z[Z["change_quarter"] == q].groupby(["placement", "event_id"])["error_pp"].apply(lambda s: np.nanmean(np.abs(s)))
                    PE[(scn, l, c)] = v
                    r[c] = float(v.mean()) if len(v) else np.nan
                lg = res.lags[res.lags["scenario"] == scn]["lag_days"] if e == "kalman" and len(res.lags) else pd.Series(dtype=float)
                r["Lag to 90% of step, d (Kalman)"] = float(lg.replace(np.inf, np.nan).median()) if lg.notna().any() else np.nan
                rows.append(r)
        if rows:
            D = pd.DataFrame(rows)

            def mk(c):
                return lambda a, b, g: psc.paired_diff(PE[(g, a, c)], PE[(g, b, c)], nb, npm).get("p", np.nan)
            out.append(_finish("P18", "P18: configuration changes, β1 estimators (SPS vs Kalman filter); rows grouped by scenario",
                               D, {c: "min" for c in cq}, {c: mk(c) for c in cq}, alpha,
                               "The adaptation lag exists for the Kalman filter only and is not ranked."))
    return out


# ---------------------------------------------------------------------------- P20
def _mt_p20(res, alpha, nb, npm) -> List[MethodTable]:
    import pv_rul_semisynth as rss
    R = res.rows[res.rows["cohort"] == 1]
    if not len(R):
        return []
    nominal = float(getattr(res.rcfg, "ci", 0.95))
    rows = []
    groups = [("All scenarios", R)] + [(rss.SCENARIO_TITLES.get(s, s), g) for s, g in R.groupby("scenario", sort=False)]
    srcs = list(R["source"].unique())
    for gname, g in groups:
        for s in srcs:
            x = g[g["source"] == s]
            if not len(x):
                continue
            m = rss._metrics(x, 10)
            rows.append({"Group": gname, "Method": s, "Rate MAE, p.p./yr": m["Rate MAE, p.p./yr"],
                         "Rate bias, p.p./yr": m["Rate bias, p.p./yr"], "RUL MAE, yr": m["RUL MAE, yr"],
                         f"CI coverage (nominal {nominal:.2f})": m["CI coverage"], "Balanced accuracy": m["Balanced accuracy"]})
    D = pd.DataFrame(rows)
    gmap = dict(groups)

    def mk(col):
        def f(a, b, g):
            G = gmap[g]
            return rss._paired(G[G["source"] == a], G[G["source"] == b], col, nb, npm).get("p", np.nan)
        return f
    better = {"Rate MAE, p.p./yr": "min", "Rate bias, p.p./yr": "zero", "RUL MAE, yr": "min",
              f"CI coverage (nominal {nominal:.2f})": ("target", nominal), "Balanced accuracy": "max"}
    return [_finish("P20", "P20: RUL algorithms on the same semi-synthetic fleet (cohort 1); rows grouped by scenario",
                    D, better, {"Rate MAE, p.p./yr": mk("abs_rate_err"), "RUL MAE, yr": mk("abs_rul_err")}, alpha,
                    "Ties: paired test over the same stations. Bias: closest to 0; coverage: closest to the nominal "
                    "level. RUL MAE '--': no station with a finite estimated and true RUL (e.g. the method classifies "
                    "all stations as censored; balanced accuracy is then 0.5).")]


# ---------------------------------------------------------------------------- public API
def method_tables(ss_res=None, fx_res=None, fcm_res=None, val_res=None, modal_res=None, fmm_res=None,
                  cfg_res=None, rul_res=None, alpha: float = 0.05, n_boot: int = 1000, n_perm: int = 2000,
                  ) -> Dict[str, MethodTable]:
    """Methods x metrics table for every protocol whose result is given (None results are skipped).

    Returns {key: MethodTable}; keys: P10_class, P10_estimators, P12_equal_budget, P12_operating_points,
    P13, P14, P15, P16, P18_detectors, P18_estimators, P20.  Use ``show_method_tables`` and
    ``export_method_tables`` for display and LaTeX / CSV export."""
    out = {}
    jobs = [("P10", ss_res, _mt_p10, ["P10_class", "P10_estimators"]),
            ("P12", fx_res, _mt_p12, ["P12_equal_budget", "P12_operating_points"]),
            ("P13", fcm_res, _mt_p13, ["P13"]), ("P14", val_res, _mt_p14, ["P14"]),
            ("P18", cfg_res, _mt_p18, ["P18_detectors", "P18_estimators"]), ("P20", rul_res, _mt_p20, ["P20"])]
    for proto, res, fn, keys in jobs:
        if res is None:
            continue
        tabs = fn(res, alpha, n_boot, n_perm)
        if proto == "P10" and len(tabs) == 1 and "estimators" in tabs[0].title:
            keys = ["P10_estimators"]
        if proto == "P12" and len(tabs) == 1 and "operating" in tabs[0].title:
            keys = ["P12_operating_points"]
        if proto == "P18" and len(tabs) == 1 and "estimators" in tabs[0].title:
            keys = ["P18_estimators"]
        out.update(dict(zip(keys, tabs)))
    for proto, res in (("P15", modal_res), ("P16", fmm_res)):
        if res is not None:
            out.update({proto: t for t in _mt_modal(res, proto, alpha, n_boot, n_perm)})
    return out


def _digits(col: str) -> int:
    return 1 if (", h" in col or ", d" in col) else (2 if ("/1000 h" in col or "p.p." in col or ", yr" in col) else 3)


def _fmt(v, col: str) -> str:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not np.isfinite(v):
        return "--"
    out = f"{v:.{_digits(col)}f}"
    return out[1:] if out.startswith("-") and float(out) == 0 else out          # no "-0.00"


def show_method_tables(T: Dict[str, MethodTable]):
    """Notebook display: bold = best, underlined = not significantly different from the best."""
    try:
        from IPython.display import display
    except Exception:                                   # pragma: no cover
        display = print
    for k, t in T.items():
        print(f"[{k}] {t.title}")
        cols = t.metric_cols
        css = t.marks.replace({BEST: "font-weight: bold", TIE: "text-decoration: underline", "": ""})

        def sty(_df, css=css, cols=cols):
            S = pd.DataFrame("", index=_df.index, columns=_df.columns)
            S[cols] = css[cols].to_numpy()
            return S
        try:
            display(t.data.style.apply(sty, axis=None).format({c: (lambda v, c=c: _fmt(v, c)) for c in cols}).hide(axis="index"))
        except Exception:                               # plain-text fallback
            display(t.data)
        nt = [c for c, v in t.tested.items() if not v]
        print("  bold = best; underlined = not significantly different from the best (paired test, Holm, "
              f"p >= alpha).  Not tested: {', '.join(nt) if nt else 'none'}.  {t.note}")


def _tex_esc(s: str) -> str:
    s = str(s)
    for a, b in (("\\", "\\textbackslash{}"), ("&", "\\&"), ("%", "\\%"), ("_", "\\_"), ("#", "\\#"),
                 ("~", "$\\approx$"), (">", "$>$"), ("<", "$<$"), ("β", "$\\beta$"), ("Δ", "$\\Delta$"), ("η", "$\\eta$")):
        s = s.replace(a, b)
    return s


def method_table_latex(t: MethodTable, label: str, wide: Optional[bool] = None) -> str:
    """LaTeX table: bold = best, underline = tie with the best.  ``wide=None``: full width (table*) for more than
    five metric columns, otherwise one column (table, \\columnwidth) as in two-column journal layouts."""
    cols = t.metric_cols
    grp = "Group" in t.data
    head = ["Method"] + cols
    if wide is None:
        wide = len(cols) > 5
    env, W = ("table*", "\\textwidth") if wide else ("table", "\\columnwidth")
    mw, cw = ("3.4cm", "1.6cm") if wide else ("2.3cm", "1.15cm")
    al = f">{{\\raggedright\\arraybackslash}}p{{{mw}}}" + f">{{\\centering\\arraybackslash}}p{{{cw}}}" * len(cols)
    L = [f"\\begin{{{env}}}[!t]", "\\centering", "\\scriptsize", "\\setlength{\\tabcolsep}{3pt}",
         f"\\caption{{{_tex_esc(t.title)}}}", f"\\label{{{label}}}",
         f"\\resizebox{{\\ifdim\\width>{W}{W}\\else\\width\\fi}}{{!}}{{%", f"\\begin{{tabular}}{{{al}}}", "\\toprule",
         " & ".join("\\textbf{" + _tex_esc(h) + "}" if i == 0 else "\\hspace{0pt}" + _tex_esc(h) for i, h in enumerate(head)) + " \\\\",
         "\\midrule"]
    last = None
    for i, r in t.data.iterrows():
        if grp and r["Group"] != last:
            if last is not None:
                L.append("\\midrule")
            L.append(f"\\multicolumn{{{len(head)}}}{{l}}{{\\textit{{{_tex_esc(r['Group'])}}}}} \\\\")
            last = r["Group"]
        cells = [_tex_esc(r["Method"])]
        for c in cols:
            s = _fmt(r[c], c); mk = t.marks.loc[i, c]
            cells.append(f"\\textbf{{{s}}}" if mk == BEST else (f"\\underline{{{s}}}" if mk == TIE else s))
        L.append(" & ".join(cells) + " \\\\")
    nt = [c for c, v in t.tested.items() if not v]
    foot = ("Bold: best value of the column" + (" within each group" if grp else "") +
            "; underlined: not significantly different from the best (paired test on the same events, Holm, "
            "$p_\\mathrm{Holm}\\geq 0.05$)." + ((" Ties not tested (no event-level outcomes)." if len(nt) == len(t.tested) else
              f" Not tested: {_tex_esc(', '.join(nt))}.") if nt else "") +
            (" " + _tex_esc(t.note) if t.note else ""))
    L += ["\\bottomrule", "\\end{tabular}}", f"\\par\\smallskip\\parbox{{{W}}}{{\\footnotesize {foot}}}", f"\\end{{{env}}}"]
    return "\n".join(L)


def export_method_tables(T: Dict[str, MethodTable], out_dir: str = "results/summary") -> Dict[str, str]:
    """CSV (values, marks, tie tests) and LaTeX (bold / underline) per table + one combined .tex file."""
    os.makedirs(out_dir, exist_ok=True)
    out, parts = {}, []
    for k, t in T.items():
        base = os.path.join(out_dir, f"methods_{k}")
        t.data.to_csv(base + ".csv", index=False)
        pd.concat([t.data[[c for c in ("Group", "Method") if c in t.data]], t.marks], axis=1).to_csv(base + "_marks.csv", index=False)
        if t.pvals is not None and len(t.pvals):
            t.pvals.to_csv(base + "_tie_tests.csv", index=False)
        tex = method_table_latex(t, f"tab:methods_{k.lower()}")
        open(base + ".tex", "w", encoding="utf-8").write(tex)
        out[k] = base + ".tex"; parts.append(tex)
    p = os.path.join(out_dir, "methods_all.tex")
    open(p, "w", encoding="utf-8").write("% Requires: booktabs, array, graphicx\n" + "\n\n".join(parts))
    out["all"] = p
    return out
