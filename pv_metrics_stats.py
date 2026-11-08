"""
pv_metrics_stats.py -- confidence intervals and paired comparisons for the summary table (section 17).

Works only on result objects that are already computed (ss_res, fx_res, fcm_res, val_res, modal_res, fmm_res);
nothing is retrained.  All tables are in English.

Statistics
----------
* Resampling unit = injected event, cluster key (placement, event_id).  All rows of a resampled event
  (model seeds, methods) are kept together, so repeated seeds on the same event are not treated as independent.
* 95 % CI of a metric: percentile cluster bootstrap (n_boot draws).  Ratio metrics (recall, precision, F1,
  macro-F1, nuisance share, MAE) are recomputed from resampled counts, not averaged per event.
* P12 / P13 equal-budget rows are interpolated from the q-curves, so no event-level outcomes exist at that
  point: their recall CI is a Wilson interval with n = number of unique events (approximate, labelled).
* Paired comparison A - B: both methods on the same events (inner join on the row key).  CI of the
  difference = paired cluster bootstrap (same resampled events for A and B); p-value = paired permutation
  test that swaps the A / B outcomes inside each event (two-sided, |delta|); Holm correction within each
  protocol (family = all pre-specified comparisons x metrics of that protocol).
* "Only A hit" / "Only B hit" (recall and cause accuracy only): discordant events, i.e. events detected
  (correctly classified) by one method but not by the other.  With n_perm draws the smallest attainable
  p-value is 1 / (n_perm + 1); use n_perm >= 10000 for the final paper tables.
* Comparisons are pre-specified (``COMPARISONS`` below); no comparison is chosen after looking at the data.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

CLUSTER = ["placement", "event_id"]
LOWER_IS_BETTER = {"Nuisance share", "MAE Δβ1, p.p.", "MAE Δβ1 (degradation), p.p.", "FPR (no injection)"}


# ============================================================================ metric specifications
# a metric = (vectors(X) -> {name: per-row float array}, combine(counts) -> value); counts may be arrays
def _ratio(n, d):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.asarray(d) > 0, np.asarray(n, float) / np.where(np.asarray(d) > 0, d, 1), np.nan)


def _v_recall(X):
    f = X["fault"].to_numpy(float) if "fault" in X else np.ones(len(X))
    return {"n": X["detected"].to_numpy(float) * f, "d": f}


def _v_nuis(X):
    f = 1 - X["fault"].to_numpy(float)
    return {"n": X["detected"].to_numpy(float) * f, "d": f}


def _v_prec(X):
    tp = X["tp"].to_numpy(float)
    return {"n": tp, "d": tp + X["fp"].to_numpy(float)}


def _v_f1(X):
    r, p = _v_recall(X), _v_prec(X)
    return {"rn": r["n"], "rd": r["d"], "pn": p["n"], "pd": p["d"]}


def _c_f1(c):
    r, p = _ratio(c["rn"], c["rd"]), _ratio(c["pn"], c["pd"])
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(r + p > 0, 2 * r * p / np.where(r + p > 0, r + p, 1), np.nan)


def _v_mean(col):
    return lambda X: {"n": X[col].to_numpy(float), "d": np.ones(len(X))}


def _v_macro(X):
    t, p = X["cls"].astype(str).to_numpy(), X["pred"].astype(str).to_numpy()
    out = {}
    for c in sorted(set(t)):
        out[f"tp|{c}"] = ((p == c) & (t == c)).astype(float)
        out[f"fp|{c}"] = ((p == c) & (t != c)).astype(float)
        out[f"fn|{c}"] = ((p != c) & (t == c)).astype(float)
    return out


def _c_macro(c):
    cls = sorted({k.split("|", 1)[1] for k in c})
    F = []
    for k in cls:
        tp, fp, fn = np.asarray(c[f"tp|{k}"], float), np.asarray(c[f"fp|{k}"], float), np.asarray(c[f"fn|{k}"], float)
        with np.errstate(divide="ignore", invalid="ignore"):
            F.append(np.where(tp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), np.nan))
    F = np.array(F)
    with np.errstate(invalid="ignore"):
        return np.nanmean(F, axis=0) if np.isfinite(F).any() else np.full(F.shape[1:], np.nan)


_c_ratio = lambda c: _ratio(c["n"], c["d"])
METRICS: Dict[str, Tuple[Callable, Callable]] = {
    "Recall": (_v_recall, _c_ratio), "Precision": (_v_prec, _c_ratio), "F1": (_v_f1, _c_f1),
    "Nuisance share": (_v_nuis, _c_ratio), "Macro-F1 (cause)": (_v_macro, _c_macro),
    "Cause accuracy": (_v_mean("ok"), _c_ratio), "FPR (no injection)": (_v_mean("ok"), _c_ratio),
    "MAE Δβ1, p.p.": (_v_mean("err"), _c_ratio), "MAE Δβ1 (degradation), p.p.": (_v_mean("err"), _c_ratio),
}


# ============================================================================ resampling core
def _clusters(X: pd.DataFrame) -> np.ndarray:
    return pd.MultiIndex.from_frame(X[CLUSTER].astype(str)).factorize()[0]


def _agg(vec: dict, cl: np.ndarray, ncl: int) -> dict:
    return {k: np.bincount(cl, weights=v, minlength=ncl) for k, v in vec.items()}


def _boot_counts(ncl: int, n_boot: int, rng) -> np.ndarray:
    D = rng.integers(0, ncl, size=(n_boot, ncl))
    C = np.zeros((n_boot, ncl))
    np.add.at(C, (np.repeat(np.arange(n_boot), ncl), D.ravel()), 1)
    return C


def _q(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return (float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))) if len(v) else (np.nan, np.nan)


def bootstrap_ci(X: pd.DataFrame, metric: str, n_boot: int = 2000, seed: int = 0) -> Tuple[float, float, float, int]:
    """Point estimate and 95 % percentile CI of ``metric`` with a cluster bootstrap over events."""
    if not len(X):
        return np.nan, np.nan, np.nan, 0
    vf, cf = METRICS[metric]
    cl = _clusters(X); ncl = int(cl.max()) + 1
    V = _agg(vf(X), cl, ncl)
    est = float(cf({k: v.sum() for k, v in V.items()}))
    C = _boot_counts(ncl, n_boot, np.random.default_rng(seed))
    lo, hi = _q(cf({k: C @ v for k, v in V.items()}))
    return est, lo, hi, ncl


def wilson(p: float, n: float, z: float = 1.959964) -> Tuple[float, float]:
    if not (np.isfinite(p) and np.isfinite(n) and n > 0):
        return np.nan, np.nan
    d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(max(0.0, c - h)), float(min(1.0, c + h))


def paired_test(A: pd.DataFrame, B: pd.DataFrame, key: List[str], metric: str, n_boot: int = 2000,
                n_perm: int = 2000, seed: int = 0) -> dict:
    """A - B on the same rows (inner join on ``key``): paired cluster-bootstrap CI and permutation p-value."""
    vf, cf = METRICS[metric]
    A = A.drop_duplicates(key).set_index(key); B = B.drop_duplicates(key).set_index(key)
    j = A.index.intersection(B.index)
    if not len(j):
        return dict(n=0, rows=0)
    A, B = A.loc[j].reset_index(), B.loc[j].reset_index()
    if metric in ("Recall", "Nuisance share", "F1") and "fault" in A:
        B["fault"] = A["fault"].to_numpy()                      # same event roles for both methods
    if "cls" in A:
        B["cls"] = A["cls"].to_numpy()
    cl = _clusters(A); ncl = int(cl.max()) + 1
    VA, VB = _agg(vf(A), cl, ncl), _agg(vf(B), cl, ncl)
    keys = sorted(set(VA) | set(VB))
    for k in keys:
        VA.setdefault(k, np.zeros(ncl)); VB.setdefault(k, np.zeros(ncl))
    sA, sB = {k: VA[k].sum() for k in keys}, {k: VB[k].sum() for k in keys}
    a, b = float(cf(sA)), float(cf(sB)); d = a - b
    rng = np.random.default_rng(seed)
    C = _boot_counts(ncl, n_boot, rng)
    dB = cf({k: C @ VA[k] for k in keys}) - cf({k: C @ VB[k] for k in keys})
    lo, hi = _q(dB)
    S = rng.integers(0, 2, size=(n_perm, ncl)).astype(float)    # 1 = swap A and B outcomes of this event
    dP = (cf({k: sA[k] + S @ (VB[k] - VA[k]) for k in keys}) - cf({k: sB[k] + S @ (VA[k] - VB[k]) for k in keys}))
    dP = dP[np.isfinite(dP)]
    p = float((1 + np.sum(np.abs(dP) >= abs(d) - 1e-12)) / (1 + len(dP))) if np.isfinite(d) and len(dP) else np.nan
    out = dict(n=ncl, rows=len(A), a=a, b=b, d=d, lo=lo, hi=hi, p=p)
    if metric not in ("Recall", "Cause accuracy"):
        pass
    elif "detected" in A:
        f = A["fault"].to_numpy(bool) if ("fault" in A and metric != "Nuisance share") else (
            ~A["fault"].to_numpy(bool) if "fault" in A else np.ones(len(A), bool))
        da, db = A["detected"].to_numpy(bool)[f], B["detected"].to_numpy(bool)[f]
        out.update(only_a=int(np.sum(da & ~db)), only_b=int(np.sum(~da & db)))
    elif "ok" in A:
        oa, ob = A["ok"].to_numpy(bool), B["ok"].to_numpy(bool)
        out.update(only_a=int(np.sum(oa & ~ob)), only_b=int(np.sum(~oa & ob)))
    return out


def holm(p: pd.Series) -> pd.Series:
    p = pd.to_numeric(p, errors="coerce"); ok = p.notna()
    out = pd.Series(np.nan, index=p.index)
    if ok.any():
        v = p[ok].sort_values(); m = len(v)
        adj = np.minimum(1, np.maximum.accumulate((m - np.arange(m)) * v.to_numpy()))
        out[v.index] = adj
    return out


# ============================================================================ event-level data per protocol
def _det_frame(E: pd.DataFrame) -> pd.DataFrame:
    """Main events (duration != short) with detection outcome and near-event TP / FP counts."""
    E = E[E["duration"] != "short"] if "duration" in E else E
    return E.assign(fault=True, tp=E.get("n_tp_detections", 0), fp=E.get("n_fp_near", 0))


def _p10_frames(res) -> Dict[str, pd.DataFrame]:
    """name -> event-level frame for P10 (detection, cause classification, degradation, beta1 error)."""
    import pv_semisynth_gen as ssg
    import pv_semisynth_eval as sse
    out = {}
    E = res.events
    if not len(E):
        return out
    out["det"] = _det_frame(E)
    key = ["placement", "model_seed", "event_id"]
    cls = res.classes
    if len(cls):
        for kind in ("detected", "oracle"):
            c = cls[cls["kind"] == kind]
            if not len(c):
                continue
            m = c.merge(E[key + ["scenario"]], on=key)
            for col in ("fuzzy", "crisp", "kmeans"):
                if col not in m or m[col].isna().all() or (kind == "oracle" and col != "fuzzy"):
                    continue
                ok = [r == ssg.SCENARIOS[s].expected_name for r, s in zip(m[col], m["scenario"])]
                v = (m.assign(ok=ok).groupby(key)["ok"].mean() >= 0.5).reset_index()
                out[f"cls_{kind}_{col}"] = v
    DG = res.degradation
    if len(DG) and "evidence_injected" in DG:
        for e, d in DG.groupby("estimator"):
            out[f"deg_{e}"] = d.assign(ok=d["evidence_injected"].astype(str).str.contains("decrease"), detected=
                                       d["evidence_injected"].astype(str).str.contains("decrease"), fault=True)
            if "evidence_clean" in d:
                out[f"degfpr_{e}"] = d.assign(ok=d["evidence_clean"].astype(str).str.contains("decrease"))
            if {"estimated_b1_pct", "expected_b1_pct"} <= set(d.columns):
                dm = d.assign(err=(d["estimated_b1_pct"] - d["expected_b1_pct"]).abs())
                out[f"degmae_{e}"] = dm[np.isfinite(dm["err"].astype(float))]
    B = getattr(res, "b1", pd.DataFrame())
    if len(B):
        B = B[B["scenario"] != "ghi_shift"]
        for e, b in B.groupby("estimator"):
            b = b.assign(err=(b["estimated_b1_pct"] - b["expected_b1_pct"]).abs())
            b = b[np.isfinite(b["err"].astype(float))]
            out[f"b1_{e}"] = b.groupby(["placement", "event_id", "plant", "quarter"], as_index=False)["err"].mean()
    return out


def _class_frame(res, method: str, col: str = "fuzzy") -> pd.DataFrame:
    """Per detected event (placement, seed, event): majority fuzzy class correct (as pv_fixes_eval._class_share)."""
    import pv_semisynth_gen as ssg
    cls, E = res.classes, res.events
    if not len(cls) or col not in cls or "method" not in cls:
        return pd.DataFrame()
    key = ["placement", "model_seed", "event_id", "method"]
    c = cls[(cls["method"] == method) & (cls["kind"] == "detected")]
    d = c.merge(E[E["method"] == method][key + ["scenario"]].drop_duplicates(), on=key)
    if not len(d):
        return pd.DataFrame()
    d = d.assign(ok=[r == ssg.SCENARIOS[s].expected_name for r, s in zip(d[col], d["scenario"])])
    return (d.groupby(key + ["scenario"])["ok"].mean() >= 0.5).reset_index()


def _class_method(proto: str, model: str) -> Optional[str]:
    """Method key behind a P12 / P13 'cause classification of detected events' row."""
    if proto == "P12":
        for m in ("S0", "S4"):
            if f"Detected, {m}" in model:
                return m
        return None
    try:
        import pv_fcm_ablation as fa
        return next((k for k, l in fa.VARIANT_LABEL.items() if model == f"Fuzzy classifier, {l} clusters"), None)
    except Exception:
        return None


P10_CLS_LABEL = {"fuzzy": "Fuzzy classifier (Mamdani rules)", "crisp": "Crisp rules", "kmeans": "K-means anomaly clusters"}


def _modal_label(m: str) -> str:
    import pv_modal_eval as pme
    if m in pme.METHODS or "|" in m:
        return pme.METHODS.get(m.split("|")[0], {}).get("label", m) + (" (" + m.split("|")[1] + ")" if "|" in m else "")
    return f"PR+ & modal without {pme.md.MODALITY_LABEL[m.split('-', 1)[1]]}"


def _modal_frame(E: pd.DataFrame, method: str, budget: float) -> pd.DataFrame:
    X = E[(E["method"] == method) & (E["budget"] == budget) & E["scored"]]
    return X.assign(fault=X["expected_alarm"].astype(bool))


# ============================================================================ CI table (one row per summary cell)
def ci_table(S: pd.DataFrame, ss_res=None, fx_res=None, fcm_res=None, val_res=None, modal_res=None, fmm_res=None,
             n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Long table: Protocol, Task, Model, Metric, Value, CI low, CI high, Events, CI method."""
    rows = []
    BOOT = "cluster bootstrap over events"

    def add(r, metric, val, lo, hi, n, how):
        rows.append({"Protocol": r["Protocol"], "Task": r["Task"], "Model": r["Model"], "Metric": metric,
                     "Value": val, "CI low": lo, "CI high": hi, "Events": n, "CI method": how})

    F10 = _p10_frames(ss_res) if ss_res is not None else {}
    vb = val_res.meta.get("ref_budget") if val_res is not None else None
    lab14 = dict(zip(val_res.points["method"], val_res.points["label"])) if val_res is not None and len(val_res.points) else {}
    for _, r in S.iterrows():
        p, task, model = r["Protocol"], str(r["Task"]), str(r["Model"])
        if p == "P10":
            if task.startswith("Fault event detection") and "det" in F10:
                for m in ("Recall", "Precision", "F1"):
                    e, lo, hi, n = bootstrap_ci(F10["det"], m, n_boot, seed); add(r, m, e, lo, hi, n, BOOT)
            elif task.startswith("Cause classification"):
                kind = "oracle" if "oracle" in model else "detected"
                col = next((c for c, l in P10_CLS_LABEL.items() if model.startswith(l)), "fuzzy")
                X = F10.get(f"cls_{kind}_{col}")
                if X is not None:
                    e, lo, hi, n = bootstrap_ci(X, "Cause accuracy", n_boot, seed); add(r, "Cause accuracy", e, lo, hi, n, BOOT)
            elif task.startswith("Degradation"):
                import pv_semisynth_eval as sse
                est = next((k for k, l in sse.EST_LABELS.items() if f"+ {l} estimator" in model), None)
                for m, nm in (("Recall", "deg"), ("FPR (no injection)", "degfpr")):
                    X = F10.get(f"{nm}_{est}")
                    if X is not None:
                        e, lo, hi, n = bootstrap_ci(X, m if m != "Recall" else "Cause accuracy", n_boot, seed)
                        add(r, m, e, lo, hi, n, BOOT)
                for m, nm in (("MAE Δβ1, p.p.", "b1"), ("MAE Δβ1 (degradation), p.p.", "degmae")):
                    X = F10.get(f"{nm}_{est}")
                    if X is not None and len(X):
                        e, lo, hi, n = bootstrap_ci(X, m, n_boot, seed); add(r, m, e, lo, hi, n, BOOT)
        elif p in ("P12", "P13"):
            n = r["Events"]
            if pd.notna(r["Recall"]):
                lo, hi = wilson(float(r["Recall"]), float(n) if pd.notna(n) else np.nan)
                add(r, "Recall", float(r["Recall"]), lo, hi, n, "Wilson, n = events (interpolated equal-budget point)")
            if pd.notna(r["Cause accuracy"]):
                res = fx_res if p == "P12" else fcm_res
                m = _class_method(p, model) if res is not None else None
                X = _class_frame(res, m) if m else pd.DataFrame()
                if len(X):
                    e, lo, hi, n = bootstrap_ci(X, "Cause accuracy", n_boot, seed)
                    add(r, "Cause accuracy", e, lo, hi, n, BOOT)
                    continue
                C = res.tables.get("classification") if res is not None else None
                nn = np.nan
                if C is not None and "N" in C and "oracle" in model.lower():
                    a = C[C["Scenario"] == "All plant-level events"]
                    nn = float(a["N"].iloc[0]) if len(a) else np.nan
                lo, hi = wilson(float(r["Cause accuracy"]), nn)
                add(r, "Cause accuracy", float(r["Cause accuracy"]), lo, hi, nn if np.isfinite(nn) else pd.NA,
                    "Wilson, n = classified events" if np.isfinite(nn) else "not available (no event-level output)")
        elif p == "P14" and val_res is not None:
            m = next((k for k, l in lab14.items() if l == model), None)
            E = val_res.events
            X = _det_frame(E[(E["method"] == m) & np.isclose(E["budget"].astype(float), float(vb))]) if m else pd.DataFrame()
            for mm in ("Recall", "Precision", "F1"):
                e, lo, hi, n = bootstrap_ci(X, mm, n_boot, seed); add(r, mm, e, lo, hi, n, BOOT)
        elif p in ("P15", "P16"):
            res = modal_res if p == "P15" else fmm_res
            if res is None:
                continue
            b = float(res.meta.get("ref_budget", 1.0))
            m = next((k for k in res.events["method"].unique() if _modal_label(k) == model), None)
            if m is None:
                continue
            X = _modal_frame(res.events, m, b)
            for mm in ("Recall", "Nuisance share", "Macro-F1 (cause)"):
                e, lo, hi, n = bootstrap_ci(X, mm, n_boot, seed); add(r, mm, e, lo, hi, n, BOOT)
    T = pd.DataFrame(rows, columns=["Protocol", "Task", "Model", "Metric", "Value", "CI low", "CI high", "Events", "CI method"])
    if len(T):
        T["Events"] = pd.to_numeric(T["Events"], errors="coerce").round().astype("Int64")
    return T


def summary_with_ci(S: pd.DataFrame, CI: pd.DataFrame, digits: int = 3) -> pd.DataFrame:
    """Compact table: one row per summary row, each metric as 'value [low, high]'."""
    if not len(CI):
        return pd.DataFrame()
    f = lambda x: "--" if not np.isfinite(x) else f"{(0.0 if abs(x) < 0.5 * 10 ** -digits else x):.{digits}f}"
    C = CI.assign(cell=[f"{f(v)} [{f(l)}, {f(h)}]" if np.isfinite(l) else f(v)
                        for v, l, h in zip(CI["Value"].astype(float), CI["CI low"].astype(float), CI["CI high"].astype(float))])
    W = C.pivot_table(index=["Protocol", "Task", "Model"], columns="Metric", values="cell", aggfunc="first", sort=False)
    order = [m for m in METRICS if m in W.columns]
    W = W[order].rename(columns=lambda c: f"{c} (95% CI)")   # no "[" after a LaTeX line break
    how = C.groupby(["Protocol", "Task", "Model"], sort=False)["CI method"].first()
    ev = C.groupby(["Protocol", "Task", "Model"], sort=False)["Events"].first()
    W.insert(0, "Events", ev); W["CI method"] = how
    W = W.reset_index()
    key = S[["Protocol", "Task", "Model"]].drop_duplicates().reset_index(drop=True).reset_index()
    W = W.merge(key, on=["Protocol", "Task", "Model"], how="left").sort_values("index").drop(columns="index")
    W["Events"] = [str(int(v)) if pd.notna(v) else "--" for v in W["Events"]]
    return W.fillna("").reset_index(drop=True)


# ============================================================================ pre-specified paired comparisons
# (protocol, comparison label, A, B, metrics); P10 entries refer to frame names of _p10_frames
P10_COMPS = [("Kalman vs SPS: beta1 change error (all quarters)", "b1_kalman", "b1_sps", ["MAE Δβ1, p.p."],
              ["placement", "event_id", "plant", "quarter"]),
             ("Kalman vs SPS: beta1 error under injected degradation", "degmae_kalman", "degmae_sps",
              ["MAE Δβ1 (degradation), p.p."], ["placement", "model_seed", "event_id"]),
             ("Kalman vs SPS: evidence of a beta1 decrease (degradation)", "deg_kalman", "deg_sps", ["Recall"],
              ["placement", "model_seed", "event_id"]),
             ("Fuzzy vs crisp rules: cause of detected events", "cls_detected_fuzzy", "cls_detected_crisp", ["Cause accuracy"],
              ["placement", "model_seed", "event_id"]),
             ("Fuzzy vs k-means clusters: cause of detected events", "cls_detected_fuzzy", "cls_detected_kmeans",
              ["Cause accuracy"], ["placement", "model_seed", "event_id"])]
CLS_COMPS = {"P12": [("Fuzzy class of detected events: S4 (calibrated) vs S0 (original rules)", "S4", "S0")],
             "P13": [("Fuzzy class of detected events: FCM hard vs k-means", "fcm_hard", "kmeans"),
                     ("Fuzzy class of detected events: FCM soft vs k-means", "fcm_soft", "kmeans")]}
P12_COMPS = [("STGNN S4 vs S0 (validation operating points)", "S4", "S0"),
             ("STGNN S2 vs S0 (validation operating points)", "S2", "S0"),
             ("STGNN S4 vs PR+ (validation operating points)", "S4", "prp_rule"),
             ("STGNN S4 vs PR rule (validation operating points)", "S4", "pr_rule"),
             ("PR+ vs PR rule (validation operating points)", "prp_rule", "pr_rule")]
P13_COMPS = [("FCM hard vs k-means (equal validation budget)", "fcm_hard", "kmeans"),
             ("FCM soft vs k-means (equal validation budget)", "fcm_soft", "kmeans")]
P14_COMPS = [("STGNN vs PR+", "stgnn", "prp"), ("STGNN (daylight) vs PR+", "stgnn_day", "prp"),
             ("Fusion PR+ OR STGNN vs PR+", "fusion_prp", "prp"), ("PR+ vs PR rule", "prp", "pr"),
             ("STGNN vs PR rule", "stgnn", "pr"), ("STGNN (daylight) vs PR rule", "stgnn_day", "pr"),
             ("Fusion vs PR rule", "fusion", "pr"), ("Fusion 75/25 vs PR rule", "fusion_75", "pr"),
             ("Fusion 25/75 vs PR rule", "fusion_25", "pr"), ("Fusion vs STGNN (daylight)", "fusion", "stgnn_day")]
MODAL_COMPS = [("STGNN vs PR+", "stgnn", "prp"), ("PR+ & modal vs PR+", "prp_m", "prp"),
               ("STGNN & modal vs STGNN", "stgnn_m", "stgnn"), ("STGNN-mm (early fusion) vs STGNN", "mm_early", "stgnn"),
               ("STGNN-mm + modal edges vs STGNN-mm", "mm_graph", "mm_early"),
               ("STGNN-mm + modal edges vs STGNN", "mm_graph", "stgnn"),
               ("STGNN-mm + edges & modal vs PR+ & modal", "mm_graph_m", "prp_m"),
               ("PR+ with shading mask + Kalman sd vs PR+", "prp_sk", "prp"),
               ("PR+ (shading, sd) & modal vs PR+ & modal", "prp_skm", "prp_m"),
               ("STGNN-fmm vs STGNN-mm + modal edges", "mm_fact", "mm_graph"), ("STGNN-fmm vs STGNN", "mm_fact", "stgnn"),
               ("STGNN-fmm vs PR+", "mm_fact", "prp"),
               ("STGNN-fmm & modal vs PR+ (shading, sd) & modal", "mm_fact_m", "prp_skm")]
DET_KEY = ["placement", "model_seed", "event_id"]


def _align_seeds(A: pd.DataFrame, B: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Seed-independent baselines (model_seed '-') are paired with every model seed of the other method,
    so that the inner join on (placement, model_seed, event_id) keeps all events of the STGNN seeds."""
    sa, sb = set(A["model_seed"].astype(str)), set(B["model_seed"].astype(str))
    if sb == {"-"} and sa != {"-"}:
        B = pd.concat([B.assign(model_seed=s) for s in A["model_seed"].unique()], ignore_index=True)
    elif sa == {"-"} and sb != {"-"}:
        A = pd.concat([A.assign(model_seed=s) for s in B["model_seed"].unique()], ignore_index=True)
    return A, B


def paired_comparisons(ss_res=None, fx_res=None, fcm_res=None, val_res=None, modal_res=None, fmm_res=None,
                       n_boot: int = 2000, n_perm: int = 2000, seed: int = 0, alpha: float = 0.05) -> pd.DataFrame:
    rows = []

    def add(proto, name, op, metric, t, fa=None):
        if not t.get("n"):
            return
        rows.append({"Protocol": proto, "Comparison (A vs B)": name, "Operating point": op, "Metric": metric,
                     "Events": t["n"], "A": t["a"], "B": t["b"], "Delta (A - B)": t["d"], "CI low": t["lo"],
                     "CI high": t["hi"], "p (permutation)": t["p"], "Only A hit": t.get("only_a", pd.NA),
                     "Only B hit": t.get("only_b", pd.NA), "Clean FA A / B": fa or ""})

    if ss_res is not None:
        F = _p10_frames(ss_res)
        for name, a, b, metrics, key in P10_COMPS:
            if a in F and b in F:
                for m in metrics:
                    A, B = F[a], F[b]
                    if m == "Recall":
                        A, B = A.assign(fault=True), B.assign(fault=True)
                    add("P10", name, "estimator evidence rule" if "deg" in a else
                        ("frozen STGNN detections" if a.startswith("cls") else "all evaluated quarters"), m,
                        paired_test(A, B, key, m, n_boot, n_perm, seed))
    for proto, res, comps, op in (("P12", fx_res, P12_COMPS, "validation operating point"),
                                  ("P13", fcm_res, P13_COMPS, "equal validation budget")):
        if res is None or not len(res.events):
            continue
        E = _det_frame(res.events)
        fa = res.points.groupby("method")["clean_fa"].mean() if "clean_fa" in res.points else pd.Series(dtype=float)
        for name, a, b in comps:
            if a in set(E["method"]) and b in set(E["method"]):
                fat = f"{fa.get(a, np.nan):.2f} / {fa.get(b, np.nan):.2f}"
                Ea, Eb = _align_seeds(E[E["method"] == a], E[E["method"] == b])
                for m in ("Recall", "F1"):
                    add(proto, name, op, m, paired_test(Ea, Eb, DET_KEY, m, n_boot, n_perm, seed), fat)
        for name, a, b in CLS_COMPS[proto]:
            A, B = _class_frame(res, a), _class_frame(res, b)
            if len(A) and len(B):        # events detected by both methods only (inner join)
                add(proto, name, "events detected by both methods", "Cause accuracy",
                    paired_test(A.drop(columns="method"), B.drop(columns="method"), DET_KEY, "Cause accuracy",
                                n_boot, n_perm, seed))
    if val_res is not None and len(val_res.events):
        b0 = float(val_res.meta.get("ref_budget"))
        E = _det_frame(val_res.events); E = E[np.isclose(E["budget"].astype(float), b0)]
        for name, a, b in P14_COMPS:
            if a in set(E["method"]) and b in set(E["method"]):
                for m in ("Recall", "F1"):
                    add("P14", name, f"equal alarm budget {b0:g} /1000 h", m,
                        paired_test(E[E["method"] == a], E[E["method"] == b], DET_KEY, m, n_boot, n_perm, seed))
    for proto, res in (("P15", modal_res), ("P16", fmm_res)):
        if res is None or not len(res.events):
            continue
        E = res.events; b0 = float(res.meta.get("ref_budget", 1.0)); have = set(E["method"])
        for name, a, b in MODAL_COMPS:
            if a in have and b in have:
                for m in ("Recall", "Nuisance share", "Macro-F1 (cause)"):
                    add(proto, name, f"equal clean alarm budget {b0:g} /1000 h", m,
                        paired_test(_modal_frame(E, a, b0), _modal_frame(E, b, b0), DET_KEY, m, n_boot, n_perm, seed))
    cols = ["Protocol", "Comparison (A vs B)", "Operating point", "Metric", "Events", "A", "B", "Delta (A - B)",
            "CI low", "CI high", "p (permutation)", "p (Holm)", "Only A hit", "Only B hit", "Clean FA A / B", "Result"]
    T = pd.DataFrame(rows)
    if not len(T):
        return pd.DataFrame(columns=cols)
    T["p (Holm)"] = T.groupby("Protocol", group_keys=False)["p (permutation)"].apply(holm)

    def verdict(r):
        if not (np.isfinite(r["CI low"]) and np.isfinite(r["p (Holm)"])) or not (r["CI low"] > 0 or r["CI high"] < 0) \
                or r["p (Holm)"] >= alpha:
            return "no difference shown"
        better_a = (r["Delta (A - B)"] < 0) if r["Metric"] in LOWER_IS_BETTER else (r["Delta (A - B)"] > 0)
        return "A better" if better_a else "B better"
    T["Result"] = T.apply(verdict, axis=1)
    return T[cols].reset_index(drop=True)


def compute(S: pd.DataFrame, ss_res=None, fx_res=None, fcm_res=None, val_res=None, modal_res=None, fmm_res=None,
            n_boot: int = 2000, n_perm: int = 2000, seed: int = 0) -> Dict[str, pd.DataFrame]:
    """{'ci': long CI table, 'summary_ci': compact 'value [CI]' table, 'paired': paired comparisons}."""
    kw = dict(ss_res=ss_res, fx_res=fx_res, fcm_res=fcm_res, val_res=val_res, modal_res=modal_res, fmm_res=fmm_res)
    CI = ci_table(S, n_boot=n_boot, seed=seed, **kw)
    return {"ci": CI, "summary_ci": summary_with_ci(S, CI),
            "paired": paired_comparisons(n_boot=n_boot, n_perm=n_perm, seed=seed, **kw)}
