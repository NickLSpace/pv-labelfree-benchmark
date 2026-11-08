"""Section 13 (ablation): k-means vs fuzzy c-means for weather regimes and technical clusters.

Hypothesis: "does a soft partition improve the final diagnosis?"

Only the two partitions that enter the STGNN are replaced (anomaly-event clustering, the
behavioural view and the fuzzy rules are untouched):

* weather regimes   -> context channels of the STGNN + regime-specific thresholds;
* technical clusters -> peer-edge prior of the multi-view graph + error-normalisation /
  threshold borrowing for plants without training data.

Variants (same k as k-means, same seeds / epochs / window / hidden, same semi-synthetic
placements as sections 10-12, same AFTER detector of section 12):

``kmeans``    existing frozen models (reference);
``fcm_hard``  FCM, hard argmax labels everywhere (isolates the algorithm change);
``fcm_soft``  FCM memberships used as soft information: context channels
              [night, (1 - night) * u_r], soft peer prior across + (1 - across) * cos(u_i, u_j),
              regime thresholds mixed as sum_r u_r * thr_r; hard argmax only where an integer
              label is required (borrowing of error statistics).

Evaluation is on the semi-synthetic test at an EQUAL false-alarm budget (clean controls,
interpolated along the threshold curve) and at the operating point chosen on validation
with the budget of section 12 — not by internal clustering indices. Internal indices
(partition coefficient, ARI to k-means) are reported for information only.

The models are trained ONCE per variant / seed and frozen; independent from the main
pipeline (nothing in run_pipeline / the notebook depends on this module).

Usage (notebook, after sections 10-12)::

    import pv_fcm_ablation as fca
    abl = fca.run_ablation(D, C, models, ss_res=ss_res, fixes_res=fx_res,
                           epochs=15, train_stride=1, device=None)
    fca.show(abl); fca.plot_all(abl); fca.export_latex(abl)
"""
from __future__ import annotations

import copy
import os
import pickle
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

import pv_semisynth_eval as sse
import pv_semisynth_gen as ssg
import pv_semisynth_sensitivity as sss
import pv_fixes_eval as fx

VARIANTS = ("kmeans", "fcm_hard", "fcm_soft")
VARIANT_LABEL = {"kmeans": "k-means (hard, reference)", "fcm_hard": "FCM, hard argmax",
                 "fcm_soft": "FCM, soft memberships"}
COLORS = {"kmeans": "#444444", "fcm_hard": "#1f77b4", "fcm_soft": "#d62728"}


def _log(*a):
    print(time.strftime("%H:%M:%S"), "[fcm]", *a, flush=True)


# ============================================================================ fuzzy c-means
@dataclass
class FCMResult:
    centers: np.ndarray
    U: np.ndarray                 # (n, c) memberships of the fitted data
    m: float
    objective: float
    n_iter: int

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return memberships(X, self.centers, self.m)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(1)


def memberships(X: np.ndarray, centers: np.ndarray, m: float = 2.0) -> np.ndarray:
    d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
    d2 = np.maximum(d2, 1e-12)
    p = 1.0 / (m - 1.0)
    inv = d2 ** (-p)
    return inv / inv.sum(1, keepdims=True)


def fcm(X: np.ndarray, c: int, m: float = 2.0, max_iter: int = 300, tol: float = 1e-6, seed: int = 0,
        n_init: int = 5) -> FCMResult:
    """Bezdek's fuzzy c-means (Euclidean); k-means++ initialisation, best of ``n_init``."""
    from sklearn.cluster import kmeans_plusplus
    X = np.asarray(X, float)
    best = None
    for s in range(n_init):
        V, _ = kmeans_plusplus(X, c, random_state=seed + s)
        J_old = np.inf
        for it in range(max_iter):
            U = memberships(X, V, m)
            W = U ** m
            V = (W.T @ X) / np.maximum(W.sum(0)[:, None], 1e-12)
            J = float((W * ((X[:, None, :] - V[None]) ** 2).sum(-1)).sum())
            if abs(J_old - J) <= tol * max(abs(J), 1.0):
                break
            J_old = J
        U = memberships(X, V, m)
        r = FCMResult(V, U, m, J, it + 1)
        if best is None or r.objective < best.objective:
            best = r
    return best


def partition_coefficient(U: np.ndarray) -> float:
    return float((U ** 2).sum(1).mean())


# ============================================================================ partitions
def _weather_frame(D: dict, C: dict) -> pd.DataFrame:
    if C.get("W") is not None:
        return C["W"]
    import pv_clustering as clu
    return clu.weather_features(D["df"], kt=D.get("kt"))


def _wr_k(C: dict) -> int:
    wr = C["wr"]
    res = getattr(wr, "result", None)
    if res is not None and getattr(res, "k", None):
        return int(res.k)
    return int(np.asarray(wr.onehot).shape[1] - 1)


def weather_fcm(D: dict, C: dict, m: float = 2.0, seed: int = 0, day_thr: float = 20.0,
                max_fit: int = 20000) -> dict:
    """FCM weather regimes with the same k, fit on TRAIN daylight hours only; regimes ordered
    by mean GHI (descending) like ``pv_clustering.weather_regimes``."""
    W = _weather_frame(D, C).reindex(D["df"].index)
    W = W.fillna(W.median())
    k = _wr_k(C)
    day = (W["GHI"] > day_thr).to_numpy()
    train = (W.index < pd.Timestamp(sse.BOUNDARY))
    res = getattr(C["wr"], "result", None)
    sc = getattr(res, "scaler", None) if res is not None else None
    if sc is None or getattr(sc, "n_features_in_", W.shape[1]) != W.shape[1]:
        sc = StandardScaler().fit(W.loc[day & train].to_numpy(float))
    Xf = sc.transform(W.loc[day & train].to_numpy(float))
    rng = np.random.default_rng(seed)
    if len(Xf) > max_fit:
        Xf = Xf[rng.choice(len(Xf), max_fit, replace=False)]
    f = fcm(Xf, k, m=m, seed=seed)
    Xd = sc.transform(W.loc[day].to_numpy(float))
    Ud = f.predict_proba(Xd)
    # order daylight regimes by mean GHI (descending)
    ghi = W.loc[day, "GHI"].to_numpy(float)
    mean_ghi = (Ud * ghi[:, None]).sum(0) / np.maximum(Ud.sum(0), 1e-12)
    order = np.argsort(-mean_ghi)
    Ud = Ud[:, order]
    T = len(W)
    U = np.zeros((T, k + 1), np.float32)
    U[~day, 0] = 1.0
    U[day, 1:] = Ud
    hard = U.argmax(1)
    regime = pd.Series(hard, index=W.index, name="weather_regime")
    km = C["wr"].regime.reindex(W.index).fillna(0).astype(int).to_numpy()
    return dict(U=U, regime=regime, k=k, m=m, fcm=f,
                pc=partition_coefficient(Ud), ari_kmeans=float(adjusted_rand_score(km[day], hard[day])),
                ambiguous=float((Ud.max(1) < 0.6).mean()), mean_max_u=float(Ud.max(1).mean()))


def _tech_features(D: dict, C: dict) -> Tuple[pd.DataFrame, str]:
    tech = C["tech"]
    F = getattr(tech, "features", None)
    pv = list(D["pv_cols"])
    if isinstance(F, pd.DataFrame) and len(F):
        return F.reindex(pv), "technical features of section 4"
    if D.get("brick") is not None and D.get("mapping") is not None:
        import pv_clustering as clu
        F = clu.technical_features(D["brick"], D["mapping"], pv)
        X = StandardScaler().fit_transform(F.to_numpy(float))
        return pd.DataFrame(X, index=pv, columns=F.columns), "technical features (rebuilt from Brick)"
    # test fixture: rated power + mean daytime performance ratio on the training period
    df, pe = D["df"], np.asarray(D["phys"]["p_expected"], float)
    tr = (df.index < pd.Timestamp(sse.BOUNDARY))
    rows = {}
    for j, c in enumerate(pv):
        p = df[c].to_numpy(float)[tr]; e = pe[tr, j]
        ok = np.isfinite(p) & (e > 0.2 * np.nanmax(e))
        rows[c] = dict(log_rated=np.log1p(D["rated"].get(c, 0.0)),
                       pr=float(np.nanmedian(p[ok] / e[ok])) if ok.any() else 1.0)
    F = pd.DataFrame(rows).T
    return pd.DataFrame(StandardScaler().fit_transform(F), index=pv, columns=F.columns), "fallback features"


def tech_fcm(D: dict, C: dict, m: float = 2.0, seed: int = 0) -> dict:
    F, src = _tech_features(D, C)
    lab_km = C["tech"].labels.reindex(F.index)
    k = int(lab_km.nunique())
    f = fcm(F.to_numpy(float), k, m=m, seed=seed, n_init=10)
    U = f.U
    hard = pd.Series(U.argmax(1), index=F.index, name="tech_cluster")
    return dict(U=pd.DataFrame(U, index=F.index), labels=hard, k=k, m=m, fcm=f, source=src,
                pc=partition_coefficient(U), ari_kmeans=float(adjusted_rand_score(lab_km.to_numpy(), hard.to_numpy())),
                ambiguous=float((U.max(1) < 0.6).mean()), mean_max_u=float(U.max(1).mean()))


def soft_peer_prior(U: pd.DataFrame, across: float = 0.5) -> pd.DataFrame:
    """across + (1 - across) * cosine(u_i, u_j); equals the hard prior for crisp memberships."""
    A = U.to_numpy(float)
    nrm = np.linalg.norm(A, axis=1, keepdims=True)
    S = (A / np.maximum(nrm, 1e-12)) @ (A / np.maximum(nrm, 1e-12)).T
    return pd.DataFrame(across + (1 - across) * np.clip(S, 0, 1), index=U.index, columns=U.index)


def _rebuild_graph(D: dict, C: dict, prior: pd.DataFrame):
    b = D.get("builder")
    if b is None or not hasattr(b, "build") or "graph_base" not in D:
        return C["graph_mv"], False
    g0 = C["graph_mv"]
    g = b.build(D["df"], D["pv_cols"], exo_cols=D["graph_base"].meta["exo_cols"], peer_fit_end=sse.BOUNDARY,
                observed=D["observed"], peer_prior=prior, extra_edges=C.get("b_edges"))
    for key in ("behav_cluster",):
        if key in getattr(g0, "meta", {}):
            g.meta[key] = g0.meta[key]
    return g, True


def make_variant_C(D: dict, C: dict, variant: str, wf: dict, tf: dict, across: float = 0.5) -> dict:
    """Copy of the cluster stage dict ``C`` with the weather regimes / technical clusters replaced."""
    if variant == "kmeans":
        return C
    import pv_clustering as clu
    Cv = dict(C)
    k = wf["k"]
    reg = wf["regime"]
    onehot = (np.eye(k + 1, dtype=np.float32)[reg.to_numpy()] if variant == "fcm_hard" else wf["U"].astype(np.float32))
    names = dict(getattr(C["wr"], "names", {}) or {})
    Cv["wr"] = SimpleNamespace(regime=reg, onehot=onehot, names=names or {r: f"R{r}" for r in range(k + 1)},
                               soft=wf["U"] if variant == "fcm_soft" else None, result=None)
    lab = tf["labels"]
    Cv["tech"] = SimpleNamespace(labels=lab, features=getattr(C["tech"], "features", None),
                                 soft=tf["U"] if variant == "fcm_soft" else None)
    prior = soft_peer_prior(tf["U"], across) if variant == "fcm_soft" else clu.peer_prior_from_clusters(lab, across)
    g, rebuilt = _rebuild_graph(D, C, prior)
    if hasattr(g, "meta"):
        g.meta = dict(g.meta); g.meta["tech_cluster"] = lab.to_dict()
    Cv["graph_mv"] = g
    Cv["peer_prior"] = prior
    if C.get("F") is not None and "cold" in C:
        try:
            Cv["cold"] = clu.cold_start_baseline(C["F"], lab)
        except Exception:                       # pragma: no cover - keep the k-means cold-start baseline
            pass
    Cv["_fcm"] = dict(variant=variant, graph_rebuilt=rebuilt)
    return Cv


# ============================================================================ models
def variant_models(D: dict, Cv: dict, variant: str, seeds: Sequence[int], cache_dir: str, epochs: int = 15,
                   train_stride: int = 1, window: int = 48, hidden: int = 32, device=None,
                   base_models: Optional[Dict[int, "sse.FrozenSTGNN"]] = None) -> Dict[int, "sse.FrozenSTGNN"]:
    if variant == "kmeans":
        assert base_models is not None
        return {s: base_models[s] for s in seeds}
    os.makedirs(cache_dir, exist_ok=True)
    out = {}
    for s in seeds:
        p = os.path.join(cache_dir, f"stgnn_frozen_multiview_{variant}_{s}.pt")
        m = None
        if os.path.exists(p):
            try:
                m = sse.FrozenSTGNN.load(p, D, Cv, device, "multiview", s, window, hidden)
            except sse.CheckpointMismatch as exc:
                _log(f"{variant} seed {s}: checkpoint not used — {exc}; retraining")
        if m is not None:
            out[s] = m
        else:
            _log(f"{variant} seed {s}: training ({epochs} epochs), then freezing")
            out[s] = sse.FrozenSTGNN.train(D, Cv, s, "multiview", window, hidden, epochs, train_stride, device, p)
        _log(f"{variant} seed {s}: {out[s].source}, weights sha256[:16] = {out[s].fingerprint()}")
    return out


# ============================================================================ detector families
def soft_stgnn_family(R: "sss.ModelRaw", eta_mode: str, U: np.ndarray,
                      scfg: Optional["sss.SensConfig"] = None) -> "fx.Family":
    """As ``fx.stgnn_family`` but regime thresholds are mixed with the soft memberships:
    thr_t = sum_r u_{t,r} * thr_r (missing thr_r -> node / cluster / POT threshold)."""
    fam = fx.stgnn_family(R, eta_mode, scfg)
    scfg = scfg or sss.SensConfig()
    calib, pv = R.calib, R.pv_cols
    tmed = sss._train_eta(R.cal, calib, pv, scfg.day_frac)
    ecal = sss.eta_variant(eta_mode, R.cal, tmed, scfg)
    phy_cal = sss._phy(R.cal, ecal)
    mu, sd = (calib["stats"]["phy_mu"], calib["stats"]["phy_sd"]) if eta_mode == "current" else \
        sss._phy_stats(phy_cal, R.cal["active"][:, :calib["n_pv"]], calib, pv)
    cal_comb, _ = sss._combined(R.cal, phy_cal, calib, mu, sd)
    cache = {}

    def thr_fn(q):
        q = float(q)
        if q not in cache:
            th = sss.thresholds_from_cal(cal_comb, R.cal["pos"], q, calib, pv)
            base_th = dict(th, regime_thresholds=None)
            out = {}
            for k, s in fam.sets.items():
                pos = s["raw"]["pos"]
                base = sss.threshold_matrix(base_th, pos, calib)
                rt = th["regime_thresholds"]
                if rt is None:
                    out[k] = base; continue
                Up = U[pos][:, :rt.shape[0]]
                T = np.zeros_like(base)
                for r in range(rt.shape[0]):
                    tr = np.where(np.isfinite(rt[r]), rt[r][None, :], base)
                    T += Up[:, r:r + 1] * tr
                out[k] = T / np.maximum(Up.sum(1, keepdims=True), 1e-12)
            cache[q] = out
        return cache[q]
    return fx.Family(fam.name + "_soft", fam.sets, thr_fn, fam.model_seed, fam.q_default)


# ============================================================================ orchestration
@dataclass
class AblationConfig:
    m: float = 2.0
    across: float = 0.5
    eta_mode: str = "med7d"
    min_len: int = 2
    zero_rule: bool = True
    fuzzy: str = "calibrated"
    budgets: Sequence[float] = (0.5, 1.0, 2.0, 5.0)     # clean-control false alarms per 1000 h
    qs: Sequence[float] = (0.9, 0.95, 0.97, 0.98, 0.99, 0.993, 0.995, 0.997, 0.998, 0.999, 0.9995)
    variants: Sequence[str] = VARIANTS


@dataclass
class AblationResult:
    points: pd.DataFrame           # validation-chosen operating point per variant / seed
    curves: pd.DataFrame           # threshold curves per variant / seed
    events: pd.DataFrame
    classes: pd.DataFrame
    partitions: pd.DataFrame
    tables: Dict[str, pd.DataFrame] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)


def _ref_budget(fixes_res, raws_km, cfg_fx) -> float:
    """Validation budget = section 12 (S0 val rate with floor); recomputed if section 12 is absent."""
    if fixes_res is not None and "val_budget" in fixes_res.points:
        v = fixes_res.points["val_budget"].dropna()
        if len(v):
            return float(v.mean())
    b = []
    for R in raws_km.values():
        fam = fx.stgnn_family(R, "current")
        thr = fam.thr_fn(R.calib["q"])
        v = fam.sets["val"]
        r = fx._rate_fast(v["comb"], thr["val"], 3)
        h = int(np.isfinite(v["comb"]).sum())
        b.append(max(r, 1000.0 * cfg_fx.min_val_budget_events / h if h else 0.0))
    return float(np.mean(b))


def run_ablation(D: dict, C: dict, models: Dict[int, "sse.FrozenSTGNN"], ss_res=None, fixes_res=None,
                 placement_seeds: Sequence[int] = (0, 1, 2), cfg: Optional[AblationConfig] = None,
                 gen_cfg=None, fuzzy_calibrated=None, epochs: int = 15, train_stride: int = 1, window: int = 48,
                 hidden: int = 32, device=None, model_dir: str = "results/cache",
                 raw_cache_dir: str = "results/fixes", out_dir: str = "results/fcm",
                 use_cache: bool = True) -> AblationResult:
    import pv_fuzzy as fz
    import pv_fuzzy_calibration as fzc
    cfg = cfg or AblationConfig()
    cfx = fx.FixesConfig(eta_mode=cfg.eta_mode, min_len_after=cfg.min_len)
    t0 = time.time()
    seeds = sorted(models)
    if ss_res is not None and getattr(ss_res, "plans", None):
        plans = dict(ss_res.plans)
    else:
        gen_cfg = gen_cfg or ssg.GenConfig(boundary=sse.BOUNDARY)
        plans = {ps: ssg.plan_semisynthetic(D, C, seed=ps, cfg=gen_cfg) for ps in placement_seeds}
    reals = {ps: ssg.apply_plan(D, C, p) for ps, p in plans.items()}     # SAME injected data for all variants
    if fuzzy_calibrated is None:
        if fixes_res is not None and fixes_res.meta.get("fuzzy_calibration"):
            fuzzy_calibrated = fz.FuzzyConfig(rule_set="calibrated", calibration=fixes_res.meta["fuzzy_calibration"])
        else:
            fuzzy_calibrated, _ = fzc.fit(D, C, fzc.CAL_PLACEMENTS, gen_cfg or ssg.GenConfig(boundary=sse.BOUNDARY),
                                          eval_placements=sorted(plans))
    fz_cfg = {"full": fz.FuzzyConfig(), "calibrated": fuzzy_calibrated}

    wf = weather_fcm(D, C, cfg.m)
    tf = tech_fcm(D, C, cfg.m)
    _log(f"weather FCM: k={wf['k']} PC={wf['pc']:.3f} ARI vs k-means={wf['ari_kmeans']:.3f} "
         f"ambiguous hours={wf['ambiguous']:.3f}")
    _log(f"technical FCM: k={tf['k']} PC={tf['pc']:.3f} ARI vs k-means={tf['ari_kmeans']:.3f} ({tf['source']})")
    part = pd.DataFrame([
        dict(Partition="Weather regimes (daylight hours)", k=wf["k"], m=cfg.m, **{
            "Partition coefficient": wf["pc"], "Mean max membership": wf["mean_max_u"],
            "Share with max membership < 0.6": wf["ambiguous"], "ARI (FCM argmax vs k-means)": wf["ari_kmeans"]}),
        dict(Partition="Technical clusters (plants)", k=tf["k"], m=cfg.m, **{
            "Partition coefficient": tf["pc"], "Mean max membership": tf["mean_max_u"],
            "Share with max membership < 0.6": tf["ambiguous"], "ARI (FCM argmax vs k-means)": tf["ari_kmeans"]})])

    P, CV, EV, CL = [], [], [], []
    budget_val = None
    graph_rebuilt = {}
    for var in cfg.variants:
        Cv = make_variant_C(D, C, var, wf, tf, cfg.across)
        graph_rebuilt[var] = Cv.get("_fcm", {}).get("graph_rebuilt", "reference")
        mods = variant_models(D, Cv, var, seeds, model_dir, epochs, train_stride, window, hidden, device, models)
        raws = fx.collect(mods, D, reals, raw_cache_dir, None, cfx.batch_size, use_cache)
        if budget_val is None:
            budget_val = _ref_budget(fixes_res, raws, cfx) if var == "kmeans" else _ref_budget(fixes_res, raws, cfx)
            _log(f"validation false-alarm budget: {budget_val:.3f} /1000 h")
        for ms, R in raws.items():
            fam = soft_stgnn_family(R, cfg.eta_mode, wf["U"]) if var == "fcm_soft" else fx.stgnn_family(R, cfg.eta_mode)
            ghis = {"val": fx._ghi_at(D["df"], R.val["ts"]), "clean": fx._ghi_at(D["df"], R.clean["ts"])}
            for ps, real in reals.items():
                ghis[ps] = fx._ghi_at(real.df, R.inj[ps]["ts"])
            zeros = ({k: fx.zero_rule_mask(s["raw"], ghis[k], R.calib["n_pv"], s["nr"], cfx) for k, s in fam.sets.items()}
                     if cfg.zero_rule else None)

            def point(q, classify):
                thr = fam.thr_fn(q)
                B = {k: fx.make_bundle(s["comb"], thr[k], s["zp"], s["eta"], s["raw"], R.pv_cols, cfg.min_len,
                                       zeros.get(k) if zeros else None) for k, s in fam.sets.items()}
                tag = dict(method=var, label=VARIANT_LABEL[var], model_seed=ms)
                summ, E, FA, cl = fx._evaluate(D, Cv, reals, B["clean"], {ps: B[ps] for ps in reals}, cfg.min_len,
                                               tag, fz_cfg[cfg.fuzzy], classify)
                return dict(tag, q=q, val_fa=fx._rate(B["val"]), **summ), E, cl

            q, vr = fx._select_q(fam, cfg.min_len, budget_val, cfx, zeros)
            row, E, cl = point(q, True)
            row["val_fa"] = vr; row["val_budget"] = budget_val
            P.append(row); EV.append(E); CL.append(cl)
            _log(f"{var} seed {ms}: q={q:.4f} val FA={vr:.2f} test FA={row['clean_fa']:.2f} "
                 f"recall={row['recall']:.3f} ({time.time() - t0:.0f}s)")
            for qq in sorted(set(map(float, cfg.qs)) | {q}):
                r2, _, _ = point(qq, False)
                CV.append(r2)
        assert all(m.unchanged() for m in mods.values()), "weights changed"

    res = AblationResult(points=pd.DataFrame(P), curves=pd.DataFrame(CV), events=pd.concat(EV, ignore_index=True),
                         classes=pd.concat([c for c in CL if len(c)], ignore_index=True) if any(len(c) for c in CL)
                         else pd.DataFrame(), partitions=part,
                         meta=dict(cfg=cfg.__dict__, seeds=seeds, placements=sorted(plans), budget_val=budget_val,
                                   graph_rebuilt=graph_rebuilt, tech_source=tf["source"], epochs=epochs,
                                   weights_unchanged=True))
    res.tables = build_tables(res)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "fcm_result.pkl"), "wb") as f:
            pickle.dump(res, f)
    _log(f"done in {time.time() - t0:.0f}s")
    return res


# ============================================================================ tables
def _interp_at(g: pd.DataFrame, budget: float, keys: Sequence[str]) -> dict:
    g = g[np.isfinite(g["clean_fa"])].sort_values(["clean_fa", "recall"])
    fa = g["clean_fa"].to_numpy(float)
    fa_u, iu = np.unique(fa, return_index=True)
    inside = len(fa_u) >= 2 and fa_u[0] <= budget <= fa_u[-1]
    r = {}
    for k in keys:
        if k not in g:
            continue
        v = g[k].to_numpy(float)[iu]; ok = np.isfinite(v)
        r[k] = (float(np.interp(budget, fa_u[ok], v[ok])) if inside and ok.sum() >= 2
                else float(g.iloc[int(np.argmin(np.abs(fa - budget)))][k]))
    r["interpolated"] = bool(inside)
    return r


def build_tables(res: AblationResult) -> Dict[str, pd.DataFrame]:
    T = {"partitions": res.partitions}
    scn = [f"R_{s}" for s in fx.SCN_ORDER]
    titles = {f"R_{k}": fx.SCN_TITLE[k] for k in fx.SCN_ORDER}
    Pt = res.points
    num = [c for c in ["q", "val_fa", "clean_fa", "bg_all", "recall", "precision", "F1", "delay_h"] + scn if c in Pt]
    A = Pt.groupby(["method", "label"], sort=False)[num].mean().reset_index()
    sd = Pt.groupby(["method"], sort=False)["recall"].std().reindex(A["method"]).to_numpy()
    A.insert(A.columns.get_loc("recall") + 1, "recall_sd", sd)
    T["operating_points"] = A.drop(columns=["method"] + [c for c in scn if c in A]).rename(columns={
        "label": "Variant", "val_fa": "Val. FA /1000 h", "clean_fa": "Test FA (controls) /1000 h",
        "bg_all": "Test FA (all plants) /1000 h", "recall": "Recall", "recall_sd": "Recall SD (seeds)",
        "precision": "Precision", "delay_h": "Delay, h (median)"})
    # equal false-alarm budgets
    budgets = list(res.meta["cfg"]["budgets"])
    ref = Pt[Pt["method"] == "kmeans"]["clean_fa"].mean() if (Pt["method"] == "kmeans").any() else np.nan
    if np.isfinite(ref):
        budgets = sorted(set(budgets) | {round(float(ref), 3)})
    rows = []
    for (m, lbl, ms), g in res.curves.groupby(["method", "label", "model_seed"], sort=False):
        for b in budgets:
            rows.append(dict(method=m, label=lbl, model_seed=ms, budget=b,
                             **_interp_at(g, b, ["recall", "precision", "F1", "bg_all"] + scn)))
    F = pd.DataFrame(rows)
    keys = [c for c in ["recall", "precision", "F1", "bg_all"] + scn if c in F]
    G = F.groupby(["budget", "method", "label"], sort=False)[keys].mean().reset_index()
    G["recall_sd"] = F.groupby(["budget", "method", "label"], sort=False)["recall"].std().values
    G["interpolated"] = F.groupby(["budget", "method", "label"], sort=False)["interpolated"].all().values
    G["_o"] = G["method"].map({v: i for i, v in enumerate(VARIANTS)})
    G = G.sort_values(["budget", "_o"]).drop(columns="_o")
    T["equal_budget"] = G[["budget", "label", "recall", "recall_sd", "precision", "F1", "bg_all", "interpolated"]].rename(
        columns={"budget": "FA budget (controls) /1000 h", "label": "Variant", "recall": "Recall",
                 "recall_sd": "Recall SD (seeds)", "precision": "Precision", "bg_all": "Test FA (all plants) /1000 h",
                 "interpolated": "Interpolated"}).reset_index(drop=True)
    # deltas vs k-means at each budget
    W = G.pivot_table(index="budget", columns="method", values="recall")
    D = pd.DataFrame(index=W.index)
    for v in ("fcm_hard", "fcm_soft"):
        if v in W and "kmeans" in W:
            D[f"ΔRecall {VARIANT_LABEL[v]} − k-means"] = W[v] - W["kmeans"]
    if len(D.columns):
        T["delta_recall"] = D.reset_index().rename(columns={"budget": "FA budget (controls) /1000 h"})
    # per-scenario recall at the reference budget (k-means operating point)
    if np.isfinite(ref):
        bsel = round(float(ref), 3)
        S = G[G["budget"] == bsel]
        T["scenario_recall"] = S[["label", "recall"] + [c for c in scn if c in S]].rename(
            columns={"label": "Variant", "recall": "Recall (all)", **titles}).reset_index(drop=True)
        T["scenario_recall"].insert(1, "FA budget (controls) /1000 h", bsel)
    # classification of detected events (operating points chosen on validation)
    if len(res.classes):
        rows = []
        for m in [v for v in VARIANTS if v in set(res.classes["method"])]:
            sr = fx._class_share(res.classes[res.classes["method"] == m], res.events[res.events["method"] == m])
            rows.append(pd.Series(sr, name=VARIANT_LABEL[m]))
        Tc = pd.DataFrame(rows).T
        Tc = Tc.reindex([s for s in fx.SCN_ORDER if s in Tc.index] + [i for i in Tc.index if i not in fx.SCN_ORDER])
        Tc.index = [fx.SCN_TITLE.get(i, "All plant-level events" if i == "all" else i) for i in Tc.index]
        T["classification"] = Tc.reset_index().rename(columns={"index": "Scenario"})
    return T


def show(res: AblationResult, digits: int = 3):
    try:
        from IPython.display import display
    except Exception:                                   # pragma: no cover
        display = print
    titles = {"partitions": "Partitions (information only; not the evaluation criterion)",
              "operating_points": "Operating points chosen on validation (same budget as section 12)",
              "equal_budget": "Equal false-alarm budget on the clean test (interpolated)",
              "delta_recall": "Recall difference to k-means at equal budget",
              "scenario_recall": "Recall by scenario at the k-means operating budget",
              "classification": "Share of detected events with the correct fuzzy class"}
    for k, t in res.tables.items():
        print(f"\n=== {titles.get(k, k)}")
        display(t.round(digits))
    print(f"\nGraph rebuilt with FCM prior: {res.meta.get('graph_rebuilt')}; technical features: {res.meta.get('tech_source')}")


def plot_all(res: AblationResult, fig_dir: Optional[str] = "results/figures") -> Dict[str, object]:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter
    figs = {}
    Cv = res.curves
    num = [c for c in Cv.columns if pd.api.types.is_numeric_dtype(Cv[c]) and c not in ("model_seed", "q", "method")]
    A = Cv.groupby(["method", "q"], sort=False)[num].mean().reset_index()
    f, ax = plt.subplots(1, 2, figsize=(14, 4.6))
    for v in [v for v in VARIANTS if v in set(A["method"])]:
        g = A[A["method"] == v].sort_values("clean_fa")
        ax[0].plot(g["clean_fa"], g["recall"], "o-", c=COLORS[v], ms=4, label=VARIANT_LABEL[v])
        ax[1].plot(g["clean_fa"], g["F1"], "o-", c=COLORS[v], ms=4, label=VARIANT_LABEL[v])
    P = res.points.groupby("method", sort=False)[["clean_fa", "recall"]].mean()
    for v in P.index:
        ax[0].scatter([P.loc[v, "clean_fa"]], [P.loc[v, "recall"]], marker="*", s=150, c=COLORS[v], edgecolor="k", zorder=5)
    for a in ax:
        for b in res.meta["cfg"]["budgets"]:
            a.axvline(b, c="grey", ls=":", lw=0.8)
        a.set_xscale("symlog", linthresh=0.1)
        a.set_xlim(left=0)
        a.xaxis.set_major_locator(FixedLocator([0, 0.1, 0.3, 1, 3, 10, 30, 100]))
        a.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
        a.set_xlabel("false alarms on control plants, per 1000 h (clean test)"); a.grid(alpha=.3)
    ax[0].set(ylabel="recall (day + week events)", title="k-means vs FCM: recall vs false alarms (stars: validation op. points)")
    ax[1].set(ylabel="F1", title="k-means vs FCM: F1 vs false alarms (dotted: budgets)")
    ax[0].legend(fontsize=8, loc="lower right")
    f.tight_layout(); figs["curves"] = f
    S = res.tables.get("scenario_recall")
    if S is not None and len(S):
        cols = [fx.SCN_TITLE[k] for k in fx.SCN_ORDER if fx.SCN_TITLE[k] in S]
        f, ax = plt.subplots(figsize=(14, 4.6))
        w = 0.8 / len(S); x = np.arange(len(cols))
        lab2v = {VARIANT_LABEL[v]: v for v in VARIANTS}
        for i, (_, r) in enumerate(S.iterrows()):
            ax.bar(x + i * w - 0.4 + w / 2, [r[c] for c in cols], w, color=COLORS[lab2v[r["Variant"]]], label=r["Variant"])
        ax.set_xticks(x); ax.set_xticklabels(cols, rotation=15, ha="right", fontsize=8)
        ax.set(ylabel="recall", ylim=(0, 1.05),
               title=f"Recall by scenario at equal budget {S['FA budget (controls) /1000 h'].iloc[0]:.2f} FA /1000 h")
        ax.legend(fontsize=8); ax.grid(alpha=.3, axis="y")
        f.tight_layout(); figs["scenarios"] = f
    if fig_dir:
        os.makedirs(fig_dir, exist_ok=True)
        for k, fg in figs.items():
            fg.savefig(os.path.join(fig_dir, f"fig_fcm_{k}.png"), dpi=150, bbox_inches="tight")
    return figs


TEX_CAPTIONS = {
    "partitions": "Fuzzy c-means partitions compared with k-means (information only)",
    "operating_points": "k-means vs fuzzy c-means: operating points chosen on validation at the false-alarm budget of section 12",
    "equal_budget": "k-means vs fuzzy c-means at equal false-alarm budgets on the clean test (semi-synthetic, frozen weights)",
    "delta_recall": "Recall difference of the FCM variants to k-means at equal false-alarm budgets",
    "scenario_recall": "Recall by scenario at the false-alarm budget of the k-means operating point",
    "classification": "Share of detected events with the correct fuzzy class (calibrated rules)",
}


def export_latex(res: AblationResult, out_dir: str = "results/fcm") -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    paths, allp = {}, ["% requires \\usepackage{booktabs,graphicx,longtable,array}", ""]
    for k, cap in TEX_CAPTIONS.items():
        t = res.tables.get(k)
        if t is None or not len(t):
            continue
        tex = sse.df_to_latex(t, cap, f"tab:fcm_{k}", digits=3)
        p = os.path.join(out_dir, f"fcm_{k}.tex")
        with open(p, "w", encoding="utf-8") as f:
            f.write(tex)
        t.to_csv(os.path.join(out_dir, f"fcm_{k}.csv"), index=False)
        paths[k] = p; allp.append(tex)
    res.points.to_csv(os.path.join(out_dir, "fcm_points_by_seed.csv"), index=False)
    res.curves.to_csv(os.path.join(out_dir, "fcm_curves_by_seed.csv"), index=False)
    p = os.path.join(out_dir, "fcm_tables.tex")
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(allp))
    paths["all"] = p
    return paths


def zip_outputs(out_dir: str = "results/fcm", fig_dir: str = "results/figures", name: str = "fcm_results.zip") -> str:
    import zipfile
    z = os.path.join(out_dir, name)
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for d in (out_dir,):
            for fn in sorted(os.listdir(d)):
                if fn.endswith((".tex", ".csv")):
                    zf.write(os.path.join(d, fn), fn)
        if fig_dir and os.path.isdir(fig_dir):
            for fn in sorted(os.listdir(fig_dir)):
                if fn.startswith("fig_fcm_"):
                    zf.write(os.path.join(fig_dir, fn), os.path.join("figures", fn))
    return z
