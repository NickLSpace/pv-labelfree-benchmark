"""Sensitivity module (q / min_len / eta_hat) on the synthetic fixture (not HKUST data).

Checks: (1) eta 'current' + checkpoint q reproduces FrozenSTGNN.score exactly (scores, thresholds,
detections); (2) recall of the recomputation = recall of pv_semisynth_eval on the same placements;
(3) weights unchanged; (4) tables / figures / LaTeX are produced."""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]
import matplotlib; matplotlib.use("Agg")
import numpy as np, pandas as pd
from _semisynth_fixture import make
import pv_semisynth_gen as ssg, pv_semisynth_eval as sse, pv_semisynth_sensitivity as sss

t0 = time.time()
D, C = make(n_pv=30, cohorts=True)
cache = os.environ.get("SENS_CACHE", "/tmp/sens_test/cache")
models = sse.frozen_models(D, C, seeds=[0], cache_dir=cache, hidden=32, epochs=5, train_stride=2, device="cpu")
m = models[0]; fp0 = m.fingerprint()
cfg_g = ssg.GenConfig(boundary=sse.BOUNDARY, n_controls=4, n_degradation=3)
plans = {s: ssg.plan_semisynthetic(D, C, seed=s, cfg=cfg_g) for s in (0, 1)}
reals = {s: ssg.apply_plan(D, C, p) for s, p in plans.items()}

# (1) exact reproduction of the frozen scorer
R = sss.collect_model(m, D, reals)
tm = sss._train_eta(R.cal, R.calib, R.pv_cols, 0.05)
cb_ref = m.score(D["df"], D["phys"], min_len=3)
ev = sss.eta_variant("current", R.clean, tm, sss.SensConfig())
comb, zp = sss._combined(R.clean, sss._phy(R.clean, ev), R.calib, R.calib["stats"]["phy_mu"], R.calib["stats"]["phy_sd"], R.not_ready)
assert np.array_equal(np.isnan(comb), np.isnan(cb_ref.combined)), "NaN pattern differs"
assert np.allclose(np.nan_to_num(comb), np.nan_to_num(cb_ref.combined), atol=1e-4), np.nanmax(np.abs(comb - cb_ref.combined))
th = {k: R.calib[k] for k in ("node_thresholds", "regime_thresholds", "cluster_threshold_prior", "threshold_pot")}
thr = sss.threshold_matrix(th, R.clean["pos"], R.calib)
assert np.allclose(thr, cb_ref.thr), "thresholds differ"
b = sss._bundle(comb, thr, zp, ev, R.clean, R.pv_cols, 3)
assert len(b.events) == len(cb_ref.events) and (b.event_mask == cb_ref.event_mask).all()
# recomputed thresholds at the checkpoint q from the stored calibration windows = checkpoint thresholds
cal_comb, _ = sss._combined(R.cal, sss._phy(R.cal, sss.eta_variant("current", R.cal, tm, sss.SensConfig())),
                            R.calib, R.calib["stats"]["phy_mu"], R.calib["stats"]["phy_sd"])
th2 = sss.thresholds_from_cal(cal_comb, R.cal["pos"], R.calib["q"], R.calib, R.pv_cols)
d = np.nanmax(np.abs(sss.threshold_matrix(th2, R.clean["pos"], R.calib) - thr))
print(f"(1) scores / thresholds / detections reproduced; recomputed-threshold max diff = {d:.2e}")
assert d < 1e-3

# (2) direct semisynth detection on the same placements
E0 = []
for s, real in reals.items():
    E0.append(sse.evaluate_detection(D, C, real, cb_ref, m.score(real.df, real.phys), sse.EvalConfig())["events"])
E0 = pd.concat(E0)

class _SS:  # minimal stand-in of SemiSynthResult for plans / reproduction
    pass
ss = _SS(); ss.plans = plans; ss.events = E0
cfg = sss.SensConfig(qs=(0.95, 0.99, 0.995, 0.999), min_lens=(1, 3, 6))
res = sss.run_sensitivity(D, C, models, ss_res=ss, cfg=cfg, cache_dir="/tmp/sens_test/out", use_cache=False)
rep = res.tables["reproduction"].set_index("Scenario")
print(rep.round(3).to_string())
assert np.allclose(rep.iloc[:, 0].values, rep.iloc[:, 1].values, equal_nan=True), "reproduction failed"
assert m.fingerprint() == fp0 and res.meta["weights_unchanged"]
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
for k in ("q_curve", "minlen_curve", "scenario", "fa_matched"):
    print("=====", k); print(res.tables[k].round(3).to_string(index=False))
figs = sss.plot_all(res, "/tmp/sens_test/fig")
tex = sss.export_latex(res, "/tmp/sens_test/out")
z = sss.zip_outputs("/tmp/sens_test/out", "/tmp/sens_test/fig")
assert os.path.exists(tex["all"]) and os.path.exists(z)
print(sorted(figs), sorted(tex)); print(f"OK ({time.time() - t0:.0f}s)")
