"""End-to-end test of pv_semisynth_gen / pv_semisynth_eval with a tiny real STGNN."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _semisynth_fixture import make
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import pv_semisynth_gen as ssg, pv_semisynth_eval as sse, pv_clustering as clu

D, C = make(n_pv=30, cohorts=True)
OUT = os.environ.get("SS_OUT", "tests/out_ss")
t0 = time.time()
plan = ssg.plan_semisynthetic(D, C, seed=0, cfg=ssg.GenConfig(n_controls=4, n_degradation=3))
print(plan.roles.value_counts().to_dict()); print(plan.summary())
real = ssg.apply_plan(D, C, plan)
print(real.labels[["scenario", "plant", "strength", "duration", "start", "n_eff_h", "true_loss_pct", "pexp_bias_pct", "weather"]].head(12).to_string())
print("plan+apply %.1fs" % (time.time() - t0))
assert real.df.index.equals(D["df"].index)
assert (real.df.loc[: "2022-12-31"].equals(D["df"].loc[: "2022-12-31"])), "train period must be untouched"
# cohorts: no injected hour outside exploitation, no plant event before the STGNN cold start ends
act = D["act"]
assert int((real.inj_mask & ~act.reindex(columns=real.inj_mask.columns)).to_numpy().sum()) == 0
info = plan.plants
pl = real.labels[real.labels["level"] == "plant"]
assert all(r.start >= info.loc[r.column, "scored_from"] for r in pl.itertuples())
assert set(plan.roles_by_cohort().index) == {"Cohort 1 (2021–2023)", "Cohort 2 (from 2022)", "Cohort 3 (2023 only)"}
assert (plan.roles_by_cohort()["control"] >= 1).all()
dg = real.labels[real.labels["level"] == "trend"]
assert set(dg["cohort"]) <= {1, 2}
d1 = dg[dg["column"] == D["pv_cols"][1]]
assert d1.empty or d1["end"].iloc[0] <= pd.Timestamp("2023-09-01"), "ramp must end at decommissioning"
print(plan.roles_by_cohort())

ck = os.path.join(OUT, "cache"); os.makedirs(ck, exist_ok=True)
t0 = time.time()
models = sse.frozen_models(D, C, seeds=[0], cache_dir=ck, hidden=32, epochs=5, train_stride=2, device="cpu")
print("train/load %.1fs" % (time.time() - t0))
# cold-start normalisation: cohort-3 nodes (no train data) borrow error stats from their cluster
import copy, torch
coh = D["ca"].cohorts["cohort"].reindex(D["pv_cols"]).to_numpy()
src = models[0].ckpt["stats_source"]
assert all(v == "cluster" for v in np.asarray(src["rec"])[: len(coh)][coh == 3]), src["rec"]
assert all(v == "personal" for v in np.asarray(src["rec"])[: len(coh)][coh == 1])
sd3 = models[0].ckpt["stats"]["rec_sd"][: len(coh)][coh == 3]
assert float(sd3.max()) < 0.9, sd3
# legacy checkpoint (saved before the fix: mu = 0, sd = 1 for cohort 3) is repaired on load
leg = copy.deepcopy(models[0].ckpt); leg.pop("stats_source")
for k in ("rec", "fore", "phy"):
    leg["stats"][f"{k}_mu"][: len(coh)][torch.as_tensor(coh == 3)] = 0.0
    leg["stats"][f"{k}_sd"][: len(coh)][torch.as_tensor(coh == 3)] = 1.0
fl = sse.FrozenSTGNN(leg, D, C, source="legacy")
assert fl.stats_repaired == int((coh == 3).sum())
assert torch.allclose(fl.ckpt["stats"]["rec_sd"], models[0].ckpt["stats"]["rec_sd"])
print("cold-start stats ok:", np.asarray(src["rec"])[: len(coh)].tolist())
try:
    models[0].build(D["df"], D["phys"]).fit(); raise SystemExit("fit not blocked")
except RuntimeError as e:
    print("blocked:", e)
# anomaly clusters A from clean events (as in stage 4)
cb = models[0].score(D["df"], D["phys"])
feat = clu.anomaly_event_features(cb.events, D["df"], D["phys"], D["pv_cols"], C["wr"].regime)
A = dict(anom=clu.anomaly_clusters(feat)) if len(feat) >= 10 else None
print("clean events:", len(cb.events))
t0 = time.time()
res = sse.run_semisynthetic(D, C, models, placement_seeds=[0, 1], estimators=("sps", "kalman"), A=A,
                            gen_cfg=ssg.GenConfig(n_controls=4, n_degradation=3), cache_dir=OUT)
print("run %.1fs" % (time.time() - t0))
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
for k, t in res.tables.items():
    print("=====", k); print(t.round(3).to_string()[:3500])
figs = sse.plot_all(res, os.path.join(OUT, "figures"))
print(sse.export_latex(res, OUT))
print(sse.zip_outputs(OUT, os.path.join(OUT, "figures")))
