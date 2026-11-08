"""Section 15 (multimodal STGNN, modal semi-synthetic scenarios) on the synthetic fixture.

Checks: modal fixture -> generator consistency (inverter sums = site power for fault scenarios,
meter fault only in the site meter, curtailment -> voltage above threshold), frozen multimodal
checkpoints (weights unchanged after collection, reload from cache), leave-one-modality-out and
robustness without modal channels, full run_modal with unimodal + multimodal models, tables,
figures, LaTeX export.  Real HKUST runs: notebook section 15 on Kaggle GPU."""
import os, sys, time, shutil, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib; matplotlib.use("Agg")
import numpy as np, pandas as pd
import _semisynth_fixture as fx
import pv_modal_data as md, pv_semisynth_modal_gen as smg, pv_stgnn_multimodal as smm, pv_modal_eval as me
import pv_semisynth_eval as sse

t0 = time.time()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out_modal"); shutil.rmtree(out, ignore_errors=True)
ck = os.path.join(out, "cache")
D, C = fx.make(cohorts=True); M = fx.make_modal(D)
print("inverters", len(M.inv), "sites", len(M.sites()))

# ---- generator consistency
plan = smg.plan_modal(D, C, M, 0, smg.ModalGenConfig())
R = smg.apply_modal(D, M, plan)
L = R.labels; print(L.groupby("scenario").size().to_dict())
assert set(L["scenario"]) >= {"inverter_trip", "phase_loss", "grid_curtail", "metering"}, set(L["scenario"])
for _, e in L.iterrows():
    c = e["column"]; ks = R.modal.inverters_of(c)
    if not ks or e["scenario"] == "drift_fault":
        continue
    w = (R.df.index >= e["eff_start"]) & (R.df.index <= e["eff_end"]) if pd.notna(e.get("eff_start")) else None
    if w is None or not w.any():
        continue
    tot = sum(R.modal.inv[k]["P"].to_numpy(float)[w] for k in ks); site = R.df[c].to_numpy(float)[w]
    ok = np.isfinite(tot) & np.isfinite(site) & (site > 100)
    if not ok.any():
        continue
    rel = np.median(np.abs(site[ok] - tot[ok]) / site[ok])
    if e["scenario"] == "metering":
        assert rel > 0.05, (e["scenario"], rel)
    else:
        assert rel < 0.05, (e["scenario"], rel)
print("generator consistency OK")

# ---- frozen multimodal models
kw = dict(cache_dir=ck, hidden=32, epochs=2, train_stride=8, device="cpu")
models = sse.frozen_models(D, C, seeds=[0], cache_dir=ck, hidden=32, epochs=2, train_stride=8, device="cpu")
mm = {v: smm.frozen_mm_models(D, C, M, seeds=[0], mm_variant=v, **kw) for v in ("mm_early", "mm_graph")}
fp = {v: m[0].fingerprint() for v, m in mm.items()}
g = mm["mm_graph"][0].graph; print("modal graph edge types", np.bincount(np.asarray(g.edge_type)))
assert np.asarray(g.edge_type).max() > np.asarray(C["graph_mv"].edge_type).max()
mm2 = smm.frozen_mm_models(D, C, M, seeds=[0], mm_variant="mm_graph", train_missing=False, **kw)
assert mm2[0].fingerprint() == fp["mm_graph"] and "checkpoint" in mm2[0].source
print("checkpoint reload OK")

res = me.run_modal(D, C, M, models=models, mm_models=mm, placement_seeds=(0, 1), out_dir=out)
for v, m in mm.items():
    assert m[0].fingerprint() == fp[v], "weights changed"
print("weights frozen OK")
me.show(res)
T = res.tables
for k in ("equal_budget", "by_scenario", "leave_one_out", "robustness", "verdict"):
    assert k in T and len(T[k]), k
meth = set(res.events["method"])
assert {"pr", "prp", "rules", "stgnn", "mm_early", "mm_graph"} <= meth, meth
figs = me.plot_all(res, out); tex = me.export_latex(res, out); z = me.zip_outputs(out, out)
assert len(figs) >= 3 and len(tex) >= 8 and os.path.exists(z)
# english only
for f in tex.values():
    s = open(f, encoding="utf-8").read(); assert not any("\u0400" <= ch <= "\u04ff" for ch in s), f
print(f"modal_test OK in {time.time() - t0:.0f}s")
