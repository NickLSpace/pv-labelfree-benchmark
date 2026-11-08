"""Section 16 (factorized multimodal STGNN, generated shading mask, Kalman sd) on the synthetic fixture.

Checks: shading mask recovers fixed obstructions injected in the fixture and reduces the spread of
log(P/P_exp) out of sample; Kalman interval coverage; factorized module (descriptive + adversarial
generative factors, cross-modal graph) trains, is frozen, reloads from the checkpoint with the same
fingerprint (STGNN + factorized weights); run_modal with the new methods, the optional
"new obstruction" scenario and robustness to removed modalities; English tables, figures, LaTeX."""
import os, sys, time, shutil, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib; matplotlib.use("Agg")
import numpy as np, pandas as pd
import _semisynth_fixture as fx
import pv_modal_data as md, pv_semisynth_modal_gen as smg, pv_stgnn_multimodal as smm, pv_modal_eval as me
import pv_semisynth_eval as sse, pv_shading as psh, pv_kalman_uncertainty as pku, pv_stgnn_factorized as pfm

t0 = time.time()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out_fmm"); shutil.rmtree(out, ignore_errors=True)
ck = os.path.join(out, "cache")
D, C = fx.make(cohorts=True)
shaded = fx.add_shading(D)                 # fixed obstructions (before the inverter channels are derived)
M = fx.make_modal(D)

# ---- shading mask
S = psh.fit_shading(D, C)
ev = psh.evaluate(D, C, S); print(ev["summary"].round(3).to_string())
for c, (a0, a1, h, d) in shaded.items():
    L = S.loss.get(c)
    assert L is not None, c
    ia = (S.az_edges[:-1] >= a0) & (S.az_edges[1:] <= a1); ie = S.el_edges[1:] <= h
    inside = L[np.ix_(ia, ie)]; inside = inside[np.isfinite(inside)]
    assert inside.size and np.median(inside) > 0.15, (c, np.median(inside) if inside.size else None)
ps = ev["per_site"]; pc = [c for c in ps.columns if c.lower() == "period"][0]
tst = ps[(ps[pc] == "test") & ps["Site"].isin([psh._short(c) for c in shaded])]
better = (tst["Spread, corrected"] < tst["Spread, raw"]).sum(); print("injected sites with lower test spread", better, "/", len(tst))
assert len(tst) and better >= max(1, len(tst) - 1)
print("shading recovery OK")
# ---- Kalman sd
K = pku.fit_uncertainty(D, C); kc = pku.evaluate(D, C, K)["calibration"]; print(kc.round(3).to_string())
assert (kc["Coverage of the 95 % interval (median)"] > 0.8).all()

# ---- models
kw = dict(cache_dir=ck, hidden=32, epochs=2, train_stride=8, device="cpu")
models = sse.frozen_models(D, C, seeds=[0], cache_dir=ck, hidden=32, epochs=2, train_stride=8, device="cpu")
mm = {"mm_graph": smm.frozen_mm_models(D, C, M, seeds=[0], mm_variant="mm_graph", **kw)}
fcfg = pfm.FMMConfig(epochs=2, stride=12)
mm["mm_fact"] = pfm.frozen_fmm_models(D, C, M, seeds=[0], fmm_cfg=fcfg, S=S, K=K, **kw)
f0 = mm["mm_fact"][0]; fp = f0.fingerprint()
assert f0.unchanged()
assert os.path.basename(smm.checkpoint_path(ck, "mm_fact", 0)) == "stgnn_mm_mm_fact_0.pt"
re = pfm.frozen_fmm_models(D, C, M, seeds=[0], train_missing=False, **kw)
assert re[0].fingerprint() == fp and "checkpoint" in re[0].source
print("checkpoint reload OK", fp)
# the fingerprint covers the factorized weights
import copy, torch
ck2 = copy.deepcopy(f0.ckpt); k0 = sorted(ck2["fmm"]["net"])[0]; ck2["fmm"]["net"][k0] = ck2["fmm"]["net"][k0] + 1e-3
assert pfm.FrozenFMM(ck2, D, C, M, seed=0, device="cpu").fingerprint() != fp
# a missing modality is generated (inputs without inverter data)
I0 = f0.inputs(D["df"], D["phys"]); I1 = f0.inputs(D["df"], D["phys"], drop=("inverter",))
assert I1.A[..., 0].sum() == 0 and I0.A[..., 0].sum() > 0
o1 = f0.fmm.complete(I1, per_group=False)
assert np.abs(o1["completed"][..., :2]).sum() > 0
print("generation OK")

gen = smg.ModalGenConfig(scenarios=tuple(smg.MODAL_SCENARIOS) + ("new_obstruction",))
res = me.run_modal(D, C, M, models=models, mm_models=mm, placement_seeds=(0, 1), gen_cfg=gen, out_dir=out,
                   shading=S, kunc=K)
assert f0.unchanged() and f0.fingerprint() == fp, "weights changed"
print("weights frozen OK")
me.show(res)
T = res.tables
for k in ("equal_budget", "by_scenario", "robustness", "verdict", "fa_shaded"):
    assert k in T and len(T[k]), k
meth = set(res.events["method"])
assert {"prp_sk", "prp_skm", "mm_fact", "mm_fact_m", "mm_fact|no shading", "mm_fact|no Kalman sd",
        "mm_fact|no modal channels", "mm_graph|no modal channels"} <= meth, meth
assert "new_obstruction" in set(res.events["scenario"])
V = T["verdict"]; assert any(V["Comparison"].str.startswith("STGNN-fmm & modal vs PR+")), V["Comparison"].tolist()
for k in ("fmm_crossmodal", "fmm_generation", "fmm_attention", "shading_summary", "kalman_calibration"):
    assert k in res.stage16 and len(res.stage16[k]), k
figs = me.plot_all(res, out); tex = me.export_latex(res, out); z = me.zip_outputs(out, out)
assert {"scenarios", "fmm_crossmodal", "fmm_attention"} <= set(figs), set(figs)
for f in tex.values():
    s = open(f, encoding="utf-8").read(); assert not any("\u0400" <= ch <= "\u04ff" for ch in s), f
for t in list(T.values()) + list(res.stage16.values()):
    s = t.to_csv(); assert not any("\u0400" <= ch <= "\u04ff" for ch in s)
print(f"fmm_test OK in {time.time() - t0:.0f}s")
