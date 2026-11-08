"""Section 18 (configuration changes: module disconnection / addition) on the synthetic fixture.

Checks: generator (one change per plant, <= 1 changed plant per technical cluster and realisation,
permanent change from t0, site = sum of inverters after the change, AC limit for module_add, clean
data untouched, the default generators of sections 10 and 15 unchanged), rule-based run without
models and without inverter data, full run with inverter data and a frozen STGNN (weights
unchanged), SPS and Kalman b1, tables / figures / LaTeX in English.  HKUST: notebook section 18."""
import os, sys, time, shutil, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib; matplotlib.use("Agg")
import numpy as np, pandas as pd
import _semisynth_fixture as fxt
import pv_semisynth_config as psc, pv_semisynth_eval as sse, pv_semisynth_gen as ssg, pv_semisynth_modal_gen as smg

t0 = time.time()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out_config"); shutil.rmtree(out, ignore_errors=True)
D, C = fxt.make(cohorts=True); M = fxt.make_modal(D)
df_ref = D["df"].copy(); inv_ref = {k: v.copy() for k, v in M.inv.items()}
base10 = ssg.plan_semisynthetic(D, C, seed=0, cfg=ssg.GenConfig(boundary=sse.BOUNDARY)).labels.copy()
base15 = smg.plan_modal(D, C, M, 0, smg.ModalGenConfig()).labels.copy()

# ---- generator
gcfg = psc.ConfigGenConfig()
plans, reals = psc.realisations(D, C, M, (0, 1), gcfg)
L = pd.concat([r.labels.assign(placement=k) for k, r in reals.items()], ignore_index=True)
print(L.groupby(["scenario", "strength"]).size().to_dict(), "realisations", len(reals))
print(L.loc[~L["effective"], ["placement", "scenario", "plant", "start", "n_eff_h"]])
assert set(L["scenario"]) == set(psc.CONFIG_SCENARIOS) and L["effective"].mean() > 0.9
for ps, p in plans.items():
    assert p.labels["column"].is_unique, "one change per plant"
for k, r in reals.items():
    lab = r.labels
    size = C["tech"].labels.value_counts()
    for cl, n in lab["cluster"].value_counts().items():
        assert n <= max(1, int(0.2 * size.get(cl, 1))), "too many changed plants of one cluster"
    for e in lab[lab["effective"]].itertuples():
        c = e.column; idx = r.df.index
        pre = idx < e.start; post = idx >= e.start
        x0, x1 = df_ref[c].to_numpy(float), r.df[c].to_numpy(float)
        assert np.allclose(np.nan_to_num(x0[pre]), np.nan_to_num(x1[pre])), "change before t0"
        day = post & (np.nan_to_num(x0) > 0.2 * np.nanmax(x0))
        ratio = np.nanmedian(x1[day] / x0[day])
        if e.scenario == "module_loss":
            assert ratio < 1 - 0.5 * e.strength, (e.scenario, e.strength, ratio)
        else:
            assert ratio > 1 + 0.3 * e.strength or e.strength < 0.1, (e.scenario, e.strength, ratio)
        # permanent: the last 30 days are changed too
        late = day & (idx >= e.inj_end - pd.Timedelta(days=30))
        assert late.any() and np.nanmedian(np.abs(x1[late] / x0[late] - 1)) > 0.02 * e.strength / 0.05 * 0.5
        ks = r.modal.inverters_of(c)
        if ks:
            tot = sum(r.modal.inv[q]["P"].reindex(idx).to_numpy(float) for q in ks)
            ok = post & np.isfinite(tot) & (np.nan_to_num(x1) > 100)
            tot0 = sum(inv_ref[q]["P"].reindex(idx).to_numpy(float) for q in ks)
            # the site meter follows the inverter sum (same absolute change)
            assert np.nanmedian(np.abs((x1[ok] - x0[ok]) - (tot[ok] - tot0[ok])) / x0[ok]) < 1e-6
            if e.scenario == "module_add":
                for q in ks:
                    a = r.modal.inv[q]["P"].to_numpy(float); b = inv_ref[q]["P"].to_numpy(float)
                    tr = inv_ref[q].index < pd.Timestamp(gcfg.boundary)
                    lim = gcfg.inv_headroom * np.nanquantile(b[tr], 0.999)
                    assert np.nanmax(np.where(np.abs(a - b) > 1e-6, a, 0)) <= lim + 1e-6, "AC limit"
assert D["df"].equals(df_ref), "clean data modified"
assert all(M.inv[k].equals(v) for k, v in inv_ref.items()), "clean inverter data modified"
assert ssg.plan_semisynthetic(D, C, seed=0, cfg=ssg.GenConfig(boundary=sse.BOUNDARY)).labels.equals(base10)
assert smg.plan_modal(D, C, M, 0, smg.ModalGenConfig()).labels.equals(base15)
print("generator OK")

# ---- rule-based, no inverter data, no models (fast path) + b1
res0 = psc.run_config(D, C, M=None, models=None, placement_seeds=(0,), out_dir=os.path.join(out, "plain"))
assert set(res0.events["method"]) == {"pr", "prp"}, set(res0.events["method"])
assert {"sps", "kalman"} <= set(res0.b1["estimator"]) and len(res0.lags)
assert {"detection", "persistence", "paired", "b1", "b1_paired"} <= set(res0.tables), set(res0.tables)
print(res0.tables["detection"].to_string()); print(res0.tables["persistence"].to_string())
print(res0.tables["b1"].to_string()); print(res0.tables["b1_paired"].to_string())
# onset recall of the 40 % changes must be high for PR (easy case)
X = res0.events[(res0.events["budget"] == 1.0) & (res0.events["method"] == "pr") & (res0.events["strength"] == 0.4)]
assert X["detected"].mean() >= 0.5, X["detected"].mean()
# PR (fixed reference) keeps alarming more than PR+ (rolling reference) 30-90 d after the change
P = res0.events[(res0.events["budget"] == 1.0) & (res0.events["strength"] == 0.4)]
s_pr = P[P["method"] == "pr"]["share_30_90"].mean(); s_pp = P[P["method"] == "prp"]["share_30_90"].mean()
print("alarm share 30-90 d, f=40%: PR", round(s_pr, 3), "PR+", round(s_pp, 3)); assert s_pr > s_pp

# ---- full run: inverter data + frozen STGNN
ck = os.path.join(out, "cache")
models = sse.frozen_models(D, C, seeds=[0], cache_dir=ck, hidden=32, epochs=2, train_stride=8, device="cpu")
fp = models[0].fingerprint()
res = psc.run_config(D, C, M=M, models=models, placement_seeds=(0, 1), out_dir=out, b1=False)
assert models[0].fingerprint() == fp and models[0].unchanged(), "weights changed"
assert {"pr", "prp", "prp_modal", "stgnn"} <= set(res.events["method"]), set(res.events["method"])
psc.show(res)
assert "paired" in res.tables and res.tables["paired"]["p (Holm)"].ge(res.tables["paired"]["p"] - 1e-12).all()
figs = psc.plot_all(res0, os.path.join(out, "fig")); figs.update(psc.plot_all(res, os.path.join(out, "fig")))
assert len(figs) >= 4
tex = psc.export_latex(res, out); tex0 = psc.export_latex(res0, os.path.join(out, "plain"))
z = psc.zip_outputs(out, os.path.join(out, "fig"))
assert os.path.exists(z) and len(tex) >= 5 and "b1" in tex0
for f in list(tex.values()) + list(tex0.values()):
    s = open(f, encoding="utf-8").read(); assert not any("\u0400" <= ch <= "\u04ff" for ch in s), f
for k, X in list(res.tables.items()) + list(res0.tables.items()):
    s = X.to_string(); assert not any("\u0400" <= ch <= "\u04ff" for ch in s), k
# cached rerun reproduces the events
res2 = psc.run_config(D, C, M=M, models=models, placement_seeds=(0, 1), out_dir=out, b1=False)
pd.testing.assert_frame_equal(res.events.reset_index(drop=True), res2.events.reset_index(drop=True))
print(f"config_test OK in {time.time() - t0:.0f}s")
