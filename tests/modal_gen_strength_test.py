"""Regression test for the modal generator (sections 15 / 16): labelled strengths.

Bug fixed: the ``metering`` branch of ``plan_modal`` did not draw its own strength (it reused the last
``grid_curtail`` factor) and the ``new_obstruction`` branch overwrote ``obstruction_block`` with
``metering_factor``.  Checks:
* every labelled strength lies in the range configured for its own scenario (all placements);
* metering strengths are drawn per event (not one value repeated for all events);
* a scenario order that starts with ``metering`` works (no strength left over from a previous scenario);
* the injected meter fault equals the labelled factor (site power after / before = strength);
* ``check_strengths`` rejects a corrupted label.
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
import _semisynth_fixture as fx
import pv_semisynth_modal_gen as smg

D, C = fx.make(cohorts=True); M = fx.make_modal(D)
cfg = smg.ModalGenConfig(scenarios=tuple(smg.MODAL_SCENARIOS) + ("new_obstruction",))
R = smg.strength_ranges(cfg)

for ps in range(5):
    L = smg.plan_modal(D, C, M, ps, cfg).labels
    for scn, g in L.groupby("scenario"):
        lo, hi = R[scn]
        assert g["strength"].between(lo, hi).all(), (ps, scn, g["strength"].tolist())
    me = L[L["scenario"] == "metering"]["strength"]
    if len(me) > 1:
        assert me.nunique() > 1, ("metering strength repeated", ps, me.tolist())
    ob = L[L["scenario"] == "new_obstruction"]["strength"]
    assert len(ob) == 0 or ob.between(*cfg.obstruction_block).all(), ob.tolist()
print("labelled strengths within configured ranges OK")

# metering first: its strength must not depend on a previous scenario
L1 = smg.plan_modal(D, C, M, 0, smg.ModalGenConfig(scenarios=("metering", "grid_curtail"))).labels
assert L1[L1["scenario"] == "metering"]["strength"].between(*cfg.metering_factor).all()
print("metering-first order OK")

# injected meter fault = labelled factor
plan = smg.plan_modal(D, C, M, 0, smg.ModalGenConfig())
Rz = smg.apply_modal(D, M, plan)
for _, e in Rz.labels[Rz.labels["scenario"] == "metering"].iterrows():
    c = e["column"]; w = (D["df"].index >= e["start"]) & (D["df"].index <= e["end"])
    b, a = D["df"][c].to_numpy(float)[w], Rz.df[c].to_numpy(float)[w]
    ok = np.isfinite(a) & np.isfinite(b) & (b > 1) & (np.abs(a - b) > 1e-9)   # changed (daylight) hours
    if ok.any():
        assert np.allclose(a[ok] / b[ok], e["strength"], atol=1e-6), (e["event_id"], np.median(a[ok] / b[ok]), e["strength"])
print("meter fault = labelled factor OK")

bad = plan.labels.copy(); i = bad.index[bad["scenario"] == "metering"][0]; bad.loc[i, "strength"] = 1.5
try:
    smg.check_strengths(bad, plan.cfg); raise AssertionError("corrupted label accepted")
except ValueError as exc:
    print("check_strengths rejects corrupted label OK:", str(exc)[:60])
print("modal_gen_strength_test PASSED")
