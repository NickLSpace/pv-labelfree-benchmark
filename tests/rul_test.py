"""Test of pv_rul on a synthetic fleet with known rates, a common 2023 step and cohorts.

Run:  python tests/rul_test.py
"""
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import matplotlib
matplotlib.use("Agg")
import pv_rul as rul  # noqa: E402

OUT = os.path.join(HERE, "out_rul")


def make_fleet(seed=0):
    rng = np.random.default_rng(seed)
    months = pd.date_range("2021-01-01", "2023-12-01", freq="MS")
    quarters = pd.date_range("2021-01-01", "2023-10-01", freq="QS")
    t = (months - months[0]).days.to_numpy() / 365.25
    tq = (quarters - quarters[0]).days.to_numpy() / 365.25
    season = 0.04 * np.sin(2 * np.pi * months.month.to_numpy() / 12)
    step = np.where(months >= "2023-01-01", 0.93, 1.0)                # common sensor shift -7 %
    stepq = np.where(quarters >= "2023-01-01", 0.93, 1.0)
    fleet_rate = -0.005                                             # common degradation -0.5 %/yr
    plants, truth, pr, cov, per = [], {}, {}, {}, []
    spec = ([(1, -0.04)] * 4 + [(1, -0.015)] * 4 + [(1, 0.0)] * 8 + [(1, 0.01)] * 2 +
            [(2, -0.03)] * 3 + [(2, 0.0)] * 2 + [(3, 0.0)] * 2)
    for i, (coh, r) in enumerate(spec):
        c = f"site/P{i:02d}"
        plants.append((c, coh))
        truth[c] = r
        lvl = 0.82 * (1 + (r + fleet_rate) * t) * step * (1 + season) * (1 + rng.normal(0, 0.012, len(t)))
        cv = np.ones(len(t))
        if coh == 2:
            cv[months.year == 2021] = 0
        if coh == 3:
            cv[months.year < 2023] = 0
        pr[c], cov[c] = lvl, cv
        b1 = (1 + (r + fleet_rate) * tq) * (1 + rng.normal(0, 0.006, len(tq)))   # b1 on harmonised data
        for q, b in zip(quarters, b1):
            if (coh == 2 and q.year == 2021) or (coh == 3 and q.year < 2023):
                continue
            per.append(dict(column=c, plant=c.split("/")[-1], period=q, theta_b1_Pexp=b))
    cohorts = pd.DataFrame([dict(column=c, plant=c.split("/")[-1], cohort=k) for c, k in plants]).set_index("column")
    ca = SimpleNamespace(pr_m=pd.DataFrame(pr, index=months), cov_m=pd.DataFrame(cov, index=months),
                         cohorts=cohorts)
    return ca, pd.DataFrame(per), truth, fleet_rate


def main():
    t0 = time.time()
    ca, per, truth, fr = make_fleet()
    pr_before = ca.pr_m.copy()
    res = rul.run_rul(ca, per_sps=per, cfg=rul.RULConfig(n_boot=1000), out_dir=OUT)
    assert ca.pr_m.equals(pr_before), "input modified"
    P = res.plants
    c1 = P[(P.cohort == 1)]
    # fleet-adjusted rate removes the common step and the common trend: rate ~ r_i - median
    fa = c1[c1.source == "PR, fleet-adjusted"].set_index("column")
    sb = c1[c1.source == "SPS b1"].set_index("column")
    err_fa = [abs(fa.loc[c, "rate_pct_per_yr"] - 100 * truth[c]) for c in fa.index]
    err_sb = [abs(sb.loc[c, "rate_pct_per_yr"] - 100 * (truth[c] + fr)) for c in sb.index]
    print("median |rate error| fleet-adjusted %.2f p.p., SPS b1 %.2f p.p." % (np.median(err_fa), np.median(err_sb)))
    assert np.median(err_fa) < 0.6 and np.median(err_sb) < 0.6
    # strongly degrading plants get finite RUL; stable and improving plants are censored, not dropped
    strong = [c for c in fa.index if truth[c] == -0.04]
    stable = [c for c in fa.index if truth[c] >= 0]
    assert all(fa.loc[c, "status"] in (rul.ST_EST, rul.ST_OPEN) for c in strong), fa.loc[strong, "status"]
    assert all("censored" in fa.loc[c, "status"] for c in stable), fa.loc[stable, "status"]
    assert len(fa) == int((ca.cohorts.cohort == 1).sum())
    for c in strong:
        r = fa.loc[c]
        assert r.rul_lo <= r.rul <= r.rul_hi
    # cohort 2 conditional, cohort 3 not estimable
    c2 = P[P.cohort == 2]
    assert c2["status"].str.startswith(rul.COND_NOTE).all() and c2["conditional"].all()
    assert P[P.cohort == 3]["status"].str.startswith(rul.ST_NA).all()
    res_skip = rul.run_rul(ca, per_sps=per, cfg=rul.RULConfig(n_boot=200, cohort2="skip"), out_dir=None)
    assert res_skip.plants[res_skip.plants.cohort == 2]["status"].str.startswith(rul.ST_NA).all()
    # without the fleet adjustment the 2023 step would bias the PR rate
    S_raw = rul.pr_series(ca, res.cfg, adjust=False)
    raw = rul.plant_rul(S_raw[stable[0]], 12, res.cfg, 6, np.random.default_rng(0))
    print("raw PR rate of a stable plant (step not removed): %.2f %%/yr" % raw["rate_pct_per_yr"])
    assert raw["rate_pct_per_yr"] < -2.5
    rul.show(res)
    os.makedirs(os.path.join(OUT, "fig"), exist_ok=True)
    fp = os.path.join(OUT, "fig", "fig_rul_forest.png")
    rul.plot_forest(res, fp)
    tex = rul.export_latex(res, OUT)
    z = rul.zip_outputs(OUT, [fp])
    assert os.path.exists(fp) and os.path.exists(z) and all(os.path.exists(p) for p in tex.values())
    print("rul_test OK in %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
