"""Semi-synthetic RUL comparison on the 2021-2023 fixture (all sources incl. frozen STGNN).

Run:  python tests/rul_semisynth_test.py
"""
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import matplotlib
matplotlib.use("Agg")
import numpy as np  # noqa: E402

import _semisynth_fixture as fxt  # noqa: E402
import pv_semisynth_eval as sse  # noqa: E402
import pv_rul as rul  # noqa: E402
import pv_rul_semisynth as rss  # noqa: E402

OUT = os.path.join(HERE, "out_rul_semisynth")


def main():
    t0 = time.time()
    D, C = fxt.make(cohorts=True)
    models = sse.frozen_models(D, C, seeds=[0], cache_dir=os.path.join(HERE, "out_config", "cache"),
                               hidden=32, epochs=2, train_stride=8, device="cpu")
    fp = {s: m.fingerprint() for s, m in models.items()}
    df0 = D["df"].copy()
    res = rss.run_rul_semisynth(D, C, models=models, placement_seeds=(0,),
                                cfg=rss.RULSSConfig(n_boot=500, n_perm=500, rul_boot=300), out_dir=OUT)
    assert D["df"].equals(df0) and all(models[s].fingerprint() == fp[s] for s in models)
    R = res.rows[res.rows.cohort == 1]
    assert set(R.source) >= {"PR, fleet-adjusted", "SPS b1", "Kalman b1", "STGNN eta"}, set(R.source)

    def bias(src, sc, role=None):
        g = R[(R.source == src) & (R.scenario == sc)]
        if role:
            g = g[g.role == role]
        return float(g.rate_err.mean())

    # fleet-adjusted PR: unaffected by the fleet-wide step, biased by +common rate
    b_step_pr, b_step_sps = bias("PR, fleet-adjusted", "plant+step", "control"), bias("SPS b1", "plant+step", "control")
    b_com_pr = bias("PR, fleet-adjusted", "plant+common", "control")
    print("controls, step: PR fleet-adj bias %.2f, SPS b1 bias %.2f | common: PR fleet-adj bias %.2f"
          % (b_step_pr, b_step_sps, b_com_pr))
    assert abs(b_step_pr) < 0.8 and b_step_sps < -1.0 and b_com_pr > 0.2
    mae = R[R.scenario == "plant"].groupby("source").abs_rate_err.mean()
    print("rate MAE, plant scenario:\n", mae.round(2).to_string())
    assert mae["PR, fleet-adjusted"] < 1.0 and mae["SPS b1"] < 1.0
    # pv_rul: STGNN as the third rate source on the real (unmodified) fixture
    r3 = rul.run_rul(D["ca"], per_sps=None, cfg=rul.RULConfig(n_boot=300), out_dir=None,
                     models=models, df=D["df"], phys=D["phys"])
    assert "STGNN eta" in r3.meta["sources"] and r3.meta["stgnn_seeds"] == 1, r3.meta
    assert all(models[s].fingerprint() == fp[s] for s in models)
    print("pv_rul sources:", r3.meta["sources"])
    rss.show(res)
    figs = rss.plot_all(res, os.path.join(OUT, "fig"))
    tex = rss.export_latex(res, OUT)
    z = rss.zip_outputs(OUT, os.path.join(OUT, "fig"))
    assert {"overall", "comparison", "paired"} <= set(tex) and os.path.exists(z) and len(figs) == 2
    print("rul_semisynth_test OK in %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
