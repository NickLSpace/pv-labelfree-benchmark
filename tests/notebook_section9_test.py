"""Executes the new notebook section 9 cells with synthetic D, C, runs, A."""
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "_common.py")).read()) if (os := __import__("os")) else None
import json, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, run_pipeline as rp
from IPython.display import display
import pv_kalman as kf_mod, pv_estimator_comparison as est_cmp
KALMAN_CFG = kf_mod.KalmanConfig(); BOUNDARY = "2023-01-01"
for ESTIMATOR in ("kalman", "sps"):
    A = rp.stage_anomaly_sps(D, C, runs, SimpleNamespace(), estimator=ESTIMATOR)
    os.makedirs("results/cache", exist_ok=True)
    for f in os.listdir("results/cache"): os.remove(os.path.join("results/cache", f))
    nb = json.load(open("pv-stgnn-fuzzy-multi-full.ipynb"))
    i9 = [i for i, c in enumerate(nb["cells"]) if "".join(c["source"]).startswith("## 9. SPS vs")][0]
    i10 = [i for i, c in enumerate(nb["cells"]) if "".join(c["source"]).startswith("## 10.")]
    for c in nb["cells"][i9:(i10[0] if i10 else None)]:
        if c["cell_type"] == "code":
            exec("".join(c["source"]), globals())
    print(ESTIMATOR, "OK"); print(cmp_res["wide"].to_string())
print(os.listdir("results"), os.listdir("results/figures"))
