exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "_common.py")).read()) if (os := __import__("os")) else None
import run_pipeline as rp, pv_report as rep, matplotlib; matplotlib.use("Agg")
args = SimpleNamespace()
for est in ("sps", "kalman"):
    A = rp.stage_anomaly_sps(D, C, runs, args, estimator=est)
    print(est, A["estimator"], A["evidence"]["evidence"].value_counts().to_dict())
    print(rp.anomaly_cache_name(est))
    os.makedirs(f"tests/out/rep_{est}", exist_ok=True)
    rep.fig_sps(A, D, f"tests/out/rep_{est}")
    print(os.listdir(f"tests/out/rep_{est}"))
