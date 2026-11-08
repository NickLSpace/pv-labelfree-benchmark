"""
Run fuzzy vs crisp anomaly classification comparison on cached model results.

Outputs:
  results/fuzzy_comparison.xlsx   — per-plant tables, ablation metrics
  results/figures/fig18_fuzzy_ablation_heatmap.png/pdf
  results/figures/fig19_fuzzy_radar.png/pdf
  results/figures/fig20_crisp_vs_fuzzy_confusion.png/pdf
  results/cache/fuzzy_comparison.pkl  — all raw results for re-use
"""

from __future__ import annotations

import os, sys, pickle, warnings
import numpy as np
import pandas as pd

# Ensure script directory (and parent) on sys.path for Kaggle
_script_dir = os.path.dirname(os.path.abspath(__file__))
for _p in [_script_dir, os.path.dirname(_script_dir)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
warnings.filterwarnings("ignore", category=RuntimeWarning)

from pv_fuzzy import (
    FuzzyConfig, fuzzy_classify_batch, fuzzy_plant_summary,
    crisp_classify_batch, get_ablation_configs,
    classification_entropy, classification_agreement, cross_seed_stability,
    CLASS_NAMES, load_and_extract_features,
)
from pv_cohorts import short
import inspect as _inspect
import pv_fuzzy as _pv_fuzzy


def load_features_compat(cache_dir, variant="multiview", seed=0, data=None, clusters=None):
    """Version-independent wrapper around ``pv_fuzzy.load_and_extract_features``.

    Older copies of pv_fuzzy.py (e.g. a stale file in a read-only Kaggle dataset)
    do not accept ``data`` / ``clusters``.  In that case the same logic is run here,
    so the caller does not depend on which pv_fuzzy.py happens to be imported.
    """
    if "data" in _inspect.signature(load_and_extract_features).parameters:
        return load_and_extract_features(cache_dir=cache_dir, variant=variant, seed=seed,
                                         data=data, clusters=clusters)
    warnings.warn(f"outdated pv_fuzzy.py imported from {_pv_fuzzy.__file__}; "
                  "using the built-in loader instead", stacklevel=2)
    from pv_clustering import anomaly_event_features
    with open(os.path.join(cache_dir, f"run_{variant}_{seed}.pkl"), "rb") as f:
        events = pickle.load(f)["events"]
    if data is None:
        with open(os.path.join(cache_dir, "data.pkl"), "rb") as f:
            data = pickle.load(f)
    if clusters is None:
        with open(os.path.join(cache_dir, "clusters.pkl"), "rb") as f:
            clusters = pickle.load(f)
    wr = clusters["wr"]
    regimes = wr.result.labels if hasattr(wr, "result") else wr.labels
    feat = anomaly_event_features(events, data["df"], data["phys"], data["pv_cols"], regimes)
    return feat, events


# ── Output directories: prefer /kaggle/working (writable), fall back to ./results
_WD = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.getcwd()
CACHE = os.path.join(_WD, "results", "cache")
OUT = os.path.join(_WD, "results")
FIG = os.path.join(OUT, "figures")
SEEDS = [0, 1, 2]  # 3 seeds for CPU; user will rerun with 5 on GPU
VARIANTS = ["baseline", "multiview"]

os.makedirs(FIG, exist_ok=True)


def run_all():
    configs = get_ablation_configs()
    all_results = {}  # (variant, config_name, seed) -> plant_summary

    for variant in VARIANTS:
        for seed in SEEDS:
            print(f"\n{'='*60}")
            print(f"  {variant} seed={seed}")
            print(f"{'='*60}")
            try:
                feat, events = load_features_compat(
                    cache_dir=CACHE, variant=variant, seed=seed)
            except Exception as e:
                print(f"  SKIP: {e}")
                continue
            print(f"  events: {len(feat)}, plants: {feat['plant'].nunique()}")

            all_plants = sorted(feat["plant"].unique())

            for cfg_name, cfg in configs.items():
                if cfg is None:
                    # Crisp baseline
                    memberships = crisp_classify_batch(feat)
                else:
                    memberships = fuzzy_classify_batch(feat, cfg)

                plant_summary = fuzzy_plant_summary(feat, memberships, all_plants)
                key = (variant, cfg_name, seed)
                all_results[key] = {
                    "feat": feat,
                    "memberships": memberships,
                    "plant_summary": plant_summary,
                    "config": cfg,
                    "n_events": len(feat),
                }
                # Quick stats
                top_class = plant_summary["most_probable_class"].value_counts()
                ent = classification_entropy(memberships)
                print(f"  {cfg_name:12s}  entropy={ent:.3f}  n_events={len(feat)}  "
                      f"top_classes: {dict(top_class.head(3))}")

    return all_results


def build_comparison_tables(all_results):
    """Build summary tables for the article."""
    # 1. Per-plant table: multiview + crisp vs multiview + fuzzy (seed 0)
    rows = []
    for variant in VARIANTS:
        for cfg_name in ["crisp", "fuzzy_full"]:
            for seed in SEEDS:
                key = (variant, cfg_name, seed)
                if key not in all_results:
                    continue
                ps = all_results[key]["plant_summary"]
                for plant in ps.index:
                    r = ps.loc[plant]
                    rows.append({
                        "variant": variant,
                        "method": cfg_name,
                        "seed": seed,
                        "plant": plant,
                        "most_probable_class": r["most_probable_class"],
                        "class_membership": r["class_membership"],
                        "n_events": r["n_events"],
                        "anomalous_hours": r["anomalous_hours"],
                        "entropy": r["class_entropy"],
                    })
    plant_table = pd.DataFrame(rows)
    plant_table.to_csv(os.path.join(OUT, "fuzzy_plant_comparison.csv"), index=False)

    # 2. Ablation metrics table
    ablation_rows = []
    for variant in VARIANTS:
        for cfg_name, cfg in get_ablation_configs().items():
            entropies = []
            n_events_list = []
            seed_summaries = []
            for seed in SEEDS:
                key = (variant, cfg_name, seed)
                if key not in all_results:
                    continue
                r = all_results[key]
                entropies.append(classification_entropy(r["memberships"]))
                n_events_list.append(r["n_events"])
                seed_summaries.append(r["plant_summary"])

            stability = cross_seed_stability(seed_summaries) if len(seed_summaries) >= 2 else np.nan

            # Agreement with crisp baseline
            agreements = []
            for seed in SEEDS:
                crisp_key = (variant, "crisp", seed)
                this_key = (variant, cfg_name, seed)
                if crisp_key in all_results and this_key in all_results:
                    agreements.append(classification_agreement(
                        all_results[crisp_key]["memberships"],
                        all_results[this_key]["memberships"]))

            ablation_rows.append({
                "variant": variant,
                "method": cfg_name,
                "mean_entropy": np.mean(entropies) if entropies else np.nan,
                "std_entropy": np.std(entropies) if entropies else np.nan,
                "cross_seed_stability": stability,
                "agreement_with_crisp": np.mean(agreements) if agreements else np.nan,
                "n_events": np.mean(n_events_list) if n_events_list else 0,
            })
    ablation_table = pd.DataFrame(ablation_rows)
    ablation_table.to_csv(os.path.join(OUT, "fuzzy_ablation_metrics.csv"), index=False)

    # 3. Class distribution table
    class_rows = []
    for variant in VARIANTS:
        for cfg_name in ["crisp", "fuzzy_full"]:
            for seed in SEEDS:
                key = (variant, cfg_name, seed)
                if key not in all_results:
                    continue
                ps = all_results[key]["plant_summary"]
                for cls in CLASS_NAMES:
                    if cls in ps.columns:
                        class_rows.append({
                            "variant": variant,
                            "method": cfg_name,
                            "seed": seed,
                            "class": cls,
                            "mean_membership": float(ps[cls].mean()),
                            "n_plants_dominant": int((ps["most_probable_class"] == cls).sum()),
                        })
    class_table = pd.DataFrame(class_rows)

    return plant_table, ablation_table, class_table


def plot_ablation_heatmap(ablation_table):
    """Heatmap: rows=ablation methods, cols=variants, values=entropy."""
    import matplotlib
    if "ipykernel" not in __import__("sys").modules:   # keep inline plots in notebooks
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, metric, title, cmap in [
        (axes[0], "mean_entropy", "Mean entropy", "YlOrRd"),
        (axes[1], "agreement_with_crisp", "Agreement with crisp", "RdYlGn"),
    ]:
        pivot = ablation_table.pivot_table(
            index="method", columns="variant", values=metric, aggfunc="mean")
        im = ax.imshow(pivot.values, aspect="auto", cmap=cmap)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=30, ha="right", fontsize=9)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=9)
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                v = pivot.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                            fontsize=8, fontweight="bold")
        ax.set_title(title, fontsize=11)
        plt.colorbar(im, ax=ax, shrink=0.8)
    plt.suptitle("Fuzzy ablation study: entropy and agreement with crisp baseline",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        plt.savefig(os.path.join(FIG, f"fig18_fuzzy_ablation_heatmap.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved fig18_fuzzy_ablation_heatmap")


def plot_radar(all_results):
    """Radar chart: mean membership per class, crisp vs fuzzy, multiview."""
    import matplotlib
    if "ipykernel" not in __import__("sys").modules:   # keep inline plots in notebooks
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), subplot_kw=dict(polar=True))
    fig.subplots_adjust(top=0.78, bottom=0.05, wspace=0.55)
    variants_labels = {"baseline": "Baseline (no multi-view)", "multiview": "Multi-view"}

    for ax, variant in zip(axes, VARIANTS):
        for cfg_name, color, ls in [("crisp", "tab:blue", "--"), ("fuzzy_full", "tab:red", "-")]:
            means = []
            for seed in SEEDS:
                key = (variant, cfg_name, seed)
                if key not in all_results:
                    means.append(np.zeros(len(CLASS_NAMES)))
                    continue
                ps = all_results[key]["plant_summary"]
                vals = [ps[c].mean() if c in ps.columns else 0.0 for c in CLASS_NAMES]
                means.append(vals)
            means_arr = np.array(means).mean(0)
            angles = np.linspace(0, 2 * np.pi, len(CLASS_NAMES), endpoint=False).tolist()
            angles += angles[:1]
            means_closed = list(means_arr) + [means_arr[0]]
            ax.plot(angles, means_closed, color=color, ls=ls, lw=2, label=cfg_name)
            ax.fill(angles, means_closed, color=color, alpha=0.12)

        # Short class labels for radar
        short_labels = ["Outage", "Clipping", "Shading",
                        "Degradation", "Overgen.", "Weather"]
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(short_labels, fontsize=8)
        ax.set_ylim(0, max(means_arr.max() * 1.35, 0.1))
        ax.set_title(variants_labels.get(variant, variant), fontsize=10, pad=12)

    # Single shared legend below suptitle
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=9,
               bbox_to_anchor=(0.5, 0.91))

    fig.suptitle("Mean class membership per plant: crisp vs fuzzy",
                 fontsize=12, y=0.96)
    for ext in ["png", "pdf"]:
        plt.savefig(os.path.join(FIG, f"fig19_fuzzy_radar.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved fig19_fuzzy_radar")


def plot_confusion(all_results):
    """Confusion matrix: crisp vs fuzzy class assignment (multiview, seed 0)."""
    import matplotlib
    if "ipykernel" not in __import__("sys").modules:   # keep inline plots in notebooks
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    short_labels = ["Outage", "Curtail.", "Shad.", "Degr.", "Overg.", "Weath."]

    for ax, variant in zip(axes, VARIANTS):
        crisp_key = (variant, "crisp", 0)
        fuzzy_key = (variant, "fuzzy_full", 0)
        if crisp_key not in all_results or fuzzy_key not in all_results:
            continue
        crisp_labels = all_results[crisp_key]["memberships"].values.argmax(1)
        fuzzy_labels = all_results[fuzzy_key]["memberships"].values.argmax(1)
        cm = confusion_matrix(crisp_labels, fuzzy_labels, labels=list(range(6)))
        cm_norm = cm.astype(float) / cm.sum(1, keepdims=True).clip(1)
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(6))
        ax.set_xticklabels(short_labels, rotation=30, ha="right", fontsize=9)
        ax.set_yticks(range(6))
        ax.set_yticklabels(short_labels, fontsize=9)
        ax.set_xlabel("Fuzzy", fontsize=10)
        ax.set_ylabel("Crisp", fontsize=10)
        ax.set_title(variant, fontsize=11)
        for i in range(6):
            for j in range(6):
                v = cm[i, j]
                if v > 0:
                    color = "white" if cm_norm[i, j] > 0.5 else "black"
                    ax.text(j, i, str(v), ha="center", va="center",
                            fontsize=8, color=color, fontweight="bold")
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.suptitle("Crisp vs fuzzy anomaly classification (seed=0, normalized rows)",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        plt.savefig(os.path.join(FIG, f"fig20_crisp_vs_fuzzy_confusion.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved fig20_crisp_vs_fuzzy_confusion")


def save_xlsx(plant_table, ablation_table, class_table, all_results):
    """Save everything to a multi-sheet xlsx (first sheet = description)."""
    path = os.path.join(OUT, "fuzzy_comparison.xlsx")
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        # Description sheet
        desc = pd.DataFrame({
            "Sheet": [
                "01_Description",
                "02_Plant_comparison",
                "03_Ablation_metrics",
                "04_Class_distribution",
                "05_Summary_multiview_crisp",
                "06_Summary_multiview_fuzzy",
                "07_Summary_baseline_crisp",
                "08_Summary_baseline_fuzzy",
            ],
            "Description": [
                "Sheet description (this sheet)",
                "Per-plant table: interpretation method, seed, class, membership, entropy",
                "Ablation: entropy, stability, agreement with crisp (all unsupervised)",
                "Class distribution: mean membership, number of dominant plants",
                "Plant summary: multiview + crisp (3 seeds)",
                "Plant summary: multiview + fuzzy (3 seeds)",
                "Plant summary: baseline + crisp (3 seeds)",
                "Plant summary: baseline + fuzzy (3 seeds)",
            ],
        })
        desc.to_excel(w, sheet_name="01_Description", index=False)
        plant_table.to_excel(w, sheet_name="02_Plant_comparison", index=False)
        ablation_table.to_excel(w, sheet_name="03_Ablation_metrics", index=False)
        class_table.to_excel(w, sheet_name="04_Class_distribution", index=False)

        for sheet_name, variant, cfg_name in [
            ("05_Summary_mw_crisp", "multiview", "crisp"),
            ("06_Summary_mw_fuzzy", "multiview", "fuzzy_full"),
            ("07_Summary_bl_crisp", "baseline", "crisp"),
            ("08_Summary_bl_fuzzy", "baseline", "fuzzy_full"),
        ]:
            rows = []
            for seed in SEEDS:
                key = (variant, cfg_name, seed)
                if key not in all_results:
                    continue
                ps = all_results[key]["plant_summary"].copy()
                ps["seed"] = seed
                rows.append(ps.reset_index())
            if rows:
                pd.concat(rows, ignore_index=True).to_excel(w, sheet_name=sheet_name, index=False)
    print(f"  Saved {path}")


def save_cache(all_results):
    """Save raw results for re-use."""
    path = os.path.join(CACHE, "fuzzy_comparison.pkl")
    # Strip feat DataFrames to save space
    slim = {}
    for key, val in all_results.items():
        slim[key] = {
            "memberships": val["memberships"],
            "plant_summary": val["plant_summary"],
            "config": val["config"],
            "n_events": val["n_events"],
        }
    with open(path, "wb") as f:
        pickle.dump(slim, f)
    print(f"  Saved cache: {path}")


def run_fuzzy_df(df_raw, ttl_path=None,
                      cache_dir=None, out_dir=None, fig_dir=None,
                      seeds=None, variants=None):
    """Run fuzzy vs crisp comparison from a DataFrame (notebook mode).

    Accepts a raw DataFrame (e.g. from ``pd.read_excel`` or loaded in a
    notebook), builds the processed data dict via ``stage_data``, then
    loads cached STGNN events and runs all 7 characterisation configs.
    No training is performed — only post-processing on cached events.

    Usage in a notebook:
        from run_fuzzy_comparison import run_from_dataframe
        all_results, plant_table, ablation_table = run_from_dataframe(
            df_raw, ttl_path="...", seeds=[0, 1, 2])

    Returns:
        all_results: dict keyed by (variant, config_name, seed)
        plant_table: pd.DataFrame
        ablation_table: pd.DataFrame
    """
    global CACHE, OUT, FIG, SEEDS, VARIANTS
    _WD = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.getcwd()
    CACHE = cache_dir or os.path.join(_WD, "results", "cache")
    OUT = out_dir or os.path.join(_WD, "results")
    FIG = fig_dir or os.path.join(OUT, "figures")
    SEEDS = seeds or [0, 1, 2]
    VARIANTS = variants or ["baseline", "multiview"]
    os.makedirs(FIG, exist_ok=True)

    # Resolve TTL path: use provided, or search common locations
    if ttl_path is None:
        from run_pipeline import _find_file
        ttl_path = _find_file('PV-generation-system-metadata.ttl')
    if ttl_path is None:
        raise FileNotFoundError(
            "TTL file not found. Pass ttl_path explicitly: "
            "run_from_dataframe(df_raw, ttl_path='/your/path/PV-generation-system-metadata.ttl')")

    # Build processed data from the raw DataFrame (or load from cache)
    data_path = os.path.join(CACHE, "data.pkl")
    clusters_path = os.path.join(CACHE, "clusters.pkl")
    if os.path.exists(data_path):
        with open(data_path, "rb") as f:
            data = pickle.load(f)
    else:
        from run_pipeline import stage_data
        data = stage_data(df_raw, ttl_path)
        os.makedirs(CACHE, exist_ok=True)
        with open(data_path, "wb") as f:
            pickle.dump(data, f)

    if os.path.exists(clusters_path):
        with open(clusters_path, "rb") as f:
            clusters = pickle.load(f)
    else:
        from run_pipeline import stage_clusters
        class _Args:
            pass
        args = _Args()
        args.ttl = ttl_path
        clusters = stage_clusters(data, args)
        with open(clusters_path, "wb") as f:
            pickle.dump(clusters, f)

    # Run all fuzzy/crisp configs using the processed data
    configs = get_ablation_configs()
    all_results = {}

    for variant in VARIANTS:
        for seed in SEEDS:
            print(f"\n{'='*60}")
            print(f"  {variant} seed={seed}")
            print(f"{'='*60}")
            try:
                feat, events = load_features_compat(
                    cache_dir=CACHE, variant=variant, seed=seed,
                    data=data, clusters=clusters)
            except Exception as e:
                print(f"  SKIP: {e}")
                continue
            print(f"  events: {len(feat)}, plants: {feat['plant'].nunique()}")
            all_plants = sorted(feat["plant"].unique())
            for cfg_name, cfg in configs.items():
                if cfg is None:
                    memberships = crisp_classify_batch(feat)
                else:
                    memberships = fuzzy_classify_batch(feat, cfg)
                plant_summary = fuzzy_plant_summary(feat, memberships, all_plants)
                key = (variant, cfg_name, seed)
                all_results[key] = {
                    "feat": feat,
                    "memberships": memberships,
                    "plant_summary": plant_summary,
                    "config": cfg,
                    "n_events": len(feat),
                }
                ent = classification_entropy(memberships)
                top_class = plant_summary["most_probable_class"].value_counts()
                print(f"  {cfg_name:12s}  entropy={ent:.3f}  n_events={len(feat)}  "
                      f"top_classes: {dict(top_class.head(3))}")

    if not all_results:
        raise RuntimeError(
            f"No STGNN runs found in {CACHE} (run_<variant>_<seed>.pkl): every run was skipped, "
            "so all tables and figures would be empty. Train the models first, e.g. "
            "run_pipeline.run_from_dataframe(df_raw, out_dir=<same results dir>, stage='stgnn').")

    # Build tables, figures, save
    plant_table, ablation_table, class_table = build_comparison_tables(all_results)
    plot_ablation_heatmap(ablation_table)
    plot_radar(all_results)
    plot_confusion(all_results)
    save_xlsx(plant_table, ablation_table, class_table, all_results)
    save_cache(all_results)
    return all_results, plant_table, ablation_table

"""
if __name__ == "__main__":
    print("="*60)
    print("  FUZZY ANOMALY CLASSIFICATION — ABLATION STUDY")
    print("="*60)
    all_results = run_all()
    print(f"\n  Total configurations: {len(all_results)}")
    if not all_results:
        raise SystemExit(f"No STGNN runs found in {CACHE} -- train the models first (run_pipeline.py).")

    print("\n  Building comparison tables...")
    plant_table, ablation_table, class_table = build_comparison_tables(all_results)
    print(f"  Plant table: {len(plant_table)} rows")
    print(f"  Ablation table: {len(ablation_table)} rows")
    print(f"  Class table: {len(class_table)} rows")

    print("\n  Generating figures...")
    plot_ablation_heatmap(ablation_table)
    plot_radar(all_results)
    plot_confusion(all_results)

    print("\n  Saving outputs...")
    save_xlsx(plant_table, ablation_table, class_table, all_results)
    save_cache(all_results)

    print("\n" + "="*60)
    print("  DONE")
    print("="*60)
    print(f"\n  Outputs in: {OUT}")
    print(f"  Figures in: {FIG}")
"""