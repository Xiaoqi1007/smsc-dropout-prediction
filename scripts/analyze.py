"""Compute performance tables and Figures 2 and 3."""

from pathlib import Path
import sys, argparse, platform

P = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(P / "src"))
import numpy as np, pandas as pd, joblib, matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from threadpoolctl import threadpool_limits
from importlib.metadata import version
from smsc.config import TEXT_REPRESENTATION, load_config
from smsc.experiment import Experiment, read_json, write_json, sha
from smsc.metrics import evaluate, METRICS
from smsc.plotting import best_f1, draw_risk
from smsc.plot_performance import draw as draw_performance
from smsc.baseline import baseline_on_saved_splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cases", type=Path)
    a = parser.parse_args()
    spec = load_config(a.config)
    run = P / "results/experiments" / spec["name"]
    e = Experiment(run)
    out = run / "reports"
    T = out / "tables"
    F = out / "figures"
    M = out / "methods"
    for p in [T, F, M]:
        p.mkdir(parents=True, exist_ok=True)
    (T / "individual_examples.csv").unlink(missing_ok=True)
    allpred = e.predictions()
    full = allpred[allpred.model.eq("full_model")].copy()
    with threadpool_limits(1):
        baseline, coverage = baseline_on_saved_splits(
            full, P / spec["inputs"]["data_path"], e, out
        )
    combined = pd.concat([full, baseline], ignore_index=True)
    baseline.to_csv(out / "private/baseline_predictions.csv", index=False)
    records = []
    classification = []
    for (model, seed), g in combined.groupby(["model", "seed"]):
        test = g[g.split.eq("test")]
        for cutoff in range(24):
            sub = test[test.session.le(cutoff)]
            records.append(
                dict(
                    model=model,
                    seed=seed,
                    cutoff=cutoff,
                    **evaluate(sub.y_true, sub.probability),
                )
            )
        for session in range(24):
            val = g[g.split.eq("validation") & g.session.eq(session)]
            test = g[g.split.eq("test") & g.session.eq(session)]
            t = best_f1(val)
            r = dict(
                model=model,
                seed=seed,
                session=session,
                threshold=t,
                n_test=len(test),
                prevalence=test.y_true.mean(),
                recall=np.nan,
                precision=np.nan,
                specificity=np.nan,
                f1=np.nan,
                flagged_fraction=np.nan,
            )
            if np.isfinite(t) and len(test):
                y = test.y_true.to_numpy()
                flag = test.probability.to_numpy() >= t
                tp = ((y == 1) & flag).sum()
                fp = ((y == 0) & flag).sum()
                fn = ((y == 1) & ~flag).sum()
                tn = ((y == 0) & ~flag).sum()
                r.update(
                    recall=tp / (tp + fn) if tp + fn else np.nan,
                    precision=tp / (tp + fp) if tp + fp else np.nan,
                    specificity=tn / (tn + fp) if tn + fp else np.nan,
                    f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
                    flagged_fraction=float(flag.mean()),
                )
            classification.append(r)
    metrics = pd.DataFrame(records)
    metrics.to_csv(T / "cumulative_metrics_by_seed.csv", index=False)
    summary = metrics.groupby(["model", "cutoff"])[METRICS].agg(["mean", "std"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary.reset_index().to_csv(T / "cumulative_metrics_summary.csv", index=False)
    c = pd.DataFrame(classification)
    c.to_csv(T / "classification_by_seed.csv", index=False)
    cs = c.groupby(["model", "session"])[
        [
            "prevalence",
            "recall",
            "precision",
            "specificity",
            "f1",
            "n_test",
            "flagged_fraction",
        ]
    ].agg(["mean", "std", "count"])
    cs.columns = ["_".join(x) for x in cs.columns]
    cs = cs.reset_index()
    cs.to_csv(T / "classification_summary.csv", index=False)
    rows = []
    for r in cs[cs.model.eq("full_model") & cs.session.le(9)].to_dict("records"):
        row = dict(Session=int(r["session"]))
        for m, label, scale, digits in [
            ("prevalence", "Dropout prevalence (%)", 100, 1),
            ("recall", "Recall (%)", 100, 1),
            ("precision", "Precision (%)", 100, 1),
            ("specificity", "Specificity (%)", 100, 1),
            ("f1", "F1", 1, 2),
            ("n_test", "Test participants (n)", 1, 1),
        ]:
            row[label] = (
                f'{r[m+"_mean"]*scale:.{digits}f} ({r[m+"_std"]*scale:.{digits}f})'
                + ("†" if r[m + "_count"] < 10 else "")
            )
        row["precision_evaluable_seeds"] = int(r["precision_count"])
        rows.append(row)
    pd.DataFrame(rows).to_csv(T / "Table2_formatted.csv", index=False)
    # Raw and calibrated metrics for every source, fusion and baseline.
    rows = []
    for (model, seed), g in (
        pd.concat([allpred, baseline])
        .query('split == "test"')
        .groupby(["model", "seed"])
    ):
        for window, hi in [("all", 23), ("early", 9)]:
            q = g[g.session.le(hi)]
            for stage, col in [("raw", "probability_raw"), ("platt", "probability")]:
                rows.append(
                    dict(
                        model=model,
                        seed=seed,
                        window=window,
                        stage=stage,
                        **evaluate(q.y_true, q[col]),
                    )
                )
    detailed = pd.DataFrame(rows)
    detailed.to_csv(T / "all_models_raw_calibrated_by_seed.csv", index=False)
    sm = detailed.groupby(["model", "window", "stage"])[METRICS].agg(
        ["mean", "std", "count"]
    )
    sm.columns = ["_".join(x) for x in sm.columns]
    sm.reset_index().to_csv(T / "all_models_raw_calibrated_summary.csv", index=False)
    settings = []
    feature_rows = []
    for seed in e.seeds:
        folder = e.seed_dir(seed)
        s = read_json(folder / "selection.json")
        pa = joblib.load(folder / "models/preassessment_xgb.joblib")["model"]
        ss = joblib.load(folder / "models/session_xgb.joblib")["model"]
        settings.append(
            dict(
                seed=seed,
                preassessment_depth=pa.max_depth,
                preassessment_trees=pa.n_estimators,
                session_depth=ss.max_depth,
                session_trees=ss.n_estimators,
                NN_best_epoch=s["final"]["epochs"][TEXT_REPRESENTATION],
            )
        )
        for branch in ["preassessment_xgb", "session_xgb", "emotion_speech_lr"]:
            payload = joblib.load(folder / f"models/{branch}.joblib")
            columns = payload.get(
                "columns", payload.get("schema", {}).get("usable_train_columns", [])
            )
            feature_rows += [dict(seed=seed, branch=branch, feature=f) for f in columns]
    pd.DataFrame(settings).to_csv(T / "training_settings_by_seed.csv", index=False)
    pd.DataFrame(feature_rows).to_csv(T / "fitted_feature_columns.csv", index=False)
    excluded = spec.get("features", {}).get("exclude_columns", [])
    assert not set(excluded) & set(pd.DataFrame(feature_rows).feature)
    write_json(
        M / "feature_exclusions.json",
        dict(
            excluded=excluded,
            retained_session=True,
            retained_duration_unit="seconds",
            emotion_proportion_reference=(
                "Neutral" if "emotion__Neutral_prop" in excluded else None
            ),
            scope="Specified structural identities; high correlation and sample-dependent rank deficiency are not exclusion criteria.",
        ),
    )
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    draw_performance(combined, F, T)
    if a.cases:
        cases = read_json(a.cases)
        rf = out / "private/risk"
        rf.mkdir(parents=True, exist_ok=True)
        audit = draw_risk(
            full,
            rf,
            (0, 9),
            "Full Text Cumulative · XGB / XGB / LR / NN",
            seed=cases["risk_seed"],
            dropout_id=cases["dropout_id"],
            completer_id=cases["completer_id"],
            endpoints=cases.get("endpoints"),
        )
        import shutil

        for ext in ["png", "pdf", "svg"]:
            shutil.copy2(rf / f"risk_ABC.{ext}", F / f"Figure_3_risk.{ext}")
        for name in [
            "group_risk_summary",
            "group_risk_by_seed",
            "validation_F1_thresholds",
        ]:
            shutil.copy2(rf / f"{name}.csv", T / f"{name}.csv")
        write_json(rf / "example_details.json", audit)
    write_json(
        M / "software_versions.json",
        dict(
            python=platform.python_version(),
            **{
                p: version(p)
                for p in [
                    "numpy",
                    "pandas",
                    "scipy",
                    "scikit-learn",
                    "xgboost",
                    "torch",
                    "matplotlib",
                    "shap",
                ]
            },
        ),
    )
    write_json(
        M / "analysis_verification.json",
        dict(
            seeds=e.seeds,
            run_manifest_sha256=sha(run / "experiment.json"),
            baseline_freshly_fitted=True,
            feature_exclusions_checked=True,
            calibration_bins="10 approximately equal-count test rows per model and split",
            ece_bins="10 equal-width bins",
            uncertainty="Sample SD across overlapping splits; not confidence intervals",
            threshold="Session-specific validation F1; highest threshold in exact ties; undefined for single-class validation",
        ),
    )
    print(out, flush=True)


if __name__ == "__main__":
    main()
