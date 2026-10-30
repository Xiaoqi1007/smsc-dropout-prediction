"""Within-treatment-arm, within-session permutation importance (30 repeats)."""

from pathlib import Path
import sys, time, argparse
import numpy as np, pandas as pd, joblib, sklearn
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from smsc.config import load_config
from smsc.utils import digest_json
from smsc.experiment import Experiment, read_csv, read_json, sha, write_json
from smsc.explain_support import (
    CachedModel,
    FrozenModel,
    WithinSessionModel,
    cohort_counts,
    load_model,
    participant_arms,
    score_auc,
    summarize_importance,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    spec = load_config(args.config)
    run = PROJECT / "results/experiments" / spec["name"]
    out = run / "reports/figure4/arm_importance"
    work = run / "reports/figure4/arm_work"
    for p in [out / "tables", out / "methods", work]:
        p.mkdir(parents=True, exist_ok=True)
    experiment = Experiment(run)
    seeds = sorted(experiment.seeds)
    wide = PROJECT / experiment.config["inputs"]["data_path"]
    assert sha(wide) in experiment.manifest["inputs"].values()
    frame = read_csv(run / "data/features.csv")
    assert (
        sha(run / "data/features.csv")
        == read_json(run / "data/DONE.json")["features_sha256"]
    )
    arms = participant_arms(frame)
    lookup = arms.set_index("id").treatment_arm
    arms.to_csv(work / "participant_arms_private.csv", index=False)
    cohort_counts_all = (
        arms.groupby("treatment_arm").size().rename("participants").reset_index()
    )
    cohort_counts_all.to_csv(out / "tables/cohort_participants_by_arm.csv", index=False)
    full = experiment.predictions("full_model")
    assert full.groupby(["seed", "id"]).split.nunique().eq(1).all()
    held = full.loc[full.split.eq("test")].copy()
    held["treatment_arm"] = held.id.map(lookup)
    cohort = cohort_counts(held)
    cohort.to_csv(out / "tables/test_cohort_by_seed_arm.csv", index=False)
    assert len(cohort.loc[cohort.treatment_arm.isin(["P-STAIR", "SC"])]) == 2 * len(
        seeds
    )
    assert (
        cohort.loc[cohort.treatment_arm.isin(["P-STAIR", "SC"]), "baseline_auroc"]
        .notna()
        .all()
    )
    hashes = {
        str(p): sha(p)
        for p in [run / "experiment.json", run / "data/features.csv", wide]
    }
    for seed in seeds:
        folder = experiment.verify_seed(seed)
        for p in [
            folder / "DONE.json",
            folder / "predictions.csv",
            folder / "calibrators.json",
            *sorted((folder / "models").glob("*.joblib")),
        ]:
            hashes[str(p)] = sha(p)
    fingerprint = digest_json(hashes)
    audits = []
    with threadpool_limits(limits=2):
        for seed in seeds:
            start = time.time()
            saved = load_model(run, seed)
            native = joblib.load(
                experiment.seed_dir(seed) / "models/full_model_calibration.joblib"
            )
            assert native.method == "sigmoid"
            params = saved["calibrator"]["parameters"]
            assert params["implementation"] == "sklearn.CalibratedClassifierCV"
            assert params["a"] > 0
            reference = FrozenModel(saved)
            np.testing.assert_allclose(
                reference.predict_probability(reference.encoded),
                saved["predictions"].probability,
                atol=1e-6,
                rtol=0,
            )
            for ai, arm in enumerate(["P-STAIR", "SC"]):
                nrep = 30
                target = work / f"{arm}_seed_{seed}_repeats{nrep}.csv"
                marker = target.with_suffix(".json")
                if (
                    target.exists()
                    and marker.exists()
                    and read_json(marker).get("fingerprint") == fingerprint
                ):
                    print(f"Complete: seed {seed} {arm}", flush=True)
                    audits.append(read_json(marker))
                    continue
                pred = saved["predictions"]
                mask = (pred.split.eq("test") & pred.id.map(lookup).eq(arm)).to_numpy()
                selected = pred.loc[mask].reset_index(drop=True)
                cached = CachedModel(reference, mask, native)
                restricted = WithinSessionModel(
                    cached, cached.original, selected.session.to_numpy()
                )
                np.testing.assert_allclose(
                    cached.base, selected.probability, atol=1e-6, rtol=0
                )
                max_error = float(np.max(abs(cached.base - selected.probability)))
                # Audit every feature against the original full-branch implementation.
                if seed == 0:
                    rng = np.random.default_rng(179)
                    for j in range(len(reference.columns)):
                        keys = restricted.keys.copy()
                        keys[:, j] = rng.permutation(keys[:, j])
                        X = restricted.restore(keys)
                        for group in restricted.groups:
                            np.testing.assert_array_equal(
                                np.sort(X[group, j]), np.sort(cached.original[group, j])
                            )
                        actual = cached.predict_proba(X)[:, 1]
                        expected = reference.predict_probability(X)
                        error = float(np.max(abs(actual - expected)))
                        max_error = max(max_error, error)
                        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
                random_seed = int(
                    np.random.SeedSequence([20261008, seed, ai]).generate_state(1)[0]
                )
                print(
                    f"Seed {seed} {arm}: {len(selected)} rows, {len(reference.columns)} features x {nrep} repeats; prediction check passed",
                    flush=True,
                )
                t = time.time()
                result = permutation_importance(
                    restricted,
                    restricted.keys,
                    selected.y_true.to_numpy(),
                    scoring=score_auc,
                    n_repeats=nrep,
                    random_state=random_seed,
                    n_jobs=1,
                    max_samples=1.0,
                )
                baseline = roc_auc_score(selected.y_true, cached.base)
                current = (
                    pd.DataFrame(result.importances.T, columns=reference.columns)
                    .rename_axis("repeat")
                    .reset_index()
                    .melt(
                        id_vars="repeat",
                        var_name="feature",
                        value_name="auroc_decrease",
                    )
                )
                current.insert(0, "seed", seed)
                current.insert(1, "treatment_arm", arm)
                current["baseline_auroc"] = baseline
                current["permuted_auroc"] = baseline - current.auroc_decrease
                assert np.isfinite(current.auroc_decrease).all()
                np.testing.assert_allclose(
                    current.loc[current.feature.eq("session"), "auroc_decrease"],
                    0,
                    atol=1e-12,
                )
                current.to_csv(target, index=False)
                support = restricted.support(reference.columns)
                support.insert(0, "seed", seed)
                support.insert(1, "treatment_arm", arm)
                support.to_csv(work / f"{arm}_seed_{seed}_support.csv", index=False)
                audit = dict(
                    fingerprint=fingerprint,
                    seed=seed,
                    arm=arm,
                    repeats=nrep,
                    features=len(reference.columns),
                    baseline_auroc=baseline,
                    calibration_slope=params["a"],
                    max_prediction_error=max_error,
                    seconds=time.time() - t,
                )
                write_json(marker, audit)
                audits.append(audit)
                print(
                    f'Completed seed {seed} {arm}: {audit["seconds"]:.1f}s', flush=True
                )
            print(f"Seed {seed} total: {time.time()-start:.1f}s", flush=True)
    records = pd.concat(
        [
            pd.read_csv(work / f"{arm}_seed_{seed}_repeats30.csv")
            for seed in seeds
            for arm in ["P-STAIR", "SC"]
        ],
        ignore_index=True,
    )
    assert records.groupby(["seed", "treatment_arm", "feature"]).size().eq(30).all()
    summaries = []
    byseeds = []
    for arm in ["P-STAIR", "SC"]:
        byseed, summary = summarize_importance(
            records.loc[records.treatment_arm.eq(arm)], seeds
        )
        summary.insert(0, "treatment_arm", arm)
        summary.insert(1, "rank", np.arange(1, len(summary) + 1))
        byseed.insert(1, "treatment_arm", arm)
        summaries.append(summary)
        byseeds.append(byseed)
    summary = pd.concat(summaries, ignore_index=True)
    byseed = pd.concat(byseeds, ignore_index=True)
    summary.to_csv(out / "tables/feature_importance_both_arms.csv", index=False)
    byseed.to_csv(out / "tables/importance_by_seed_both_arms.csv", index=False)
    records.to_csv(out / "tables/permutation_repeats.csv", index=False)
    support = pd.concat(
        [
            pd.read_csv(work / f"{arm}_seed_{seed}_support.csv")
            for seed in seeds
            for arm in ["P-STAIR", "SC"]
        ],
        ignore_index=True,
    )
    support.to_csv(out / "tables/session_feature_support.csv", index=False)
    compare = (
        summaries[0]
        .drop(columns=["treatment_arm", "n_seeds", "seeds_using_feature"])
        .rename(
            columns={
                "rank": "pstair_rank",
                "importance_mean": "pstair_mean",
                "importance_sd": "pstair_sd",
            }
        )
        .merge(
            summaries[1][
                ["feature", "rank", "importance_mean", "importance_sd"]
            ].rename(
                columns={
                    "rank": "sc_rank",
                    "importance_mean": "sc_mean",
                    "importance_sd": "sc_sd",
                }
            ),
            on="feature",
            validate="one_to_one",
        )
    )
    paired = byseed.pivot(
        index=["seed", "feature"], columns="treatment_arm", values="importance_mean"
    )
    assert not paired.isna().any().any()
    paired["pstair_minus_sc"] = paired["P-STAIR"] - paired["SC"]
    paired.reset_index().to_csv(out / "tables/arm_difference_by_seed.csv", index=False)
    diff = (
        paired.groupby("feature")
        .pstair_minus_sc.agg(["mean", "std"])
        .rename(columns={"mean": "difference_mean", "std": "difference_sd"})
    )
    compare = compare.merge(diff, on="feature").sort_values("pstair_rank")
    compare.to_csv(out / "tables/arm_comparison_all_features.csv", index=False)
    assert all(sha(Path(p)) == digest for p, digest in hashes.items())
    write_json(
        out / "methods/protocol.json",
        dict(
            source_run=str(run),
            wide=str(wide),
            wide_sha256=sha(wide),
            source_hashes=hashes,
            fingerprint=fingerprint,
            seeds=seeds,
            repeats=30,
            random_seed=20261008,
            random_stream="SeedSequence([20261008, seed, arm_index]), arm order P-STAIR, SC",
            engine="sklearn.inspection.permutation_importance",
            sklearn_version=sklearn.__version__,
            calibration="Saved sklearn CalibratedClassifierCV(method=sigmoid) predict_proba; validation only; fixed",
            fusion="Equal mean of four raw probabilities",
            permutation="Independent columns within treatment arm and session; all sessions jointly",
            aggregation=f"Mean over 30 repeats per seed; equal mean and sample SD over {len(seeds)} seeds",
            population="All observed test session rows 0-23 within arm; no participant weighting",
            text="Saved text NN raw score, not tokens",
            fixed_context="Session cannot change by construction",
            uncertainty="SD across overlapping participant splits, not confidence intervals; exploratory descriptive comparison",
            cache_verification="Every feature in both arms of seed 0 checked against saved component prediction; <=1e-12 difference",
            source_files_unchanged=all(
                sha(Path(p)) == value for p, value in hashes.items()
            ),
            audits=audits,
        ),
    )
    print("ALL COMPLETE", flush=True)


if __name__ == "__main__":
    main()
