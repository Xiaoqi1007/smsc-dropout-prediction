"""Fresh clinician-rating baseline on the same participant splits."""

from pathlib import Path
import numpy as np, pandas as pd, joblib
from .calibration import fit_sklearn_platt, apply_platt
from .experiment import write_json, sha

SCORES = [
    "session_note_struct__session_engagement_score",
    "session_note_struct__learning_engagement_score",
    "session_note_struct__trauma_parenting_engagement_score",
    "session_note_struct__motivation_to_continue_score",
]


def normalize_ids(values):
    return (
        values.fillna("")
        .astype(str)
        .str.strip()
        .str.replace(r"^(\d+)\.0$", r"\1", regex=True)
    )


def fit_calibration(probability, outcome):
    params, _ = fit_sklearn_platt(probability, outcome)
    return dict(method="platt", parameters=params)


def calibrated(probability, fitted):
    return apply_platt(probability, fitted["parameters"])


def baseline_on_saved_splits(full, wide, experiment, output):
    """Fit the nine clinician-rating predictors and calibrate on validation records."""
    TABLES = Path(output) / "tables"
    METHODS = Path(output) / "methods"
    PRIVATE = Path(output) / "private"
    for p in [TABLES, METHODS, PRIVATE]:
        p.mkdir(parents=True, exist_ok=True)

    def csv_table(frame, name, private=False):
        frame.to_csv((PRIVATE if private else TABLES) / (name + ".csv"), index=False)

    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.exceptions import ConvergenceWarning
    import warnings

    raw = pd.read_csv(wide, dtype={"id": str}, low_memory=False)
    raw["id"] = normalize_ids(raw.id)
    for c in ["session", "ever_dropout", "next_session_dropout", *SCORES]:
        original = raw[c]
        raw[c] = pd.to_numeric(original, errors="coerce")
        if c in SCORES and (
            (original.notna() & raw[c].isna()).any()
            or not raw[c].dropna().isin([1, 2, 3, 4, 5]).all()
        ):
            raise ValueError(f"Clinician score coding must be 1–5 or missing: {c}")
    raw = raw.loc[
        raw.id.ne("")
        & raw[["session", "ever_dropout", "next_session_dropout"]].notna().all(axis=1)
        & raw.session.between(*experiment.config["session_window"])
    ].copy()
    keys = ["id", "session"]
    if raw.duplicated(keys).any():
        raise ValueError("Duplicate participant/session in baseline source")
    reference = (
        full.loc[full.seed.eq(experiment.seeds[0]), keys + ["y_true"]]
        .sort_values(keys)
        .reset_index(drop=True)
    )
    source = (
        raw[keys + ["ever_dropout"]]
        .rename(columns={"ever_dropout": "y_true"})
        .sort_values(keys)
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(reference, source, check_dtype=False)
    flags = [c + "_missing" for c in SCORES]
    raw[flags] = raw[SCORES].isna().astype(int).to_numpy()
    features = ["session", *SCORES, *flags]
    recorded = raw[SCORES].notna().any(axis=1)
    coverage = pd.DataFrame(
        [
            dict(
                participant_sessions=len(raw),
                participants=raw.id.nunique(),
                rows_with_any_rating=int(recorded.sum()),
                participants_with_any_rating=raw.loc[recorded, "id"].nunique(),
            )
        ]
    )
    (rows, coefficients, protocols) = ([], [], [])
    for seed, part in full.groupby("seed", sort=True):
        pd.testing.assert_frame_equal(
            part[keys + ["y_true"]].sort_values(keys).reset_index(drop=True),
            source,
            check_dtype=False,
        )
        joined = part.drop(columns=["model"]).merge(
            raw[keys + SCORES + flags], on=keys, validate="one_to_one"
        )
        train = joined.loc[joined.split.eq("train")]
        counts = np.bincount(train.y_true.astype(int), minlength=2)
        if (counts == 0).any():
            raise ValueError("Baseline training requires both outcome classes")
        weights = 1 / train.id.map(train.id.value_counts()).to_numpy(float)
        weights *= len(train) / (2 * counts[train.y_true.astype(int).to_numpy()])
        weights *= len(weights) / weights.sum()
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scaler", StandardScaler()),
                (
                    "lr",
                    LogisticRegression(
                        C=1.0,
                        penalty="l2",
                        solver="lbfgs",
                        max_iter=5000,
                        random_state=int(seed),
                    ),
                ),
            ]
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model.fit(train[features], train.y_true, lr__sample_weight=weights)
        matrix = model[:-1].transform(joined[features])
        (unique_matrix, inverse) = np.unique(matrix, axis=0, return_inverse=True)
        p = model["lr"].predict_proba(unique_matrix)[:, 1][inverse]
        valid = joined.split.eq("validation").to_numpy()
        fitted_calibration = fit_calibration(p[valid], joined.loc[valid, "y_true"])
        out = joined[["seed", "split", "id", "session", "y_true"]].copy()
        out["probability_raw"] = p
        out["probability"] = calibrated(p, fitted_calibration)
        out["model"] = "clinician_baseline"
        rows.append(out)
        for col, value, median, mean, scale in zip(
            features,
            model["lr"].coef_[0],
            model["imputer"].statistics_,
            model["scaler"].mean_,
            model["scaler"].scale_,
        ):
            coefficients.append(
                dict(
                    seed=seed,
                    feature=col,
                    standardized_coefficient=value,
                    training_imputation=median,
                    training_mean=mean,
                    training_scale=scale,
                )
            )
        joblib.dump(model, PRIVATE / f"baseline_seed_{seed}.joblib")
        protocols.append(
            dict(
                seed=int(seed),
                calibration=fitted_calibration,
                intercept=float(model["lr"].intercept_[0]),
            )
        )
    result = pd.concat(rows, ignore_index=True)
    compare = ["seed", "split", "id", "session", "y_true"]
    pd.testing.assert_frame_equal(
        full[compare].sort_values(compare).reset_index(drop=True),
        result[compare].sort_values(compare).reset_index(drop=True),
        check_dtype=False,
    )
    csv_table(coverage, "baseline_rating_coverage")
    csv_table(pd.DataFrame(coefficients), "baseline_coefficients_and_preprocessing")
    csv_table(result, "baseline_predictions", private=True)
    write_json(
        METHODS / "baseline_protocol.json",
        dict(
            features=features,
            model="Fixed L2 logistic regression, C=1, lbfgs, max_iter=5000",
            fit_partition="train only; participant splits copied from the final model",
            missing="Four flags before median imputation; all-missing training columns filled with 0",
            weights="Inverse participant row count × inverse row-class frequency, normalized to mean 1",
            calibration="platt",
            calibration_partition="validation only; no test-based tuning",
            seeds=protocols,
            data_sha256=sha(wide),
        ),
    )
    return (result, coverage)
