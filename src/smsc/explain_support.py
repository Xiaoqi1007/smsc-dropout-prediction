"""Adapters for explaining a frozen four-source model."""

from pathlib import Path
import json, os
import numpy as np, pandas as pd, joblib
from sklearn.metrics import roc_auc_score
from .experiment import Experiment, read_csv, read_json, sha
from .models import branch_columns, predict_branch
from .config import TEXT_MODE, TEXT_REPRESENTATION
from .training import BRANCHES
from ._core.fusion import predict_average as predict_fusion
from .calibration import apply_platt

LABEL_MAP = {}
_label_path = os.environ.get("SMSC_FEATURE_LABELS")
if _label_path:
    LABEL_MAP = json.loads(Path(_label_path).read_text())
EPROP = [
    f"emotion__{e}_prop"
    for e in ["Neutral", "Happy", "Sad", "Angry", "Fearful", "Disgusted", "Surprised"]
]


def label(c):
    return LABEL_MAP.get(c, c.replace("__", " ").replace("_", " ").capitalize())


def calibrated(p, c):
    return apply_platt(p, c["parameters"])


def aligned(table, frame):
    assert not table.duplicated(["id", "session"]).any()
    keys = pd.MultiIndex.from_frame(frame[["id", "session"]])
    actual = pd.MultiIndex.from_frame(table[["id", "session"]])
    assert set(keys) == set(actual)
    return table.set_index(["id", "session"]).loc[keys].reset_index()


def load_model(run, seed):
    experiment = Experiment(run)
    folder = experiment.verify_seed(seed)
    assert (
        sha(Path(run) / "data/features.csv")
        == read_json(Path(run) / "data/DONE.json")["features_sha256"]
    )
    frame = read_csv(Path(run) / "data/features.csv")
    long = read_csv(folder / "predictions.csv")
    models = {
        b: joblib.load(folder / f"models/{b}.joblib")
        for b in BRANCHES
        if b != "text_only"
    }
    fitted = joblib.load(folder / "models/full_model.joblib")
    calibrator = read_json(folder / "calibrators.json")["full_model"]
    predictions = aligned(long[long.model.eq("full_model")], frame)
    predictions["text_only"] = aligned(
        long[long.model.eq("text_only")], frame
    ).probability_raw.to_numpy()
    scores = {b: predict_branch(m, frame) for b, m in models.items()}
    for b, p in scores.items():
        np.testing.assert_allclose(
            p,
            aligned(long[long.model.eq(b)], frame).probability_raw,
            atol=1e-12,
            rtol=0,
        )
    matrix = np.column_stack(
        [predictions.text_only if b == "text_only" else scores[b] for b in BRANCHES]
    )
    raw = predict_fusion(fitted, matrix)
    np.testing.assert_allclose(raw, predictions.probability_raw, atol=1e-12, rtol=0)
    np.testing.assert_allclose(
        calibrated(raw, calibrator), predictions.probability, atol=1e-12, rtol=0
    )
    return dict(
        run=Path(run),
        seed=seed,
        mode=TEXT_MODE,
        representation=TEXT_REPRESENTATION,
        frame=frame,
        predictions=predictions,
        models=models,
        fusion=fitted,
        calibrator=calibrator,
    )


def participant_arms(frame):
    columns = [
        c
        for c in ["speech__arm", "session_note_struct__session_note_arm"]
        if c in frame
    ]
    if not columns:
        raise ValueError(
            "Saved feature snapshot has no recognized treatment-arm columns."
        )
    if frame.id.isna().any() or frame.id.astype(str).str.strip().eq("").any():
        raise ValueError("Missing participant IDs.")
    raw = frame[["id", *columns]].melt(id_vars="id").dropna(subset=["value"])
    raw["value"] = raw.value.astype(str).str.strip().str.lower()
    raw = raw.loc[raw.value.ne("")].copy()
    mapping = {
        "pstair": "P-STAIR",
        "p-stair": "P-STAIR",
        "sc": "SC",
        "supportive counseling": "SC",
    }
    unknown = sorted(set(raw.value) - set(mapping))
    if unknown:
        raise ValueError(f"Unknown treatment-arm codes; verify coding: {unknown}")
    raw["treatment_arm"] = raw.value.map(mapping)
    conflicts = raw.groupby("id").treatment_arm.nunique()
    if conflicts.gt(1).any():
        raise ValueError(
            f"Conflicting treatment arms for {int(conflicts.gt(1).sum())} participants."
        )
    assigned = raw.groupby("id").treatment_arm.first()
    people = frame[["id"]].drop_duplicates().copy()
    people["treatment_arm"] = people.id.map(assigned).fillna("NO_RECORDED_ARM")
    sources = raw.groupby("id").variable.agg(lambda x: "; ".join(sorted(set(x))))
    people["arm_sources"] = people.id.map(sources).fillna("No recorded arm")
    return people.reset_index(drop=True)


def case_metadata(frame, verification):
    """Describe the participant-session represented by a SHAP result."""
    case_id = str(verification["patient_id"])
    session = int(verification["session"])
    observed = frame.loc[frame.id.eq(case_id) & frame.session.eq(session)]
    if len(observed) != 1:
        raise ValueError("A SHAP example must match one observed participant-session")
    outcome = int(observed.ever_dropout.iloc[0])
    if outcome not in (0, 1):
        raise ValueError("SHAP examples require a known binary outcome")
    arms = participant_arms(frame).set_index("id").treatment_arm
    arm = arms.loc[case_id]
    if arm not in ("P-STAIR", "SC"):
        raise ValueError("The example has no treatment-arm assignment")
    return dict(
        session=session,
        outcome="Dropout" if outcome else "Completed",
        treatment_arm=arm,
    )


def cohort_counts(predictions):
    records = []
    for (seed, arm), group in predictions.groupby(["seed", "treatment_arm"], sort=True):
        people = group[["id", "y_true"]].drop_duplicates()
        if people.id.duplicated().any():
            raise ValueError("Outcome is inconsistent within participant.")
        records.append(
            dict(
                seed=seed,
                treatment_arm=arm,
                test_rows=len(group),
                test_patients=len(people),
                dropout_patients=int(people.y_true.eq(1).sum()),
                non_dropout_patients=int(people.y_true.eq(0).sum()),
                dropout_rows=int(group.y_true.eq(1).sum()),
                non_dropout_rows=int(group.y_true.eq(0).sum()),
                n_sessions=int(group.session.nunique()),
                baseline_auroc=(
                    roc_auc_score(group.y_true, group.probability)
                    if group.y_true.nunique() == 2
                    else np.nan
                ),
            )
        )
    return pd.DataFrame(records)


class FrozenModel:
    """Reversible numeric input table around one saved, calibrated full model."""

    def __init__(self, saved):
        self.saved = saved
        self.columns = sorted(
            set().union(
                *(set(branch_columns(model)) for model in saved["models"].values())
            )
        )
        self.inputs = saved["frame"][self.columns].reset_index(drop=True).copy()
        self.inputs["text_nn_score"] = saved["predictions"].text_only.to_numpy()
        self.columns = list(self.inputs)
        (self.lookup, codes) = ({}, {})
        for column in self.columns:
            (indices, values) = pd.factorize(
                self.inputs[column], sort=False, use_na_sentinel=False
            )
            self.lookup[column] = values
            codes[column] = indices.astype(float)
        self.encoded = pd.DataFrame(codes)

    def fit(self, X, y=None):
        raise RuntimeError("This frozen adapter must not be fitted.")

    def decode(self, matrix):
        matrix = np.asarray(matrix, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.columns):
            raise ValueError("Unexpected model-input shape.")
        restored = {}
        for j, column in enumerate(self.columns):
            indices = matrix[:, j].astype(int)
            if (
                not np.array_equal(indices, matrix[:, j])
                or (indices < 0).any()
                or (indices >= len(self.lookup[column])).any()
            ):
                raise ValueError(f"Invalid lookup code for {column}")
            restored[column] = self.lookup[column].take(indices).to_numpy()
        return pd.DataFrame(restored)

    def predict_frame(self, rows):
        outputs = []
        for start in range(0, len(rows), 2048):
            batch = rows.iloc[start : start + 2048]
            scores = {
                name: predict_branch(model, batch)
                for (name, model) in self.saved["models"].items()
            }
            scores["text_only"] = batch["text_nn_score"].to_numpy(float)
            matrix = np.column_stack([scores[name] for name in BRANCHES])
            raw = predict_fusion(self.saved["fusion"], matrix)
            outputs.append(calibrated(raw, self.saved["calibrator"]))
        return np.concatenate(outputs)

    def predict_probability(self, X):
        return self.predict_frame(self.decode(X))

    def predict_proba(self, X):
        probability = self.predict_probability(X)
        return np.column_stack([1 - probability, probability])


def score_auc(estimator, X, y):
    return roc_auc_score(y, estimator.predict_proba(X)[:, 1])


def importance_label(column):
    if column == "text_nn_score":
        return "Text NN score"
    if column.startswith("emotion__derived_prior_before_session_") and column.endswith(
        "_prop"
    ):
        emotion = column.removeprefix(
            "emotion__derived_prior_before_session_"
        ).removesuffix("_prop")
        return f"{emotion} emotion proportion (history)"
    if column in EPROP:
        return (
            column.split("__")[1].removesuffix("_prop")
            + " emotion proportion (current)"
        )
    return label(column)


def summarize_importance(records, seeds):
    if set(records.seed.unique()) != set(seeds):
        raise ValueError("Seed results are incomplete")
    by_seed = records.groupby(["seed", "feature"]).agg(
        importance_mean=("auroc_decrease", "mean"),
        permutation_sd=("auroc_decrease", "std"),
        n_repeats=("auroc_decrease", "size"),
    )
    features = sorted(records.feature.unique())
    index = pd.MultiIndex.from_product([seeds, features], names=["seed", "feature"])
    by_seed = by_seed.reindex(index)
    by_seed["used_by_model"] = by_seed.n_repeats.notna()
    by_seed[["importance_mean", "permutation_sd", "n_repeats"]] = by_seed[
        ["importance_mean", "permutation_sd", "n_repeats"]
    ].fillna(0)
    by_seed = by_seed.reset_index()
    summary = (
        by_seed.groupby("feature", as_index=False)
        .agg(
            importance_mean=("importance_mean", "mean"),
            importance_sd=("importance_mean", "std"),
            n_seeds=("seed", "nunique"),
            seeds_using_feature=("used_by_model", "sum"),
        )
        .sort_values(["importance_mean", "feature"], ascending=[False, True])
    )
    summary["label"] = summary.feature.map(importance_label)
    return (by_seed, summary.reset_index(drop=True))


class WithinSessionModel:
    """Translate sklearn's shuffled ordering keys into within-session donor rows."""

    def __init__(self, model, X, sessions):
        self.model = model
        self.original = np.asarray(X, dtype=float).copy()
        self.sessions = np.asarray(sessions)
        if len(self.sessions) != len(self.original) or pd.isna(self.sessions).any():
            raise ValueError("Every test row must have an observed session.")
        self.groups = [
            np.flatnonzero(self.sessions == session)
            for session in pd.unique(self.sessions)
        ]
        self.keys = np.broadcast_to(
            np.arange(len(X))[:, None], self.original.shape
        ).copy()

    def fit(self, X, y=None):
        raise RuntimeError("This frozen adapter must not be fitted.")

    def restore(self, keys):
        keys = np.asarray(keys)
        if keys.shape != self.keys.shape:
            raise ValueError("Use the complete test matrix: max_samples=1.0.")
        changed = np.flatnonzero(np.any(keys != self.keys, axis=0))
        if len(changed) and (
            not np.array_equal(np.sort(keys[:, changed], axis=0), self.keys[:, changed])
        ):
            raise ValueError(
                "Each key column must be a permutation of the original row positions."
            )
        restored = self.original.copy()
        for column in changed:
            for rows in self.groups:
                ranks = np.argsort(
                    np.argsort(keys[rows, column], kind="stable"), kind="stable"
                )
                donors = rows[ranks]
                restored[rows, column] = self.original[donors, column]
        return restored

    def predict_proba(self, keys):
        return self.model.predict_proba(self.restore(keys))

    def support(self, feature_names):
        records = []
        for rows in self.groups:
            for column, feature in enumerate(feature_names):
                unique = len(np.unique(self.original[rows, column]))
                records.append(
                    dict(
                        session=self.sessions[rows[0]],
                        feature=feature,
                        n_rows=len(rows),
                        n_unique_including_missing=unique,
                        can_change=unique > 1,
                    )
                )
        return pd.DataFrame(records)


class CachedModel:
    """Reuse unaffected branch predictions; preserve the reference estimator exactly."""

    def __init__(self, reference, mask, native):
        (self.reference, self.native) = (reference, native)
        self.original = reference.encoded.loc[mask].to_numpy()
        self.frame = reference.decode(self.original)
        self.columns = reference.columns
        self.models = reference.saved["models"]
        self.branch_columns = {
            b: set(branch_columns(m)) for (b, m) in self.models.items()
        }
        scores = {b: predict_branch(m, self.frame) for (b, m) in self.models.items()}
        scores["text_only"] = self.frame.text_nn_score.to_numpy(float)
        self.matrix = np.column_stack([scores[b] for b in BRANCHES])
        self.base = self.native.predict_proba(
            predict_fusion(reference.saved["fusion"], self.matrix)[:, None]
        )[:, 1]

    def fit(self, X, y=None):
        raise RuntimeError("Frozen")

    def predict_proba(self, X):
        X = np.asarray(X)
        assert X.shape == self.original.shape
        changed = np.flatnonzero(np.any(X != self.original, axis=0))
        if not len(changed):
            p = self.base
        else:
            frame = self.frame.copy()
            names = set()
            for j in changed:
                name = self.columns[j]
                names.add(name)
                vals = X[:, j].astype(int)
                assert np.array_equal(vals, X[:, j])
                frame[name] = self.reference.lookup[name].take(vals).to_numpy()
            matrix = self.matrix.copy()
            for i, b in enumerate(BRANCHES):
                if b == "text_only":
                    if b.replace("text_only", "text_nn_score") in names:
                        matrix[:, i] = frame.text_nn_score.to_numpy(float)
                elif self.branch_columns[b] & names:
                    matrix[:, i] = predict_branch(self.models[b], frame)
            raw = predict_fusion(self.reference.saved["fusion"], matrix)
            p = self.native.predict_proba(raw[:, None])[:, 1]
        return np.column_stack([1 - p, p])


def direction_for_feature(adapter, column, overrides=None):
    """Training-defined contrasts of the frozen model, without test outcomes."""
    saved = adapter.saved
    result = dict(
        seed=saved["seed"],
        feature=column,
        kind="unavailable",
        status="not_in_model",
        effect_pp=np.nan,
        low=np.nan,
        high=np.nan,
        n_rows=0,
        nonmonotone=False,
    )
    if column not in adapter.columns:
        return result
    splits = saved["predictions"].split.to_numpy().copy()
    splits[~adapter.arm_mask] = "excluded"
    train = adapter.inputs.loc[splits == "train"].copy()
    if column.startswith("pre_assessment__"):
        train_ids = saved["frame"].loc[splits == "train", "id"]
        train = train.loc[~train_ids.duplicated().to_numpy()]
    values = train[column].dropna()
    numeric = pd.to_numeric(values, errors="coerce")
    numeric_ok = len(values) > 0 and np.isfinite(numeric.to_numpy(float)).all()
    unique = np.sort(numeric.unique()) if numeric_ok else []
    if numeric_ok and len(unique) == 2 and (set(unique) == {0, 1}):
        kind = "binary"
    elif numeric_ok and (
        len(unique) > 12
        or column in ["session", "text_nn_score"]
        or any(
            (
                token in column
                for token in [
                    "score__",
                    "_score",
                    "days_since",
                    "age_years",
                    "totalnumberofchildren",
                    "_count",
                    "_words",
                    "_prop",
                    "_ratio",
                    "_fraction",
                    "_seconds",
                    "_mins",
                    "_second",
                    "n_snippets",
                ]
            )
        )
    ):
        kind = "continuous"
    else:
        kind = "categorical"
    kind = (overrides or {}).get(column, kind)
    if kind not in ["continuous", "binary", "categorical"]:
        raise ValueError(f"Invalid direction type for {column}: {kind}")
    result["kind"] = kind
    if kind == "categorical":
        result["status"] = "categorical_or_no_numeric_support"
        return result
    if not numeric_ok:
        raise ValueError(
            f"Numeric direction requested without numeric support: {column}"
        )
    rows = adapter.inputs.loc[splits == "test"].copy()
    observed = pd.to_numeric(rows[column], errors="coerce")
    rows = rows.loc[np.isfinite(observed)].copy()
    result["n_rows"] = len(rows)
    if not len(rows) or len(unique) < 2:
        result["status"] = "no_observed_variation"
        return result
    grid = (
        np.array([0.0, 1.0])
        if kind == "binary"
        else np.quantile(numeric, [0.1, 0.2, 0.25, 0.4, 0.5, 0.6, 0.75, 0.8, 0.9])
    )
    if kind == "binary" and (not set(unique).issubset({0, 1})):
        raise ValueError(f"Binary direction requires 0/1 coding: {column}")
    (low_index, high_index) = (0, 1) if kind == "binary" else (2, 6)
    result.update(low=float(grid[low_index]), high=float(grid[high_index]))
    if grid[low_index] >= grid[high_index]:
        result["status"] = "no_contrast_variation"
        return result
    means = []
    for value in grid:
        changed = rows.copy()
        changed[column] = (
            values.loc[numeric.eq(value)].iloc[0] if kind == "binary" else value
        )
        means.append(100 * adapter.predict_frame(changed).mean())
    increments = np.diff(means)
    result.update(
        status="ok",
        effect_pp=float(means[high_index] - means[low_index]),
        nonmonotone=bool((increments > 0.01).any() and (increments < -0.01).any()),
    )
    return result


def compact_contributions(table, max_features):
    """Keep the text branch visible; preserve all omitted effects as one signed sum."""
    table = table.sort_values("abs_contribution_pp", ascending=False).reset_index(
        drop=True
    )
    shown = table.head(max_features).copy()
    text = table.loc[table.feature.eq("text_nn_score")]
    if len(text) != 1:
        raise ValueError("Expected one text-branch SHAP value")
    if "text_nn_score" not in set(shown.feature):
        shown = pd.concat([shown.iloc[:-1], text], ignore_index=True)
        shown = shown.sort_values("abs_contribution_pp", ascending=False)
    other = table.loc[~table.feature.isin(shown.feature)]
    if len(other):
        shown = pd.concat(
            [
                shown,
                pd.DataFrame(
                    [
                        dict(
                            feature="__other__",
                            label=f"Other inputs ({len(other)})",
                            contribution_pp=other.contribution_pp.sum(),
                            mc_se_pp=np.nan,
                            abs_contribution_pp=abs(other.contribution_pp.sum()),
                        )
                    ]
                ),
            ],
            ignore_index=True,
        )
    np.testing.assert_allclose(
        shown.contribution_pp.sum(), table.contribution_pp.sum(), atol=1e-08
    )
    return shown.reset_index(drop=True)
