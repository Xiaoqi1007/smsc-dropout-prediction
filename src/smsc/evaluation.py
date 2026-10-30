"""Held-out metrics and independent reconstruction from newly fitted models."""

import joblib
import numpy as np
import pandas as pd
from .experiment import Experiment, write_json
from .metrics import evaluate, METRICS
from .models import predict_branch, predict_text
from .config import TEXT_MODE, training_config
from ._core import pipeline
from .training import BRANCHES
from threadpoolctl import threadpool_limits


def evaluate_run(run, project):
    experiment = Experiment(run)
    predictions = experiment.predictions()
    rows = []
    for (seed, model), group in predictions[predictions.split.eq("test")].groupby(
        ["seed", "model"]
    ):
        for window, hi in [("all", 23), ("early", 9)]:
            part = group[group.session.between(0, hi)]
            rows.append(
                dict(
                    seed=seed,
                    model=model,
                    window=window,
                    **evaluate(part.y_true, part.probability),
                )
            )
    out = experiment.path / "tables"
    out.mkdir(exist_ok=True)
    by_seed = pd.DataFrame(rows)
    by_seed.to_csv(out / "metrics_by_seed.csv", index=False)
    summary = by_seed.groupby(["model", "window"])[METRICS].agg(["mean", "std"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary.reset_index().to_csv(out / "metrics_summary.csv", index=False)
    return out / "metrics_summary.csv"


def verify_run(run, project):
    experiment = Experiment(run)
    config = training_config(experiment.config, project)
    base, bundle, splits, audit = pipeline.load_inputs(config)
    frame = bundle.frame
    checks = []
    with threadpool_limits(1):
        for seed in experiment.seeds:
            folder = experiment.verify_seed(seed)
            masks = {
                k: v
                for k, v in base.participant_masks(bundle, splits, config, seed).items()
                if k in ["train", "validation", "test"]
            }
            people = {k: set(frame.loc[v, "id"]) for k, v in masks.items()}
            assert not people["train"] & (people["validation"] | people["test"])
            assert not people["validation"] & people["test"]
            mode = TEXT_MODE
            current = base.load_current_embedding(config, bundle, mode, seed)
            matrix = base.cumulative_history(config, bundle, current)[0]
            raw = {}
            for branch in BRANCHES:
                payload = joblib.load(folder / "models" / f"{branch}.joblib")
                raw[branch] = (
                    predict_text(payload, matrix)
                    if branch == "text_only"
                    else predict_branch(payload, frame)
                )
            assert (
                joblib.load(folder / "models/session_xgb.joblib")[
                    "model"
                ].__class__.__name__
                == "XGBClassifier"
            )
            assert (
                joblib.load(folder / "models/preassessment_xgb.joblib")[
                    "model"
                ].__class__.__name__
                == "XGBClassifier"
            )
            assert (
                joblib.load(folder / "models/emotion_speech_lr.joblib")[
                    "model"
                ].__class__.__name__
                == "LogisticRegression"
            )
            raw["full_model"] = np.mean(
                np.column_stack([raw[b] for b in BRANCHES]), axis=1
            )
            saved = pd.read_csv(folder / "predictions.csv", dtype={"id": str})
            max_error = 0.0
            for model, values in raw.items():
                native = joblib.load(folder / "models" / f"{model}_calibration.joblib")
                fresh = frame[["id", "session"]].assign(
                    raw=values, calibrated=native.predict_proba(values[:, None])[:, 1]
                )
                aligned = fresh.merge(
                    saved[saved.model.eq(model)],
                    on=["id", "session"],
                    validate="one_to_one",
                )
                assert len(aligned) == len(frame)
                error = max(
                    abs(aligned.raw - aligned.probability_raw).max(),
                    abs(aligned.calibrated - aligned.probability).max(),
                )
                assert error < 1e-7, (seed, model, error)
                max_error = max(max_error, float(error))
            checks.append(
                dict(
                    seed=seed,
                    participants={k: len(v) for k, v in people.items()},
                    max_probability_error=max_error,
                )
            )
    result = dict(
        seeds=checks,
        architecture="XGB/XGB/LR/NN",
        scope="Fresh downstream fits from existing private features and embeddings; raw-media and selector fitting excluded",
    )
    destination = experiment.path / "verification.json"
    write_json(destination, result)
    return destination
