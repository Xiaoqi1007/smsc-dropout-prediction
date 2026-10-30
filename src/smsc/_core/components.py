"""Fit component models on training data with validation-based early stopping."""

from __future__ import annotations
from ..config import TEXT_REPRESENTATION
from .xgb import xgb_parameters
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from . import pipeline as pipeline
from .preassessment import filter_preassessment

COMPONENTS = ("preassessment_xgb", "session_xgb", "emotion_speech_lr")


def fit_text_model(config, base, matrix, y, ids, masks, seed):
    """Fit the text network and retain its best validation checkpoint."""
    for first, second in [
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ]:
        if first not in masks or second not in masks:
            continue
        if set(ids[masks[first]]) & set(ids[masks[second]]):
            raise ValueError("Participant overlap in text fitting")
    probability, history, epoch, state, scaler = pipeline.fit_nn(
        config, matrix, y, ids, masks, seed
    )
    payload = dict(
        scaler=scaler,
        epochs=int(epoch),
        input_dim=matrix.shape[1],
        nn_config=config["nn"],
        state_dict=state,
        training_policy="validation_early_stopping",
        n_fit_participants=int(len(np.unique(ids[masks["train"]]))),
        n_fit_rows=int(masks["train"].sum()),
    )
    return (probability, payload, history)


def assert_separate(frame, masks, config):
    groups = {
        key: set(frame.loc[mask, config["id_column"]]) for key, mask in masks.items()
    }
    keys = list(groups)
    for i, key in enumerate(keys):
        if not groups[key]:
            raise ValueError(f"Empty participant set: {key}")
        for other in keys[i + 1 :]:
            if groups[key] & groups[other]:
                raise ValueError(f"Participant overlap: {key}, {other}")


def fit_components(
    config,
    base,
    frame,
    audit,
    embeddings,
    train,
    prediction_masks,
    seed,
    threads,
    settings,
    destination=None,
):
    from xgboost import XGBClassifier

    y, ids = (frame.ever_dropout.to_numpy(), frame[config["id_column"]].to_numpy())
    outputs = {name: {} for name in prediction_masks}
    columns, filter_audit = filter_preassessment(config, frame, train)
    unique_train = train & ~frame[config["id_column"]].duplicated().to_numpy()
    raw = (
        frame[columns]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )
    imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(
        raw.loc[unique_train]
    )
    matrix = imputer.transform(raw).astype(np.float32)
    parameters = xgb_parameters()
    parameters.update(config["xgb"]["candidates"][settings["xgb_candidate"]])
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=threads,
        random_state=seed,
        verbosity=0,
        **parameters,
    )
    model.fit(
        matrix[unique_train],
        y[unique_train],
        sample_weight=base.training_weights(config, ids[unique_train], y[unique_train]),
    )
    for name, mask in prediction_masks.items():
        outputs[name]["preassessment_xgb"] = model.predict_proba(matrix[mask])[
            :, 1
        ].astype(float)
    if destination:
        destination.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            dict(model=model, preprocessor=imputer, columns=columns),
            destination / "preassessment_xgb.joblib",
        )
        pd.DataFrame(filter_audit).to_csv(
            destination / "preassessment_filter.csv", index=False
        )
    session_columns = [
        c
        for c in audit["tabular_columns"]
        if c.startswith(tuple(config["session_prefixes"]))
        or c in config["session_columns"]
    ]
    for stage, columns in [
        ("session_xgb", session_columns),
        ("emotion_speech_lr", audit["video_columns"]),
    ]:
        matrix, transformer, schema = pipeline.prepare_block(
            base,
            frame,
            columns,
            train,
            config["max_categorical_levels"],
            scale_numeric=(stage == "session_xgb"),
            dtype="float64" if stage == "emotion_speech_lr" else "float32",
        )
        scaler = None
        if stage == "session_xgb":
            from sklearn.metrics import roc_auc_score

            if "validation" not in prediction_masks:
                raise ValueError("Session tuning requires validation participants")
            best = -np.inf
            search = []
            for index, candidate in enumerate(config["xgb"]["candidates"]):
                params = xgb_parameters()
                params.update(candidate)
                fitted = XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    tree_method="hist",
                    n_jobs=threads,
                    random_state=seed,
                    verbosity=0,
                    **params,
                )
                fitted.fit(
                    matrix[train],
                    y[train],
                    sample_weight=base.training_weights(config, ids[train], y[train]),
                )
                held = prediction_masks["validation"]
                score = float(
                    roc_auc_score(y[held], fitted.predict_proba(matrix[held])[:, 1])
                )
                search.append(dict(candidate=index, validation_auroc=score, **params))
                if score > best:
                    model, best, selected = (fitted, score, index)
            settings["session_xgb_candidate"] = selected
            if destination:
                pd.DataFrame(search).assign(
                    selected=lambda d: d.candidate.eq(selected)
                ).to_csv(destination / "session_xgb_validation_search.csv", index=False)
        else:
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler().fit(matrix[train])
            matrix = scaler.transform(matrix).astype(np.float32)
            model = LogisticRegression(
                C=config["source_lr"]["C"],
                max_iter=config["source_lr"]["max_iter"],
                penalty="l2",
                solver="lbfgs",
                random_state=seed,
            )
            model.fit(
                matrix[train],
                y[train],
                sample_weight=base.training_weights(config, ids[train], y[train]),
            )
        for name, mask in prediction_masks.items():
            outputs[name][stage] = model.predict_proba(matrix[mask])[:, 1]
        if destination:
            joblib.dump(
                dict(
                    model=model,
                    preprocessor=transformer,
                    schema=schema,
                    scaler=scaler,
                    input_dtype=(
                        "float64" if stage == "emotion_speech_lr" else "float32"
                    ),
                ),
                destination / (stage + ".joblib"),
            )
    representation = TEXT_REPRESENTATION
    masks = dict(train=train, **prediction_masks)
    probabilities, payload, history = fit_text_model(
        config, base, embeddings[representation], y, ids, masks, seed
    )
    settings["epochs"][representation] = payload["epochs"]
    if destination:
        pd.DataFrame(history).to_csv(
            destination / ("text_" + representation + "_history.csv"), index=False
        )
    for name in prediction_masks:
        outputs[name]["text_" + representation] = probabilities[name].astype(float)
    if destination:
        joblib.dump(payload, destination / ("text_" + representation + ".joblib"))
    return outputs
