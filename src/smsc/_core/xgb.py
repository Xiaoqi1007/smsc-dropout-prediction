"""Validation-based XGBoost selection."""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier


def fit_preassessment_xgb(config, frame, columns, masks, seed, threads):
    from . import embeddings

    base = embeddings
    first = ~frame[config["id_column"]].duplicated().to_numpy()
    train, validation = (masks["train"] & first, masks["validation"] & first)
    y = frame["ever_dropout"].to_numpy(dtype=int)
    ids = frame[config["id_column"]].to_numpy()
    numeric = (
        frame[columns]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    imputer.fit(numeric.loc[train])
    matrix = imputer.transform(numeric).astype(np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError(
            "Nonfinite pre-assessment matrix after training-median imputation"
        )
    settings = config.get("xgb", {})
    grid = settings.get(
        "candidates",
        [
            dict(max_depth=2, n_estimators=100),
            dict(max_depth=2, n_estimators=200),
            dict(max_depth=3, n_estimators=100),
            dict(max_depth=3, n_estimators=200),
        ],
    )
    if not grid:
        raise ValueError("xgb.candidates must not be empty")
    weights = base.training_weights(config, ids[train], y[train])
    rows, winner, best, best_index = ([], None, -np.inf, None)
    allowed = {
        "max_depth",
        "n_estimators",
        "learning_rate",
        "min_child_weight",
        "subsample",
        "colsample_bytree",
        "reg_lambda",
        "reg_alpha",
        "gamma",
    }
    for index, candidate in enumerate(grid):
        if set(candidate) - allowed:
            raise ValueError("Unsupported XGBoost search parameter")
        parameters = xgb_parameters()
        parameters.update(candidate)
        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=threads,
            random_state=seed,
            verbosity=0,
            **parameters,
        )
        model.fit(matrix[train], y[train], sample_weight=weights)
        score = float(
            roc_auc_score(y[validation], model.predict_proba(matrix[validation])[:, 1])
        )
        rows.append(dict(candidate=index, validation_auroc=score, **parameters))
        if score > best:
            winner, best, best_index = (model, score, index)
    for row in rows:
        row["selected"] = row["candidate"] == best_index
    probabilities = {
        split: winner.predict_proba(matrix[masks[split]])[:, 1]
        for split in ("validation", "test")
    }
    schema = dict(
        raw_requested=columns,
        encoded_columns=columns,
        usable_train_columns=columns,
        n_unique_training_participants=int(train.sum()),
        n_unique_validation_participants=int(validation.sum()),
        imputation="Median of unique training participants",
        selection_metric="Validation participant-level AUROC",
        selected_candidate=best_index,
        validation_auroc=best,
        refit_train_plus_validation=False,
        categorical_policy="Supplied numeric coding retained; no new outcome-based screening",
    )
    bundle = dict(model=winner, preprocessor=imputer, schema=schema)
    history = [
        dict(
            n_unique_training_participants=int(train.sum()),
            selected_candidate=best_index,
            validation_auroc=best,
        )
    ]
    return (bundle, probabilities, history, schema, rows)


def xgb_parameters(candidate=None):
    parameters = dict(
        learning_rate=0.05,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.0,
    )
    parameters.update(candidate or {})
    return parameters
