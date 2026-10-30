"""Dataset loading, preprocessing, and neural-network fitting."""

from __future__ import annotations
import re
from dataclasses import replace
from . import embeddings


def feature_audit(config, frame):
    base = embeddings
    reserved = {config["id_column"], *config["cache_targets"]}
    tab, video, ignored, unknown = ([], [], {}, [])
    explicit = set(config["tabular_columns"])
    missing = explicit - set(frame.columns)
    if missing:
        raise ValueError(f"Explicit tabular columns absent: {sorted(missing)}")
    for column in frame:
        unsafe = column in reserved or bool(base.LEAKAGE_PATTERN.search(column))
        if column in {
            "session_info__derived_session_current",
            "derived_session_current",
            "session_current_missing",
        }:
            ignored[column] = (
                "duplicate session number; no replacement missingness flag"
            )
        elif column == "emotion__derived_prior_before_session_emotion_n_snippets":
            ignored[column] = "excluded prior-session emotion snippet count"
        elif unsafe:
            ignored[column] = "identifier/outcome/future information"
        elif re.search(
            "source_files.*Sessions\\d*|(?:_text$|task_category)", column, re.I
        ):
            ignored[column] = "source-file artifact or unstructured text/category"
        elif column in config["ignored_columns"] or (
            config["ignored_prefixes"]
            and column.startswith(tuple(config["ignored_prefixes"]))
        ):
            ignored[column] = "excluded by configuration"
        elif column.startswith(tuple(config["video_prefixes"])):
            if column in explicit or column.startswith(
                tuple(config["tabular_prefixes"])
            ):
                raise ValueError(
                    f"Feature belongs to both tabular and video blocks: {column}"
                )
            video.append(column)
        elif column in explicit or column.startswith(tuple(config["tabular_prefixes"])):
            tab.append(column)
        else:
            unknown.append(column)
    if unknown:
        raise ValueError(
            "Unclassified columns: declare structured pre-assessment/session columns or add exclusions in ignored_columns/ignored_prefixes. "
            + repr(unknown)
        )
    if not tab or not video:
        raise ValueError("Both tabular and emotion/speech blocks must have features")
    counts = {
        prefix: sum((c.startswith(prefix) for c in tab))
        for prefix in config["tabular_prefixes"]
    }
    pre_columns = [
        c for c in tab if c.startswith(("pre__", "pre_assessment__", "preassessment__"))
    ]
    warnings = (
        []
        if pre_columns
        else [
            "No recognized pre-assessment prefix found. Declare the pre-assessment input columns before fitting."
        ]
    )
    return {
        "tabular_columns": tab,
        "video_columns": video,
        "excluded_columns": ignored,
        "tabular_counts_by_prefix": counts,
        "warnings": warnings,
        "warning": "Feature names alone do not establish temporal availability. Inputs must contain only information available at the prediction session.",
    }


def load_inputs(config):
    import numpy as np
    import pandas as pd

    base = embeddings
    raw_session = pd.read_csv(config["data_path"], usecols=[config["session_column"]])[
        config["session_column"]
    ]
    numeric_session = pd.to_numeric(raw_session, errors="coerce")
    present = numeric_session.dropna()
    if (~np.isfinite(present) | (present < 0) | (present % 1 != 0)).any():
        raise ValueError("Session values must be finite nonnegative integers")
    bundle = base.load_dataset(config)
    pre_audit = None
    if config["preassessment"].get("enabled", False):
        from .preassessment import build_preassessment

        original_keys = bundle.frame[
            [config["id_column"], config["session_column"]]
        ].copy()
        augmented, pre_audit = build_preassessment(config, bundle.frame)
        if not original_keys.equals(
            augmented[[config["id_column"], config["session_column"]]]
        ):
            raise ValueError(
                "Pre-assessment merge changed row identity/order; embedding alignment would be invalid"
            )
        bundle = replace(bundle, frame=augmented)
    if not len(bundle.frame):
        raise ValueError("Empty analysis dataset")
    for target in config["analysis_targets"]:
        if (
            target == "ever_dropout"
            and bundle.frame.groupby(config["id_column"])[target].nunique().gt(1).any()
        ):
            raise ValueError("ever_dropout must be constant within participant")
    excluded = config.get("excluded_feature_columns", [])
    absent = set(excluded) - set(bundle.frame.columns)
    if absent:
        raise ValueError(
            f"Requested feature exclusions missing from source: {sorted(absent)}"
        )
    if excluded:
        bundle = replace(bundle, frame=bundle.frame.drop(columns=excluded))
    audit = feature_audit(config, bundle.frame)
    audit["user_excluded_columns"] = excluded
    audit["retained_session_key"] = config["session_column"]
    audit["session_window"] = config.get("session_window")
    audit["session_min"] = int(bundle.frame[config["session_column"]].min())
    audit["session_max"] = int(bundle.frame[config["session_column"]].max())
    audit["cache_n_rows"] = bundle.cache_n_rows or len(bundle.frame)
    audit["excluded_outside_window"] = audit["cache_n_rows"] - len(bundle.frame)
    audit["target_definition"] = (
        "Final follow-up ever_dropout, unchanged by session cutoff"
    )
    if pre_audit is not None:
        audit["preassessment_merge"] = pre_audit
    splits = base.load_split_rows(config)
    return (base, bundle, splits, audit)


def prepare_block(
    base, frame, columns, train, max_levels, scale_numeric=True, dtype="float32"
):
    """Infer types, cardinality, imputation and categories from training data only."""
    import numpy as np
    import pandas as pd

    train_raw, excluded = base.prepare_source_frame(
        frame.loc[train], columns, max_levels
    )
    converted_columns = {}
    for column in train_raw:
        if pd.api.types.is_numeric_dtype(train_raw[column]):
            converted_columns[column] = pd.to_numeric(
                frame[column], errors="coerce"
            ).replace([np.inf, -np.inf], np.nan)
        else:
            values = (
                frame[column]
                .astype("string")
                .str.strip()
                .replace("", pd.NA)
                .astype(object)
            )
            values[pd.isna(values)] = np.nan
            converted_columns[column] = values
    converted = pd.DataFrame(converted_columns, index=frame.index)
    preprocessor, usable = base.build_preprocessor(converted.loc[train])
    if not scale_numeric:
        numeric = next(
            (pipe for name, pipe, _ in preprocessor.transformers if name == "numeric"),
            None,
        )
        if numeric is not None:
            numeric.set_params(scale="passthrough")
    preprocessor.fit(converted.loc[train])
    matrix = preprocessor.transform(converted).astype(dtype)
    if not np.isfinite(matrix).all():
        raise ValueError("Non-finite preprocessed tabular block")
    return (
        matrix,
        preprocessor,
        {
            "raw_requested": columns,
            "usable_train_columns": usable,
            "excluded_high_cardinality_train": excluded,
            "encoded_columns": preprocessor.get_feature_names_out().tolist(),
        },
    )


def fit_nn(config, matrix, y, ids, masks, seed):
    import numpy as np
    from sklearn.preprocessing import StandardScaler

    base = embeddings
    torch = base.torch
    if torch is None:
        raise RuntimeError("PyTorch is required for model training")
    nnc = config["nn"]
    base.set_seed(seed)
    device = base.resolve_device(config)
    scaler = StandardScaler().fit(matrix[masks["train"]])
    xtrain = torch.from_numpy(
        scaler.transform(matrix[masks["train"]]).astype(np.float32)
    )
    yt = torch.from_numpy(y[masks["train"]].astype(np.float32))
    weights = base.training_weights(config, ids[masks["train"]], y[masks["train"]])
    wt = torch.from_numpy(weights.astype(np.float32))
    xval = torch.from_numpy(
        scaler.transform(matrix[masks["validation"]]).astype(np.float32)
    ).to(device)
    yv = torch.from_numpy(y[masks["validation"]].astype(np.float32)).to(device)
    wv = torch.from_numpy(
        base.participant_weights(
            ids[masks["validation"]], config["participant_equal_weights"]
        ).astype(np.float32)
    ).to(device)
    model = base.EmbeddingNN(
        matrix.shape[1], int(nnc["hidden_units"]), float(nnc["dropout"])
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(nnc["learning_rate"]),
        weight_decay=float(nnc["l2_strength"]),
    )
    best, best_state, stale, history = (float("inf"), None, 0, [])
    for epoch in range(1, int(nnc["epochs"]) + 1):
        model.train()
        order = torch.randperm(
            len(yt), generator=torch.Generator().manual_seed(seed * 1000003 + epoch)
        )
        losses, sums = (0.0, 0.0)
        for start in range(0, len(yt), int(nnc["batch_size"])):
            ix = order[start : start + int(nnc["batch_size"])]
            xb, yb, wb = (xtrain[ix].to(device), yt[ix].to(device), wt[ix].to(device))
            optimizer.zero_grad(set_to_none=True)
            each = torch.nn.functional.binary_cross_entropy_with_logits(
                model(xb), yb, reduction="none"
            )
            loss = (each * wb).sum() / wb.sum()
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward()
            optimizer.step()
            losses += float((each.detach() * wb).sum().cpu())
            sums += float(wb.sum().cpu())
        model.eval()
        with torch.no_grad():
            each = torch.nn.functional.binary_cross_entropy_with_logits(
                model(xval), yv, reduction="none"
            )
            val = float(((each * wv).sum() / wv.sum()).cpu())
        if not np.isfinite(val):
            raise ValueError("Non-finite validation loss")
        history.append(
            {"epoch": epoch, "train_loss": losses / sums, "validation_bce": val}
        )
        if val < best - float(nnc["min_delta"]):
            best, stale, best_epoch = (val, 0, epoch)
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
        else:
            stale += 1
        if stale >= int(nnc["patience"]):
            break
    if best_state is None:
        raise ValueError("No trained epoch; nn.epochs must be positive")
    model.load_state_dict(best_state)
    probability = {
        split: base.predict_torch_model(
            model, scaler, matrix[mask], device, int(nnc["prediction_batch_size"])
        )
        for split, mask in masks.items()
        if split in ("validation", "test")
    }
    if any((not np.isfinite(p).all() for p in probability.values())):
        raise ValueError("Non-finite probabilities")
    return (probability, history, best_epoch, best_state, scaler)
