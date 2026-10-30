"""Validation-tuned component models with fixed average fusion."""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from . import calibration
from .config import TEXT_MODE, TEXT_REPRESENTATION, load_config, training_config
from .experiment import (
    Experiment,
    SCHEMA_VERSION,
    sha,
    digest,
    write_json,
    read_json,
    validate_predictions,
)
from .models import predict_branch, predict_text
from .privacy import feature_snapshot, save_text_hashes
from ._core import pipeline, components, fusion

BRANCHES = ["preassessment_xgb", "session_xgb", "text_only", "emotion_speech_lr"]
FUSIONS = {"full_model": BRANCHES}


def text_cache(config, base, frame, mode, seed):
    cache = (
        Path(config["embedding_root"]) / config["modes"][mode]["directory"] / "cache"
    )
    key = "shared"
    path = cache / "text" / f"{key}.csv"
    text = pd.read_csv(path, dtype={"id": str}, float_precision="round_trip").fillna(
        {"text": ""}
    )
    metadata = read_json(base.embedding_paths(config, mode, seed)["metadata"])
    if base.stable_digest(text.text) != metadata["text_digest"]:
        raise ValueError(f"Text and embedding content differ: {path}")
    if (
        base.stable_digest(text.id + "::" + text.session.astype(str))
        != metadata["row_key_digest"]
    ):
        raise ValueError(f"Text and embedding identities differ: {path}")
    if text.duplicated(["id", "session"]).any():
        raise ValueError("Duplicate full-text rows")
    aligned = frame[["id", "session"]].merge(
        text, on=["id", "session"], validate="one_to_one", how="left"
    )
    if aligned.text.isna().any():
        raise ValueError("Some modeling sessions have no text cache row")
    return aligned, path


def initialize(specification, project, threads=2):
    project = Path(project).resolve()
    config = training_config(specification, project)
    base, bundle, split_rows, audit = pipeline.load_inputs(config)
    base.torch.set_num_threads(threads)
    mode = TEXT_MODE
    paths = {
        Path(config["data_path"]),
        Path(config["split_path"]),
        Path(config["preassessment"]["cm_path"]),
    }
    for seed in specification["seeds"]:
        base.load_current_embedding(config, bundle, mode, seed)
        _, text_path = text_cache(config, base, bundle.frame, mode, seed)
        paths.add(text_path)
        paths.update(
            base.embedding_paths(config, mode, seed)[key]
            for key in ["embedding", "done", "metadata"]
        )

    def portable(path):
        try:
            return str(path.resolve().relative_to(project))
        except ValueError:
            return str(path.resolve())

    manifest = dict(
        schema_version=SCHEMA_VERSION,
        config=specification,
        models=BRANCHES + list(FUSIONS),
        inputs={portable(p): sha(p) for p in sorted(paths)},
        code={
            str(p.relative_to(Path(__file__).parent)): sha(p)
            for p in sorted(Path(__file__).parent.rglob("*"))
            if p.suffix in (".py", ".json")
        },
        protocol=dict(
            splits="Participant-specific training, validation and test splits",
            tuning="XGBoost: validation participant-level AUROC; NN: validation BCE early stopping",
            fusion="Fixed equal mean of raw component probabilities",
            final_fit="Training participants only; training and validation remain separate",
            calibration="Validation only, equal session-row weights",
            representation="Fixed by configuration for every seed",
            limitations=[
                "Validation is reused for XGBoost tuning, NN early stopping and calibration.",
            ],
        ),
    )
    run = project / "results" / "experiments" / specification["name"]
    if (run / "experiment.json").exists():
        if read_json(run / "experiment.json") != manifest:
            raise ValueError(
                "Experiment configuration, code or inputs changed. Use a new experiment name."
            )
    else:
        if run.exists() and any(run.iterdir()):
            raise ValueError(
                "Output directory already contains an incompatible experiment"
            )
        write_json(run / "experiment.json", manifest)
    data_dir = run / "data"
    data_dir.mkdir(exist_ok=True)
    frame_path = data_dir / "features.csv"
    if not frame_path.exists():
        feature_snapshot(bundle.frame, audit).to_csv(frame_path, index=False)
        write_json(data_dir / "feature_audit.json", audit)
        write_json(data_dir / "DONE.json", {"features_sha256": sha(frame_path)})
    elif sha(frame_path) != read_json(data_dir / "DONE.json")["features_sha256"]:
        raise ValueError("Saved feature snapshot changed")
    return Experiment(run), config, base, bundle, split_rows, audit


def train_seed(experiment, config, base, bundle, split_rows, audit, seed, threads):
    folder = experiment.seed_dir(seed)
    if (folder / "DONE.json").exists():
        experiment.verify_seed(seed)
        print(f"Seed {seed}: complete", flush=True)
        return
    folder.mkdir(parents=True, exist_ok=True)
    frame = bundle.frame
    masks = {
        k: v
        for k, v in base.participant_masks(bundle, split_rows, config, seed).items()
        if k in ["train", "validation", "test"]
    }
    components.assert_separate(frame, masks, config)
    splits = frame[["id", "session"]].copy()
    for name, mask in masks.items():
        splits.loc[mask, "split"] = name
    splits.to_csv(folder / "splits.csv", index=False)
    mode = TEXT_MODE
    representation = TEXT_REPRESENTATION
    current = base.load_current_embedding(config, bundle, mode, seed)
    matrix = base.cumulative_history(config, bundle, current)[0]
    text, _ = text_cache(config, base, frame, mode, seed)
    save_text_hashes(text, folder / "text_hashes.json")
    embeddings = {representation: matrix}
    y = frame.ever_dropout.to_numpy()
    # Parameter selection sees training and validation only, never test labels.
    from ._core.preassessment import filter_preassessment
    from ._core.xgb import fit_preassessment_xgb

    columns, _ = filter_preassessment(config, frame, masks["train"])
    tuning_masks = dict(
        train=masks["train"], validation=masks["validation"], test=masks["validation"]
    )
    _, _, _, schema, search = fit_preassessment_xgb(
        config, frame, columns, tuning_masks, seed, threads
    )
    validation_tuning = dict(
        selection_split="validation",
        selection_metric="participant-level AUROC",
        xgb_search=search,
    )
    settings = dict(xgb_candidate=int(schema["selected_candidate"]), epochs={})
    pd.DataFrame(search).to_csv(folder / "xgb_validation_search.csv", index=False)
    print(f"Seed {seed}: validation XGBoost selection complete", flush=True)
    models_dir = folder / "models"
    fit_train = masks["train"]
    prediction_masks = {k: masks[k] for k in ["validation", "test"]}
    components.fit_components(
        config,
        base,
        frame,
        audit,
        embeddings,
        fit_train,
        prediction_masks,
        seed,
        threads,
        settings,
        destination=models_dir,
    )
    (models_dir / f"text_{representation}.joblib").replace(
        models_dir / "text_only.joblib"
    )
    raw = {
        branch: predict_branch(joblib.load(models_dir / f"{branch}.joblib"), frame)
        for branch in components.COMPONENTS
    }
    raw["text_only"] = predict_text(
        joblib.load(models_dir / "text_only.joblib"), matrix
    )
    selected = {}
    for name, columns in FUSIONS.items():
        fitted = fusion.average_specification(len(columns))
        raw[name] = fusion.predict_average(
            fitted, np.column_stack([raw[c] for c in columns])
        )
        joblib.dump(fitted, models_dir / f"{name}.joblib")
        selected[name] = dict(
            columns=columns,
            strategy="fixed_simple_average",
            weights=[1 / len(columns)] * len(columns),
        )
    calibrators, predictions = {}, []
    for model in BRANCHES + list(FUSIONS):
        parameters, native = calibration.fit_sklearn_platt(
            raw[model][masks["validation"]], y[masks["validation"]]
        )
        calibrator = dict(method="platt", parameters=parameters)
        joblib.dump(native, models_dir / f"{model}_calibration.joblib")
        probability = native.predict_proba(raw[model][:, None])[:, 1]
        calibrators[model] = calibrator
        predictions.append(
            frame[["id", "session"]].assign(
                seed=seed,
                split=splits.split.to_numpy(),
                model=model,
                y_true=y,
                probability_raw=raw[model],
                probability=probability,
            )
        )
    table = validate_predictions(
        pd.concat(predictions, ignore_index=True), [seed], experiment.manifest["models"]
    )
    table.to_csv(folder / "predictions.csv", index=False)
    write_json(folder / "calibrators.json", calibrators)
    write_json(
        folder / "selection.json",
        dict(
            final=settings,
            validation_tuning=validation_tuning,
            fusion=selected,
            final_fit_participants=int(frame.loc[fit_train, "id"].nunique()),
            final_fit_rows=int(fit_train.sum()),
        ),
    )
    files = [p for p in folder.rglob("*") if p.is_file() and p.name != "DONE.json"]
    write_json(
        folder / "DONE.json",
        dict(
            experiment_digest=digest(experiment.manifest),
            files={str(p.relative_to(folder)): sha(p) for p in files},
        ),
    )
    print(f"Seed {seed}: saved; test metrics not evaluated during training", flush=True)


def train(config_path, project, seed=None, threads=2):
    specification = load_config(config_path)
    if seed is not None and seed not in specification["seeds"]:
        raise ValueError("Requested seed is not declared in the configuration")
    with threadpool_limits(limits=threads):
        experiment, config, base, bundle, splits, audit = initialize(
            specification, project, threads
        )
        for selected_seed in [seed] if seed is not None else experiment.seeds:
            train_seed(
                experiment, config, base, bundle, splits, audit, selected_seed, threads
            )
    return experiment.path
