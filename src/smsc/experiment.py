"""Versioned experiment storage shared by training and every analysis."""

from pathlib import Path
import json
import re
import numpy as np
import pandas as pd

from .utils import sha256_file as sha, digest_json as digest

SCHEMA_VERSION = 1
KEYS = ["seed", "split", "id", "session", "model"]
COLUMNS = KEYS + ["y_true", "probability_raw", "probability"]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def read_csv(path):
    return pd.read_csv(
        path, dtype={"id": str}, float_precision="round_trip", low_memory=False
    )


def validate_predictions(frame, seeds=None, models=None):
    missing = set(COLUMNS) - set(frame)
    if missing:
        raise ValueError(f"Prediction columns missing: {sorted(missing)}")
    if frame.empty or frame.duplicated(KEYS).any():
        raise ValueError("Empty or duplicate prediction records")
    if not frame.split.isin(["train", "validation", "test"]).all():
        raise ValueError("Unknown split")
    if frame.groupby(["seed", "id"]).split.nunique().gt(1).any():
        raise ValueError("A participant crosses splits within a seed")
    if (
        not frame.y_true.isin([0, 1]).all()
        or frame.groupby("id").y_true.nunique().gt(1).any()
    ):
        raise ValueError("Invalid or inconsistent outcomes")
    if frame.id.isna().any() or frame.id.str.strip().eq("").any():
        raise ValueError("Missing participant ID")
    for column in ["seed", "session"]:
        if (
            not np.isfinite(frame[column]).all()
            or not frame[column].ge(0).all()
            or not frame[column].mod(1).eq(0).all()
        ):
            raise ValueError(f"Invalid integer column: {column}")
    for column in ["probability_raw", "probability"]:
        if (
            not np.isfinite(frame[column]).all()
            or not frame[column].between(0, 1).all()
        ):
            raise ValueError(f"Invalid {column}")
    if seeds is not None and set(frame.seed) != set(seeds):
        raise ValueError(f"Expected seeds {seeds}; found {sorted(frame.seed.unique())}")
    for seed, group in frame.groupby("seed"):
        expected_models = set(models or group.model.unique())
        if set(group.model) != expected_models:
            raise ValueError(f"Seed {seed}: model set differs from manifest")
        reference = None
        for model, part in group.groupby("model"):
            if set(part.split) not in (
                {"train", "validation", "test"},
                {"train", "test"},
            ):
                raise ValueError(f"Seed {seed}, {model}: incomplete splits")
            keys = part.set_index(["split", "id", "session"]).y_true.sort_index()
            if reference is not None:
                pd.testing.assert_series_equal(keys, reference)
            reference = keys
    return frame


class Experiment:
    def __init__(self, run):
        self.path = Path(run).expanduser().resolve()
        manifest = self.path / "experiment.json"
        if not manifest.is_file():
            raise FileNotFoundError(
                f"{self.path} is not a unified experiment: experiment.json is missing. "
                "Expected an experiment directory."
            )
        self.manifest = read_json(manifest)
        if self.manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Unsupported experiment schema version")
        self.config = self.manifest["config"]
        self.seeds = self.config["seeds"]

    def seed_dir(self, seed):
        if seed not in self.seeds:
            raise ValueError(f"Seed {seed} is not declared in this experiment")
        return self.path / "seeds" / f"seed_{seed}"

    def verify_seed(self, seed):
        folder = self.seed_dir(seed)
        marker = folder / "DONE.json"
        if not marker.is_file():
            raise FileNotFoundError(f"Seed {seed} is incomplete: {marker}")
        done = read_json(marker)
        if done["experiment_digest"] != digest(self.manifest):
            raise ValueError(f"Seed {seed}: experiment definition changed")
        for name, expected in done["files"].items():
            path = folder / name
            if not path.is_file() or sha(path) != expected:
                raise ValueError(f"Seed {seed}: missing or modified artifact {name}")
        return folder

    def predictions(self, model=None):
        frames = [
            read_csv(self.verify_seed(seed) / "predictions.csv") for seed in self.seeds
        ]
        frame = validate_predictions(
            pd.concat(frames, ignore_index=True), self.seeds, self.manifest["models"]
        )
        expected_splits = {"train", "validation", "test"}
        for _, part in frame.groupby(["seed", "model"]):
            if set(part.split) != expected_splits:
                raise ValueError(
                    "Prediction splits must include training, validation and test records"
                )
        lo, hi = self.config["session_window"]
        if not frame.session.between(lo, hi).all():
            raise ValueError(
                "Prediction sessions are outside the declared training window"
            )
        if model is not None:
            if model not in self.manifest["models"]:
                raise ValueError(
                    f'Unknown model {model}; available: {self.manifest["models"]}'
                )
            frame = frame[frame.model.eq(model)].copy()
        return frame


def check_name(name):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise ValueError(
            "Experiment name must contain only letters, numbers, underscores, dots or hyphens"
        )
    return name
