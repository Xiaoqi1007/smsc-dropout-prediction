"""Embedding alignment, participant splits, and training weights."""

from __future__ import annotations
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - exercised on non-training nodes
    torch = None
    nn = None

from ..utils import digest_values as stable_digest, normalize_id

DEFAULT_CONFIG = json.loads((Path(__file__).parent / "defaults.json").read_text())

LEAKAGE_PATTERN = re.compile(
    r"(^|__)(?:ever_dropout|next_session_dropout|target|true_label|label|"
    r"last_session|observed_last_session)(?:$|__)",
    flags=re.IGNORECASE,
)


def deep_update(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


@dataclass(frozen=True)
class DatasetBundle:
    frame: pd.DataFrame
    row_key_digest: str
    cache_row_key_digest: str | None = None
    cache_n_rows: int | None = None
    cache_positions: np.ndarray | None = None


def load_dataset(config: Mapping[str, Any]) -> DatasetBundle:
    path = Path(config["data_path"])
    if not path.exists():
        raise FileNotFoundError(f"Wide table not found: {path}")
    id_column = config["id_column"]
    session_column = config["session_column"]
    # Align embeddings before selecting the rows used for model fitting.
    targets = list(config.get("cache_targets", config["targets"]))
    frame = pd.read_csv(path, dtype={id_column: str}, low_memory=False)
    missing = {id_column, session_column, *targets} - set(frame.columns)
    if missing:
        raise ValueError(f"Wide table is missing required columns: {sorted(missing)}")
    frame[id_column] = frame[id_column].map(normalize_id)
    frame[session_column] = pd.to_numeric(frame[session_column], errors="coerce")
    present = frame[session_column].dropna()
    if ((~np.isfinite(present)) | (present < 0) | (present % 1 != 0)).any():
        raise ValueError("Session values must be finite nonnegative integers")
    for target in targets:
        frame[target] = pd.to_numeric(frame[target], errors="coerce")
    keep = frame[id_column].ne("") & frame[session_column].notna()
    keep &= frame[targets].notna().all(axis=1)
    frame = frame.loc[keep].reset_index(drop=True)
    frame[session_column] = frame[session_column].astype(int)
    if frame.duplicated([id_column, session_column]).any():
        raise ValueError("Expected exactly one row per id + session.")
    for target in targets:
        if not frame[target].isin([0, 1]).all():
            raise ValueError(f"{target} must contain only 0/1 after filtering.")
        frame[target] = frame[target].astype(int)
    row_keys = frame[id_column].astype(str) + "::" + frame[session_column].astype(str)
    cache_digest, cache_n = stable_digest(row_keys), len(frame)
    from .paths import session_bounds

    bounds = session_bounds(config)
    if bounds is None:
        return DatasetBundle(frame=frame, row_key_digest=cache_digest)
    positions = np.flatnonzero(frame[session_column].between(*bounds).to_numpy())
    if not len(positions):
        raise ValueError("No rows in the configured session window")
    selected = frame.iloc[positions].reset_index(drop=True)
    if selected.groupby(id_column).ever_dropout.nunique().gt(1).any():
        raise ValueError("ever_dropout must be constant within participant")
    return DatasetBundle(
        selected,
        stable_digest(row_keys.iloc[positions]),
        cache_digest,
        cache_n,
        positions,
    )


def load_split_rows(config: Mapping[str, Any]) -> pd.DataFrame:
    path = Path(config["split_path"])
    if not path.exists():
        raise FileNotFoundError(f"Authoritative split file not found: {path}")
    rows = pd.read_csv(path, dtype={"doc_id": str}, low_memory=False)
    required = {"doc_id", "seed", "split"}
    if not required.issubset(rows.columns):
        raise ValueError(f"Split file must contain {sorted(required)}")
    rows["_id"] = rows["doc_id"].map(normalize_id)
    rows["seed"] = pd.to_numeric(rows["seed"], errors="raise").astype(int)
    rows["_split"] = (
        rows["split"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({"val": "validation", "valid": "validation"})
    )
    unknown = set(rows["_split"].unique()) - {"train", "validation", "test"}
    if unknown:
        raise ValueError(f"Unknown split labels: {sorted(unknown)}")
    return rows


def participant_masks(
    bundle: DatasetBundle,
    split_rows: pd.DataFrame,
    config: Mapping[str, Any],
    seed: int,
) -> dict[str, np.ndarray]:
    id_column = config["id_column"]
    assignment = split_rows.loc[split_rows["seed"].eq(seed), ["_id", "_split"]]
    assignment = assignment.drop_duplicates()
    conflicts = assignment.groupby("_id")["_split"].nunique()
    if conflicts.gt(1).any():
        examples = conflicts[conflicts.gt(1)].index.tolist()[:10]
        raise ValueError(
            f"Seed {seed} assigns participants to multiple splits: {examples}"
        )
    assignment = assignment.drop_duplicates("_id")
    frame_ids = set(bundle.frame[id_column].unique())
    missing = frame_ids - set(assignment["_id"])
    if missing:
        raise ValueError(
            f"Seed {seed} split is missing {len(missing)} participants; "
            f"examples: {sorted(missing)[:10]}"
        )
    ids = {
        name: set(assignment.loc[assignment["_split"].eq(name), "_id"]) & frame_ids
        for name in ("train", "validation", "test")
    }
    ids["development"] = ids["train"] | ids["validation"]
    if ids["train"] & ids["validation"] or ids["development"] & ids["test"]:
        raise AssertionError("Participant leakage detected in split assignment.")
    masks = {
        name: bundle.frame[id_column].isin(values).to_numpy()
        for name, values in ids.items()
    }
    for name in ("train", "validation", "test"):
        if not masks[name].any():
            raise ValueError(f"Seed {seed} has an empty {name} split.")
    return masks


def cache_key(config: Mapping[str, Any], mode: str, seed: int) -> str:
    return "shared" if config["modes"][mode]["shared_embedding"] else f"seed_{seed}"


def embedding_paths(config: Mapping[str, Any], mode: str, seed: int) -> dict[str, Path]:
    directory = (
        Path(config["embedding_root"])
        / config["modes"][mode]["directory"]
        / "cache"
        / "embeddings"
        / cache_key(config, mode, seed)
    )
    return {
        "directory": directory,
        "embedding": directory / "embeddings.npy",
        "done": directory / "embeddings_done.npy",
        "metadata": directory / "embeddings_metadata.json",
    }


def load_current_embedding(
    config: Mapping[str, Any], bundle: DatasetBundle, mode: str, seed: int
) -> np.ndarray:
    paths = embedding_paths(config, mode, seed)
    missing = [
        str(paths[key])
        for key in ("embedding", "done", "metadata")
        if not paths[key].exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Completed embedding cache is missing; the training command does not "
            "re-embeds text. Missing: " + ", ".join(missing)
        )
    metadata = json.loads(paths["metadata"].read_text())
    expected_digest = bundle.cache_row_key_digest or bundle.row_key_digest
    expected_n = (
        bundle.cache_n_rows if bundle.cache_n_rows is not None else len(bundle.frame)
    )
    native_window = metadata.get("row_scope") == "session_window"
    if native_window:
        if (
            metadata.get("session_window") != config.get("session_window")
            or config.get("session_window") is None
        ):
            raise ValueError("Native embedding cache session window mismatch")
        expected_digest, expected_n = bundle.row_key_digest, len(bundle.frame)
    if metadata.get("row_key_digest") != expected_digest:
        raise RuntimeError(
            f"Embedding row order mismatch for mode={mode}, seed={seed}."
        )
    matrix = np.load(paths["embedding"], mmap_mode="r")
    done = np.load(paths["done"], mmap_mode="r").astype(bool)
    if matrix.ndim != 2 or matrix.shape[0] != expected_n:
        raise RuntimeError(f"Embedding cache has the wrong shape: {paths['embedding']}")
    if done.shape != (expected_n,) or not done.all():
        raise RuntimeError(f"Embedding cache is incomplete: {paths['directory']}")
    metadata_dim = metadata.get("embedding_dim")
    if metadata_dim is not None and int(metadata_dim) != matrix.shape[1]:
        raise RuntimeError(
            f"Embedding dimension mismatch: metadata={metadata_dim}, matrix={matrix.shape[1]}"
        )
    if bundle.cache_positions is not None:
        return (
            matrix
            if native_window
            else np.asarray(matrix[bundle.cache_positions], dtype=np.float32)
        )
    return matrix


def cumulative_history(
    config: Mapping[str, Any], bundle: DatasetBundle, matrix: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    id_column = config["id_column"]
    session_column = config["session_column"]
    ordering = bundle.frame[[id_column, session_column]].copy()
    ordering["_position"] = np.arange(len(ordering), dtype=int)
    ordering = ordering.sort_values(
        [id_column, session_column, "_position"], kind="mergesort"
    )
    output = np.empty(matrix.shape, dtype=np.float32)
    counts = np.zeros(len(ordering), dtype=int)
    for _, participant_rows in ordering.groupby(id_column, sort=False):
        positions = participant_rows["_position"].to_numpy(dtype=int)
        values = np.asarray(matrix[positions], dtype=np.float64)
        cumulative_sum = np.cumsum(values, axis=0)
        denominators = np.arange(1, len(positions) + 1, dtype=np.float64)
        output[positions] = (cumulative_sum / denominators[:, None]).astype(np.float32)
        counts[positions] = denominators.astype(int)
    return output, counts


def participant_weights(ids: Sequence[str], enabled: bool) -> np.ndarray:
    ids = pd.Series(ids, dtype=str)
    if enabled:
        counts = ids.map(ids.value_counts()).to_numpy(dtype=float)
        weights = 1.0 / counts
    else:
        weights = np.ones(len(ids), dtype=float)
    return weights * (len(weights) / weights.sum())


def class_weights(y: np.ndarray, enabled: bool) -> np.ndarray:
    if not enabled:
        return np.ones(len(y), dtype=float)
    counts = np.bincount(y.astype(int), minlength=2).astype(float)
    if np.any(counts == 0):
        return np.ones(len(y), dtype=float)
    per_class = len(y) / (2.0 * counts)
    return per_class[y.astype(int)]


def training_weights(
    config: Mapping[str, Any], ids: Sequence[str], y: np.ndarray
) -> np.ndarray:
    weights = participant_weights(ids, bool(config["participant_equal_weights"]))
    weights *= class_weights(y, bool(config["balanced_classes"]))
    return weights * (len(weights) / weights.sum())


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def resolve_device(config: Mapping[str, Any]) -> str:
    requested = str(config["nn"].get("device", "auto"))
    if requested == "auto":
        return "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
    return requested


if nn is not None:
    from .. import models

    EmbeddingNN = models.EmbeddingNN


def predict_torch_model(
    model: Any,
    scaler: StandardScaler,
    x: np.ndarray,
    device: str,
    batch_size: int,
) -> np.ndarray:
    assert torch is not None
    model.eval()
    scaled = scaler.transform(np.asarray(x, dtype=np.float32)).astype(np.float32)
    parts: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(scaled), batch_size):
            xb = torch.from_numpy(scaled[start : start + batch_size]).to(device)
            parts.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(parts) if parts else np.empty(0, dtype=float)


def prepare_source_frame(
    raw: pd.DataFrame, columns: Sequence[str], max_levels: int
) -> tuple[pd.DataFrame, list[str]]:
    output = pd.DataFrame(index=raw.index)
    high_cardinality: list[str] = []
    for column in columns:
        series = raw[column]
        if pd.api.types.is_bool_dtype(series):
            output[column] = series.astype(float)
        elif pd.api.types.is_numeric_dtype(series):
            output[column] = pd.to_numeric(series, errors="coerce")
        else:
            numeric = pd.to_numeric(series, errors="coerce")
            nonmissing = series.dropna()
            convertible = (
                numeric.notna().sum() / len(nonmissing) if len(nonmissing) else 1.0
            )
            if convertible >= 0.95:
                output[column] = numeric
            else:
                cleaned = series.astype("string").str.strip().replace("", pd.NA)
                if cleaned.nunique(dropna=True) > max_levels:
                    high_cardinality.append(column)
                else:
                    object_values = cleaned.astype(object)
                    object_values[pd.isna(object_values)] = np.nan
                    output[column] = object_values
    if output.shape[1] == 0:
        raise ValueError("No usable features remain for a source block.")
    return output, high_cardinality


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # scikit-learn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(x_fit: pd.DataFrame) -> tuple[ColumnTransformer, list[str]]:
    usable = [column for column in x_fit.columns if x_fit[column].notna().any()]
    if not usable:
        raise ValueError("Every source feature is missing in the fit split.")
    numeric = [c for c in usable if pd.api.types.is_numeric_dtype(x_fit[c])]
    categorical = [c for c in usable if c not in numeric]
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", make_one_hot_encoder()),
                    ]
                ),
                categorical,
            )
        )
    return (
        ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.0),
        usable,
    )
