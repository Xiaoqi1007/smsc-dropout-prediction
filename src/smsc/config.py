"""Experiment inputs and adjustable training settings."""

from pathlib import Path
import copy
from .experiment import read_json, check_name

TEXT_MODE = "full_text"
TEXT_REPRESENTATION = "cumulative_history"


def load_config(path):
    config = read_json(path)
    features = config.get("features", {})
    excluded = features.get("exclude_columns", [])
    if (
        set(features) - {"exclude_columns"}
        or not isinstance(excluded, list)
        or any(not isinstance(c, str) for c in excluded)
        or len(set(excluded)) != len(excluded)
    ):
        raise ValueError("Feature exclusions must contain distinct column names")
    if {"id", "session", "ever_dropout", "next_session_dropout"} & set(excluded):
        raise ValueError("Cannot exclude identity, session key, or target columns")
    check_name(config["name"])
    seeds = config["seeds"]
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or any(type(s) is not int or s < 0 for s in seeds)
    ):
        raise ValueError("Seeds must be distinct nonnegative integers")
    lo, hi = config["session_window"]
    if any(type(s) is not int for s in [lo, hi]) or not 0 <= lo <= hi:
        raise ValueError("Invalid session window")
    return config


def training_config(specification, project):
    from ._core import embeddings
    from ._core.preassessment import settings

    project = Path(project).resolve()
    config = copy.deepcopy(embeddings.DEFAULT_CONFIG)
    embeddings.deep_update(config, specification.get("training", {}))
    for key, value in specification["inputs"].items():
        if key == "preassessment_path":
            config["preassessment"]["cm_path"] = str(project / value)
        else:
            config[key] = str(project / value)
    config["session_window"] = dict(
        zip(["min", "max"], specification["session_window"])
    )
    config["seeds"] = specification["seeds"]
    config["excluded_feature_columns"] = specification.get("features", {}).get(
        "exclude_columns", []
    )
    config["preassessment"] = settings(config)
    return config
