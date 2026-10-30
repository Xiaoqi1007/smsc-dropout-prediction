"""Participant-level splits stratified by outcome and attendance duration."""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import sklearn
from ..utils import sha256_file, normalize_id
from sklearn.model_selection import train_test_split


def load_inputs(root, config):
    """Read only the columns used to define the participant cohort and strata."""
    id_col, session = config["id_column"], config["session_column"]
    targets = config["targets"]
    frame = pd.read_csv(
        root / config["wide_path"],
        dtype={id_col: str},
        usecols=[id_col, session, *targets],
    )
    frame[id_col] = frame[id_col].map(normalize_id)
    for column in [session, *targets]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.loc[
        frame[id_col].ne("")
        & frame[session].notna()
        & frame[targets].notna().all(axis=1)
    ].copy()
    frame[session] = frame[session].astype(int)
    if frame.empty or frame.duplicated([id_col, session]).any():
        raise ValueError("Expected a nonempty table with one row per id and session")
    if not frame[targets].isin([0, 1]).all().all():
        raise ValueError("Targets must contain only 0/1")

    label_id, last_col = config["last_session_id_column"], config["last_session_column"]
    labels = pd.read_csv(
        root / config["last_session_path"],
        dtype={label_id: str},
        usecols=[label_id, last_col],
    )
    labels[label_id] = labels[label_id].map(normalize_id)
    labels[last_col] = pd.to_numeric(labels[last_col], errors="coerce")
    labels = labels.drop_duplicates()
    if labels[label_id].duplicated().any():
        raise ValueError("Conflicting last-session records")
    last = labels.set_index(label_id)[last_col].reindex(sorted(frame[id_col].unique()))
    if not np.isfinite(last.to_numpy(dtype=float)).all():
        raise ValueError("Every included participant needs a numeric last session")
    return frame, last


def participant_table(frame, last, target, config):
    summary = frame.groupby(config["id_column"], sort=True)[target].agg(["min", "max"])
    # String labels determine the class order used by stratified sampling.
    strata = np.select(
        [
            summary["max"].eq(0),
            summary["min"].eq(0) & summary["max"].eq(1),
            summary["min"].eq(1),
        ],
        config["target_stratum_labels"],
        default="unknown",
    )
    groups = pd.cut(
        last.reindex(summary.index),
        bins=[-np.inf, *config["last_session_cutpoints"], np.inf],
        labels=config["last_session_bin_labels"],
        include_lowest=True,
        right=True,
    )
    return pd.DataFrame(
        {
            "target_stratum": strata,
            "last_session_group": groups.astype("string").fillna("missing").astype(str),
        },
        index=summary.index,
    )


def split_ids(ids, table, size, seed):
    ids = np.asarray(sorted(set(ids)), dtype=object)
    table = table.reindex(ids)
    strata = table.target_stratum + "|" + table.last_session_group
    counts = strata.value_counts()
    n_test = max(1, int(np.ceil(len(ids) * size)))
    if (
        len(counts) < 2
        or counts.min() < 2
        or min(n_test, len(ids) - n_test) < len(counts)
    ):
        raise ValueError(
            "Joint stratification is infeasible; no fallback split was generated"
        )
    return train_test_split(
        ids, test_size=size, random_state=seed, shuffle=True, stratify=strata.to_numpy()
    )


def generate_assignments(frame, last, config):
    """Generate participant-level training, validation and test assignments."""
    if config["strategy"] != "target_and_last_session":
        raise ValueError(
            "This workflow requires joint outcome and last-session stratification"
        )
    if len(config["seeds"]) != len(set(config["seeds"])):
        raise ValueError("Seeds must be unique")
    if len(config["targets"]) != len(set(config["targets"])):
        raise ValueError("Targets must be unique")
    rows, audit = [], []
    for target in config["targets"]:
        table = participant_table(frame, last, target, config)
        for seed in config["seeds"]:
            development, test = split_ids(table.index, table, config["test_size"], seed)
            validation_seed = seed + config["validation_seed_offset"]
            train, validation = split_ids(
                development,
                table,
                config["validation_size_within_development"],
                validation_seed,
            )
            for split, ids in [
                ("train", train),
                ("validation", validation),
                ("test", test),
            ]:
                rows.extend(
                    {"target": target, "seed": seed, "id": who, "split": split}
                    for who in ids
                )
            for stage, stage_seed, left, right in [
                ("development_vs_test", seed, development, test),
                ("train_vs_validation", validation_seed, train, validation),
            ]:
                audit.append(
                    dict(
                        target=target,
                        seed=seed,
                        random_state=stage_seed,
                        stage=stage,
                        strategy=config["strategy"],
                        n_left=len(left),
                        n_right=len(right),
                    )
                )
    assignments = (
        pd.DataFrame(rows).sort_values(["target", "seed", "id"]).reset_index(drop=True)
    )
    return assignments, pd.DataFrame(audit)


def compare_assignments(generated, reference):
    """Compare complete participant assignments, independent of row order."""
    keys = ["target", "seed", "id"]
    tables = []
    for frame in [generated, reference]:
        frame = frame[[*keys, "split"]].copy()
        frame["id"] = frame["id"].map(normalize_id)
        frame["split"] = (
            frame["split"]
            .astype(str)
            .str.strip()
            .str.lower()
            .replace({"val": "validation", "valid": "validation"})
        )
        if frame[keys].isna().any().any() or frame.id.eq("").any():
            raise ValueError("Missing assignment key")
        if frame.duplicated(keys).any():
            raise ValueError("Duplicate participant assignment")
        if not frame.split.isin(["train", "validation", "test"]).all():
            raise ValueError("Unknown split label")
        tables.append(frame)
    joined = tables[0].merge(
        tables[1],
        on=keys,
        how="outer",
        suffixes=("_generated", "_reference"),
        indicator=True,
        validate="one_to_one",
    )
    matched = joined._merge.eq("both") & joined.split_generated.eq(
        joined.split_reference
    )
    return {
        "generated_rows": len(tables[0]),
        "reference_rows": len(tables[1]),
        "matched_rows": int(matched.sum()),
        "mismatched_or_missing_rows": int((~matched).sum()),
        "exact_match": bool(len(joined) and matched.all()),
    }


def export_splits(root, config):
    root = Path(root)
    frame, last = load_inputs(root, config)
    assignments, audit = generate_assignments(frame, last, config)
    training = assignments.loc[assignments.target.eq(config["training_target"])].copy()
    if training.empty:
        raise ValueError("Training target is not among the generated targets")
    checks = {}
    # Reference tables are used only after generation, never as split inputs.
    if config.get("reference_path"):
        reference = pd.read_csv(root / config["reference_path"], dtype={"id": str})
        checks["provided_assignments"] = compare_assignments(assignments, reference)
    if config.get("training_reference_path"):
        reference = pd.read_csv(
            root / config["training_reference_path"],
            usecols=["doc_id", "seed", "split"],
            dtype={"doc_id": str},
        )
        reference = reference.rename(columns={"doc_id": "id"}).drop_duplicates()
        reference["target"] = config["training_target"]
        checks["current_training_assignments"] = compare_assignments(
            training, reference
        )
    output = root / config["output_dir"]
    if root.resolve() / "results" not in output.resolve().parents:
        raise ValueError("Split outputs must be placed below results/")
    output.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(output / "participant_split_assignments.csv", index=False)
    training.rename(columns={"id": "doc_id"})[["doc_id", "seed", "split"]].to_csv(
        output / "training_splits.csv", index=False
    )
    audit.to_csv(output / "split_strategy_audit.csv", index=False)
    hashes = {
        key: sha256_file(root / config[key])
        for key in [
            "wide_path",
            "last_session_path",
            "reference_path",
            "training_reference_path",
        ]
        if config.get(key)
    }
    report = {
        "versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "participants": int(frame[config["id_column"]].nunique()),
        "assignment_rows": len(assignments),
        "stage_count": len(audit),
        "checks": checks,
        "input_sha256": hashes,
    }
    (output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "split_config.json").write_text(json.dumps(config, indent=2) + "\n")
    if any(not check["exact_match"] for check in checks.values()):
        raise ValueError(
            "Generated splits differ from reference assignments; see verification.json"
        )
    return output, report
