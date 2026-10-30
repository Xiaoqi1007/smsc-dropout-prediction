"""Join baseline measures and select training-set features."""

from __future__ import annotations
import re
from ..utils import normalize_id as canonical_id
import numpy as np
import pandas as pd

DEFAULTS = {
    "enabled": False,
    "cm_path": "data/CM.csv",
    "label_path": "data/label.csv",
    "id_map_path": None,
    "cm_id_column": "record_id",
    "label_id_column": "record_id",
    "label_target_column": "ever_dropout",
    "event_column": "event",
    "pre_events": [],
    "conflict_policy": "error",
    "prefix": "pre_assessment__",
    "numeric_parse_threshold": 0.90,
    "max_missing_rate": 0.40,
    "drop_exact_duplicates": True,
    "max_dominant_fraction": 0.95,
    "min_binary_minority_count": 5,
    "exclude_columns": [],  # No named clinical baseline exclusions by default.
    "exclude_name_patterns": [
        r"^redcap_",
        r"_timestamp$",
        r"_complete$",
        r"(^|__)(ever_dropout|next_session_dropout|target|label|last_session|observed_last_session)($|__)",
    ],
    "lr_C": 1.0,
    "lr_max_iter": 5000,
}


def settings(config):
    return {**DEFAULTS, **config.get("preassessment", {})}


def equal_value(a, b):
    try:
        return float(a) == float(b)
    except (ValueError, TypeError):
        return str(a).strip() == str(b).strip()


def build_preassessment(config, frame):
    cfg = settings(config)
    if not cfg["enabled"]:
        return frame, {"enabled": False}
    if cfg["conflict_policy"] not in ["error", "most_complete"]:
        raise ValueError("preassessment.conflict_policy must be error or most_complete")
    idc = config["id_column"]
    cm = pd.read_csv(cfg["cm_path"], dtype={cfg["cm_id_column"]: str}, low_memory=False)
    processed = cfg.get("source_format") == "processed_by_id"
    if processed:
        if cfg["cm_id_column"] not in cm:
            raise ValueError(
                "Processed feature table is missing its participant ID column"
            )
        cfg["event_column"] = "_processed_event"
        cfg["pre_events"] = ["processed_baseline"]
        cm[cfg["event_column"]] = "processed_baseline"
    if {cfg["cm_id_column"], cfg["event_column"]} - set(cm):
        raise ValueError("Baseline source is missing its identifier or event column")
    mapping = {}
    if cfg["id_map_path"]:
        m = pd.read_csv(cfg["id_map_path"], dtype=str)
        if {"original_id", "canonical_id"} - set(m):
            raise ValueError("ID map must contain original_id,canonical_id")
        a, b = m.original_id.map(canonical_id), m.canonical_id.map(canonical_id)
        if a.eq("").any() or b.eq("").any() or a.duplicated().any():
            raise ValueError("Invalid duplicate/empty ID map")
        mapping = dict(zip(a, b))
        if any(mapping.get(v, v) != v for v in mapping.values()):
            raise ValueError("ID mappings must be direct, without chains/cycles")

    def match_id(x):
        s = canonical_id(x)
        return mapping.get(s, s)

    keys = frame[idc].map(match_id)
    identities = pd.DataFrame(
        {"original": frame[idc], "matched": keys}
    ).drop_duplicates()
    if identities.groupby("matched").original.nunique().gt(1).any():
        raise ValueError("Multiple modeling identifiers map to one participant")
    pre = cm.loc[cm[cfg["event_column"]].isin(cfg["pre_events"])].copy()
    if pre.empty:
        raise ValueError("No matching pre-assessment events")
    pre["_match_id"] = pre[cfg["cm_id_column"]].map(match_id)
    if processed and pre["_match_id"].duplicated().any():
        raise ValueError(
            "Processed pre-assessment table must have one row per canonical ID"
        )
    missing_cm_ids = int(pre["_match_id"].eq("").sum())
    pre = pre.loc[pre["_match_id"].ne("")].copy()
    # Empty export fields retained until train-only screening. Preserve source order.
    pairs = []
    conflicts = []
    for canonical, source_columns in cfg.get("column_aliases", {}).items():
        present = [column for column in source_columns if column in pre]
        if not present:
            continue
        values = pre[present]
        for index, row in values.iterrows():
            observed = row.dropna().tolist()
            if any(not equal_value(observed[0], value) for value in observed[1:]):
                conflicts.append(
                    {
                        "id": pre.at[index, "_match_id"],
                        "feature": canonical,
                        "kind": "column_alias",
                    }
                )
        pre[canonical] = values.bfill(axis=1).iloc[:, 0]
        pre = pre.drop(columns=[column for column in present if column != canonical])
        pairs.append({"feature": canonical, "sources": present})
    forbidden = {
        cfg["cm_id_column"],
        cfg["event_column"],
        "_match_id",
        config.get("session_column", "session"),
        idc,
        *config.get("cache_targets", []),
        *config.get("analysis_targets", []),
        *cfg["exclude_columns"],
    }
    candidates = [
        c
        for c in pre
        if c not in forbidden
        and not any(re.search(p, c, re.I) for p in cfg["exclude_name_patterns"])
    ]
    rows = []
    for rid, group in pre.groupby("_match_id", sort=False):
        # Prefer the most complete record before resolving ties.
        order = (
            group[candidates]
            .notna()
            .sum(axis=1)
            .sort_values(ascending=False, kind="stable")
            .index
        )
        ordered = group.loc[order, candidates]
        row = {"_match_id": rid}
        # Scan each group once instead of repeating a DataFrame .loc for every
        # questionnaire item. Preserve completeness-first conflict semantics.
        for col, values_array in zip(candidates, ordered.to_numpy().T):
            values = values_array[pd.notna(values_array)]
            if len(values) > 1 and any(
                not equal_value(values[0], x) for x in values[1:]
            ):
                conflicts.append(
                    {"id": rid, "feature": col, "kind": "duplicate_pre_rows"}
                )
            row[cfg["prefix"] + col] = values[0] if len(values) else np.nan
        rows.append(row)
    if conflicts and cfg["conflict_policy"] == "error":
        affected = sorted({x["feature"] for x in conflicts})
        raise ValueError(
            f"Conflicting pre-assessment responses ({len(conflicts)}). Review source data, or explicitly choose documented most_complete policy. Fields: {affected[:20]}"
        )
    one = pd.DataFrame(rows).set_index("_match_id")
    collision = set(frame) & set(one)
    if collision:
        raise ValueError(f"Pre-assessment columns already exist: {sorted(collision)}")
    # Reindex to the immutable modeling row sequence, not an inner join.
    joined = one.reindex(keys.to_numpy()).set_axis(frame.index)
    augmented = pd.concat([frame, joined], axis=1)
    if len(augmented) != len(frame) or not augmented[frame.columns].equals(frame):
        raise AssertionError("Pre-assessment join changed original cache rows")
    label_report = {"used_for_cohort_filter": False, "available": False}
    if cfg["label_path"]:
        labels = pd.read_csv(
            cfg["label_path"], dtype={cfg["label_id_column"]: str}, low_memory=False
        )
        for required in [cfg["label_id_column"], cfg["label_target_column"]]:
            if required not in labels:
                raise ValueError(f"label.csv missing {required}")
        labels["_match_id"] = labels[cfg["label_id_column"]].map(match_id)
        labels["_outcome"] = pd.to_numeric(
            labels[cfg["label_target_column"]], errors="coerce"
        )
        known = labels.loc[labels["_outcome"].isin([0, 1]) & labels["_match_id"].ne("")]
        if known.groupby("_match_id")._outcome.nunique().gt(1).any():
            raise ValueError("Conflicting label.csv outcomes for canonical participant")
        lookup = known.drop_duplicates("_match_id").set_index("_match_id")._outcome
        yc = cfg["label_target_column"]
        if yc not in frame:
            raise ValueError(
                f"Cannot cross-check label outcome {yc} against wide table"
            )
        check = keys.map(lookup)
        if (check.notna() & check.ne(frame[yc])).any():
            raise ValueError(
                "label.csv and modeling wide table disagree on participant outcome"
            )
        label_report = {
            "used_for_cohort_filter": False,
            "available": True,
            "participants_matched": int(keys[check.notna()].nunique()),
            "model_participants_absent_from_resolved_labels": sorted(
                set(keys) - set(lookup.index)
            ),
        }
    audit = {
        "enabled": True,
        "cm_pre_rows": len(pre),
        "cm_rows_missing_id": missing_cm_ids,
        "pre_participants": len(one),
        "model_participants": frame[idc].nunique(),
        "model_participants_matched": int(keys[keys.isin(one.index)].nunique()),
        "unmatched_model_ids": identities.loc[
            ~identities.matched.isin(one.index), "original"
        ].tolist(),
        "pairs": pairs,
        "conflicts": conflicts,
        "conflict_policy": cfg["conflict_policy"],
        "label_crosscheck": label_report,
        "raw_candidate_columns": list(one),
        "id_matching": identities.to_dict("records"),
        "row_order_preserved": True,
        "unmatched_policy": "retain and train-median-impute; never drop cohort rows",
    }
    return augmented, audit


def filter_preassessment(config, frame, train_mask):
    cfg = settings(config)
    prefix = cfg["prefix"]
    columns = [c for c in frame if c.startswith(prefix)]
    # Baseline repeats across sessions; count people, never session rows.
    train = frame.loc[train_mask].drop_duplicates(config["id_column"])
    decisions, kept = [], []
    for col in columns:
        raw = train[col]
        numeric = pd.to_numeric(raw, errors="coerce").replace([np.inf, -np.inf], np.nan)
        nonmissing = raw.notna().sum()
        fraction = numeric.notna().sum() / nonmissing if nonmissing else 0
        vals = numeric.dropna()
        decision = "keep"
        if fraction < cfg["numeric_parse_threshold"]:
            decision = "non_numeric_or_text"
        elif numeric.isna().mean() > cfg["max_missing_rate"]:
            decision = "high_missing"
        elif vals.nunique() <= 1:
            decision = "constant"
        elif vals.value_counts(normalize=True).iloc[0] >= cfg["max_dominant_fraction"]:
            decision = "near_zero_variance"
        elif (
            vals.nunique() == 2
            and vals.value_counts().min() < cfg["min_binary_minority_count"]
        ):
            decision = "rare_binary"
        decisions.append(
            {
                "feature": col,
                "decision": decision,
                "numeric_parse_fraction_train": fraction,
                "missing_rate_train": float(numeric.isna().mean()),
                "n_train_participants": len(train),
            }
        )
        if decision == "keep":
            kept.append(col)
    if kept and cfg.get("drop_exact_duplicates", True):
        numeric = (
            train[kept]
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
        )
        dup = set(numeric.columns[numeric.T.duplicated()])
        kept = [c for c in kept if c not in dup]
        for row in decisions:
            if row["feature"] in dup:
                row["decision"] = "exact_duplicate"
    if not kept:
        raise ValueError("No usable pre-assessment features in the training partition")
    return kept, decisions
