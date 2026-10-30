"""Limit saved analysis inputs to predictors and content hashes."""

from pathlib import Path

from .experiment import write_json
from .utils import digest_values as _digest


STRUCTURAL_FIELDS = {
    "session_info__derived_session_current",
    "speech__participant_total_words",
    "speech__total_role_labeled_words",
    "emotion__n_snippets",
    "emotion__total_video_length_mins",
    "emotion__Neutral_prop",
    "emotion__derived_prior_before_session_Neutral_prop",
}


def feature_snapshot(frame, audit):
    """Return structured predictors, row keys, outcomes, and structural controls."""
    allowed = {
        "id",
        "session",
        "ever_dropout",
        "next_session_dropout",
        *audit["tabular_columns"],
        *audit["video_columns"],
        *STRUCTURAL_FIELDS,
    }
    columns = [column for column in frame if column in allowed]
    if any(column.endswith("_text") for column in columns):
        raise ValueError("Unstructured text cannot be included in a feature snapshot")
    return frame.loc[:, columns].copy()


def save_text_hashes(text, destination):
    """Save ordered content and row-key digests without saving source text."""
    hashes = {
        "n_rows": len(text),
        "text_sha256": _digest(text["text"]),
        "row_key_sha256": _digest(text["id"] + "::" + text["session"].astype(str)),
    }
    write_json(Path(destination), hashes)
    return hashes
