"""Baseline, session, speech, and audio-emotion feature construction."""

import re
import numpy as np
import pandas as pd

EMOTIONS = ("Neutral", "Happy", "Sad", "Angry", "Fearful", "Disgusted", "Surprised")


def word_count(value):
    return len(re.findall(r"\b\w+\b", "" if pd.isna(value) else str(value), re.UNICODE))


def apply_aliases(frame, aliases):
    frame = frame.copy()
    for output, sources in aliases.items():
        columns = [name for name in sources if name in frame]
        if not columns:
            continue
        values = frame[columns]
        if (
            values.apply(lambda row: row.dropna().astype(str).nunique(), axis=1)
            .gt(1)
            .any()
        ):
            raise ValueError("Conflicting values in aliased columns")
        frame[output] = values.bfill(axis=1).iloc[:, 0]
        frame = frame.drop(columns=[name for name in columns if name != output])
    return frame


def baseline_features(frame, specification):
    """Score a local codebook without embedding source field names in the code."""
    frame = apply_aliases(frame, specification.get("aliases", {}))
    identifier = specification.get("id_column", "record_id")
    if specification.get("event_values"):
        frame = frame.loc[
            frame[specification["event_column"]].isin(specification["event_values"])
        ].copy()
    output = pd.DataFrame({"record_id": frame[identifier].astype(str)})
    for item in specification["features"]:
        operation = item["operation"]
        columns = item["columns"]
        if operation == "copy":
            values = frame[columns[0]]
        elif operation == "map":
            values = frame[columns[0]].astype("string").map(item["values"])
        elif operation == "age_years":
            dates = frame[columns].apply(pd.to_datetime, errors="coerce")
            values = (dates.iloc[:, 1] - dates.iloc[:, 0]).dt.total_seconds() / (
                365.25 * 86400
            )
        else:
            values = frame[columns].apply(pd.to_numeric, errors="coerce")
            low, high = item.get("range", [None, None])
            if low is not None and ((values < low) | (values > high)).any().any():
                raise ValueError("Questionnaire response is outside the scoring range")
            for column in item.get("reverse", []):
                values[column] = low + high - values[column]
            required = item.get("minimum_items", len(columns))
            observed = values.notna().sum(axis=1)
            if operation == "sum":
                values = values.sum(axis=1, min_count=required)
            elif operation == "mean":
                values = values.mean(axis=1).where(observed >= required)
            else:
                raise ValueError("Unknown scoring operation")
        output[item["name"]] = values.to_numpy()
    records = []
    for _, group in output.groupby("record_id", sort=False):
        if group.nunique(dropna=True).gt(1).any():
            raise ValueError("Conflicting baseline records")
        records.append(group.bfill().iloc[0])
    return pd.DataFrame(records).reset_index(drop=True)


def session_metadata(frame, field_mapping):
    """Map locally specified note fields and calculate session timing."""
    result = frame[list(field_mapping)].rename(columns=field_mapping).copy()
    result["id"] = result.id.astype(str)
    result["session"] = pd.to_numeric(result.session, errors="raise").astype(int)
    result = result.sort_values(["id", "session"], kind="stable")
    if result.duplicated(["id", "session"]).any():
        raise ValueError("Duplicate session metadata")
    if "session_date" in result:
        date = pd.to_datetime(result.session_date, errors="coerce")
        first = (
            pd.to_datetime(result.session0_date, errors="coerce")
            if "session0_date" in result
            else date.where(result.session.eq(0)).groupby(result.id).transform("first")
        )
        previous = date.groupby(result.id).ffill().groupby(result.id).shift()
        result["session_info__derived_days_since_session0"] = (date - first).dt.days
        result["session_info__derived_days_since_previous_session"] = (
            date - previous
        ).dt.days
    if "note_text" in result:
        result["session_note_struct__session_note_word_count"] = result.note_text.map(
            word_count
        )
    return result


def extract_session_records(frame, specification):
    """Convert repeated source-form fields using a privately supplied field map."""
    identifier = specification["id_column"]
    rows = []
    for form in specification["sessions"]:
        source = list(form["fields"])
        present = frame[source].notna().any(axis=1)
        part = frame.loc[present, [identifier, *source]].rename(columns=form["fields"])
        part = part.rename(columns={identifier: "id"})
        part["session"] = int(form["session"])
        rows.append(part)
    if not rows:
        return pd.DataFrame(columns=["id", "session"])
    result = pd.concat(rows, ignore_index=True)
    if result.duplicated(["id", "session"]).any():
        merged = []
        for _, group in result.groupby(["id", "session"], sort=False):
            if group.nunique(dropna=True).gt(1).any():
                raise ValueError("Conflicting session records")
            merged.append(group.bfill().iloc[0])
        result = pd.DataFrame(merged)
    return result.reset_index(drop=True)


def outcome_sessions(attendance, outcomes):
    """Keep known outcomes and add pre-assessment rows to observed attendance."""
    known = outcomes.loc[
        outcomes.ever_dropout.isin([0, 1]), ["id", "ever_dropout", "last_session"]
    ].copy()
    if known.id.duplicated().any():
        raise ValueError("Conflicting participant outcomes")
    baseline = known[["id"]].assign(session=0)
    sessions = pd.concat([attendance[["id", "session"]], baseline]).drop_duplicates()
    result = sessions.merge(known, on="id", how="inner", validate="many_to_one")
    if result.session.gt(result.last_session).any():
        raise ValueError("Attendance extends beyond the recorded endpoint")
    result["next_session_dropout"] = (
        result.ever_dropout.eq(1) & result.session.eq(result.last_session)
    ).astype(int)
    return (
        result.drop(columns="last_session")
        .sort_values(["id", "session"], kind="stable")
        .reset_index(drop=True)
    )


def segments_from_recordings(manifest, reader):
    """Attach opaque participant-session keys through a local recording manifest."""
    rows = []
    for item in manifest.itertuples(index=False):
        payload = reader(item.path)
        duration = float(payload["video_info"]["video_length_seconds"])
        for segment in payload["snippet_emotions"]:
            start = float(segment["start_second"])
            rows.append(
                {
                    "id": str(item.id),
                    "session": int(item.session),
                    "emotion": segment["predicted_emotion"],
                    "start_second": start,
                    "duration_seconds": min(30, max(0, duration - start)),
                }
            )
    return pd.DataFrame(rows)


def speech_features(chunks):
    """Aggregate role-tagged text within each session and treatment-task category."""
    frame = chunks.copy()
    frame["participant_words"] = frame.participant_text.map(word_count)
    frame["clinician_words"] = frame.clinician_text.map(word_count)
    rows = []
    for (participant, session), group in frame.groupby(["id", "session"], sort=False):
        participant_words = int(group.participant_words.sum())
        clinician_words = int(group.clinician_words.sum())
        total = participant_words + clinician_words
        row = {
            "id": str(participant),
            "session": int(session),
            "speech__participant_total_words": participant_words,
            "speech__clinician_total_words": clinician_words,
            "speech__total_role_labeled_words": total,
            "speech__participant_clinician_word_ratio": (
                participant_words / clinician_words if clinician_words else np.nan
            ),
            "speech__participant_text_fraction": (
                participant_words / total if total else np.nan
            ),
        }
        for column in ["arm", "language"]:
            if column in group:
                values = group[column].dropna().unique()
                if len(values) > 1:
                    raise ValueError("Conflicting session metadata")
                row["speech__" + column] = values[0] if len(values) else np.nan
        for category, count in (
            group.groupby("task_category").participant_words.sum().items()
        ):
            name = re.sub(r"[^a-z0-9]+", "_", str(category).lower()).strip("_")
            row["speech__participant_words_" + name] = int(count)
        rows.append(row)
    return pd.DataFrame(rows)


def emotion_features(segments):
    """Count current segments and pool counts from strictly earlier sessions."""
    records = []
    for (participant, session), group in segments.groupby(
        ["id", "session"], sort=False
    ):
        counts = group.emotion.value_counts().reindex(EMOTIONS, fill_value=0)
        if not group.emotion.isin(EMOTIONS).all():
            raise ValueError("Unknown audio-emotion category")
        row = {
            "id": str(participant),
            "session": int(session),
            "emotion__n_snippets": len(group),
            "emotion__dominant_emotion_by_count": counts.idxmax(),
            "emotion__dominant_emotion_count": int(counts.max()),
            "emotion__dominant_emotion_prop": counts.max() / len(group),
        }
        for emotion in EMOTIONS:
            row[f"emotion__{emotion}_count"] = int(counts[emotion])
            row[f"emotion__{emotion}_prop"] = counts[emotion] / len(group)
        if "duration_seconds" in group:
            duration = float(group.duration_seconds.sum())
            row["emotion__total_video_length_seconds"] = duration
            row["emotion__total_video_length_mins"] = duration / 60
        if "start_second" in group:
            row["emotion__first_start_second"] = group.start_second.min()
            row["emotion__last_start_second"] = group.start_second.max()
        records.append(row)
    result = (
        pd.DataFrame(records)
        .sort_values(["id", "session"], kind="stable")
        .reset_index(drop=True)
    )
    total = (
        result.groupby("id").emotion__n_snippets.cumsum() - result.emotion__n_snippets
    )
    for emotion in EMOTIONS:
        name = f"emotion__{emotion}_count"
        previous = result.groupby("id")[name].cumsum() - result[name]
        result[f"emotion__derived_prior_before_session_{emotion}_prop"] = previous.div(
            total.where(total > 0)
        )
    return result


def merge_modalities(sessions, *modalities):
    result = sessions.copy()
    keys = ["id", "session"]
    if result.duplicated(keys).any():
        raise ValueError("Duplicate participant-session keys")
    for frame in modalities:
        if frame.duplicated(keys).any():
            raise ValueError("Duplicate modality records")
        if (set(result) & set(frame)) - set(keys):
            raise ValueError("Overlapping modality columns")
        result = result.merge(
            frame, on=keys, how="left", sort=False, validate="one_to_one"
        )
    return result


def combine_text_sources(records, sessions):
    """Join ordered baseline responses, session notes, and transcript excerpts."""
    records = records.sort_values(
        ["id", "session", "source_order", "excerpt_order"], kind="stable"
    )
    text = records.groupby(["id", "session"], sort=False).text.agg(
        lambda values: "\n".join(
            str(value).strip()
            for value in values
            if pd.notna(value) and str(value).strip()
        )
    )
    result = sessions[["id", "session"]].merge(
        text.rename("text"), on=["id", "session"], how="left", validate="one_to_one"
    )
    result["text"] = result.text.fillna("")
    return result
