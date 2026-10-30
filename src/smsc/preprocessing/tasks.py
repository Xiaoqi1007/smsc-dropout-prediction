"""Dialogue chunking and task segmentation from ordered protocol cues."""

from pathlib import Path
from collections import Counter
import json
import math
import re
import unicodedata
import numpy as np
import pandas as pd

SESSION_TEMPLATES = {}
TASK_META = {}
MAX_DIALOGUE_CHUNK_WORDS = 200
KEEP_STANDARD_TWO_TURN_PAIR_INTACT = True
BOUNDARY_WEIGHT = 8.0
CONTENT_WEIGHT = 2.0
POSITION_WEIGHT = 1.5
LENGTH_WEIGHT = 0.8
COMPETITOR_WEIGHT = 0.8
NEXT_BOUNDARY_WEIGHT = 2.5
NO_BOUNDARY_PENALTY = 2.0
BOUNDARY_LOOKAHEAD_CHUNKS = 2
BOUNDARY_SECOND_CHUNK_DECAY = 0.35
BOUNDARY_MAX_DISTINCT_HITS = 2
MIN_TASK_SPAN_CHUNKS = 1
MAX_TASK_LENGTH_RATIO = 4.0
CONTENT_LENGTH_NORMALIZATION = "sqrt"
USE_SEMANTIC_CUE_MATCHING = True
SEMANTIC_MODEL_NAME_OR_PATH = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
SEMANTIC_MODEL_DEVICE = None
SEMANTIC_BATCH_SIZE = 64
SEMANTIC_SHOW_PROGRESS_BAR = False
BOUNDARY_SEMANTIC_THRESHOLD = 0.62
CONTENT_SEMANTIC_THRESHOLD = 0.58
SEMANTIC_MIN_CUE_TOKENS = 3
SEMANTIC_MIN_MATCH_STRENGTH = 0.55
FALLBACK_ALLOW_MISSING_TASKS = True
FALLBACK_TASK_INCLUDED_REWARD = 3.0
FALLBACK_TASK_SKIP_PENALTY = 2.0
FALLBACK_RELAX_MAX_TASK_LENGTH = True
SC_CHECKIN_EXPECTED_END = 0.15
SC_CHECKIN_MAX_END = 0.4
SC_CHECKIN_MIN_CHUNKS = 1
SC_SAFETY_EXPECTED_START = 0.82
SC_SAFETY_MIN_START_RATIO = 0.05
SC_SAFETY_MAX_BLOCK_RATIO = 0.35
SC_SAFETY_POSITION_WEIGHT = 0.8
SC_SAFETY_LENGTH_WEIGHT = 1.2
SC_MIN_SAFETY_EVIDENCE = 1.5
SC_FORCE_THREE_PARTS = True
SC_FALLBACK_SAFETY_MIN_CHUNKS = 1
SC_FALLBACK_SAFETY_MAX_CHUNKS = 2
SC_FALLBACK_END_SEARCH_RATIO = 0.45
SC_SAFETY_MIN_SEGMENTS = 1
SC_SAFETY_MAX_SEGMENTS = 2
SC_SAFETY_MIN_GAP_CHUNKS = 1
SC_SECOND_SAFETY_MIN_EVIDENCE = 1.5
SC_SAFETY_MAX_SEGMENT_CHUNKS = 8
SC_SECOND_SEGMENT_PENALTY = 1.0
ALLOW_POST_SAFETY_SUPPORT = True
NORM_LINE_PATTERN = re.compile(
    "^\\s*(?P<timestamp>\\[\\s*(?P<start>[\\d.]+)\\s*-\\s*(?P<end>[\\d.]+)\\s*\\]\\s*)?(?P<speaker>Clinician|Participant)\\s*(?:[\\(（]\\s*(?P<original_speaker>[^)）]+?)\\s*[\\)）])?\\s*:\\s*(?P<text>.*)$",
    re.I,
)


def parse_normalized_transcript(text, part=1):
    """Parse role-tagged turns with optional timestamps and anonymous speaker labels."""
    turns = []
    for raw in str(text).splitlines():
        line = raw.strip()
        if not line:
            continue
        match = NORM_LINE_PATTERN.match(line)
        if not match:
            if turns:
                turns[-1]["text"] += " " + line
            continue
        original_speaker = match.group("original_speaker")
        turns.append(
            {
                "speaker": match.group("speaker").capitalize(),
                "original_speaker": (
                    original_speaker.strip() if original_speaker else None
                ),
                "text": match.group("text").strip(),
                "start": float(match.group("start")) if match.group("start") else None,
                "end": float(match.group("end")) if match.group("end") else None,
                "part": part,
            }
        )
    for index, turn in enumerate(turns):
        turn["turn_index"] = index
    return turns


def count_words(text):
    """Unicode-aware word count used only for chunk-length control."""
    return len(re.findall("\\b\\w+\\b", str(text), flags=re.UNICODE))


def _turn_word_count(turn):
    return count_words(turn.get("text", ""))


def _is_standard_two_turn_pair(turns):
    """Identify an adjacent clinician–participant exchange."""
    return len(turns) == 2 and Counter(
        (turn.get("speaker") for turn in turns)
    ) == Counter({"Clinician": 1, "Participant": 1})


def _greedy_min_group_count(weights, max_words):
    """Count contiguous groups under a maximum word count."""
    if not weights:
        return 0
    groups = 1
    running = 0
    for weight in weights:
        if running and running + weight > max_words:
            groups += 1
            running = weight
        else:
            running += weight
    return groups


def _balanced_partition_bounded_turns(turns, max_words):
    """Split a sequence whose individual utterances are all <= max_words."""
    if not turns:
        return []
    weights = [_turn_word_count(turn) for turn in turns]
    total_words = sum(weights)
    if total_words <= max_words:
        return [turns]
    n_groups = _greedy_min_group_count(weights, max_words)
    target = total_words / max(n_groups, 1)
    n_turns = len(turns)
    prefix = [0]
    for weight in weights:
        prefix.append(prefix[-1] + weight)
    infinity = float("inf")
    dp = [[infinity] * (n_turns + 1) for _ in range(n_groups + 1)]
    previous = [[-1] * (n_turns + 1) for _ in range(n_groups + 1)]
    dp[0][0] = 0.0
    for group_number in range(1, n_groups + 1):
        min_end = group_number
        max_end = n_turns - (n_groups - group_number)
        for end in range(min_end, max_end + 1):
            for start in range(group_number - 1, end):
                if dp[group_number - 1][start] == infinity:
                    continue
                group_words = prefix[end] - prefix[start]
                if group_words > max_words:
                    continue
                cost = dp[group_number - 1][start] + (group_words - target) ** 2
                if cost < dp[group_number][end]:
                    dp[group_number][end] = cost
                    previous[group_number][end] = start
    if previous[n_groups][n_turns] < 0:
        groups = []
        current = []
        current_words = 0
        for turn, weight in zip(turns, weights):
            if current and current_words + weight > max_words:
                groups.append(current)
                current = []
                current_words = 0
            current.append(turn)
            current_words += weight
        if current:
            groups.append(current)
        return groups
    boundaries = [n_turns]
    end = n_turns
    for group_number in range(n_groups, 0, -1):
        start = previous[group_number][end]
        boundaries.append(start)
        end = start
    boundaries.reverse()
    return [
        turns[boundaries[index] : boundaries[index + 1]]
        for index in range(len(boundaries) - 1)
    ]


def _balanced_split_turns(turns, max_words):
    """Split only between utterances."""
    if not turns:
        return []
    groups = []
    bounded_buffer = []

    def flush_buffer():
        nonlocal bounded_buffer
        if bounded_buffer:
            groups.extend(_balanced_partition_bounded_turns(bounded_buffer, max_words))
            bounded_buffer = []

    for turn in turns:
        if _turn_word_count(turn) > max_words:
            flush_buffer()
            groups.append([turn])
        else:
            bounded_buffer.append(turn)
    flush_buffer()
    return groups


def _new_raw_chunk(turn):
    return {
        "all_turns": [turn],
        "split_from_oversized_chunk": False,
        "original_raw_chunk_word_count": _turn_word_count(turn),
        "original_raw_chunk_turn_count": 1,
    }


def _split_raw_chunk_if_needed(raw_chunk):
    turns = raw_chunk["all_turns"]
    total_words = sum((_turn_word_count(turn) for turn in turns))
    keep_standard_pair = (
        KEEP_STANDARD_TWO_TURN_PAIR_INTACT and _is_standard_two_turn_pair(turns)
    )
    if total_words <= MAX_DIALOGUE_CHUNK_WORDS or keep_standard_pair:
        raw_chunk["original_raw_chunk_word_count"] = total_words
        raw_chunk["original_raw_chunk_turn_count"] = len(turns)
        raw_chunk["split_from_oversized_chunk"] = False
        return [raw_chunk]
    turn_groups = _balanced_split_turns(turns, MAX_DIALOGUE_CHUNK_WORDS)
    return [
        {
            "all_turns": group,
            "split_from_oversized_chunk": True,
            "original_raw_chunk_word_count": total_words,
            "original_raw_chunk_turn_count": len(turns),
        }
        for group in turn_groups
    ]


def _materialize_chunk(raw_chunk, chunk_index):
    turns = raw_chunk["all_turns"]
    clinician_turns = [turn for turn in turns if turn["speaker"] == "Clinician"]
    participant_turns = [turn for turn in turns if turn["speaker"] == "Participant"]
    clinician_text = " ".join((turn["text"] for turn in clinician_turns)).strip()
    participant_text = " ".join((turn["text"] for turn in participant_turns)).strip()
    first_turn = turns[0]
    last_turn = turns[-1]
    word_count = sum((_turn_word_count(turn) for turn in turns))
    original_speakers = [
        turn.get("original_speaker") for turn in turns if turn.get("original_speaker")
    ]
    return {
        "chunk_index": chunk_index,
        "clinician_text": clinician_text,
        "participant_text": participant_text,
        "combined_text": " ".join(
            (turn["text"] for turn in turns if str(turn.get("text", "")).strip())
        ).strip(),
        "turns": turns,
        "start_turn_index": first_turn["turn_index"],
        "end_turn_index": last_turn["turn_index"],
        "start_time": first_turn.get("start"),
        "end_time": last_turn.get("end"),
        "n_turns": len(turns),
        "n_words": word_count,
        "n_clinician_turns": len(clinician_turns),
        "n_participant_turns": len(participant_turns),
        "first_speaker": first_turn["speaker"],
        "last_speaker": last_turn["speaker"],
        "first_original_speaker": first_turn.get("original_speaker"),
        "last_original_speaker": last_turn.get("original_speaker"),
        "original_speakers": original_speakers,
        "split_from_oversized_chunk": raw_chunk.get(
            "split_from_oversized_chunk", False
        ),
        "original_raw_chunk_word_count": raw_chunk.get(
            "original_raw_chunk_word_count", word_count
        ),
        "original_raw_chunk_turn_count": raw_chunk.get(
            "original_raw_chunk_turn_count", len(turns)
        ),
        "contains_single_oversized_utterance": len(turns) == 1
        and word_count > MAX_DIALOGUE_CHUNK_WORDS,
    }


def build_dialogue_chunks(turns):
    """Build dialogue chunks in two stages."""
    if not turns:
        return []
    raw_chunks = []
    current = None
    for turn in turns:
        speaker = turn["speaker"]
        if current is None:
            current = _new_raw_chunk(turn)
            continue
        current_has_participant = any(
            (existing["speaker"] == "Participant" for existing in current["all_turns"])
        )
        if speaker == "Clinician" and current_has_participant:
            raw_chunks.append(current)
            current = _new_raw_chunk(turn)
            continue
        current["all_turns"].append(turn)
    if current is not None:
        raw_chunks.append(current)
    refined_raw_chunks = []
    for raw_chunk in raw_chunks:
        refined_raw_chunks.extend(_split_raw_chunk_if_needed(raw_chunk))
    chunks = [
        _materialize_chunk(raw_chunk, chunk_index)
        for chunk_index, raw_chunk in enumerate(refined_raw_chunks)
    ]
    flattened_indices = [
        turn["turn_index"] for chunk in chunks for turn in chunk["turns"]
    ]
    expected_indices = [turn["turn_index"] for turn in turns]
    if flattened_indices != expected_indices:
        raise RuntimeError(
            "Chunk construction changed the original turn order or lost turns."
        )
    for chunk in chunks:
        if (
            chunk["split_from_oversized_chunk"]
            and chunk["n_turns"] > 1
            and (chunk["n_words"] > MAX_DIALOGUE_CHUNK_WORDS)
        ):
            raise RuntimeError(
                f"Balanced chunk splitting produced a multi-utterance chunk with {chunk['n_words']} words."
            )
    return chunks


def hit_patterns(text, patterns):
    normalized = normalize_text(text)
    return [index for index, regex in enumerate(patterns) if regex.search(normalized)]


def format_timestamp_from_chunk(chunk):
    if chunk.get("start_time") is None or chunk.get("end_time") is None:
        return ""
    return f"[{chunk['start_time']:.2f} - {chunk['end_time']:.2f}] "


def _normalized_content_score(weighted_hit_sum, block_length):
    if CONTENT_LENGTH_NORMALIZATION == "sqrt":
        return weighted_hit_sum / math.sqrt(max(block_length, 1))
    if CONTENT_LENGTH_NORMALIZATION == "log":
        return weighted_hit_sum / (1.0 + math.log1p(max(block_length, 1)))
    return weighted_hit_sum


_SEMANTIC_MODEL = None
_SEMANTIC_CUE_EMBEDDING_CACHE = {}


def _semantic_cue_text(cue):
    """Replace literal cue wildcards with natural text for embedding."""
    return re.sub("\\s*\\.\\.\\.\\s*", " ", str(cue)).strip()


def _semantic_token_count(text):
    return len(re.findall("[a-z0-9']+", normalize_text(text)))


def _semantic_cue_is_eligible(cue):
    return _semantic_token_count(_semantic_cue_text(cue)) >= SEMANTIC_MIN_CUE_TOKENS


def get_semantic_model():
    global _SEMANTIC_MODEL
    if _SEMANTIC_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "Semantic cue matching requires the 'sentence-transformers' package. Install it with `pip install sentence-transformers`, or set USE_SEMANTIC_CUE_MATCHING=False."
            ) from exc
        kwargs = {}
        if SEMANTIC_MODEL_DEVICE:
            kwargs["device"] = SEMANTIC_MODEL_DEVICE
        _SEMANTIC_MODEL = SentenceTransformer(SEMANTIC_MODEL_NAME_OR_PATH, **kwargs)
    return _SEMANTIC_MODEL


def _encode_semantic_texts(texts):
    """Encode texts as normalized float32 vectors."""
    texts = [str(text).strip() for text in texts]
    model = get_semantic_model()
    embedding_dim = model.get_sentence_embedding_dimension()
    output = np.zeros((len(texts), embedding_dim), dtype=np.float32)
    nonempty = [index for index, text in enumerate(texts) if text]
    if nonempty:
        output[nonempty] = model.encode(
            [texts[index] for index in nonempty],
            batch_size=SEMANTIC_BATCH_SIZE,
            show_progress_bar=SEMANTIC_SHOW_PROGRESS_BAR,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)
    return output


def _get_cue_embeddings(cues):
    """Encode each unique cue once and reuse it for every session."""
    cleaned = [_semantic_cue_text(cue) for cue in cues]
    missing = [
        text
        for text in dict.fromkeys(cleaned)
        if text and text not in _SEMANTIC_CUE_EMBEDDING_CACHE
    ]
    if missing:
        embeddings = _encode_semantic_texts(missing)
        for text, embedding in zip(missing, embeddings):
            _SEMANTIC_CUE_EMBEDDING_CACHE[text] = embedding
    if not cleaned:
        dim = get_semantic_model().get_sentence_embedding_dimension()
        return np.zeros((0, dim), dtype=np.float32)
    dim = get_semantic_model().get_sentence_embedding_dimension()
    zero = np.zeros(dim, dtype=np.float32)
    return np.vstack(
        [_SEMANTIC_CUE_EMBEDDING_CACHE.get(text, zero) for text in cleaned]
    ).astype(np.float32)


def _ensure_chunk_semantic_units(chunks):
    """Encode every original utterance exactly once."""
    if not USE_SEMANTIC_CUE_MATCHING or not chunks:
        return
    if all(("_semantic_unit_embeddings" in chunk for chunk in chunks)):
        return
    all_texts = []
    owners = []
    unit_metadata = [[] for _ in chunks]
    for chunk_index, chunk in enumerate(chunks):
        for unit_index, turn in enumerate(chunk.get("turns", [])):
            text = str(turn.get("text", "")).strip()
            unit_metadata[chunk_index].append(
                {
                    "speaker": turn.get("speaker"),
                    "original_speaker": turn.get("original_speaker"),
                    "speaker_label": _turn_speaker_label(turn),
                    "turn_index": turn.get("turn_index"),
                    "text": text,
                }
            )
            all_texts.append(text)
            owners.append((chunk_index, unit_index))
        if not unit_metadata[chunk_index]:
            unit_metadata[chunk_index].append(
                {
                    "speaker": "Empty",
                    "original_speaker": None,
                    "speaker_label": "Empty",
                    "turn_index": None,
                    "text": "",
                }
            )
            all_texts.append("")
            owners.append((chunk_index, 0))
    all_embeddings = _encode_semantic_texts(all_texts)
    embedding_dim = all_embeddings.shape[1]
    grouped = [
        np.zeros((len(unit_metadata[index]), embedding_dim), dtype=np.float32)
        for index in range(len(chunks))
    ]
    for embedding, (chunk_index, unit_index) in zip(all_embeddings, owners):
        grouped[chunk_index][unit_index] = embedding
    for chunk_index, chunk in enumerate(chunks):
        chunk["_semantic_units"] = unit_metadata[chunk_index]
        chunk["_semantic_unit_embeddings"] = grouped[chunk_index]


def _semantic_similarity_by_chunk(chunks, cues):
    """Find the strongest utterance-level cue similarity in each chunk."""
    n_chunks = len(chunks)
    n_cues = len(cues)
    scores = np.full((n_chunks, n_cues), -1.0, dtype=np.float32)
    best_units = [[None for _ in range(n_cues)] for _ in range(n_chunks)]
    if not USE_SEMANTIC_CUE_MATCHING or n_chunks == 0 or n_cues == 0:
        return (scores, best_units)
    _ensure_chunk_semantic_units(chunks)
    cue_embeddings = _get_cue_embeddings(cues)
    eligible = np.array([_semantic_cue_is_eligible(cue) for cue in cues])
    for chunk_index, chunk in enumerate(chunks):
        unit_embeddings = chunk["_semantic_unit_embeddings"]
        similarities = unit_embeddings @ cue_embeddings.T
        best_indices = np.argmax(similarities, axis=0)
        best_scores = similarities[best_indices, np.arange(n_cues)]
        best_scores[~eligible] = -1.0
        scores[chunk_index] = best_scores
        for cue_index, unit_index in enumerate(best_indices):
            if eligible[cue_index]:
                best_units[chunk_index][cue_index] = chunk["_semantic_units"][
                    int(unit_index)
                ]
    return (scores, best_units)


def _semantic_strength(similarity, threshold):
    """Convert a threshold-passing cosine similarity to a [0, 1] strength."""
    similarity = float(similarity)
    if similarity < threshold:
        return 0.0
    scaled = (similarity - threshold) / max(1.0 - threshold, 1e-08)
    return float(
        min(
            1.0,
            SEMANTIC_MIN_MATCH_STRENGTH
            + (1.0 - SEMANTIC_MIN_MATCH_STRENGTH) * max(0.0, scaled),
        )
    )


def _turn_speaker_label(turn):
    role = str(turn.get("speaker", "")).strip() or "Unknown"
    original = turn.get("original_speaker")
    if original:
        return f"{role} ({original})"
    return role


def _exact_match_details_by_utterance(chunk, patterns):
    """Match each regular expression separately against each original utterance."""
    details = {}
    for turn in chunk.get("turns", []):
        normalized_text = normalize_text(turn.get("text", ""))
        for cue_index, regex in enumerate(patterns):
            if cue_index in details:
                continue
            match = regex.search(normalized_text)
            if match is None:
                continue
            details[cue_index] = {
                "speaker": turn.get("speaker"),
                "original_speaker": turn.get("original_speaker"),
                "speaker_label": _turn_speaker_label(turn),
                "turn_index": turn.get("turn_index"),
                "text": turn.get("text", ""),
                "normalized_text": normalized_text,
                "char_position": int(match.start()),
            }
    return details


def build_task_cache(chunks, task):
    """Cache utterance-level hybrid exact + semantic cue evidence."""
    meta = TASK_META[task]
    n_chunks = len(chunks)
    boundary_semantic, boundary_best_units = _semantic_similarity_by_chunk(
        chunks, meta["boundary_text"]
    )
    content_semantic, content_best_units = _semantic_similarity_by_chunk(
        chunks, meta["content_text"]
    )
    boundary_score = np.zeros(n_chunks, dtype=float)
    boundary_hit_indices = [[] for _ in range(n_chunks)]
    boundary_hit_details = [[] for _ in range(n_chunks)]
    for start_chunk in range(n_chunks):
        evidence = []
        for offset in range(BOUNDARY_LOOKAHEAD_CHUNKS):
            chunk_index = start_chunk + offset
            if chunk_index >= n_chunks:
                break
            chunk = chunks[chunk_index]
            exact_match_details = _exact_match_details_by_utterance(
                chunk, meta["boundary"]
            )
            exact_hits = set(exact_match_details)
            decay = 1.0 if offset == 0 else BOUNDARY_SECOND_CHUNK_DECAY
            for cue_index, regex in enumerate(meta["boundary"]):
                exact_hit = cue_index in exact_hits
                similarity = float(boundary_semantic[chunk_index, cue_index])
                semantic_strength = _semantic_strength(
                    similarity, BOUNDARY_SEMANTIC_THRESHOLD
                )
                if not exact_hit and semantic_strength == 0:
                    continue
                cue_weight = (
                    float(meta["boundary_weights"][cue_index])
                    if cue_index < len(meta["boundary_weights"])
                    else 1.0
                )
                match_strength = max(1.0 if exact_hit else 0.0, semantic_strength)
                within_turn_bonus = 1.0
                if exact_hit:
                    exact_detail = exact_match_details[cue_index]
                    matched_speaker = exact_detail["speaker_label"]
                    matched_original_speaker = exact_detail["original_speaker"]
                    matched_turn_index = exact_detail["turn_index"]
                    matched_text = exact_detail["text"]
                    normalized_source = exact_detail["normalized_text"]
                    char_position = exact_detail["char_position"]
                    if char_position >= 0 and normalized_source:
                        within_turn_bonus += 0.35 * (
                            1.0 - char_position / len(normalized_source)
                        )
                else:
                    best_unit = boundary_best_units[chunk_index][cue_index]
                    matched_speaker = (
                        best_unit["speaker_label"] if best_unit is not None else None
                    )
                    matched_original_speaker = (
                        best_unit["original_speaker"] if best_unit is not None else None
                    )
                    matched_turn_index = (
                        best_unit["turn_index"] if best_unit is not None else None
                    )
                    matched_text = best_unit["text"] if best_unit is not None else None
                if exact_hit and semantic_strength > 0:
                    match_type = "exact+semantic"
                elif exact_hit:
                    match_type = "exact"
                else:
                    match_type = "semantic"
                evidence.append(
                    {
                        "strength": float(
                            decay * cue_weight * match_strength * within_turn_bonus
                        ),
                        "cue_index": cue_index,
                        "chunk_index": chunk_index,
                        "match_type": match_type,
                        "cosine_similarity": similarity if similarity >= 0 else None,
                        "matched_speaker": matched_speaker,
                        "matched_original_speaker": matched_original_speaker,
                        "matched_turn_index": matched_turn_index,
                        "matched_text": matched_text,
                        "clinician_text": chunk["clinician_text"],
                        "participant_text": chunk["participant_text"],
                    }
                )
        best_by_cue = {}
        for detail in evidence:
            cue_index = detail["cue_index"]
            previous = best_by_cue.get(cue_index)
            if previous is None or detail["strength"] > previous["strength"]:
                best_by_cue[cue_index] = detail
        ranked = sorted(
            best_by_cue.values(), key=lambda item: item["strength"], reverse=True
        )[:BOUNDARY_MAX_DISTINCT_HITS]
        boundary_score[start_chunk] = sum((item["strength"] for item in ranked))
        boundary_hit_indices[start_chunk] = [item["cue_index"] for item in ranked]
        boundary_hit_details[start_chunk] = ranked
    n_cues = len(meta["content"])
    content_strengths = np.zeros((n_cues, n_chunks), dtype=np.float32)
    content_occurrences = [[] for _ in range(n_cues)]
    content_match_details = [{} for _ in range(n_cues)]
    for chunk_index, chunk in enumerate(chunks):
        exact_match_details = _exact_match_details_by_utterance(chunk, meta["content"])
        exact_hits = set(exact_match_details)
        for cue_index, regex in enumerate(meta["content"]):
            exact_hit = cue_index in exact_hits
            similarity = float(content_semantic[chunk_index, cue_index])
            semantic_strength = _semantic_strength(
                similarity, CONTENT_SEMANTIC_THRESHOLD
            )
            if not exact_hit and semantic_strength == 0:
                continue
            match_strength = max(1.0 if exact_hit else 0.0, semantic_strength)
            if exact_hit:
                exact_detail = exact_match_details[cue_index]
                matched_speaker = exact_detail["speaker_label"]
                matched_original_speaker = exact_detail["original_speaker"]
                matched_turn_index = exact_detail["turn_index"]
                matched_text = exact_detail["text"]
            else:
                best_unit = content_best_units[chunk_index][cue_index]
                matched_speaker = (
                    best_unit["speaker_label"] if best_unit is not None else None
                )
                matched_original_speaker = (
                    best_unit["original_speaker"] if best_unit is not None else None
                )
                matched_turn_index = (
                    best_unit["turn_index"] if best_unit is not None else None
                )
                matched_text = best_unit["text"] if best_unit is not None else None
            if exact_hit and semantic_strength > 0:
                match_type = "exact+semantic"
            elif exact_hit:
                match_type = "exact"
            else:
                match_type = "semantic"
            content_strengths[cue_index, chunk_index] = match_strength
            content_occurrences[cue_index].append(chunk_index)
            content_match_details[cue_index][chunk_index] = {
                "match_type": match_type,
                "cosine_similarity": similarity if similarity >= 0 else None,
                "match_strength": float(match_strength),
                "matched_speaker": matched_speaker,
                "matched_original_speaker": matched_original_speaker,
                "matched_turn_index": matched_turn_index,
                "matched_text": matched_text,
            }
    return {
        "boundary_score": boundary_score,
        "boundary_hit_indices": boundary_hit_indices,
        "boundary_hit_details": boundary_hit_details,
        "content_strengths": content_strengths,
        "content_occurrences": content_occurrences,
        "content_match_details": content_match_details,
    }


def full_block_content_evidence(cache, task, start_chunk, end_chunk, chunks):
    strengths = cache["content_strengths"]
    if strengths.shape[0] == 0 or end_chunk <= start_chunk:
        return {
            "raw_count": 0,
            "weighted_sum": 0.0,
            "normalized_score": 0.0,
            "cue_indices": [],
            "details": [],
        }
    interval_strengths = strengths[:, start_chunk:end_chunk]
    max_strengths = interval_strengths.max(axis=1)
    cue_indices = np.flatnonzero(max_strengths > 0).tolist()
    weights = TASK_META[task]["content_weights"]
    weighted_sum = float(
        sum(
            (
                (weights[index] if index < len(weights) else 1.0)
                * float(max_strengths[index])
                for index in cue_indices
            )
        )
    )
    normalized_score = _normalized_content_score(weighted_sum, end_chunk - start_chunk)
    details = []
    for cue_index in cue_indices:
        positions = [
            index
            for index in cache["content_occurrences"][cue_index]
            if start_chunk <= index < end_chunk
        ]
        examples = []
        for index in positions[:3]:
            match_detail = cache["content_match_details"][cue_index].get(index, {})
            examples.append(
                {
                    "chunk_index": index,
                    "clinician_text": chunks[index]["clinician_text"],
                    "participant_text": chunks[index]["participant_text"],
                    "match_type": match_detail.get("match_type"),
                    "cosine_similarity": match_detail.get("cosine_similarity"),
                    "match_strength": match_detail.get("match_strength"),
                    "matched_speaker": match_detail.get("matched_speaker"),
                    "matched_original_speaker": match_detail.get(
                        "matched_original_speaker"
                    ),
                    "matched_turn_index": match_detail.get("matched_turn_index"),
                    "matched_text": match_detail.get("matched_text"),
                }
            )
        details.append(
            {
                "cue_index": cue_index,
                "chunk_indices": positions,
                "max_match_strength": float(max_strengths[cue_index]),
                "examples": examples,
            }
        )
    return {
        "raw_count": len(cue_indices),
        "weighted_sum": weighted_sum,
        "normalized_score": normalized_score,
        "cue_indices": cue_indices,
        "details": details,
    }


def _competitor_content_score(
    caches, template, task_index, start_chunk, end_chunk, chunks
):
    scores = []
    for adjacent_index in (task_index - 1, task_index + 1):
        if 0 <= adjacent_index < len(template):
            adjacent_task = template[adjacent_index]
            evidence = full_block_content_evidence(
                caches[adjacent_index], adjacent_task, start_chunk, end_chunk, chunks
            )
            scores.append(evidence["normalized_score"])
    return max(scores, default=0.0)


def ordered_dp_segment(chunks, template):
    """Optimize ordered task blocks over complete dialogue chunks."""
    n_chunks = len(chunks)
    n_tasks = len(template)
    if n_tasks == 0 or n_chunks < n_tasks:
        return None
    caches = [build_task_cache(chunks, task) for task in template]
    expected_length = n_chunks / n_tasks
    negative_infinity = -1e18
    dp = np.full((n_tasks + 1, n_chunks + 1), negative_infinity)
    backpointer = np.full((n_tasks + 1, n_chunks + 1), -1, dtype=int)
    dp[0, 0] = 0.0
    for task_number in range(1, n_tasks + 1):
        task_index = task_number - 1
        task = template[task_index]
        cache = caches[task_index]
        expected_start = task_index / n_tasks
        for end_chunk in range(task_number, n_chunks + 1):
            for start_chunk in range(task_number - 1, end_chunk):
                if dp[task_number - 1, start_chunk] <= negative_infinity / 2:
                    continue
                block_length = end_chunk - start_chunk
                if block_length < MIN_TASK_SPAN_CHUNKS:
                    continue
                if block_length > MAX_TASK_LENGTH_RATIO * expected_length:
                    continue
                content_evidence = full_block_content_evidence(
                    cache, task, start_chunk, end_chunk, chunks
                )
                content_score = content_evidence["normalized_score"]
                competitor_score = _competitor_content_score(
                    caches, template, task_index, start_chunk, end_chunk, chunks
                )
                boundary_score = float(cache["boundary_score"][start_chunk])
                position_fit = 1.0 - abs(
                    start_chunk / max(n_chunks - 1, 1) - expected_start
                )
                length_penalty = abs(
                    math.log(max(block_length, 1) / max(expected_length, 1))
                )
                next_boundary = 0.0
                if task_index + 1 < n_tasks and end_chunk < n_chunks:
                    next_boundary = float(
                        caches[task_index + 1]["boundary_score"][end_chunk]
                    )
                score = (
                    BOUNDARY_WEIGHT * boundary_score
                    + CONTENT_WEIGHT * content_score
                    + POSITION_WEIGHT * position_fit
                    + NEXT_BOUNDARY_WEIGHT * next_boundary
                    - COMPETITOR_WEIGHT * competitor_score
                    - LENGTH_WEIGHT * length_penalty
                )
                if boundary_score == 0 and task_number > 1:
                    score -= NO_BOUNDARY_PENALTY
                candidate = dp[task_number - 1, start_chunk] + score
                if candidate > dp[task_number, end_chunk]:
                    dp[task_number, end_chunk] = candidate
                    backpointer[task_number, end_chunk] = start_chunk
    if dp[n_tasks, n_chunks] <= negative_infinity / 2:
        return None
    boundaries = [n_chunks]
    end_chunk = n_chunks
    for task_number in range(n_tasks, 0, -1):
        start_chunk = int(backpointer[task_number, end_chunk])
        boundaries.append(start_chunk)
        end_chunk = start_chunk
    boundaries = boundaries[::-1]
    spans = []
    for task_index, task in enumerate(template):
        start_chunk = boundaries[task_index]
        end_chunk = boundaries[task_index + 1]
        cache = caches[task_index]
        content_evidence = full_block_content_evidence(
            cache, task, start_chunk, end_chunk, chunks
        )
        boundary_indices = sorted(set(cache["boundary_hit_indices"][start_chunk]))
        spans.append(
            {
                "task": task,
                "start_chunk": start_chunk,
                "end_chunk": end_chunk,
                "boundary_score": float(cache["boundary_score"][start_chunk]),
                "boundary_hits": [
                    TASK_META[task]["boundary_text"][index]
                    for index in boundary_indices
                    if index < len(TASK_META[task]["boundary_text"])
                ],
                "boundary_hit_details": cache["boundary_hit_details"][start_chunk],
                "content_score": float(content_evidence["normalized_score"]),
                "content_hits": [
                    TASK_META[task]["content_text"][index]
                    for index in content_evidence["cue_indices"]
                    if index < len(TASK_META[task]["content_text"])
                ],
                "content_hit_details": content_evidence["details"],
                "selection_reason": "ordered_dp_chunk_level",
                "forced_assignment": False,
            }
        )
    return spans


def ordered_dp_segment_allow_missing_tasks(chunks, template):
    """Find ordered task blocks while allowing unsupported tasks to be skipped."""
    n_chunks = len(chunks)
    n_tasks = len(template)
    if n_chunks == 0 or n_tasks == 0:
        return (None, list(template))
    caches = [build_task_cache(chunks, task) for task in template]
    expected_length = max(n_chunks / max(min(n_tasks, n_chunks), 1), 1.0)
    negative_infinity = -1e18
    dp = np.full((n_tasks + 1, n_chunks + 1), negative_infinity, dtype=float)
    backpointer = [[None] * (n_chunks + 1) for _ in range(n_tasks + 1)]
    dp[0, 0] = 0.0
    for task_index in range(n_tasks):
        task = template[task_index]
        cache = caches[task_index]
        expected_start = task_index / max(n_tasks, 1)
        for start_chunk in range(n_chunks + 1):
            current_score = dp[task_index, start_chunk]
            if current_score <= negative_infinity / 2:
                continue
            skipped_score = current_score - FALLBACK_TASK_SKIP_PENALTY
            if skipped_score > dp[task_index + 1, start_chunk]:
                dp[task_index + 1, start_chunk] = skipped_score
                backpointer[task_index + 1][start_chunk] = {
                    "action": "skip",
                    "previous_task_index": task_index,
                    "previous_chunk": start_chunk,
                }
            for end_chunk in range(start_chunk + 1, n_chunks + 1):
                block_length = end_chunk - start_chunk
                if (
                    not FALLBACK_RELAX_MAX_TASK_LENGTH
                    and block_length > MAX_TASK_LENGTH_RATIO * expected_length
                ):
                    continue
                content_evidence = full_block_content_evidence(
                    cache, task, start_chunk, end_chunk, chunks
                )
                content_score = content_evidence["normalized_score"]
                competitor_score = _competitor_content_score(
                    caches, template, task_index, start_chunk, end_chunk, chunks
                )
                boundary_score = float(cache["boundary_score"][start_chunk])
                position_fit = 1.0 - abs(
                    start_chunk / max(n_chunks - 1, 1) - expected_start
                )
                length_penalty = abs(
                    math.log(max(block_length, 1) / max(expected_length, 1))
                )
                next_boundary = 0.0
                if task_index + 1 < n_tasks and end_chunk < n_chunks:
                    next_boundary = float(
                        caches[task_index + 1]["boundary_score"][end_chunk]
                    )
                block_score = (
                    BOUNDARY_WEIGHT * boundary_score
                    + CONTENT_WEIGHT * content_score
                    + POSITION_WEIGHT * position_fit
                    + NEXT_BOUNDARY_WEIGHT * next_boundary
                    - COMPETITOR_WEIGHT * competitor_score
                    - LENGTH_WEIGHT * length_penalty
                    + FALLBACK_TASK_INCLUDED_REWARD
                )
                if boundary_score == 0 and start_chunk > 0:
                    block_score -= NO_BOUNDARY_PENALTY
                candidate = current_score + block_score
                if candidate > dp[task_index + 1, end_chunk]:
                    dp[task_index + 1, end_chunk] = candidate
                    backpointer[task_index + 1][end_chunk] = {
                        "action": "use",
                        "previous_task_index": task_index,
                        "previous_chunk": start_chunk,
                        "task_index": task_index,
                        "start_chunk": start_chunk,
                        "end_chunk": end_chunk,
                    }
    if dp[n_tasks, n_chunks] <= negative_infinity / 2:
        return (None, list(template))
    selected_blocks = []
    omitted_task_indices = []
    task_cursor = n_tasks
    chunk_cursor = n_chunks
    while task_cursor > 0:
        pointer = backpointer[task_cursor][chunk_cursor]
        if pointer is None:
            return (None, list(template))
        if pointer["action"] == "skip":
            omitted_task_indices.append(task_cursor - 1)
            task_cursor = pointer["previous_task_index"]
            chunk_cursor = pointer["previous_chunk"]
        else:
            selected_blocks.append(
                {
                    "task_index": pointer["task_index"],
                    "start_chunk": pointer["start_chunk"],
                    "end_chunk": pointer["end_chunk"],
                }
            )
            task_cursor = pointer["previous_task_index"]
            chunk_cursor = pointer["previous_chunk"]
    if chunk_cursor != 0:
        return (None, list(template))
    selected_blocks.reverse()
    omitted_task_indices = sorted(set(omitted_task_indices))
    omitted_tasks = [template[index] for index in omitted_task_indices]
    spans = []
    for block in selected_blocks:
        task_index = block["task_index"]
        task = template[task_index]
        start_chunk = block["start_chunk"]
        end_chunk = block["end_chunk"]
        cache = caches[task_index]
        content_evidence = full_block_content_evidence(
            cache, task, start_chunk, end_chunk, chunks
        )
        boundary_indices = sorted(set(cache["boundary_hit_indices"][start_chunk]))
        spans.append(
            {
                "task": task,
                "start_chunk": start_chunk,
                "end_chunk": end_chunk,
                "boundary_score": float(cache["boundary_score"][start_chunk]),
                "boundary_hits": [
                    TASK_META[task]["boundary_text"][index]
                    for index in boundary_indices
                    if index < len(TASK_META[task]["boundary_text"])
                ],
                "boundary_hit_details": cache["boundary_hit_details"][start_chunk],
                "content_score": float(content_evidence["normalized_score"]),
                "content_hits": [
                    TASK_META[task]["content_text"][index]
                    for index in content_evidence["cue_indices"]
                    if index < len(TASK_META[task]["content_text"])
                ],
                "content_hit_details": content_evidence["details"],
                "selection_reason": "ordered_dp_error_fallback_allow_missing_tasks",
                "forced_assignment": True,
                "fallback_used": True,
                "omitted_tasks": omitted_tasks,
            }
        )
    if not spans:
        return (None, list(template))
    assigned = []
    for span in spans:
        assigned.extend(range(span["start_chunk"], span["end_chunk"]))
    if assigned != list(range(n_chunks)):
        raise RuntimeError(
            "Fallback segmentation did not cover every chunk exactly once."
        )
    return (spans, omitted_tasks)


def _make_chunk_span(
    task,
    start_chunk,
    end_chunk,
    chunks,
    cache=None,
    selection_reason="model_selected",
    forced_assignment=False,
    safety_segment_number=None,
    safety_segment_count=None,
):
    if cache is None:
        cache = build_task_cache(chunks, task)
    content_evidence = full_block_content_evidence(
        cache, task, start_chunk, end_chunk, chunks
    )
    if start_chunk < len(chunks):
        boundary_indices = sorted(set(cache["boundary_hit_indices"][start_chunk]))
        boundary_score = float(cache["boundary_score"][start_chunk])
        boundary_details = cache["boundary_hit_details"][start_chunk]
    else:
        boundary_indices = []
        boundary_score = 0.0
        boundary_details = []
    return {
        "task": task,
        "start_chunk": start_chunk,
        "end_chunk": end_chunk,
        "boundary_score": boundary_score,
        "boundary_hits": [
            TASK_META[task]["boundary_text"][index]
            for index in boundary_indices
            if index < len(TASK_META[task]["boundary_text"])
        ],
        "boundary_hit_details": boundary_details,
        "content_score": float(content_evidence["normalized_score"]),
        "content_hits": [
            TASK_META[task]["content_text"][index]
            for index in content_evidence["cue_indices"]
            if index < len(TASK_META[task]["content_text"])
        ],
        "content_hit_details": content_evidence["details"],
        "selection_reason": selection_reason,
        "forced_assignment": bool(forced_assignment),
        "safety_segment_number": safety_segment_number,
        "safety_segment_count": safety_segment_count,
    }


def _select_sc_checkin_end(chunks, check_task, latest_allowed_end):
    n_chunks = len(chunks)
    cache = build_task_cache(chunks, check_task)
    upper = min(
        max(SC_CHECKIN_MIN_CHUNKS, latest_allowed_end),
        max(SC_CHECKIN_MIN_CHUNKS, int(math.ceil(n_chunks * SC_CHECKIN_MAX_END))),
    )
    upper = min(upper, max(1, n_chunks - 2))
    best_end = min(max(1, SC_CHECKIN_MIN_CHUNKS), upper)
    best_score = -1e18
    for end_chunk in range(max(1, SC_CHECKIN_MIN_CHUNKS), upper + 1):
        evidence = full_block_content_evidence(cache, check_task, 0, end_chunk, chunks)
        position_fit = 1.0 - abs(end_chunk / max(n_chunks, 1) - SC_CHECKIN_EXPECTED_END)
        length_penalty = max(
            0.0, end_chunk / max(n_chunks, 1) - SC_CHECKIN_EXPECTED_END
        )
        score = (
            CONTENT_WEIGHT * evidence["normalized_score"]
            + POSITION_WEIGHT * position_fit
            - LENGTH_WEIGHT * length_penalty
        )
        if score > best_score:
            best_score = score
            best_end = end_chunk
    return (best_end, cache)


_SC_FALLBACK_SCHEDULE_PATTERNS = [
    re.compile(pattern, re.I)
    for pattern in [
        "\\bnext\\s+(week|session|appointment)\\b",
        "\\bschedul\\w*\\b",
        "\\bappointment\\b",
        "\\bwhat\\s+time\\b",
        "\\bdoes\\s+.*\\s+work\\s+for\\s+you\\b",
        "\\bcan\\s+you\\s+make\\s+it\\b",
        "\\bbarrier\\w*\\b",
        "\\battend\\w*\\b",
        "\\bcoming\\s+next\\s+week\\b",
        "\\bproxima\\s+(semana|sesion)\\b",
        "\\bpróxima\\s+(semana|sesión)\\b",
        "\\bnos\\s+vemos\\b",
        "\\ba\\s+que\\s+hora\\b",
        "\\ba\\s+qué\\s+hora\\b",
        "\\basist\\w*\\b",
        "\\bcita\\b",
    ]
]


def _fallback_sc_safety_interval(chunks, lower_bound):
    """Force one compact dialogue-chunk interval as safety/attendance."""
    n_chunks = len(chunks)
    if n_chunks < 3:
        raise ValueError("SC 2–22 requires at least three dialogue chunks.")
    search_start = max(
        lower_bound + 1,
        int(math.floor(n_chunks * (1.0 - SC_FALLBACK_END_SEARCH_RATIO))),
    )
    search_start = min(search_start, n_chunks - 1)
    candidates = []
    for chunk_index in range(search_start, n_chunks):
        text = normalize_text(chunks[chunk_index]["combined_text"])
        lexical = sum(
            (bool(regex.search(text)) for regex in _SC_FALLBACK_SCHEDULE_PATTERNS)
        )
        end_bonus = chunk_index / max(n_chunks - 1, 1)
        score = 3.0 * lexical + end_bonus
        candidates.append((score, lexical, chunk_index))
    _, _, anchor = max(candidates)
    start_chunk = anchor
    end_chunk = min(n_chunks, start_chunk + SC_FALLBACK_SAFETY_MAX_CHUNKS)
    if start_chunk <= lower_bound:
        start_chunk = min(n_chunks - 1, lower_bound + 1)
        end_chunk = min(n_chunks, start_chunk + 1)
    return (start_chunk, end_chunk)


def _enumerate_sc_safety_candidates(chunks, safety_task, lower_bound):
    """Enumerate compact cue-supported safety/attendance intervals."""
    n_chunks = len(chunks)
    cache = build_task_cache(chunks, safety_task)
    start_min = max(
        lower_bound + 1, int(math.floor(n_chunks * SC_SAFETY_MIN_START_RATIO))
    )
    start_min = min(start_min, n_chunks - 1)
    relative_max_length = max(1, int(math.ceil(n_chunks * SC_SAFETY_MAX_BLOCK_RATIO)))
    max_length = min(relative_max_length, SC_SAFETY_MAX_SEGMENT_CHUNKS)
    candidates = []
    for start_chunk in range(start_min, n_chunks):
        end_max = min(n_chunks, start_chunk + max_length)
        for end_chunk in range(start_chunk + 1, end_max + 1):
            evidence = full_block_content_evidence(
                cache, safety_task, start_chunk, end_chunk, chunks
            )
            boundary = float(cache["boundary_score"][start_chunk])
            total_evidence = boundary + evidence["weighted_sum"]
            if total_evidence <= 0:
                continue
            cue_chunk_positions = sorted(
                {
                    chunk_index
                    for detail in evidence["details"]
                    for chunk_index in detail["chunk_indices"]
                }
            )
            boundary_positions = sorted(
                {
                    detail["chunk_index"]
                    for detail in cache["boundary_hit_details"][start_chunk]
                }
            )
            supported_positions = sorted(
                set(cue_chunk_positions) | set(boundary_positions)
            )
            block_length = end_chunk - start_chunk
            cue_density = len(supported_positions) / max(block_length, 1)
            position_fit = 1.0 - abs(
                start_chunk / max(n_chunks - 1, 1) - SC_SAFETY_EXPECTED_START
            )
            length_ratio = block_length / max(n_chunks, 1)
            score = (
                BOUNDARY_WEIGHT * boundary
                + CONTENT_WEIGHT * evidence["normalized_score"]
                + SC_SAFETY_POSITION_WEIGHT * position_fit
                + 2.0 * cue_density
                - SC_SAFETY_LENGTH_WEIGHT * length_ratio
                - 0.15 * max(0, block_length - 2)
            )
            candidates.append(
                {
                    "start_chunk": start_chunk,
                    "end_chunk": end_chunk,
                    "score": float(score),
                    "total_evidence": float(total_evidence),
                    "boundary_score": boundary,
                    "content_score": float(evidence["normalized_score"]),
                    "cue_density": float(cue_density),
                    "supported_positions": supported_positions,
                }
            )
    return (candidates, cache)


def _intervals_are_compatible(first, second):
    """Check separation between non-overlapping safety intervals."""
    return first["end_chunk"] + SC_SAFETY_MIN_GAP_CHUNKS <= second["start_chunk"]


def _select_one_or_two_sc_safety_intervals(chunks, safety_task, lower_bound):
    """Select up to two separated safety intervals from cue-supported candidates."""
    candidates, cache = _enumerate_sc_safety_candidates(
        chunks, safety_task, lower_bound
    )
    eligible = [
        candidate
        for candidate in candidates
        if candidate["total_evidence"] >= SC_MIN_SAFETY_EVIDENCE
    ]
    if not eligible:
        if not SC_FORCE_THREE_PARTS:
            return ([], cache, "sc_safety_not_found", False)
        start_chunk, end_chunk = _fallback_sc_safety_interval(chunks, lower_bound)
        return (
            [
                {
                    "start_chunk": start_chunk,
                    "end_chunk": end_chunk,
                    "score": 0.0,
                    "total_evidence": 0.0,
                }
            ],
            cache,
            "sc_safety_forced_fallback",
            True,
        )
    best_single = max(
        eligible,
        key=lambda item: (
            item["score"],
            item["total_evidence"],
            -(item["end_chunk"] - item["start_chunk"]),
        ),
    )
    selected = [best_single]
    if SC_SAFETY_MAX_SEGMENTS >= 2:
        best_pair = None
        best_pair_objective = -1e18
        sorted_candidates = sorted(
            eligible, key=lambda item: (item["start_chunk"], item["end_chunk"])
        )
        for first_index, first in enumerate(sorted_candidates):
            for second in sorted_candidates[first_index + 1 :]:
                if not _intervals_are_compatible(first, second):
                    continue
                if second["total_evidence"] < SC_SECOND_SAFETY_MIN_EVIDENCE:
                    continue
                pair_objective = (
                    first["score"] + second["score"] - SC_SECOND_SEGMENT_PENALTY
                )
                if pair_objective > best_pair_objective:
                    best_pair_objective = pair_objective
                    best_pair = [first, second]
        if best_pair is not None and best_pair_objective > best_single["score"] + 1e-09:
            selected = best_pair
    selected = sorted(
        selected, key=lambda item: (item["start_chunk"], item["end_chunk"])
    )
    return (
        selected,
        cache,
        (
            "sc_safety_two_cue_supported_segments"
            if len(selected) == 2
            else "sc_safety_one_cue_supported_segment"
        ),
        False,
    )


def flexible_sc_segment(chunks, template):
    """Assign SC check-in, safety, and supportive counseling blocks."""
    n_chunks = len(chunks)
    if n_chunks < 3:
        raise ValueError(
            f"SC 2–22 needs at least 3 dialogue chunks; received {n_chunks}."
        )
    check_task, support_task, safety_task = template
    check_end, check_cache = _select_sc_checkin_end(
        chunks, check_task, latest_allowed_end=n_chunks - 2
    )
    check_end = min(max(1, check_end), n_chunks - 2)
    safety_intervals, safety_cache, safety_reason, safety_forced = (
        _select_one_or_two_sc_safety_intervals(
            chunks, safety_task, lower_bound=check_end
        )
    )
    if not safety_intervals:
        raise RuntimeError(
            "SC_FORCE_THREE_PARTS is enabled but no safety interval was produced."
        )
    covered_after_checkin = sum(
        (
            interval["end_chunk"] - interval["start_chunk"]
            for interval in safety_intervals
        )
    )
    if covered_after_checkin >= n_chunks - check_end:
        last_interval = safety_intervals[-1]
        last_interval["end_chunk"] = max(
            last_interval["start_chunk"] + 1, last_interval["end_chunk"] - 1
        )
    support_cache = build_task_cache(chunks, support_task)
    spans = [
        _make_chunk_span(
            check_task,
            0,
            check_end,
            chunks,
            check_cache,
            selection_reason="sc_opening_checkin",
            forced_assignment=False,
        )
    ]
    cursor = check_end
    safety_count = len(safety_intervals)
    for safety_number, interval in enumerate(safety_intervals, start=1):
        safety_start = interval["start_chunk"]
        safety_end = interval["end_chunk"]
        if cursor < safety_start:
            spans.append(
                _make_chunk_span(
                    support_task,
                    cursor,
                    safety_start,
                    chunks,
                    support_cache,
                    selection_reason="sc_residual_support",
                    forced_assignment=False,
                )
            )
        spans.append(
            _make_chunk_span(
                safety_task,
                safety_start,
                safety_end,
                chunks,
                safety_cache,
                selection_reason=safety_reason,
                forced_assignment=safety_forced,
                safety_segment_number=safety_number,
                safety_segment_count=safety_count,
            )
        )
        cursor = safety_end
    if cursor < n_chunks:
        spans.append(
            _make_chunk_span(
                support_task,
                cursor,
                n_chunks,
                chunks,
                support_cache,
                selection_reason="sc_residual_support",
                forced_assignment=False,
            )
        )
    spans = [span for span in spans if span["end_chunk"] > span["start_chunk"]]
    present = {span["task"] for span in spans}
    required = {check_task, support_task, safety_task}
    if present != required:
        missing = required - present
        raise RuntimeError(
            f"SC 2–22 segmentation is missing required tasks: {sorted(missing)}"
        )
    safety_span_count = sum((span["task"] == safety_task for span in spans))
    if not SC_SAFETY_MIN_SEGMENTS <= safety_span_count <= SC_SAFETY_MAX_SEGMENTS:
        raise RuntimeError(
            f"SC safety/attendance segment count is outside the allowed range: {safety_span_count}"
        )
    assigned = []
    for span in sorted(
        spans, key=lambda item: (item["start_chunk"], item["end_chunk"])
    ):
        assigned.extend(range(span["start_chunk"], span["end_chunk"]))
    if assigned != list(range(n_chunks)):
        raise RuntimeError("SC segmentation did not cover every chunk exactly once.")
    return sorted(spans, key=lambda span: (span["start_chunk"], span["end_chunk"]))


def assign_spans_to_chunks(chunks, spans):
    for segment_index, span in enumerate(spans, 1):
        for chunk_index in range(span["start_chunk"], span["end_chunk"]):
            chunk = chunks[chunk_index]
            chunk["task"] = span["task"]
            chunk["segment_index"] = segment_index
            chunk["boundary_score"] = span["boundary_score"]
            chunk["content_score"] = span["content_score"]
            chunk["boundary_hits"] = "; ".join(span["boundary_hits"])
            chunk["content_hits"] = "; ".join(span["content_hits"])
            chunk["selection_reason"] = span.get("selection_reason", "model_selected")
            chunk["forced_assignment"] = span.get("forced_assignment", False)
    return chunks


def process_session(turns, session, arm):
    template = get_session_template(session, arm)
    if not template:
        raise ValueError(f"No template for arm={arm}, session={session}")
    chunks = build_dialogue_chunks(turns)
    if not chunks:
        raise ValueError("No dialogue chunks found.")
    fallback_used = False
    omitted_tasks = []
    if arm == "sc" and 2 <= int(session) <= 22:
        spans = flexible_sc_segment(chunks, template)
        mode = "sc_flexible_utterance_matching_balanced_chunks"
    else:
        spans = ordered_dp_segment(chunks, template)
        mode = "ordered_dp_utterance_matching_balanced_chunks"
        if spans is None and FALLBACK_ALLOW_MISSING_TASKS:
            spans, omitted_tasks = ordered_dp_segment_allow_missing_tasks(
                chunks, template
            )
            if spans is not None:
                fallback_used = True
                mode = "ordered_dp_utterance_matching_fallback_allow_missing_tasks"
    if spans is None:
        raise ValueError("Unable to segment dialogue chunks even after fallback.")
    assign_spans_to_chunks(chunks, spans)
    rows = []
    evidence_rows = []
    for segment_index, span in enumerate(spans, 1):
        task = span["task"]
        first_chunk = chunks[span["start_chunk"]]
        last_chunk = chunks[span["end_chunk"] - 1]
        evidence_rows.append(
            {
                "segment_index": segment_index,
                "task": task,
                "task_label": TASK_META[task]["label"],
                "task_category": TASK_META[task]["category"],
                "start_chunk": span["start_chunk"],
                "end_chunk_exclusive": span["end_chunk"],
                "n_chunks": span["end_chunk"] - span["start_chunk"],
                "start_turn": first_chunk["start_turn_index"],
                "end_turn": last_chunk["end_turn_index"],
                "first_speaker": first_chunk["first_speaker"],
                "last_speaker": last_chunk["last_speaker"],
                "boundary_score": span["boundary_score"],
                "content_score": span["content_score"],
                "boundary_hits": json.dumps(span["boundary_hits"], ensure_ascii=False),
                "content_hits": json.dumps(span["content_hits"], ensure_ascii=False),
                "boundary_hit_details": json.dumps(
                    span["boundary_hit_details"], ensure_ascii=False
                ),
                "content_hit_details": json.dumps(
                    span["content_hit_details"], ensure_ascii=False
                ),
                "selection_reason": span.get("selection_reason", "model_selected"),
                "forced_assignment": span.get("forced_assignment", False),
                "safety_segment_number": span.get("safety_segment_number"),
                "safety_segment_count": span.get("safety_segment_count"),
                "fallback_used": fallback_used,
                "omitted_tasks": json.dumps(omitted_tasks, ensure_ascii=False),
            }
        )
        for chunk_index in range(span["start_chunk"], span["end_chunk"]):
            chunk = chunks[chunk_index]
            rows.append(
                {
                    "task": task,
                    "task_label": TASK_META[task]["label"],
                    "task_category": TASK_META[task]["category"],
                    "segment_index": segment_index,
                    "chunk_in_task": chunk_index - span["start_chunk"] + 1,
                    "chunk_global": chunk_index + 1,
                    "clinician_text": chunk["clinician_text"],
                    "participant_text": chunk["participant_text"],
                    "combined_text": chunk["combined_text"],
                    "n_turns": chunk["n_turns"],
                    "n_words": chunk["n_words"],
                    "split_from_oversized_chunk": chunk["split_from_oversized_chunk"],
                    "contains_single_oversized_utterance": chunk[
                        "contains_single_oversized_utterance"
                    ],
                    "original_raw_chunk_word_count": chunk[
                        "original_raw_chunk_word_count"
                    ],
                    "original_raw_chunk_turn_count": chunk[
                        "original_raw_chunk_turn_count"
                    ],
                    "original_speakers": json.dumps(
                        chunk["original_speakers"], ensure_ascii=False
                    ),
                    "start_turn_index": chunk["start_turn_index"],
                    "end_turn_index": chunk["end_turn_index"],
                    "n_clinician_turns": chunk["n_clinician_turns"],
                    "n_participant_turns": chunk["n_participant_turns"],
                    "first_speaker": chunk["first_speaker"],
                    "last_speaker": chunk["last_speaker"],
                    "first_original_speaker": chunk["first_original_speaker"],
                    "last_original_speaker": chunk["last_original_speaker"],
                    "boundary_score": span["boundary_score"],
                    "content_score": span["content_score"],
                    "boundary_hits": "; ".join(span["boundary_hits"]),
                    "content_hits": "; ".join(span["content_hits"]),
                    "selection_reason": span.get("selection_reason", "model_selected"),
                    "forced_assignment": span.get("forced_assignment", False),
                    "safety_segment_number": span.get("safety_segment_number"),
                    "safety_segment_count": span.get("safety_segment_count"),
                    "fallback_used": fallback_used,
                    "omitted_tasks": "; ".join(omitted_tasks),
                }
            )
    return (
        rows,
        evidence_rows,
        chunks,
        spans,
        mode,
        template,
        fallback_used,
        omitted_tasks,
    )


def write_annotated_session_txt(
    output_path,
    pid,
    session,
    arm,
    language,
    chunks,
    spans,
    mode,
    fallback_used=False,
    omitted_tasks=None,
):
    lines = [
        "=" * 100,
        f"ID: {pid}",
        f"Session: {session}",
        f"Arm: {arm}",
        f"Language: {language}",
        f"Segmentation mode: {mode}",
        f"Dialogue chunks: {len(chunks)}",
        f"Chunk word limit: {MAX_DIALOGUE_CHUNK_WORDS}",
        f"Oversized raw chunks split: {sum((chunk['split_from_oversized_chunk'] for chunk in chunks))}",
        f"Standalone oversized utterances: {sum((chunk['contains_single_oversized_utterance'] for chunk in chunks))}",
        "Cue matching unit: original utterance",
        f"Fallback used: {fallback_used}",
        "Omitted template tasks: "
        + (" | ".join(omitted_tasks) if omitted_tasks else "None"),
        "All original turns retained: True",
        "=" * 100,
        "",
    ]
    for segment_index, span in enumerate(spans, 1):
        task = span["task"]
        first_chunk = chunks[span["start_chunk"]]
        last_chunk = chunks[span["end_chunk"] - 1]
        lines.extend(
            [
                "#" * 100,
                f"TASK BLOCK {segment_index}",
                f"Task: {task}",
                f"Task label: {TASK_META[task]['label']}",
                f"Task category: {TASK_META[task]['category']}",
                f"Chunk range: {span['start_chunk']} to {span['end_chunk'] - 1}",
                f"Number of chunks: {span['end_chunk'] - span['start_chunk']}",
                f"First speaker: {first_chunk['first_speaker']}",
                f"Last speaker: {last_chunk['last_speaker']}",
                f"Boundary score: {span['boundary_score']:.4f}",
                f"Content score: {span['content_score']:.4f}",
                f"Selection reason: {span.get('selection_reason', 'model_selected')}",
                f"Forced assignment: {span.get('forced_assignment', False)}",
                "Safety segment: "
                + (
                    f"{span.get('safety_segment_number')}/{span.get('safety_segment_count')}"
                    if span.get("safety_segment_number") is not None
                    else "N/A"
                ),
                "",
                "MATCHED BOUNDARY CUES:",
            ]
        )
        if span["boundary_hit_details"]:
            for detail in span["boundary_hit_details"]:
                cue_index = detail["cue_index"]
                cue = (
                    TASK_META[task]["boundary_text"][cue_index]
                    if cue_index < len(TASK_META[task]["boundary_text"])
                    else str(cue_index)
                )
                similarity = detail.get("cosine_similarity")
                similarity_text = (
                    f"{similarity:.4f}" if similarity is not None else "N/A"
                )
                lines.extend(
                    [
                        f"  - Cue: {cue}",
                        f"    Chunk {detail['chunk_index']} | Match type: {detail.get('match_type', 'exact')}",
                        f"    Matched speaker/source: {detail.get('matched_speaker', 'Unknown')}",
                        f"    Matched turn index: {detail.get('matched_turn_index', 'N/A')}",
                        f"    Matched text: {detail.get('matched_text') or chunks[detail['chunk_index']]['combined_text']}",
                        f"    Cosine similarity: {similarity_text}",
                        f"    Match strength: {detail['strength']:.4f}",
                    ]
                )
        else:
            lines.append("  - None")
        lines.extend(["", "MATCHED CONTENT CUES:"])
        if span["content_hit_details"]:
            for detail in span["content_hit_details"]:
                cue_index = detail["cue_index"]
                cue = (
                    TASK_META[task]["content_text"][cue_index]
                    if cue_index < len(TASK_META[task]["content_text"])
                    else str(cue_index)
                )
                positions = ", ".join((str(x) for x in detail["chunk_indices"]))
                lines.append(f"  - Cue: {cue} | matched chunk(s): {positions}")
                for example in detail["examples"]:
                    similarity = example.get("cosine_similarity")
                    similarity_text = (
                        f"{similarity:.4f}" if similarity is not None else "N/A"
                    )
                    lines.append(
                        f"    Chunk {example['chunk_index']} | Match type: {example.get('match_type', 'exact')} | Cosine similarity: {similarity_text}"
                    )
                    if example.get("matched_text"):
                        lines.append(
                            f"    Matched {example.get('matched_speaker', 'source')} (turn {example.get('matched_turn_index', 'N/A')}): {example['matched_text']}"
                        )
                    if example["clinician_text"]:
                        lines.append(f"    Clinician: {example['clinician_text']}")
                    if example["participant_text"]:
                        lines.append(f"    Participant: {example['participant_text']}")
        else:
            lines.append("  - None")
        lines.extend(["", "ASSIGNED DIALOGUE CHUNKS:", "-" * 100])
        for chunk_index in range(span["start_chunk"], span["end_chunk"]):
            chunk = chunks[chunk_index]
            lines.append(
                f"[Chunk {chunk_index}] words={chunk['n_words']} | turns={chunk['n_turns']} | split_from_oversized={chunk['split_from_oversized_chunk']}"
            )
            for turn in chunk["turns"]:
                if turn.get("start") is not None and turn.get("end") is not None:
                    turn_timestamp = f"[{turn['start']:.2f} - {turn['end']:.2f}] "
                else:
                    turn_timestamp = ""
                lines.append(
                    f"  {turn_timestamp}{_turn_speaker_label(turn)}: {turn.get('text', '')}"
                )
            lines.append("")
        lines.append("")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def normalize_text(text):
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join((ch for ch in text if not unicodedata.combining(ch)))
    text = text.lower().replace("’", "'").replace("–", "-").replace("—", "-")
    text = re.sub("[^a-z0-9' ]+", " ", text)
    return re.sub("\\s+", " ", text).strip()


def parse_cue_list(value):
    if pd.isna(value) or str(value).strip() == "":
        return []
    s = str(value).strip()
    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            return [str(x).strip() for x in obj if str(x).strip()]
    except json.JSONDecodeError:
        pass
    out = []
    for x in s.splitlines():
        x = x.strip()
        if x.startswith("•"):
            x = x[1:].strip()
        if x:
            out.append(x)
    return out


def cue_to_regex(cue):
    """
    Convert a cue to a tolerant regular expression.
    The literal token "..." permits up to 120 intervening characters.
    """
    cue = normalize_text(cue)
    parts = [re.escape(x.strip()) for x in cue.split(" ... ") if x.strip()]
    body = ".{0,120}".join((p.replace("\\ ", "\\s+") for p in parts))
    return re.compile("(?<!\\w)" + body + "(?!\\w)", re.IGNORECASE)


def load_task_config(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Task configuration file does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xlsm"}:
        return pd.read_excel(path, sheet_name="module_tasks", engine="openpyxl")
    if suffix == ".xls":
        return pd.read_excel(path, sheet_name="module_tasks", engine="xlrd")
    raise ValueError(f"Unsupported task configuration format: {suffix}")


def get_session_template(session, arm):
    return SESSION_TEMPLATES.get((str(arm).lower(), int(session)))


def configure_tasks(path):
    global SESSION_TEMPLATES, TASK_META
    tasks_df = load_task_config(path)
    required = {
        "arm",
        "module",
        "task_order",
        "task",
        "task_label",
        "task_category",
        "task_count_in_module",
        "cue_boundary",
        "cue_words",
    }
    missing = required - set(tasks_df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    tasks_df["arm_key"] = (
        tasks_df["arm"]
        .astype(str)
        .str.lower()
        .replace(
            {
                "p-stair": "pstair",
                "parenting stair": "pstair",
                "supportive counseling": "sc",
            }
        )
    )
    tasks_df["module"] = pd.to_numeric(tasks_df["module"]).astype(int)
    tasks_df["task_order"] = pd.to_numeric(tasks_df["task_order"]).astype(int)
    tasks_df = tasks_df.sort_values(["arm_key", "module", "task_order"]).reset_index(
        drop=True
    )
    SESSION_TEMPLATES = {
        (arm, int(module)): group["task"].tolist()
        for (arm, module), group in tasks_df.groupby(["arm_key", "module"], sort=False)
    }
    boundary_df = Counter()
    content_df = Counter()
    parsed_rows = []
    for row in tasks_df.to_dict("records"):
        boundary_text = parse_cue_list(row["cue_boundary"])
        content_text = parse_cue_list(row["cue_words"])
        parsed_rows.append((row, boundary_text, content_text))
        boundary_df.update(set((normalize_text(x) for x in boundary_text)))
        content_df.update(set((normalize_text(x) for x in content_text)))
    n_tasks = max(len(tasks_df), 1)

    def cue_idf(cue, df_counter):
        frequency = df_counter.get(normalize_text(cue), 0)
        return math.log((n_tasks + 1) / (frequency + 1)) + 1.0

    TASK_META = {}
    for row, boundary_text, content_text in parsed_rows:
        TASK_META[row["task"]] = {
            "label": row["task_label"],
            "category": row["task_category"],
            "boundary_text": boundary_text,
            "content_text": content_text,
            "boundary": [cue_to_regex(x) for x in boundary_text],
            "content": [cue_to_regex(x) for x in content_text],
            "boundary_weights": np.array(
                [cue_idf(x, boundary_df) for x in boundary_text], dtype=float
            ),
            "content_weights": np.array(
                [cue_idf(x, content_df) for x in content_text], dtype=float
            ),
        }
