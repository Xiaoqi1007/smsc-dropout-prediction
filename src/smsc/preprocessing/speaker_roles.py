"""Rule-based speaker roles and transcript turn parsing."""

from collections import defaultdict
import json
import re

MAIN_SPEAKER_FLIP_BLOCK_INITIAL_RATIO_MAX = 0.8

MAIN_SPEAKER_FLIP_BLOCK_CONSECUTIVE_RATIO = 0.9

SPEAKER_LINE_PATTERN = re.compile(
    "\n    ^\\s*\n    (?P<timestamp>\\[\\s*[\\d.]+\\s*-\\s*[\\d.]+\\s*\\]\\s*)?\n    (?P<speaker>[A-Z])\\s*:\\s*\n    (?P<text>.*)\n    $\n    ",
    re.VERBOSE,
)

STRONG_CLINICIAN_PHRASES = [
    "share my screen",
    "can you see my screen",
    "i'm going to share my screen",
    "i am going to share my screen",
    "i'm going to ask you questions",
    "i am going to ask you questions",
    "i want to ask you",
    "i'd like to ask you",
    "last time we talked about",
    "last time we spoke",
    "since our last session",
    "since the last time we met",
    "let's start by",
    "let's begin with",
    "let's move on to",
    "before we finish",
    "before we wrap up",
    "i want to check in",
    "i'd like to check in",
    "i want to review",
    "i'd like to review",
    "today we're going to",
    "today we are going to",
    "in this treatment",
    "for homework",
    "your homework",
    "between now and our next session",
    "i'm going to send you the homework",
    "i'll send you the homework",
    "how are you feeling about your safety",
    "how do you feel about your safety",
    "do you feel safe",
    "are you feeling safe",
    "are you having any suicidal thoughts",
    "any thoughts about suicide",
    "any thoughts of hurting yourself",
    "any thoughts of harming yourself",
    "what time works for you",
    "does this time work for you",
    "are you available next week",
    "same time next week",
    "our next session",
    "next appointment",
    "i'll see you next week",
    "let me look at my calendar",
    "voy a compartir mi pantalla",
    "puede ver mi pantalla",
    "le voy a hacer unas preguntas",
    "voy a hacerle unas preguntas",
    "la última vez hablamos",
    "desde nuestra última sesión",
    "vamos a empezar",
    "vamos a revisar",
    "antes de terminar",
    "cómo se siente acerca de su seguridad",
    "ha tenido pensamientos suicidas",
    "qué hora le funciona",
    "puede asistir la próxima semana",
    "nos vemos la próxima semana",
]

WEAK_CLINICIAN_REGEX_PATTERNS = [
    "\\bwhat is your\\b",
    "\\bwhat's your\\b",
    "\\bhow often\\b",
    "\\bwhen did\\b",
    "\\bwhen was\\b",
    "\\bhow long\\b",
    "\\bhow many\\b",
    "\\bhow much\\b",
    "\\bhow have you been\\b",
    "\\bhow was your week\\b",
    "\\bwhat happened after that\\b",
    "\\bwhat do you think\\b",
    "\\bhow do you feel\\b",
    "\\bhow are you feeling\\b",
    "\\bwhat makes you\\b",
    "\\bwhat would you like\\b",
    "\\bcan you tell me\\b",
    "\\bcould you tell me\\b",
    "\\bhave you talked to\\b",
    "\\bdoes that sound familiar\\b",
    "\\bdo you remember\\b",
    "\\btell me more\\b",
    "\\bit sounds like\\b",
    "\\bsounds like\\b",
    "\\bwhat are you hoping\\b",
    "\\bwhat do you mean\\b",
    "\\bcuántas veces\\b",
    "\\bcuándo pasó\\b",
    "\\bcuánto tiempo\\b",
    "\\bcómo se siente\\b",
    "\\bqué piensa\\b",
    "\\bpuede contarme\\b",
]

STRONG_PARTICIPANT_PHRASES = [
    "my case worker",
    "my caseworker",
    "my case planner",
    "my therapist",
    "when i was a child",
    "when i was younger",
    "when i was little",
    "when i was growing up",
    "what happened to me",
    "when that happened to me",
    "i was abused",
    "i went through",
    "i felt scared",
    "i don't feel safe",
    "i do not feel safe",
    "i feel overwhelmed",
    "i have trouble sleeping",
    "mi hija",
    "mi hijo",
    "mis hijos",
    "mi esposo",
    "mi novio",
    "mi novia",
    "mi pareja",
    "mi familia",
    "mi madre",
    "mi mamá",
    "mi padre",
    "mi papá",
    "mi trabajo",
    "mi jefe",
    "cuando era niña",
    "cuando era niño",
    "me sentí asustada",
    "no me siento segura",
    "no me siento seguro",
]

MEDIUM_PARTICIPANT_PHRASES = [
    "my daughter",
    "my son",
    "my child",
    "my children",
    "my kids",
    "my husband",
    "my boyfriend",
    "my girlfriend",
    "my partner",
    "my family",
    "my mother",
    "my mom",
    "my father",
    "my dad",
    "my job",
    "my boss",
    "at my job",
]

WEAK_PARTICIPANT_REGEX_PATTERNS = [
    "\\bi feel\\b",
    "\\bi felt\\b",
    "\\bi went\\b",
    "\\bi had\\b",
    "\\bi was\\b",
    "\\bwhen i was\\b",
    "\\bwhat happened to me\\b",
    "\\bmy (daughter|son|child|children|kids|family|job|boss)\\b",
    "\\bme siento\\b",
    "\\bme sentí\\b",
    "\\byo fui\\b",
    "\\byo tenía\\b",
    "\\bcuando yo era\\b",
    "\\bmi (hija|hijo|familia|trabajo|jefe)\\b",
]

BACKCHANNEL_TURN_RE = re.compile(
    "^(?:(?:hi|hello|hey|good (?:morning|afternoon|evening)|hola|buenos dias|buenas tardes|how are you(?: doing| feeling)?|i'?m (?:good|fine|okay|ok|well)|and you|okay|ok|yeah|yes|no|yep|mm-?hmm+|uh-?huh|wow|right|exactly|sure|thank you|thanks|you'?re welcome|gracias|de nada|bye|bye-?bye|good|great|all right|alright)[\\s.,!?-]*)+$",
    re.I,
)

BACKCHANNEL_MAX_WORDS = 6


def is_backchannel_turn(text):
    text = str(text).strip()
    if not text:
        return True
    if count_words(text) > BACKCHANNEL_MAX_WORDS:
        return False
    return bool(BACKCHANNEL_TURN_RE.match(text))


GREETING_NAME_RE = re.compile(
    "\\b(?:hi|hello|good (?:morning|afternoon|evening)|bye(?:-bye)?|hola|buenos dias|buenas tardes|adios)[,.!]?\\s+(?:ms|mrs|miss|mr|señora|señor|sra|sr)\\.?\\s+\\w+",
    re.I,
)

FIRST_PERSON_REGEX = re.compile(
    "\\b(i|i'm|im|i’ve|ive|i've|i'd|id|i’ll|i'll|ill|me|my|mine|myself|yo|mi|mis|mío|mía|míos|mías)\\b",
    re.I,
)

SECOND_PERSON_REGEX = re.compile(
    "\\b(you|you're|youre|you’ve|youve|you've|you'd|youd|you’ll|you'll|youll|your|yours|yourself|usted|ustedes|tú|tu|te|ti|su|sus)\\b",
    re.I,
)

WEAK_CLINICIAN_REGEX = [
    re.compile(pattern, re.I) for pattern in WEAK_CLINICIAN_REGEX_PATTERNS
]

WEAK_PARTICIPANT_REGEX = [
    re.compile(pattern, re.I) for pattern in WEAK_PARTICIPANT_REGEX_PATTERNS
]

PRONOUN_RATIO_WEIGHT = 5.0

WEAK_TURN_RATIO_WEIGHT = 3.0

MINOR_ADJACENCY_WEIGHT = 3.0

MINOR_ADJACENCY_MIN_NEIGHBORS = 6

PRONOUN_MIN_INFORMATIVE_WORDS = 30

ANCHOR_LOW_CONFIDENCE_MARGIN = 1.0

ANCHOR_HIGH_CONFIDENCE_MARGIN = 3.0

VERY_STRONG_PARTICIPANT_RATE = 3

VERY_STRONG_CLINICIAN_RATE = 3

PARTICIPANT_OVERRIDE_RATIO = 2.5

CLINICIAN_OVERRIDE_RATIO = 2.5

MIN_FINAL_ROLE_MARGIN = 1.0

MINOR_LOW_VOLUME_WORDS = 30

MINOR_LOW_VOLUME_TURNS = 5


def parse_turns_from_transcript(text):
    turns = []
    current_speaker = None
    current_timestamp = ""
    current_text_parts = []
    current_raw_parts = []
    for raw_line in str(text).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = SPEAKER_LINE_PATTERN.match(line)
        if match:
            if current_speaker is not None:
                turns.append(
                    {
                        "speaker": current_speaker,
                        "timestamp": current_timestamp,
                        "text": " ".join(current_text_parts).strip(),
                        "raw_line": " ".join(current_raw_parts).strip(),
                    }
                )
            current_speaker = match.group("speaker")
            current_timestamp = (match.group("timestamp") or "").strip()
            current_text_parts = [match.group("text").strip()]
            current_raw_parts = [line]
        elif current_speaker is not None:
            current_text_parts.append(line)
            current_raw_parts.append(line)
    if current_speaker is not None:
        turns.append(
            {
                "speaker": current_speaker,
                "timestamp": current_timestamp,
                "text": " ".join(current_text_parts).strip(),
                "raw_line": " ".join(current_raw_parts).strip(),
            }
        )
    return turns


def count_words(text):
    return len(re.findall("\\b[\\w’'-]+\\b", str(text), flags=re.UNICODE))


def compile_phrase_pattern(phrases):
    """Compile one longest-first alternation pattern."""
    unique_phrases = sorted(
        {str(phrase).strip().lower() for phrase in phrases if str(phrase).strip()},
        key=len,
        reverse=True,
    )
    if not unique_phrases:
        return None
    alternatives = "|".join((re.escape(phrase) for phrase in unique_phrases))
    return re.compile(f"(?<!\\w)(?:{alternatives})(?!\\w)", re.I)


STRONG_CLINICIAN_PATTERN = compile_phrase_pattern(STRONG_CLINICIAN_PHRASES)

STRONG_PARTICIPANT_PATTERN = compile_phrase_pattern(STRONG_PARTICIPANT_PHRASES)

MEDIUM_PARTICIPANT_PATTERN = compile_phrase_pattern(MEDIUM_PARTICIPANT_PHRASES)


def find_phrase_occurrences(text, compiled_pattern):
    """Return all non-overlapping strong-cue occurrences in text."""
    if compiled_pattern is None:
        return []
    return [match.group(0).lower() for match in compiled_pattern.finditer(str(text))]


def strong_evidence_points(hit_count):
    """Score strong lexical cues using cumulative hit thresholds."""
    hit_count = int(hit_count)
    if hit_count <= 0:
        return 0.0
    if hit_count == 1:
        return 1.0
    if hit_count == 2:
        return 5.0
    return 10.0


def turn_matches_any_regex(text, regexes):
    return any((regex.search(str(text)) for regex in regexes))


def compute_speaker_evidence(turns):
    """Calculate clinician and participant lexical evidence for each anonymous speaker."""
    grouped = defaultdict(list)
    for turn in turns:
        grouped[turn["speaker"]].append(turn)
    stats = {}
    for speaker, speaker_turns in grouped.items():
        texts = [str(turn.get("text", "")) for turn in speaker_turns]
        all_text = " ".join(texts)
        n_turns = len(speaker_turns)
        words = sum((count_words(text) for text in texts))
        chars = sum((len(text) for text in texts))
        clinician_strong_hits = find_phrase_occurrences(
            all_text, STRONG_CLINICIAN_PATTERN
        )
        participant_strong_hits = find_phrase_occurrences(
            all_text, STRONG_PARTICIPANT_PATTERN
        )
        participant_medium_hits = find_phrase_occurrences(
            all_text, MEDIUM_PARTICIPANT_PATTERN
        )
        clinician_strong_hit_count = len(clinician_strong_hits)
        participant_strong_hit_count = len(participant_strong_hits)
        clinician_strong_points = strong_evidence_points(clinician_strong_hit_count)
        participant_strong_points = strong_evidence_points(participant_strong_hit_count)
        participant_medium_hit_count = len(participant_medium_hits)
        participant_medium_points = float(participant_medium_hit_count)
        informative_texts = [text for text in texts if not is_backchannel_turn(text)]
        informative_turn_count = len(informative_texts)
        informative_words = sum((count_words(text) for text in informative_texts))
        informative_all_text = " ".join(informative_texts)
        clinician_greeting_name_hits = GREETING_NAME_RE.findall(all_text)
        clinician_medium_hit_count = len(clinician_greeting_name_hits)
        clinician_medium_points = float(clinician_medium_hit_count)
        clinician_weak_turn_count = sum(
            (
                turn_matches_any_regex(text, WEAK_CLINICIAN_REGEX)
                for text in informative_texts
            )
        )
        participant_weak_turn_count = sum(
            (
                turn_matches_any_regex(text, WEAK_PARTICIPANT_REGEX)
                for text in informative_texts
            )
        )
        clinician_weak_turn_ratio = clinician_weak_turn_count / max(
            informative_turn_count, 1
        )
        participant_weak_turn_ratio = participant_weak_turn_count / max(
            informative_turn_count, 1
        )
        first_person_count = len(FIRST_PERSON_REGEX.findall(informative_all_text))
        second_person_count = len(SECOND_PERSON_REGEX.findall(informative_all_text))
        pronoun_total = first_person_count + second_person_count
        if pronoun_total > 0:
            first_person_ratio = first_person_count / pronoun_total
            second_person_ratio = second_person_count / pronoun_total
        else:
            first_person_ratio = 0.0
            second_person_ratio = 0.0
        pronoun_volume_scale = min(
            1.0, informative_words / max(PRONOUN_MIN_INFORMATIVE_WORDS, 1)
        )
        clinician_evidence = (
            clinician_strong_points
            + clinician_medium_points
            + PRONOUN_RATIO_WEIGHT * pronoun_volume_scale * second_person_ratio
            + WEAK_TURN_RATIO_WEIGHT * clinician_weak_turn_ratio
        )
        participant_evidence = (
            participant_strong_points
            + participant_medium_points
            + PRONOUN_RATIO_WEIGHT * pronoun_volume_scale * first_person_ratio
            + WEAK_TURN_RATIO_WEIGHT * participant_weak_turn_ratio
        )
        role_score = clinician_evidence - participant_evidence
        stats[speaker] = {
            "turns": int(n_turns),
            "words": int(words),
            "chars": int(chars),
            "clinician_strong_hit_count": int(clinician_strong_hit_count),
            "participant_strong_hit_count": int(participant_strong_hit_count),
            "clinician_strong_points": float(clinician_strong_points),
            "participant_strong_points": float(participant_strong_points),
            "participant_medium_hit_count": int(participant_medium_hit_count),
            "participant_medium_points": float(participant_medium_points),
            "participant_medium_hits": participant_medium_hits,
            "clinician_medium_hit_count": int(clinician_medium_hit_count),
            "clinician_medium_points": float(clinician_medium_points),
            "informative_turns": int(informative_turn_count),
            "informative_words": int(informative_words),
            "pronoun_volume_scale": float(pronoun_volume_scale),
            "clinician_strong_hits": clinician_strong_hits,
            "participant_strong_hits": participant_strong_hits,
            "clinician_strong_indicator": int(clinician_strong_hit_count > 0),
            "participant_strong_indicator": int(participant_strong_hit_count > 0),
            "first_person_count": int(first_person_count),
            "second_person_count": int(second_person_count),
            "first_person_ratio": float(first_person_ratio),
            "second_person_ratio": float(second_person_ratio),
            "clinician_weak_turn_count": int(clinician_weak_turn_count),
            "participant_weak_turn_count": int(participant_weak_turn_count),
            "clinician_weak_turn_ratio": float(clinician_weak_turn_ratio),
            "participant_weak_turn_ratio": float(participant_weak_turn_ratio),
            "clinician_evidence_score": float(clinician_evidence),
            "participant_evidence_score": float(participant_evidence),
            "self_role_score": float(role_score),
        }
    return stats


def choose_two_main_anchors(stats):
    ranked = sorted(
        stats,
        key=lambda speaker: (
            stats[speaker]["words"],
            stats[speaker]["chars"],
            stats[speaker]["turns"],
            speaker,
        ),
        reverse=True,
    )
    if not ranked:
        return (None, None, [], 0.0, "low")
    if len(ranked) == 1:
        return (ranked[0], None, ranked, 0.0, "low")
    top_two = ranked[:2]
    clinician_anchor = max(
        top_two, key=lambda speaker: stats[speaker]["self_role_score"]
    )
    participant_anchor = min(
        top_two, key=lambda speaker: stats[speaker]["self_role_score"]
    )
    margin = abs(
        stats[clinician_anchor]["self_role_score"]
        - stats[participant_anchor]["self_role_score"]
    )
    if margin >= ANCHOR_HIGH_CONFIDENCE_MARGIN:
        confidence = "high"
    elif margin >= ANCHOR_LOW_CONFIDENCE_MARGIN:
        confidence = "medium"
    else:
        confidence = "low"
    return (clinician_anchor, participant_anchor, top_two, float(margin), confidence)


def strong_contradictory_override(initial_role, speaker_stats):
    C = float(speaker_stats["clinician_evidence_score"])
    P = float(speaker_stats["participant_evidence_score"])
    if (
        initial_role == "Clinician"
        and P >= VERY_STRONG_PARTICIPANT_RATE
        and (P >= PARTICIPANT_OVERRIDE_RATIO * max(C, 1e-12))
    ):
        return ("Participant", True, "participant_evidence_override")
    if (
        initial_role == "Participant"
        and C >= VERY_STRONG_CLINICIAN_RATE
        and (C >= CLINICIAN_OVERRIDE_RATIO * max(P, 1e-12))
    ):
        return ("Clinician", True, "clinician_evidence_override")
    return (initial_role, False, "")


def compute_consecutive_dialogue_ratio(turns, role_map):
    """Return the proportion of adjacent dialogue-turn pairs assigned the same role."""
    role_sequence = []
    for turn in turns:
        speaker = turn["speaker"]
        if speaker not in role_map:
            raise KeyError(f"Speaker {speaker!r} is missing from the role map.")
        role = role_map[speaker]
        if role not in {"Clinician", "Participant"}:
            raise ValueError(f"Unexpected role {role!r} for speaker {speaker!r}.")
        role_sequence.append(role)
    total_adjacent_pairs = max(len(role_sequence) - 1, 0)
    if total_adjacent_pairs == 0:
        return 0.0
    same_role_pairs = sum(
        (
            previous_role == current_role
            for previous_role, current_role in zip(role_sequence, role_sequence[1:])
        )
    )
    return float(same_role_pairs / total_adjacent_pairs)


def apply_contradictory_overrides_with_role_guard(
    initial_role_map, speaker_stats, turns, main_speakers
):
    """Evaluate contradictory-evidence overrides and apply two guards."""
    main_speakers = set(main_speakers)
    tentative_role_map = {}
    override_details = {}
    for speaker, initial_role in initial_role_map.items():
        proposed_role, proposed, reason = strong_contradictory_override(
            initial_role, speaker_stats[speaker]
        )
        tentative_role_map[speaker] = proposed_role
        override_details[speaker] = {
            "initial_role": initial_role,
            "proposed_role": proposed_role,
            "override_proposed": bool(proposed),
            "override_reason": reason,
            "override_applied": bool(proposed),
            "override_blocked_high_consecutive_ratio": False,
            "override_blocked_all_same_role": False,
            "is_main_speaker": speaker in main_speakers,
        }
    initial_consecutive_ratio = compute_consecutive_dialogue_ratio(
        turns, initial_role_map
    )
    tentative_consecutive_ratio = compute_consecutive_dialogue_ratio(
        turns, tentative_role_map
    )
    proposed_main_speaker_overrides = [
        speaker
        for speaker, details in override_details.items()
        if details["override_proposed"] and details["is_main_speaker"]
    ]
    consecutive_ratio_guard_triggered = (
        initial_consecutive_ratio < MAIN_SPEAKER_FLIP_BLOCK_INITIAL_RATIO_MAX
        and tentative_consecutive_ratio > MAIN_SPEAKER_FLIP_BLOCK_CONSECUTIVE_RATIO
        and bool(proposed_main_speaker_overrides)
    )
    if consecutive_ratio_guard_triggered:
        for speaker in proposed_main_speaker_overrides:
            tentative_role_map[speaker] = initial_role_map[speaker]
            override_details[speaker]["override_applied"] = False
            override_details[speaker]["override_blocked_high_consecutive_ratio"] = True
    post_main_speaker_guard_ratio = compute_consecutive_dialogue_ratio(
        turns, tentative_role_map
    )
    initial_roles = set(initial_role_map.values())
    tentative_roles = set(tentative_role_map.values())
    any_override_still_active = any(
        (details["override_applied"] for details in override_details.values())
    )
    all_same_role_guard_triggered = (
        len(initial_roles) >= 2
        and len(tentative_roles) == 1
        and any_override_still_active
    )
    if all_same_role_guard_triggered:
        final_role_map = initial_role_map.copy()
        for details in override_details.values():
            if details["override_applied"]:
                details["override_applied"] = False
                details["override_blocked_all_same_role"] = True
    else:
        final_role_map = tentative_role_map
    final_consecutive_ratio = compute_consecutive_dialogue_ratio(turns, final_role_map)
    ratio_diagnostics = {
        "initial_consecutive_dialogue_ratio": initial_consecutive_ratio,
        "tentative_consecutive_dialogue_ratio": tentative_consecutive_ratio,
        "post_main_speaker_guard_consecutive_dialogue_ratio": post_main_speaker_guard_ratio,
        "final_consecutive_dialogue_ratio": final_consecutive_ratio,
        "main_speaker_flip_block_initial_ratio_max": MAIN_SPEAKER_FLIP_BLOCK_INITIAL_RATIO_MAX,
        "main_speaker_flip_block_threshold": MAIN_SPEAKER_FLIP_BLOCK_CONSECUTIVE_RATIO,
    }
    return (
        final_role_map,
        override_details,
        all_same_role_guard_triggered,
        consecutive_ratio_guard_triggered,
        ratio_diagnostics,
    )


def assign_speakers_to_roles(turns):
    """Assign roles from lexical evidence, speaker anchors, and turn consistency."""
    stats = compute_speaker_evidence(turns)
    if not stats:
        diagnostics = {
            "clinician_anchor": None,
            "participant_anchor": None,
            "anchor_score_margin": 0.0,
            "anchor_confidence": "low",
            "needs_manual_review": False,
            "critical_review_reasons": [],
            "critical_review_rows": [],
            "has_minor_speaker_low_evidence": False,
            "minor_speaker_low_evidence_reasons": [],
            "minor_speaker_low_evidence_rows": [],
            "speaker_stats": {},
            "minor_speakers": [],
            "minor_assignment_details": {},
            "role_map_before_override": {},
            "role_map_after_override": {},
            "override_details": {},
            "override_guard_triggered": False,
            "consecutive_ratio_guard_triggered": False,
            "blocked_override_speakers": [],
            "blocked_all_same_role_speakers": [],
            "blocked_high_consecutive_ratio_speakers": [],
            "initial_consecutive_dialogue_ratio": 0.0,
            "tentative_consecutive_dialogue_ratio": 0.0,
            "post_main_speaker_guard_consecutive_dialogue_ratio": 0.0,
            "final_consecutive_dialogue_ratio": 0.0,
            "main_speaker_flip_block_initial_ratio_max": MAIN_SPEAKER_FLIP_BLOCK_INITIAL_RATIO_MAX,
            "main_speaker_flip_block_threshold": MAIN_SPEAKER_FLIP_BLOCK_CONSECUTIVE_RATIO,
        }
        return ({}, stats, diagnostics, "no_speaker_turns_found")
    clinician_anchor, participant_anchor, anchors, anchor_margin, anchor_confidence = (
        choose_two_main_anchors(stats)
    )
    if participant_anchor is None:
        only_speaker = clinician_anchor
        initial_role_map = {
            only_speaker: (
                "Clinician"
                if stats[only_speaker]["self_role_score"] >= 0
                else "Participant"
            )
        }
        final_role_map = initial_role_map.copy()
        diagnostics = {
            "clinician_anchor": (
                only_speaker if final_role_map[only_speaker] == "Clinician" else None
            ),
            "participant_anchor": (
                only_speaker if final_role_map[only_speaker] == "Participant" else None
            ),
            "anchor_score_margin": 0.0,
            "anchor_confidence": "low",
            "needs_manual_review": True,
            "critical_review_reasons": ["single_speaker_file"],
            "critical_review_rows": [
                {
                    "speaker": only_speaker,
                    "initial_role": initial_role_map[only_speaker],
                    "final_role": final_role_map[only_speaker],
                    "review_reason": "single_speaker_file",
                    "is_anchor": True,
                    "self_role_score": stats[only_speaker]["self_role_score"],
                    "clinician_evidence_score": stats[only_speaker][
                        "clinician_evidence_score"
                    ],
                    "participant_evidence_score": stats[only_speaker][
                        "participant_evidence_score"
                    ],
                    "words": stats[only_speaker]["words"],
                    "turns": stats[only_speaker]["turns"],
                }
            ],
            "has_minor_speaker_low_evidence": False,
            "minor_speaker_low_evidence_reasons": [],
            "minor_speaker_low_evidence_rows": [],
            "speaker_stats": stats,
            "minor_speakers": [],
            "minor_assignment_details": {},
            "role_map_before_override": initial_role_map.copy(),
            "role_map_after_override": final_role_map.copy(),
            "override_details": {},
            "override_guard_triggered": False,
            "consecutive_ratio_guard_triggered": False,
            "blocked_override_speakers": [],
            "blocked_all_same_role_speakers": [],
            "blocked_high_consecutive_ratio_speakers": [],
            "initial_consecutive_dialogue_ratio": 0.0,
            "tentative_consecutive_dialogue_ratio": 0.0,
            "post_main_speaker_guard_consecutive_dialogue_ratio": 0.0,
            "final_consecutive_dialogue_ratio": 0.0,
            "main_speaker_flip_block_initial_ratio_max": MAIN_SPEAKER_FLIP_BLOCK_INITIAL_RATIO_MAX,
            "main_speaker_flip_block_threshold": MAIN_SPEAKER_FLIP_BLOCK_CONSECUTIVE_RATIO,
        }
        return (final_role_map, stats, diagnostics, "single_speaker_sign_rule")
    initial_role_map = {
        clinician_anchor: "Clinician",
        participant_anchor: "Participant",
    }
    minor_speakers = [speaker for speaker in stats if speaker not in initial_role_map]
    minor_details = {}
    speaker_sequence = [turn["speaker"] for turn in turns]
    adjacency_counts = {
        speaker: {"clinician_anchor": 0, "participant_anchor": 0}
        for speaker in minor_speakers
    }
    for index, current_speaker in enumerate(speaker_sequence):
        if current_speaker not in adjacency_counts:
            continue
        for neighbor_index in (index - 1, index + 1):
            if not 0 <= neighbor_index < len(speaker_sequence):
                continue
            neighbor = speaker_sequence[neighbor_index]
            if neighbor == clinician_anchor:
                adjacency_counts[current_speaker]["clinician_anchor"] += 1
            elif neighbor == participant_anchor:
                adjacency_counts[current_speaker]["participant_anchor"] += 1
    for speaker in minor_speakers:
        R = float(stats[speaker]["self_role_score"])
        counts = adjacency_counts[speaker]
        total_anchor_neighbors = (
            counts["clinician_anchor"] + counts["participant_anchor"]
        )
        if total_anchor_neighbors >= MINOR_ADJACENCY_MIN_NEIGHBORS:
            adjacency_balance = (
                counts["participant_anchor"] - counts["clinician_anchor"]
            ) / total_anchor_neighbors
        else:
            adjacency_balance = 0.0
        adjusted_score = R + MINOR_ADJACENCY_WEIGHT * adjacency_balance
        initial_role = "Clinician" if adjusted_score >= 0 else "Participant"
        initial_role_map[speaker] = initial_role
        minor_details[speaker] = {
            "self_role_score": R,
            "adjacency_balance": float(adjacency_balance),
            "adjacency_adjusted_score": float(adjusted_score),
            "initial_role": initial_role,
            "words": int(stats[speaker]["words"]),
            "turns": int(stats[speaker]["turns"]),
        }
    (
        final_role_map,
        override_details,
        override_guard_triggered,
        consecutive_ratio_guard_triggered,
        consecutive_ratio_diagnostics,
    ) = apply_contradictory_overrides_with_role_guard(
        initial_role_map, stats, turns, main_speakers=anchors
    )
    blocked_all_same_role_speakers = [
        speaker
        for speaker, details in override_details.items()
        if details["override_blocked_all_same_role"]
    ]
    blocked_high_consecutive_ratio_speakers = [
        speaker
        for speaker, details in override_details.items()
        if details["override_blocked_high_consecutive_ratio"]
    ]
    blocked_override_speakers = sorted(
        set(blocked_all_same_role_speakers + blocked_high_consecutive_ratio_speakers)
    )
    critical_review_reasons = []
    critical_review_rows = []
    minor_review_reasons = []
    minor_review_rows = []
    if override_guard_triggered:
        critical_review_reasons.append("contradictory_override_blocked_all_same_role")
        critical_review_rows.append(
            {
                "speaker": "__OVERRIDE_GUARD__",
                "initial_role": json.dumps(initial_role_map, ensure_ascii=False),
                "final_role": json.dumps(final_role_map, ensure_ascii=False),
                "review_reason": "contradictory_override_blocked_all_same_role",
                "is_anchor": True,
                "self_role_score": None,
                "clinician_evidence_score": None,
                "participant_evidence_score": None,
                "words": None,
                "turns": None,
            }
        )
    if consecutive_ratio_guard_triggered:
        critical_review_reasons.append(
            "main_speaker_override_blocked_high_consecutive_ratio"
        )
        critical_review_rows.append(
            {
                "speaker": "__CONSECUTIVE_RATIO_GUARD__",
                "initial_role": json.dumps(initial_role_map, ensure_ascii=False),
                "final_role": json.dumps(final_role_map, ensure_ascii=False),
                "review_reason": "main_speaker_override_blocked_low_to_high_consecutive_ratio",
                "is_anchor": True,
                "self_role_score": None,
                "clinician_evidence_score": None,
                "participant_evidence_score": None,
                "words": None,
                "turns": None,
                "initial_consecutive_dialogue_ratio": consecutive_ratio_diagnostics[
                    "initial_consecutive_dialogue_ratio"
                ],
                "tentative_consecutive_dialogue_ratio": consecutive_ratio_diagnostics[
                    "tentative_consecutive_dialogue_ratio"
                ],
                "main_speaker_flip_block_initial_ratio_max": MAIN_SPEAKER_FLIP_BLOCK_INITIAL_RATIO_MAX,
                "main_speaker_flip_block_threshold": MAIN_SPEAKER_FLIP_BLOCK_CONSECUTIVE_RATIO,
                "blocked_speakers": blocked_high_consecutive_ratio_speakers,
            }
        )
    for speaker, initial_role in initial_role_map.items():
        override_detail = override_details[speaker]
        final_role = final_role_map[speaker]
        overridden = override_detail["override_applied"]
        override_blocked = override_detail["override_blocked_all_same_role"]
        override_blocked_high_consecutive = override_detail[
            "override_blocked_high_consecutive_ratio"
        ]
        override_reason = override_detail["override_reason"]
        is_anchor = speaker in anchors
        if is_anchor:
            reasons = []
            if anchor_confidence == "low":
                reasons.append("low_anchor_margin")
            if overridden:
                reasons.append(f"anchor_{override_reason}")
            if override_blocked:
                reasons.append(f"anchor_{override_reason}_blocked_all_same_role")
            if override_blocked_high_consecutive:
                reasons.append(
                    f"anchor_{override_reason}_blocked_high_consecutive_ratio"
                )
            if reasons:
                critical_review_reasons.extend(reasons)
                critical_review_rows.append(
                    {
                        "speaker": speaker,
                        "initial_role": initial_role,
                        "final_role": final_role,
                        "review_reason": "; ".join(sorted(set(reasons))),
                        "is_anchor": True,
                        "self_role_score": stats[speaker]["self_role_score"],
                        "clinician_evidence_score": stats[speaker][
                            "clinician_evidence_score"
                        ],
                        "participant_evidence_score": stats[speaker][
                            "participant_evidence_score"
                        ],
                        "words": stats[speaker]["words"],
                        "turns": stats[speaker]["turns"],
                    }
                )
        else:
            reasons = []
            if abs(stats[speaker]["self_role_score"]) < MIN_FINAL_ROLE_MARGIN:
                reasons.append("weak_self_role_margin")
            if (
                stats[speaker]["words"] < MINOR_LOW_VOLUME_WORDS
                or stats[speaker]["turns"] < MINOR_LOW_VOLUME_TURNS
            ):
                reasons.append("low_volume_minor_speaker")
            if overridden:
                reasons.append(override_reason)
            if override_blocked:
                reasons.append(f"{override_reason}_blocked_all_same_role")
            if override_blocked_high_consecutive:
                reasons.append(f"{override_reason}_blocked_high_consecutive_ratio")
            if reasons:
                minor_review_reasons.extend(reasons)
                minor_review_rows.append(
                    {
                        "speaker": speaker,
                        "initial_role": initial_role,
                        "final_role": final_role,
                        "review_reason": "; ".join(sorted(set(reasons))),
                        "is_anchor": False,
                        "self_role_score": stats[speaker]["self_role_score"],
                        "clinician_evidence_score": stats[speaker][
                            "clinician_evidence_score"
                        ],
                        "participant_evidence_score": stats[speaker][
                            "participant_evidence_score"
                        ],
                        "words": stats[speaker]["words"],
                        "turns": stats[speaker]["turns"],
                    }
                )
    if final_role_map.get(clinician_anchor) == final_role_map.get(participant_anchor):
        critical_review_reasons.append("anchor_roles_converged_after_override")
        critical_review_rows.append(
            {
                "speaker": "__ANCHOR_PAIR__",
                "initial_role": f"{clinician_anchor}=Clinician; {participant_anchor}=Participant",
                "final_role": f"{clinician_anchor}={final_role_map.get(clinician_anchor)}; {participant_anchor}={final_role_map.get(participant_anchor)}",
                "review_reason": "anchor_roles_converged_after_override",
                "is_anchor": True,
                "self_role_score": None,
                "clinician_evidence_score": None,
                "participant_evidence_score": None,
                "words": None,
                "turns": None,
            }
        )
    diagnostics = {
        "clinician_anchor": clinician_anchor,
        "participant_anchor": participant_anchor,
        "anchor_score_margin": float(anchor_margin),
        "anchor_confidence": anchor_confidence,
        "needs_manual_review": bool(critical_review_rows),
        "critical_review_reasons": sorted(set(critical_review_reasons)),
        "critical_review_rows": critical_review_rows,
        "has_minor_speaker_low_evidence": bool(minor_review_rows),
        "minor_speaker_low_evidence_reasons": sorted(set(minor_review_reasons)),
        "minor_speaker_low_evidence_rows": minor_review_rows,
        "speaker_stats": stats,
        "minor_speakers": minor_speakers,
        "minor_assignment_details": minor_details,
        "role_map_before_override": initial_role_map.copy(),
        "role_map_after_override": final_role_map.copy(),
        "override_details": override_details,
        "override_guard_triggered": override_guard_triggered,
        "consecutive_ratio_guard_triggered": consecutive_ratio_guard_triggered,
        "blocked_override_speakers": blocked_override_speakers,
        "blocked_all_same_role_speakers": blocked_all_same_role_speakers,
        "blocked_high_consecutive_ratio_speakers": blocked_high_consecutive_ratio_speakers,
        **consecutive_ratio_diagnostics,
    }
    decision_rule = (
        "two_anchor_Rscore_plus_contradictory_override"
        + (
            "_with_high_consecutive_main_speaker_guard_triggered"
            if consecutive_ratio_guard_triggered
            else "_with_high_consecutive_main_speaker_guard"
        )
        + (
            "_with_all_same_role_guard_triggered"
            if override_guard_triggered
            else "_with_all_same_role_guard"
        )
    )
    return (final_role_map, stats, diagnostics, decision_rule)


def rewrite_transcript_as_clinician_participant(text, role_map):
    output_lines = []
    for raw_line in str(text).splitlines():
        line = raw_line.strip()
        if not line:
            output_lines.append("")
            continue
        match = SPEAKER_LINE_PATTERN.match(line)
        if not match:
            output_lines.append(line)
            continue
        old_speaker = match.group("speaker")
        timestamp = (match.group("timestamp") or "").strip()
        text_only = (match.group("text") or "").strip()
        if old_speaker not in role_map:
            raise KeyError(f"Speaker {old_speaker!r} is missing from role_map.")
        new_label = role_map[old_speaker]
        display_label = f"{new_label} ({old_speaker})"
        if timestamp:
            output_lines.append(f"{timestamp} {display_label}: {text_only}")
        else:
            output_lines.append(f"{display_label}: {text_only}")
    return "\n".join(output_lines).strip()
