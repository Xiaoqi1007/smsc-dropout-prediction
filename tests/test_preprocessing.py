"""Feature construction and alignment on generated records."""

from pathlib import Path
import copy
import json
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from smsc.preprocessing.features import (
    baseline_features,
    emotion_features,
    speech_features,
    merge_modalities,
    session_metadata,
)
from smsc.preprocessing.embeddings import write_embedding_cache
from smsc.preprocessing.splits import generate_assignments
from smsc.preprocessing.speaker_roles import assign_speakers_to_roles
from smsc.preprocessing.tasks import build_dialogue_chunks
from smsc.preprocessing import tasks
from smsc.preprocessing.transcription import build_turns, render_transcripts
from smsc._core import embeddings
from smsc.utils import digest_values


class PreprocessingTests(unittest.TestCase):
    def test_session_dates_use_only_preceding_observations(self):
        frame = pd.DataFrame(
            {
                "id": ["a"] * 4,
                "session": [1, 2, 3, 4],
                "session_date": [None, "2020-01-05", None, "2020-01-10"],
            }
        )
        mapping = {name: name for name in frame}
        result = session_metadata(frame, mapping)
        self.assertTrue(
            pd.isna(result.session_info__derived_days_since_session0.iloc[0])
        )
        self.assertEqual(
            result.session_info__derived_days_since_previous_session.iloc[3], 5
        )

    def test_scoring_reverses_items_and_preserves_incomplete_scores(self):
        frame = pd.DataFrame({"person": ["a", "b"], "q1": [1, np.nan], "q2": [3, 2]})
        spec = {
            "id_column": "person",
            "features": [
                {
                    "name": "score",
                    "operation": "sum",
                    "columns": ["q1", "q2"],
                    "range": [1, 5],
                    "reverse": ["q1"],
                }
            ],
        }
        scored = baseline_features(frame, spec)
        self.assertEqual(scored.score.iloc[0], 8)
        self.assertTrue(pd.isna(scored.score.iloc[1]))

    def test_prior_emotions_pool_segments_without_current_or_future_sessions(self):
        frame = pd.DataFrame(
            {
                "id": ["a"] * 4,
                "session": [1, 1, 2, 3],
                "emotion": ["Happy", "Happy", "Angry", "Fearful"],
            }
        )
        scored = emotion_features(frame)
        self.assertTrue(
            pd.isna(scored.emotion__derived_prior_before_session_Happy_prop.iloc[0])
        )
        self.assertEqual(
            scored.emotion__derived_prior_before_session_Happy_prop.iloc[1], 1
        )
        self.assertAlmostEqual(
            scored.emotion__derived_prior_before_session_Happy_prop.iloc[2], 2 / 3
        )

    def test_speech_counts_sum_tasks_and_define_zero_denominator_as_missing(self):
        chunks = pd.DataFrame(
            {
                "id": ["a", "a", "b"],
                "session": [1, 1, 1],
                "participant_text": ["one two", "three", "one"],
                "clinician_text": ["one", "two", ""],
                "task_category": ["checkin", "review", "checkin"],
            }
        )
        result = speech_features(chunks).set_index("id")
        self.assertEqual(result.loc["a", "speech__participant_total_words"], 3)
        self.assertEqual(
            result.loc["a", "speech__participant_clinician_word_ratio"], 1.5
        )
        self.assertTrue(
            pd.isna(result.loc["b", "speech__participant_clinician_word_ratio"])
        )

    def test_modality_merge_preserves_unobserved_modalities(self):
        sessions = pd.DataFrame({"id": ["a", "a"], "session": [0, 1]})
        modality = pd.DataFrame({"id": ["a"], "session": [1], "value": [3]})
        merged = merge_modalities(sessions, modality)
        self.assertEqual(len(merged), 2)
        self.assertTrue(pd.isna(merged.value.iloc[0]))

    def test_embedding_writer_is_readable_by_the_training_pipeline(self):
        frame = pd.DataFrame(
            {"id": ["a", "a"], "session": [0, 1], "text": ["one", "two three"]}
        )

        def encoder(text):
            return np.asarray([[len(value), 1] for value in text], dtype=np.float32)

        with tempfile.TemporaryDirectory() as directory:
            write_embedding_cache(frame, directory, encoder)
            config = copy.deepcopy(embeddings.DEFAULT_CONFIG)
            config["embedding_root"] = directory
            keys = frame.id + "::" + frame.session.astype(str)
            bundle = embeddings.DatasetBundle(frame, digest_values(keys))
            actual = embeddings.load_current_embedding(config, bundle, "full_text", 0)
            np.testing.assert_array_equal(actual, encoder(frame.text))

    def test_dialogue_pairs_remain_intact_when_above_chunk_word_limit(self):
        turns = [
            {"turn_index": i, "speaker": speaker, "text": "word " * 140}
            for i, speaker in enumerate(["Clinician", "Participant"])
        ]
        chunks = build_dialogue_chunks(turns)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["n_turns"], 2)
        self.assertEqual(chunks[0]["n_words"], 280)

    def test_task_segmentation_loads_a_local_protocol(self):
        protocol = pd.DataFrame(
            {
                "arm": ["pstair"] * 2,
                "module": [1, 1],
                "task_order": [1, 2],
                "task": ["opening", "closing"],
                "task_label": ["Opening", "Closing"],
                "task_category": ["opening", "closing"],
                "task_count_in_module": [2, 2],
                "cue_boundary": ["begin", "finish"],
                "cue_words": ["begin", "finish"],
            }
        )
        turns = [
            {"turn_index": i, "speaker": role, "text": text}
            for i, (role, text) in enumerate(
                [
                    ("Clinician", "begin now"),
                    ("Participant", "begin"),
                    ("Clinician", "finish now"),
                    ("Participant", "finish"),
                ]
            )
        ]
        semantic = tasks.USE_SEMANTIC_CUE_MATCHING
        try:
            tasks.USE_SEMANTIC_CUE_MATCHING = False
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "protocol.csv"
                protocol.to_csv(path, index=False)
                tasks.configure_tasks(path)
                rows, *_ = tasks.process_session(turns, 1, "pstair")
            self.assertEqual(sum(row["n_turns"] for row in rows), 4)
            self.assertEqual(
                {row["task_category"] for row in rows}, {"opening", "closing"}
            )
        finally:
            tasks.USE_SEMANTIC_CUE_MATCHING = semantic

    def test_word_timestamps_form_anonymous_speaker_turns(self):
        source = {
            "segments": [
                {
                    "words": [
                        {
                            "word": "Hello",
                            "speaker": "speaker_one",
                            "start": 0,
                            "end": 1,
                        },
                        {"word": ",", "speaker": "speaker_one", "start": 1, "end": 1.1},
                        {
                            "word": "world",
                            "speaker": "speaker_one",
                            "start": 1.1,
                            "end": 2,
                        },
                    ]
                }
            ]
        }
        turns, mapping = build_turns(source)
        plain, _ = render_transcripts(turns)
        self.assertEqual(len(mapping), 1)
        self.assertIn("Hello, world", plain)

    def test_speaker_roles_follow_language_evidence(self):
        turns = [
            {
                "speaker": "A",
                "text": "I want to ask you questions. Let us review your homework.",
            },
            {
                "speaker": "B",
                "text": "My daughter and my family are at my job. I feel overwhelmed.",
            },
        ]
        roles, *_ = assign_speakers_to_roles(turns)
        self.assertEqual(roles, {"A": "Clinician", "B": "Participant"})

    def test_split_generation_keeps_all_sessions_of_a_person_together(self):
        frame = pd.DataFrame(
            {
                "id": [f"sample_{i}" for i in range(120)],
                "session": [0] * 120,
                "ever_dropout": [i % 2 for i in range(120)],
            }
        )
        last = pd.Series([0, 5, 14] * 40, index=frame.id)
        config = json.loads((ROOT / "configs/splits.json").read_text())
        config["seeds"] = [1, 7]
        first, _ = generate_assignments(frame, last, config)
        second, _ = generate_assignments(frame, last, config)
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(len(first), 240)
        self.assertTrue(first.groupby(["seed", "id"]).split.nunique().eq(1).all())


if __name__ == "__main__":
    unittest.main()
