"""Checks for text-free snapshots and data-driven example labels."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd

P = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(P / "src"))

from smsc.explain_support import case_metadata
from smsc.privacy import feature_snapshot, save_text_hashes


class OutputPrivacyTests(unittest.TestCase):
    def test_snapshot_keeps_predictors_without_source_text(self):
        frame = pd.DataFrame(
            {
                "id": ["subject_a"],
                "session": [2],
                "ever_dropout": [1],
                "x": [0.7],
                "speech__participant_text_fraction": [0.4],
                "speech__participant_total_words": [40],
                "transcript_chunk_001_text": ["private source content"],
                "session_note_chunk_001_text": ["private clinical note"],
            }
        )
        audit = {
            "tabular_columns": ["x"],
            "video_columns": ["speech__participant_text_fraction"],
        }
        saved = feature_snapshot(frame, audit)
        self.assertIn("x", saved)
        self.assertIn("speech__participant_text_fraction", saved)
        self.assertIn("speech__participant_total_words", saved)
        self.assertFalse(any(c.endswith("_text") for c in saved))
        pd.testing.assert_series_equal(saved["x"], frame["x"])

    def test_text_hashes_contain_no_text_or_identifiers(self):
        text = pd.DataFrame(
            {
                "id": ["subject_a", "subject_b"],
                "session": [1, 2],
                "text": ["private statement", "private note"],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "hashes.json"
            first = save_text_hashes(text, destination)
            raw = destination.read_text()
            for value in [
                "subject_a",
                "subject_b",
                "private statement",
                "private note",
            ]:
                self.assertNotIn(value, raw)
            self.assertEqual(first, json.loads(raw))
            second = save_text_hashes(
                text.assign(text=["changed", "private note"]), destination
            )
            self.assertNotEqual(first["text_sha256"], second["text_sha256"])
            self.assertEqual(first["row_key_sha256"], second["row_key_sha256"])

    def test_case_labels_follow_actual_session_arm_and_outcome(self):
        frame = pd.DataFrame(
            {
                "id": ["subject_a"],
                "session": [5],
                "ever_dropout": [0],
                "speech__arm": ["sc"],
            }
        )
        result = case_metadata(frame, {"patient_id": "subject_a", "session": 5})
        self.assertEqual(
            result, {"session": 5, "outcome": "Completed", "treatment_arm": "SC"}
        )
        with self.assertRaises(ValueError):
            case_metadata(frame, {"patient_id": "subject_a", "session": 3})


if __name__ == "__main__":
    unittest.main()
