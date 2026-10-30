"""Selection uses a supplied training prior and preserves excerpt order."""

from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from smsc.excerpts import greedy_kl, build_states, review_sample


class ExcerptTests(unittest.TestCase):
    def test_greedy_criterion_uses_kl_change_from_the_current_prefix(self):
        probabilities = {(0,): 0.35, (1,): 0.8, (2,): 0.1, (0, 1): 0.81, (1, 2): 0.5}
        calls = []

        def predict(subsets):
            calls.extend(subsets)
            return np.array([probabilities[subset] for subset in subsets])

        result = greedy_kl(["one", "two", "three"], predict, 0.2, budget=2)
        self.assertEqual([row["excerpt_index"] for row in result], [1, 2])
        self.assertEqual(result[0]["previous_probability"], 0.2)
        self.assertEqual(result[1]["previous_probability"], 0.8)
        self.assertIn((0, 1), calls)
        self.assertNotIn((1, 0), calls)

    def test_test_records_cannot_change_the_training_states_or_prior(self):
        frame = pd.DataFrame(
            {
                "id": ["a", "a", "b", "b", "c", "d"],
                "session": [1] * 6,
                "excerpt_order": [0, 1, 0, 1, 0, 0],
                "text": ["one", "two", "three", "four", "five", "six"],
                "ever_dropout": [0, 0, 1, 1, 1, 1],
            }
        )
        assignments = pd.DataFrame(
            {
                "doc_id": ["a", "b", "c", "d"],
                "seed": [0] * 4,
                "split": ["train", "train", "validation", "test"],
            }
        )

        def encoder(texts):
            return np.array([[len(text), 1] for text in texts], dtype=np.float32)

        original = build_states(frame, assignments, 0, encoder, states_per_session=3)
        changed = frame.copy()
        changed.loc[changed.id.eq("d"), ["text", "ever_dropout"]] = ["different", 0]
        repeated = build_states(changed, assignments, 0, encoder, states_per_session=3)
        self.assertEqual(original["training_prior"], 0.5)
        np.testing.assert_array_equal(original["matrix"], repeated["matrix"])
        self.assertNotIn("test", original["split"])

    def test_review_sample_applies_the_per_participant_cap(self):
        rows = pd.DataFrame(
            {
                "id": ["a", "a", "b"],
                "seed": [0, 0, 0],
                "session": [1, 2, 1],
                "excerpt_index": [0, 0, 0],
                "ever_dropout": [1, 1, 1],
                "standalone_probability_shift": [0.2, 0.3, 0.1],
            }
        )
        selected = review_sample(rows)
        self.assertEqual(len(selected), 2)
        self.assertEqual(selected.iloc[0].session, 2)


if __name__ == "__main__":
    unittest.main()
