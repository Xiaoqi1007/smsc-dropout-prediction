"""Tests for participant isolation, training-only transforms and frozen explanations."""

from pathlib import Path
import sys, tempfile, unittest, json, copy
import numpy as np, pandas as pd

P = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(P / "src"))
from smsc._core import embeddings, pipeline
from smsc.models import predict_branch
from smsc.config import load_config
from smsc.calibration import fit_sklearn_platt, apply_platt
from smsc.plotting import best_f1
from smsc.experiment import validate_predictions
from smsc.explain_support import WithinSessionModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class PipelineTests(unittest.TestCase):

    def test_single_scaling_preserves_small_numeric_differences(self):
        frame = pd.DataFrame({"x": 1 + np.arange(6) * 1e-8})
        train = np.ones(6, dtype=bool)
        matrix, transformer, schema = pipeline.prepare_block(
            embeddings, frame, ["x"], train, 50, scale_numeric=False, dtype="float64"
        )
        scaler = StandardScaler().fit(matrix)
        transformed = scaler.transform(matrix).astype(np.float32)
        model = LogisticRegression().fit(transformed, [0, 0, 0, 1, 1, 1])
        payload = dict(
            model=model,
            preprocessor=transformer,
            schema=schema,
            scaler=scaler,
            input_dtype="float64",
        )
        np.testing.assert_allclose(
            predict_branch(payload, frame),
            model.predict_proba(transformed)[:, 1],
            atol=1e-12,
        )

    def test_invalid_seed_configuration(self):
        original = json.loads((P / "configs/analysis.json").read_text())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(original))
            self.assertEqual(load_config(path)["seeds"], original["seeds"])
            for seeds in [[0, 0], [-1], []]:
                candidate = copy.deepcopy(original)
                candidate["seeds"] = seeds
                path.write_text(json.dumps(candidate))
                with self.assertRaises(ValueError):
                    load_config(path)

    def test_preprocessing_fit_isolated(self):
        train = np.array([True] * 4 + [False] * 2)
        frame = pd.DataFrame(
            {
                "x": [0.0, 1.0, 2.0, 3.0, 1000.0, -1000.0],
                "cat": ["a", "b", "a", "b", "unseen", "unseen"],
            }
        )
        x, prep, schema = pipeline.prepare_block(
            embeddings, frame, ["x", "cat"], train, 50, scale_numeric=False
        )
        self.assertNotIn("unseen", " ".join(schema["encoded_columns"]))
        np.testing.assert_array_equal(x[:, 0], frame.x)
        scale = StandardScaler().fit(x[train])
        self.assertEqual(scale.mean_[0], 1.5)
        model = LogisticRegression().fit(scale.transform(x[train]), [0, 0, 1, 1])
        payload = dict(model=model, preprocessor=prep, schema=schema, scaler=scale)
        np.testing.assert_allclose(
            predict_branch(payload, frame),
            model.predict_proba(scale.transform(x))[:, 1],
            atol=1e-06,
        )

    def test_calibration_reload_formula(self):
        p = np.linspace(0.05, 0.95, 20)
        params, native = fit_sklearn_platt(p, np.array([0, 1] * 10))
        np.testing.assert_allclose(
            apply_platt(p, params), native.predict_proba(p[:, None])[:, 1], atol=1e-12
        )

    def test_exclusions_cannot_remove_identity_or_outcome(self):
        original = json.loads((P / "configs/analysis.json").read_text())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            for c in ["id", "session", "ever_dropout", "next_session_dropout"]:
                original["features"]["exclude_columns"] = [c]
                path.write_text(json.dumps(original))
                with self.assertRaises(ValueError):
                    load_config(path)

    def test_thresholds_use_validation_only_and_undefined_single_class(self):
        x = pd.DataFrame(
            dict(
                split=["validation"] * 3, y_true=[0, 1, 1], probability=[0.2, 0.5, 0.8]
            )
        )
        self.assertEqual(best_f1(x), 0.5)
        self.assertTrue(np.isnan(best_f1(x[x.y_true.eq(1)])))
        with self.assertRaises(ValueError):
            best_f1(x.assign(split="test"))

    def test_permutations_stay_within_session(self):
        X = np.arange(24).reshape(8, 3)
        sessions = np.repeat([1, 2], 4)
        m = WithinSessionModel(None, X, sessions)
        keys = m.keys.copy()
        keys[:, 1] = np.random.default_rng(1).permutation(keys[:, 1])
        changed = m.restore(keys)
        for rows in m.groups:
            np.testing.assert_array_equal(
                np.sort(changed[rows, 1]), np.sort(X[rows, 1])
            )
        np.testing.assert_array_equal(changed[:, [0, 2]], X[:, [0, 2]])

    def test_cross_partition_participant_rejected(self):
        x = pd.DataFrame(
            dict(
                seed=[0, 0],
                split=["train", "test"],
                id=["a", "a"],
                session=[0, 1],
                model=["full_model"] * 2,
                y_true=[1, 1],
                probability_raw=[0.3, 0.4],
                probability=[0.4, 0.5],
            )
        )
        with self.assertRaises(ValueError):
            validate_predictions(x)


if __name__ == "__main__":
    unittest.main()
