"""Validation-only sklearn Platt calibration on logit probabilities."""

import numpy as np
from scipy.special import expit, logit
from sklearn.base import BaseEstimator, ClassifierMixin

EPS = 1e-12


def score_from_probability(probability):
    p = np.asarray(probability, dtype=float)
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Probabilities must be finite and inside [0, 1]")
    return logit(np.clip(p, EPS, 1 - EPS))


class ProbabilityScoreClassifier(ClassifierMixin, BaseEstimator):
    """Frozen score adapter: expose logit(p) without fitting another classifier."""

    def __init__(self):
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = 1

    def fit(self, X, y):
        raise RuntimeError(
            "Score adapter must be frozen; base probabilities are already fitted"
        )

    def decision_function(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != 1:
            raise ValueError("Expected one column of raw probabilities")
        return score_from_probability(X[:, 0])

    def predict_proba(self, X):
        self.decision_function(X)
        p = np.asarray(X, dtype=float)[:, 0]
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.decision_function(X) >= 0).astype(int)


def fit_sklearn_platt(probability, outcome):
    """Fit through sklearn's public API on supplied validation rows only.

    FrozenEstimator keeps the component models fixed. Calibration fits a
    sigmoid to logit probabilities using the supplied validation observations.
    """
    import sklearn
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.frozen import FrozenEstimator

    p = np.asarray(probability, dtype=float)
    score = score_from_probability(p)
    y = np.asarray(outcome, dtype=float)
    if score.ndim != 1 or y.shape != score.shape or set(np.unique(y)) != {0.0, 1.0}:
        raise ValueError("Calibration requires matching vectors and both binary labels")
    fitted = CalibratedClassifierCV(
        FrozenEstimator(ProbabilityScoreClassifier()),
        method="sigmoid",
        ensemble=False,
        n_jobs=1,
    )
    fitted.fit(p[:, None], y.astype(int))
    # Store coefficients for the equivalent sigmoid formula.
    sigmoid = fitted.calibrated_classifiers_[0].calibrators[0]
    a, b = -float(sigmoid.a_), -float(sigmoid.b_)
    parameters = dict(
        a=a,
        b=b,
        n_rows=len(y),
        n_positive=int(y.sum()),
        n_negative=int(len(y) - y.sum()),
        implementation="sklearn.CalibratedClassifierCV",
        sklearn_version=sklearn.__version__,
        method="sigmoid",
        score="logit(probability_raw)",
        frozen=True,
        weighting="equal_session_rows",
        success=True,
    )
    if not np.isfinite([a, b]).all():
        raise RuntimeError("Nonfinite sklearn Platt coefficients")
    np.testing.assert_allclose(
        apply_platt(p, parameters),
        fitted.predict_proba(p[:, None])[:, 1],
        rtol=0,
        atol=1e-12,
    )
    return parameters, fitted


def apply_platt(probability, parameters):
    return expit(
        parameters["a"] * score_from_probability(probability) + parameters["b"]
    )
