"""Equal-weight fusion of source probabilities."""

import numpy as np


def average_specification(n_components):
    if type(n_components) is not int or n_components < 1:
        raise ValueError("n_components must be a positive integer")
    return dict(strategy="fixed_simple_average", n_components=n_components)


def predict_average(specification, probabilities):
    if specification.get("strategy") != "fixed_simple_average":
        raise ValueError("Only fixed average fusion is supported")
    x = np.asarray(probabilities, dtype=float)
    if x.ndim != 2 or x.shape[1] != specification.get("n_components") or x.shape[1] < 1:
        raise ValueError(
            "Component count does not match the saved average specification"
        )
    if not np.isfinite(x).all() or np.any((x < 0) | (x > 1)):
        raise ValueError("All component probabilities must be finite and in [0, 1]")
    return x.mean(axis=1)
