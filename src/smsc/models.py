"""Predictions from saved model payloads."""

import numpy as np
import pandas as pd
import torch
from torch import nn


def branch_columns(payload):
    return list(
        payload["columns"]
        if "columns" in payload
        else payload["schema"]["usable_train_columns"]
    )


def predict_branch(payload, frame):
    transformer = payload["preprocessor"]
    if "columns" in payload:
        converted = (
            frame[payload["columns"]]
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
        )
    else:
        converted = pd.DataFrame(index=frame.index)
        # Types are taken from the fitted transformer.
        for name, _, columns in transformer.transformers_:
            if name == "remainder":
                continue
            for column in columns:
                if name == "numeric":
                    converted[column] = pd.to_numeric(
                        frame[column], errors="coerce"
                    ).replace([np.inf, -np.inf], np.nan)
                elif name == "categorical":
                    values = (
                        frame[column]
                        .astype("string")
                        .str.strip()
                        .replace("", pd.NA)
                        .astype(object)
                    )
                    values[pd.isna(values)] = np.nan
                    converted[column] = values
                else:
                    raise ValueError(f"Unknown transformer: {name}")
    matrix = transformer.transform(converted).astype(
        payload.get("input_dtype", "float32")
    )
    if not np.isfinite(matrix).all():
        raise ValueError("Nonfinite transformed input")
    if payload.get("scaler") is not None:
        matrix = payload["scaler"].transform(matrix).astype(np.float32)
    return payload["model"].predict_proba(matrix)[:, 1].astype(float)


class EmbeddingNN(nn.Module):
    def __init__(self, input_dim, hidden_units, dropout):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_units, 1),
        )

    def forward(self, x):
        return self.network(x).squeeze(1)


def predict_text(payload, matrix):
    config = payload["nn_config"]
    model = EmbeddingNN(payload["input_dim"], config["hidden_units"], config["dropout"])
    model.load_state_dict(payload["state_dict"])
    model.eval()
    matrix = (
        payload["scaler"]
        .transform(np.asarray(matrix, dtype=np.float32))
        .astype(np.float32)
    )
    with torch.no_grad():
        parts = [
            torch.sigmoid(model(torch.from_numpy(matrix[i : i + 512]))).numpy()
            for i in range(0, len(matrix), 512)
        ]
    return np.concatenate(parts)
