"""Random-subset prediction and greedy KL selection of transcript excerpts."""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.special import logit


def bernoulli_kl(probability, reference):
    p = np.clip(np.asarray(probability, dtype=float), 1e-12, 1 - 1e-12)
    q = np.clip(np.asarray(reference, dtype=float), 1e-12, 1 - 1e-12)
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))


def greedy_kl(excerpts, predict, training_prior, budget=5):
    """Select the candidate with the largest KL change at each step."""
    selected = []
    reference = float(training_prior)
    if not 0 < reference < 1 or budget < 1:
        raise ValueError(
            "Selection requires a nondegenerate training prior and a positive budget"
        )
    records = []
    for step in range(min(budget, len(excerpts))):
        remaining = [i for i in range(len(excerpts)) if i not in selected]
        subsets = [tuple(sorted(selected + [candidate])) for candidate in remaining]
        probabilities = np.asarray(predict(subsets), dtype=float)
        if (
            probabilities.shape != (len(remaining),)
            or not np.isfinite(probabilities).all()
            or not ((probabilities >= 0) & (probabilities <= 1)).all()
        ):
            raise ValueError("Invalid subset probabilities")
        gain = bernoulli_kl(probabilities, reference)
        best = int(np.argmax(gain))
        chosen = remaining[best]
        probability = float(probabilities[best])
        records.append(
            {
                "step": step + 1,
                "excerpt_index": chosen,
                "probability": probability,
                "previous_probability": reference,
                "kl_gain": float(gain[best]),
                "step_log_odds_change": float(
                    logit(np.clip(probability, 1e-12, 1 - 1e-12))
                    - logit(np.clip(reference, 1e-12, 1 - 1e-12))
                ),
            }
        )
        selected.append(chosen)
        reference = probability
    if records:
        standalone = np.asarray(
            predict([(record["excerpt_index"],) for record in records]), dtype=float
        )
        for record, probability in zip(records, standalone):
            record["standalone_probability"] = float(probability)
            record["standalone_probability_shift"] = float(probability - training_prior)
            record["standalone_log_odds_change"] = float(
                logit(np.clip(probability, 1e-12, 1 - 1e-12))
                - logit(np.clip(training_prior, 1e-12, 1 - 1e-12))
            )
    return records


def subset_inputs(excerpts, subsets, encoder, budget, maximum_excerpts):
    text = [" ".join(excerpts[index] for index in sorted(subset)) for subset in subsets]
    embedding = np.asarray(encoder(text), dtype=np.float32)
    counts = np.asarray(
        [
            [len(subset) / max(budget, 1), len(excerpts) / max(maximum_excerpts, 1)]
            for subset in subsets
        ],
        dtype=np.float32,
    )
    return np.column_stack([embedding, counts])


def build_states(
    excerpts, assignments, seed, encoder, states_per_session=100, budget=5
):
    """Sample subset size and membership independently of content and outcome."""
    split = assignments.loc[
        assignments.seed.eq(seed), ["doc_id", "split"]
    ].drop_duplicates()
    if split.doc_id.duplicated().any():
        raise ValueError("A participant appears in more than one split")
    split = (
        split.set_index("doc_id")
        .split.str.lower()
        .replace({"val": "validation", "valid": "validation"})
    )
    groups = list(
        excerpts.sort_values(["id", "session", "excerpt_order"], kind="stable").groupby(
            ["id", "session"], sort=False
        )
    )
    maximum_excerpts = max(
        len(group)
        for (participant, _), group in groups
        if split.loc[str(participant)] == "train"
    )
    rng = np.random.default_rng(seed)
    matrix, labels, partitions = [], [], []
    train_labels = []
    for (participant, _), group in groups:
        partition = str(split.loc[str(participant)])
        if partition == "test":
            continue
        if group.ever_dropout.nunique() != 1:
            raise ValueError("Conflicting excerpt outcomes")
        outcome = int(group.ever_dropout.iloc[0])
        if partition == "train":
            train_labels.append(outcome)
        text = group.text.astype(str).tolist()
        subsets = []
        for _ in range(states_per_session):
            count = int(rng.integers(1, min(budget, len(text)) + 1))
            subsets.append(
                tuple(sorted(rng.choice(len(text), size=count, replace=False).tolist()))
            )
        matrix.append(subset_inputs(text, subsets, encoder, budget, maximum_excerpts))
        labels.extend([outcome] * len(subsets))
        partitions.extend([partition] * len(subsets))
    if not train_labels or len(set(train_labels)) != 2:
        raise ValueError("Subset training requires both outcome classes")
    return {
        "matrix": np.vstack(matrix),
        "outcome": np.asarray(labels, dtype=int),
        "split": np.asarray(partitions),
        "training_prior": float(np.mean(train_labels)),
        "maximum_excerpts": maximum_excerpts,
        "budget": budget,
    }


def train_predictor(
    states,
    output,
    seed=0,
    width=256,
    depth=1,
    dropout=0.3,
    learning_rate=1e-5,
    epochs=80,
    batch_size=4096,
    patience=8,
):
    """Fit the two-class subset MLP using validation cross-entropy."""
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    layers = [tf.keras.layers.Input(shape=(states["matrix"].shape[1],))]
    for _ in range(depth):
        layers.extend(
            [
                tf.keras.layers.Dense(width, activation="gelu"),
                tf.keras.layers.Dropout(dropout),
            ]
        )
    layers.append(tf.keras.layers.Dense(2, activation="softmax"))
    model = tf.keras.Sequential(layers)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="sparse_categorical_crossentropy",
    )
    train = states["split"] == "train"
    validation = states["split"] == "validation"
    if not train.any() or not validation.any():
        raise ValueError("Training and validation states are required")
    model.fit(
        states["matrix"][train],
        states["outcome"][train],
        validation_data=(states["matrix"][validation], states["outcome"][validation]),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=patience,
                min_delta=1e-4,
                restore_best_weights=True,
            )
        ],
        verbose=2,
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    model.save(output / "subset_predictor.keras")
    metadata = {
        key: states[key].item() if isinstance(states[key], np.ndarray) else states[key]
        for key in ["training_prior", "maximum_excerpts", "budget"]
    }
    metadata["embedding_dim"] = int(states["matrix"].shape[1] - 2)
    (output / "subset_predictor.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return model


def select_test_excerpts(
    excerpts, assignments, seed, encoder, model, metadata, budget=5
):
    held = set(
        assignments.loc[
            assignments.seed.eq(seed) & assignments.split.eq("test"), "doc_id"
        ].astype(str)
    )
    frame = excerpts.loc[excerpts.id.astype(str).isin(held)].sort_values(
        ["id", "session", "excerpt_order"], kind="stable"
    )
    rows = []
    for (participant, session), group in frame.groupby(["id", "session"], sort=False):
        text = group.text.astype(str).tolist()

        def predict(subsets):
            matrix = subset_inputs(
                text, subsets, encoder, metadata["budget"], metadata["maximum_excerpts"]
            )
            return model.predict(matrix, verbose=0)[:, 1]

        selected = greedy_kl(text, predict, metadata["training_prior"], budget)
        for record in selected:
            record.update(
                id=str(participant),
                session=int(session),
                seed=seed,
                ever_dropout=int(group.ever_dropout.iloc[0]),
                text=text[record["excerpt_index"]],
                training_prior=metadata["training_prior"],
            )
            rows.append(record)
    return pd.DataFrame(rows)


def review_sample(selected, per_direction=5, first_session=1, last_session=9):
    """Choose a review sample with at most one excerpt per participant and split."""
    frame = selected.loc[selected.session.between(first_session, last_session)].copy()
    shift = frame.standalone_probability_shift
    frame["direction"] = np.where(shift > 0, "dropout", "completion")
    frame = frame.loc[
        ((frame.ever_dropout == 1) & (shift > 0))
        | ((frame.ever_dropout == 0) & (shift < 0))
    ]
    frame["magnitude"] = frame.standalone_probability_shift.abs()
    frame = frame.sort_values(
        ["seed", "direction", "magnitude", "id", "session", "excerpt_index"],
        ascending=[True, True, False, True, True, True],
        kind="stable",
    )
    frame = frame.drop_duplicates(["seed", "direction", "id"])
    return (
        frame.groupby(["seed", "direction"], sort=False)
        .head(per_direction)
        .drop(columns="magnitude")
    )
