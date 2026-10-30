"""Build random subset states, fit the subset predictor, and select test excerpts."""

from pathlib import Path
import argparse
import json
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from smsc.excerpts import (
    build_states,
    train_predictor,
    select_test_excerpts,
    review_sample,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["states", "train", "select", "review"])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--splits", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-8B")
    parser.add_argument("--predictor", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dimension", type=int, default=4096)
    parser.add_argument("--budget", type=int, default=5)
    parser.add_argument("--training-budget", type=int, default=5)
    parser.add_argument("--states-per-session", type=int, default=100)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.command == "train":
        with np.load(args.input, allow_pickle=False) as saved:
            states = {name: saved[name] for name in saved.files}
        train_predictor(states, args.output, seed=args.seed)
        return
    frame = pd.read_csv(args.input, dtype={"id": str})
    if args.command == "review":
        review_sample(frame).to_csv(args.output, index=False)
        return
    if args.splits is None:
        parser.error("--splits is required for states and select")
    from smsc.preprocessing.embeddings import QwenEncoder

    encoder = QwenEncoder(args.model, args.device, dimension=args.dimension)
    assignments = pd.read_csv(args.splits, dtype={"doc_id": str})
    if args.command == "states":
        states = build_states(
            frame,
            assignments,
            args.seed,
            encoder,
            args.states_per_session,
            args.training_budget,
        )
        np.savez_compressed(args.output, **states)
    else:
        if args.predictor is None:
            parser.error("--predictor is required for select")
        import tensorflow as tf

        tf.config.set_visible_devices([], "GPU")
        model = tf.keras.models.load_model(args.predictor / "subset_predictor.keras")
        metadata = json.loads((args.predictor / "subset_predictor.json").read_text())
        if metadata["embedding_dim"] != args.dimension:
            parser.error("--dimension must match the trained subset predictor")
        selected = select_test_excerpts(
            frame, assignments, args.seed, encoder, model, metadata, args.budget
        )
        selected.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
