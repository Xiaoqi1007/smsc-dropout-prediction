"""Session text encoding and caches used by the dropout models."""

from pathlib import Path
import json
import numpy as np
from ..utils import digest_values


class QwenEncoder:
    def __init__(
        self,
        model="Qwen/Qwen3-Embedding-8B",
        device=None,
        dimension=4096,
        max_length=32768,
        batch_size=8,
    ):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model, device=device, truncate_dim=dimension)
        self.model.max_seq_length = max_length
        self.batch_size = batch_size
        self.dimension = dimension

    def __call__(self, texts):
        return self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)


def write_embedding_cache(text_rows, root, encoder):
    """Keep the supplied row order and write both text and embedding fingerprints."""
    root = Path(root) / "text_input_full_text/cache"
    text_folder = root / "text"
    embedding_folder = root / "embeddings/shared"
    text_folder.mkdir(parents=True, exist_ok=True)
    embedding_folder.mkdir(parents=True, exist_ok=True)
    if text_rows.duplicated(["id", "session"]).any():
        raise ValueError("Duplicate text rows")
    rows = text_rows[["id", "session", "text"]].fillna({"text": ""})
    matrix = np.asarray(encoder(rows.text.tolist()), dtype=np.float32)
    if matrix.ndim != 2 or len(matrix) != len(rows) or not np.isfinite(matrix).all():
        raise ValueError("Invalid session embeddings")
    rows.to_csv(text_folder / "shared.csv", index=False)
    np.save(embedding_folder / "embeddings.npy", matrix)
    np.save(embedding_folder / "embeddings_done.npy", np.ones(len(rows), dtype=bool))
    metadata = {
        "n_rows": len(rows),
        "embedding_dim": matrix.shape[1],
        "row_key_digest": digest_values(
            rows.id.astype(str) + "::" + rows.session.astype(str)
        ),
        "text_digest": digest_values(rows.text),
        "representation": "current_session_text",
    }
    for path in [
        embedding_folder / "embeddings_metadata.json",
        text_folder / "shared_metadata.json",
    ]:
        path.write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata
