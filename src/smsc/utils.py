"""Hashing and identifier normalization used across analysis modules."""

import hashlib
import json
import re
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def digest_values(values):
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8", errors="replace"))
        digest.update(b"\x1e")
    return digest.hexdigest()


def sha256_text(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


def normalize_id(value):
    import pandas as pd

    if pd.isna(value):
        return ""
    text = str(value).strip()
    return text[:-2] if re.fullmatch(r"\d+\.0", text) else text
