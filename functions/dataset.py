import os
import re
import unicodedata
from collections.abc import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from definitions import DATA_DIR


def _get_embedding(emb_dir: str, model: str, texts: Sequence[str]) -> np.ndarray:
    if os.path.exists(emb_dir):
        x = np.load(emb_dir)
    else:
        x = embed_texts(model, texts)
        np.save(emb_dir, x)
    return x


def embed_texts(model: str, texts: Sequence[str]) -> np.ndarray:
    enc = SentenceTransformer(model)
    # noinspection PyTypeChecker
    return enc.encode(
        [f"query: {text}" for text in texts],  # intfloat/multilingual-e5-large was trained with "query: " prefix
        normalize_embeddings=True,
        batch_size=64,
    )


def slugify(value: str, allow_unicode=False) -> str:
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize("NFKC", value)
    else:
        value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^\w\s-]", "", value.lower())
    return re.sub(r"[-\s]+", "-", value).strip("-_")


def get_emb_dir(dataset_name: str, model_name: str) -> str:
    return os.path.join(DATA_DIR, f"{slugify(dataset_name)}_emb_{slugify(model_name)}")
