import os
import re
import unicodedata
from collections.abc import Sequence

import numpy as np
import torch.mps
from sentence_transformers import SentenceTransformer

from vmf_hac.definitions import DATA_DIR


def _get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.mps.is_available():
        return "mps"
    else:
        return "cpu"


def _get_embedding(emb_dir: str, model: str, texts: Sequence[str]) -> np.ndarray:
    if os.path.exists(emb_dir):
        x = np.load(emb_dir)
    else:
        x = embed_texts(model, texts)
        np.save(emb_dir, x)
    return x


def _prepare_texts(model_name: str, texts: Sequence[str]) -> list[str]:
    """
    Prepare texts according to the embedding model.

    This helper is intended for symmetric tasks such as clustering.
    """

    # E5 models require the "passage:" prefix for document embeddings.
    if model_name.startswith("intfloat/multilingual-e5"):
        return [f"passage: {text}" for text in texts]

    # All other models are used without prompts for clustering.
    return list(texts)


def embed_texts(model: str, texts: Sequence[str]) -> np.ndarray:
    enc = SentenceTransformer(
        model,
        device=_get_device(),
        trust_remote_code=True,  # needed by several newer embedding models
    )

    prepared_texts = _prepare_texts(model, texts)

    print(f"Encoding {len(texts)} texts...")

    # noinspection PyTypeChecker
    return enc.encode(
        prepared_texts,
        normalize_embeddings=True,
        batch_size=32,
        convert_to_numpy=True,
        show_progress_bar=True
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
