import gc
import logging
import os
import re
import unicodedata
from collections.abc import Sequence

import numpy as np
import torch.mps
from sentence_transformers import SentenceTransformer
from transformers import BitsAndBytesConfig

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


def _initial_batch_size(model_name: str) -> int:
    name = model_name.lower()
    # Conservative defaults for large embedding models
    if any(x in name for x in ("27b", "14b", "12b", "8b")):
        return 2
    if "large" in name:
        return 8
    return 32


def embed_texts(encoding_model: str, texts: Sequence[str]) -> np.ndarray:
    """
    Encode texts with automatic OOM recovery by shrinking batch size.
    Optional env override: EMBED_BATCH_SIZE
    """
    logger = logging.getLogger(__name__)
    prepared_texts = _prepare_texts(encoding_model, texts)

    # Allow manual override from env, otherwise use heuristic
    batch_size = int(os.getenv("EMBED_BATCH_SIZE", _initial_batch_size(encoding_model)))

    torch.cuda.empty_cache()

    enc = SentenceTransformer(
        encoding_model,
        trust_remote_code=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
        #model_kwargs={"device_map": "auto"},
        model_kwargs={
            "quantization_config": BitsAndBytesConfig(load_in_4bit=True)
        },

    )

    while batch_size >= 1:
        try:
            logger.info(
                "Encoding %d texts with model=%s batch_size=%d",
                len(prepared_texts),
                encoding_model,
                batch_size,
            )
            return enc.encode_document(
                prepared_texts,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
            )
        except torch.OutOfMemoryError:
            if not torch.cuda.is_available():
                raise
            logger.warning(
                "CUDA OOM for model=%s batch_size=%d; retrying with smaller batch size",
                encoding_model,
                batch_size,
            )
            gc.collect()
            torch.cuda.empty_cache()
            if batch_size == 1:
                raise
            batch_size = max(1, batch_size // 2)

    raise RuntimeError(f"Failed to encode texts for model={encoding_model}")


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
