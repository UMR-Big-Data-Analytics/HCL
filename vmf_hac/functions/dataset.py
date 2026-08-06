import gc
import logging
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
        x = ensure_valid_embeddings(np.load(emb_dir), source=emb_dir)
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


def _empty_device_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.mps.is_available():
        torch.mps.empty_cache()


def _is_out_of_memory(exc: BaseException) -> bool:
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    message = str(exc).lower()
    return any(marker in message for marker in ("out of memory", "invalid buffer size", "can't allocate"))


def ensure_valid_embeddings(x: np.ndarray, source: str) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if not np.isfinite(x).all():
        n_bad_rows = int((~np.isfinite(x)).any(axis=1).sum())
        raise ValueError(
            f"Embeddings from {source} contain NaN/Inf values ({n_bad_rows}/{x.shape[0]} rows affected). "
            "This usually means the encoder ran in float16 and overflowed; regenerate the embeddings."
        )
    return x


MAX_SEQ_LENGTH = 4096
"""
Upper bound on tokens per document.

Some encoders advertise very long contexts (KaLM/Gemma 3 reports 131072 tokens). Without a cap the
attention matrix of a single long document can require tens of GiB, which shrinking the batch size
cannot recover from. Clustering does not benefit from such long contexts, so documents are
truncated. Override with EMBED_MAX_SEQ_LENGTH.
"""


def _max_seq_length() -> int:
    return int(os.getenv("EMBED_MAX_SEQ_LENGTH", MAX_SEQ_LENGTH))


def _initial_batch_size(model_name: str) -> int:
    name = model_name.lower()
    # if any(x in name for x in ("27b", "14b", "12b", "8b")):
    #    return 2
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

    _empty_device_cache()

    enc = SentenceTransformer(
        encoding_model,
        trust_remote_code=True,
        device=_get_device(),
        # model_kwargs={"device_map": "auto"},
        model_kwargs={
            "dtype": "float64",
        },
    )

    max_seq_length = _max_seq_length()
    if enc.max_seq_length is None or enc.max_seq_length > max_seq_length:
        logger.info(
            "Capping max_seq_length from %s to %d for model=%s", enc.max_seq_length, max_seq_length, encoding_model
        )
        enc.max_seq_length = max_seq_length

    while batch_size >= 1:
        try:
            logger.info(
                "Encoding %d texts with model=%s batch_size=%d",
                len(prepared_texts),
                encoding_model,
                batch_size,
            )
            embeddings = enc.encode_document(
                prepared_texts,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
            )
            return ensure_valid_embeddings(embeddings, source=f"model={encoding_model}")  # ty:ignore[invalid-argument-type]
        except (torch.OutOfMemoryError, RuntimeError) as exc:
            if not _is_out_of_memory(exc):
                raise
            logger.warning(
                "Out of memory for model=%s batch_size=%d (%s); retrying with smaller batch size",
                encoding_model,
                batch_size,
                exc,
            )
            gc.collect()
            _empty_device_cache()
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
