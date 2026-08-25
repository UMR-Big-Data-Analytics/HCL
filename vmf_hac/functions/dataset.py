import gc
import logging
import os
import re
import unicodedata
from collections.abc import Sequence
from typing import Any

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
    # Do NOT upcast here - keep the original dtype (typically float32).
    # Callers that require float64 (e.g. VmfHAC.fit) cast locally.
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
    if _parameter_count_billions(name) >= 8:
        return 1
    if "large" in name:
        return 8
    return 32


def _parameter_count_billions(model_name: str) -> float:
    match = re.search(r"(?:^|[-_/])(\d+(?:\.\d+)?)b(?:$|[-_/])", model_name.lower())
    return float(match.group(1)) if match else 0


def _model_kwargs(encoding_model: str) -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"dtype": torch.float32}

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    kwargs: dict[str, Any] = {"dtype": compute_dtype}
    if _parameter_count_billions(encoding_model) >= 12:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = {"": 0}
    return kwargs


def _load_encoder(encoding_model: str) -> SentenceTransformer:
    model_kwargs = _model_kwargs(encoding_model)
    if "quantization_config" in model_kwargs:
        logging.getLogger(__name__).info("Loading model=%s with 4-bit quantization", encoding_model)

    enc = SentenceTransformer(
        encoding_model,
        trust_remote_code=True,
        device=_get_device(),
        model_kwargs=model_kwargs,
    )
    max_seq_length = _max_seq_length()
    if enc.max_seq_length is None or enc.max_seq_length > max_seq_length:
        logging.getLogger(__name__).info(
            "Capping max_seq_length from %s to %d for model=%s", enc.max_seq_length, max_seq_length, encoding_model
        )
        enc.max_seq_length = max_seq_length
    return enc


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

    enc: SentenceTransformer | None = None
    try:
        enc = _load_encoder(encoding_model)
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
                if batch_size == 1:
                    raise
                batch_size = max(1, batch_size // 2)
                logger.warning(
                    "Out of memory for model=%s; retrying with batch_size=%d (%s)",
                    encoding_model,
                    batch_size,
                    exc,
                )
                gc.collect()
                _empty_device_cache()
    finally:
        if enc is not None:
            del enc
        gc.collect()
        _empty_device_cache()

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
