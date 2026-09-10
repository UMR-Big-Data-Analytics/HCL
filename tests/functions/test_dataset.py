import numpy as np

from vmf_hac.entity import dataset as entity_dataset
from vmf_hac.functions import dataset


def test_embedding_validation_uses_bounded_chunks(monkeypatch):
    embeddings = np.ones((12, 2), dtype=np.float32)
    largest_checked_chunk = 0
    original_isfinite = np.isfinite

    def tracking_isfinite(values):
        nonlocal largest_checked_chunk
        largest_checked_chunk = max(largest_checked_chunk, values.nbytes)
        return original_isfinite(values)

    monkeypatch.setattr(dataset, "EMBEDDING_VALIDATION_CHUNK_BYTES", 16)
    monkeypatch.setattr(dataset.np, "isfinite", tracking_isfinite)

    result = dataset.ensure_valid_embeddings(embeddings, source="test")

    assert result is embeddings
    assert largest_checked_chunk <= 16


def test_embedding_validation_reports_all_invalid_rows(monkeypatch):
    embeddings = np.ones((12, 2), dtype=np.float32)
    embeddings[[1, 5, 11], 0] = np.nan
    monkeypatch.setattr(dataset, "EMBEDDING_VALIDATION_CHUNK_BYTES", 16)

    with np.testing.assert_raises_regex(ValueError, "3/12 rows affected"):
        dataset.ensure_valid_embeddings(embeddings, source="test")


def test_cached_embeddings_are_memory_mapped_without_a_full_scan(tmp_path, monkeypatch):
    embeddings = np.ones((12, 2), dtype=np.float32)
    np.save(tmp_path / "embeddings.npy", embeddings)
    monkeypatch.setattr(entity_dataset, "get_emb_dir", lambda *_: str(tmp_path))

    def fail_if_scanned(*_args, **_kwargs):
        raise AssertionError("cached mmap was scanned")

    monkeypatch.setattr(entity_dataset, "ensure_valid_embeddings", fail_if_scanned)

    cached_dataset = entity_dataset.TextDataset(name="test", encoding_model="test-model")
    loaded = cached_dataset.embeddings

    assert isinstance(loaded, np.memmap)


def test_cached_dataset_lookup_does_not_evaluate_dataset_truthiness(monkeypatch):
    class CachedDataset:
        def __bool__(self):
            raise AssertionError("cached dataset truthiness was evaluated")

    cached_dataset = CachedDataset()
    monkeypatch.setattr(
        entity_dataset.DatasetManager,
        "_get_cached_dataset",
        lambda *_: cached_dataset,
    )

    result = entity_dataset.DatasetManager.get_mteb_clustering_data("test", "test-model")

    assert result is cached_dataset
