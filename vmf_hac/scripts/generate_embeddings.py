import gc
import json
import logging
from collections.abc import Generator

import torch
from tqdm import tqdm

from vmf_hac.definitions import ROOT_DIR
from vmf_hac.entity.dataset import DatasetFactory, DatasetManager, TextDatasets


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def iterator() -> Generator[tuple[str, TextDatasets]]:
    config_path = ROOT_DIR / "config.json"
    with config_path.open() as f:
        config = json.load(f)
    for model in config["models"]:
        for dataset_id in config["datasets"]:
            text_dataset = DatasetFactory.from_string(dataset_id)
            yield model, text_dataset


def main() -> None:
    failed_pairs = []
    successful_pairs = []

    for model_id, dataset in tqdm(iterator()):
        try:
            print(f"Generate embeddings for {model_id} and {dataset}")
            manager = DatasetManager(model_id)
            manager.get(dataset)
            successful_pairs.append((model_id, dataset.value))
            logger.info(f"✓ Successfully generated embeddings for {model_id} and {dataset.value}")
        except Exception as e:
            failed_pairs.append((model_id, dataset.value, str(e)))
            logger.error(
                f"✗ Failed to generate embeddings for {model_id} and {dataset.value}: {e}",
                exc_info=True,
            )
            continue
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    logger.info("=" * 80)
    logger.info("EMBEDDING GENERATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Successfully processed: {len(successful_pairs)} pairs")
    logger.info(f"Failed: {len(failed_pairs)} pairs")

    if failed_pairs:
        logger.warning("\nFailed pairs:")
        for model_id, dataset_id, error in failed_pairs:
            logger.warning(f"  - {model_id} + {dataset_id}")
            logger.warning(f"    Error: {error}")

    if successful_pairs:
        logger.info("\nSuccessfully generated:")
        for model_id, dataset_id in successful_pairs:
            logger.info(f"  - {model_id} + {dataset_id}")


if __name__ == "__main__":
    main()