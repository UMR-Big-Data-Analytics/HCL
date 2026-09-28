import argparse
import gc
import logging
from collections.abc import Generator

import torch
from tqdm import tqdm

from hcl.entity.dataset import DatasetFactory, DatasetManager, TextDatasets
from hcl.utils import get_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def iterator() -> Generator[tuple[str, TextDatasets]]:
    config = get_config()
    for model in config["models"]:
        for dataset_id in config["datasets"]:
            text_dataset = DatasetFactory.from_string(dataset_id)
            yield model, text_dataset


def generate(model_id: str, dataset: TextDatasets) -> None:
    try:
        print(f"Generate embeddings for {model_id} and {dataset}")
        manager = DatasetManager(model_id)
        manager.get(dataset)
        logger.info(f"✓ Successfully generated embeddings for {model_id} and {dataset.value.technical_name}")
    except Exception as e:
        logger.error(
            f"✗ Failed to generate embeddings for {model_id} and {dataset.value.technical_name}: {e}",
            exc_info=True,
        )
        raise
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate text embeddings for configured or selected model/dataset pairs."
    )
    parser.add_argument("--model", type=str, help="Model ID from config.json (requires --dataset).")
    parser.add_argument("--dataset", type=str, help="Dataset ID from config.json (requires --model).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if bool(args.model) != bool(args.dataset):
        msg = "--model and --dataset must be provided together."
        raise ValueError(msg)

    if args.model and args.dataset:
        dataset = DatasetFactory.from_string(args.dataset)
        generate(args.model, dataset)
    else:
        for model_id, dataset in tqdm(iterator()):
            generate(model_id, dataset)


if __name__ == "__main__":
    main()
