import json
import logging
from collections.abc import Generator

from tqdm import tqdm

from entity.dataset import DatasetFactory, DatasetManager, TextDatasets


def iterator() -> Generator[tuple[str, TextDatasets]]:
    with open("config.json") as f:
        config = json.load(f)
    for model in config["models"]:
        for dataset_id in config["datasets"]:
            text_dataset = DatasetFactory.from_string(dataset_id)
            yield model, text_dataset


if __name__ == "__main__":
    for model_id, dataset in tqdm(iterator()):
        DatasetManager(model_id).get(dataset)
    logging.info("Done")
