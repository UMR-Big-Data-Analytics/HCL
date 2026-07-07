import json
import logging
from collections.abc import Generator

from tqdm import tqdm

from vmf_hac.definitions import ROOT_DIR
from vmf_hac.entity.dataset import DatasetFactory, DatasetManager, TextDatasets


def iterator() -> Generator[tuple[str, TextDatasets]]:
    config_path = ROOT_DIR / "config.json"
    with config_path.open() as f:
        config = json.load(f)
    for model in config["models"]:
        for dataset_id in config["datasets"]:
            text_dataset = DatasetFactory.from_string(dataset_id)
            yield model, text_dataset


def main() -> None:
    for model_id, dataset in tqdm(iterator()):
        DatasetManager(model_id).get(dataset)
    logging.info("Done")


if __name__ == "__main__":
    main()
