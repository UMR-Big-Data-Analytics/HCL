import json

from sentence_transformers import SentenceTransformer

from vmf_hac.definitions import ROOT_DIR


def main():
    with open(ROOT_DIR / "config.json") as f:
        models = json.loads(f.read())["models"]

    for model in models:
        SentenceTransformer(
            model,
            trust_remote_code=True,
        )


if __name__ == "__main__":
    main()
