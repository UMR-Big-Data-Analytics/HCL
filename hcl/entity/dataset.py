import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, cast

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.datasets import fetch_20newsgroups

from hcl.functions.dataset import embed_texts, get_emb_dir


@dataclass(frozen=True)
class DatasetInfo:
    technical_name: str
    visual_name: str


class TextDatasets(Enum):
    ARXIV_CLUSTERING_P2P = DatasetInfo("mteb/arxiv-clustering-p2p", "ArxivClusteringP2P")
    BIORXIV_CLUSTERING_P2P = DatasetInfo("mteb/biorxiv-clustering-p2p", "BiorxivClusteringP2P.V2")
    """
    @article{geigle:2021:arxiv,
      archiveprefix = {arXiv},
      author = {Gregor Geigle and
    Nils Reimers and
    Andreas R{\"u}ckl{\'e} and
    Iryna Gurevych},
      eprint = {2104.07081},
      journal = {arXiv preprint},
      title = {TWEAC: Transformer with Extendable QA Agent Classifiers},
      url = {http://arxiv.org/abs/2104.07081},
      volume = {abs/2104.07081},
      year = {2021},
    }
    """
    REDDIT_CLUSTERING = DatasetInfo("mteb/reddit-clustering", "RedditClusteringV2")
    """
    @article{geigle:2021:arxiv,
      archiveprefix = {arXiv},
      author = {Gregor Geigle and
    Nils Reimers and
    Andreas R{\"u}ckl{\'e} and
    Iryna Gurevych},
      eprint = {2104.07081},
      journal = {arXiv preprint},
      title = {TWEAC: Transformer with Extendable QA Agent Classifiers},
      url = {http://arxiv.org/abs/2104.07081},
      volume = {abs/2104.07081},
      year = {2021},
    }
    """
    REDDIT_CLUSTERING_P2P = DatasetInfo("mteb/reddit-clustering-p2p", "RedditClusteringP2P")
    """
    @article{geigle:2021:arxiv,
      archiveprefix = {arXiv},
      author = {Gregor Geigle and
    Nils Reimers and
    Andreas R{\"u}ckl{\'e} and
    Iryna Gurevych},
      eprint = {2104.07081},
      journal = {arXiv preprint},
      title = {TWEAC: Transformer with Extendable QA Agent Classifiers},
      url = {http://arxiv.org/abs/2104.07081},
      volume = {abs/2104.07081},
      year = {2021},
    }
    """
    STACKEXCHANGE_CLUSTERING = DatasetInfo("mteb/stackexchange-clustering", "StackexchangeClustering")
    """
    @misc{pham2025vnmtebvietnamesemassivetext,
        title={VN-MTEB: Vietnamese Massive Text Embedding Benchmark},
        author={Loc Pham and Tung Luu and Thu Vo and Minh Nguyen and Viet Hoang},
        year={2025},
        eprint={2507.21500},
        archivePrefix={arXiv},
        primaryClass={cs.CL},
        url={https://arxiv.org/abs/2507.21500}
    }
    """
    STACKEXCHANGE_CLUSTERING_VN = DatasetInfo("GreenNode/stackexchange-clustering-vn", "StackexchangeClusteringVN")
    """
    @misc{pham2025vnmtebvietnamesemassivetext,
        title={VN-MTEB: Vietnamese Massive Text Embedding Benchmark},
        author={Loc Pham and Tung Luu and Thu Vo and Minh Nguyen and Viet Hoang},
        year={2025},
        eprint={2507.21500},
        archivePrefix={arXiv},
        primaryClass={cs.CL},
        url={https://arxiv.org/abs/2507.21500}
    }
    """
    ALLO_PROF_CLUSTERING_S2S = DatasetInfo("mteb/AlloProfClusteringS2S", "AlloProfClusteringS2S")
    """
    @inproceedings{katz-etal-2024-knowledge,
      address = {Miami, Florida, USA},
      author = {Katz, Uri  and
    Levy, Mosh  and
    Goldberg, Yoav},
      booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2024},
      month = nov,
      pages = {8838--8855},
      publisher = {Association for Computational Linguistics},
      title = {Knowledge Navigator: {LLM}-guided Browsing Framework for Exploratory Search in Scientific Literature},
      url = {https://aclanthology.org/2024.findings-emnlp.516},
      year = {2024},
    }
    """
    CLUSTREC_COVID = DatasetInfo("Uri-ka/ClusTREC-Covid", "ClusTREC-Covid")
    DIGIKALAMAG_CLUSTERING = DatasetInfo("mteb/DigikalamagClustering", "DigikalamagClustering")
    LIVEDOOR_NEWS_CLUSTERING_V2 = DatasetInfo("mteb/LivedoorNewsClustering.v2", "LivedoorNewsClusteringV2")
    """
    @misc{banar2025mtebnle5nlembeddingbenchmark,
      archiveprefix = {arXiv},
      author = {Nikolay Banar and Ehsan Lotfi and Jens Van Nooten and Cristina Arhiliuc and Marija Kliocaite and Walter Daelemans},
      eprint = {2509.12340},
      primaryclass = {cs.CL},
      title = {MTEB-NL and E5-NL: Embedding Benchmark and Models for Dutch},
      url = {https://arxiv.org/abs/2509.12340},
      year = {2025},
    }
    """
    OPEN_TENTER_CLUSTERING_P2P = DatasetInfo("clips/mteb-nl-opentender-cls-pr", "OpenTenderClusteringP2P")
    """
    @inproceedings{wehrli-etal-2023-german,
        title = "{G}erman Text Embedding Clustering Benchmark",
        author = "Wehrli, Silvan  and
          Arnrich, Bert  and
          Irrgang, Christopher",
        editor = "Georges, Munir  and
          Herygers, Aaricia  and
          Friedrich, Annemarie  and
          Roth, Benjamin",
        booktitle = "Proceedings of the 19th Conference on Natural Language Processing (KONVENS 2023)",
        month = sep,
        year = "2023",
        address = "Ingolstadt, Germany",
        publisher = "Association for Computational Lingustics",
        url = "https://aclanthology.org/2023.konvens-main.20",
        pages = "187--201",
    }
    """
    TEN_K_GNAD_CLUSTERING_P2P = DatasetInfo("slvnwhrl/tenkgnad-clustering-p2p", "10KgnadClusteringP2P")
    r"""
    @article{kasmaee2024chemteb,
      author = {Kasmaee, Ali Shiraee and Khodadad, Mohammad and Saloot, Mohammad Arshi and Sherck, Nick and Dokas, Stephen and Mahyar, Hamidreza and Samiee, Soheila},
      journal = {arXiv preprint arXiv:2412.00532},
      title = {ChemTEB: Chemical Text Embedding Benchmark, an Overview of Embedding Models Performance \& Efficiency on a Specific Domain},
      year = {2024},
    }
    """
    WIKIPEDIA_CHEMISTRY_TOPIC_CLUSTERING = DatasetInfo("BASF-AI/WikipediaEasy10Clustering", "WikipediaEasy10Clustering")
    r"""
    @article{kasmaee2024chemteb,
      author = {Kasmaee, Ali Shiraee and Khodadad, Mohammad and Saloot, Mohammad Arshi and Sherck, Nick and Dokas, Stephen and Mahyar, Hamidreza and Samiee, Soheila},
      journal = {arXiv preprint arXiv:2412.00532},
      title = {ChemTEB: Chemical Text Embedding Benchmark, an Overview of Embedding Models Performance \& Efficiency on a Specific Domain},
      year = {2024},
    }
    """
    WIKIPEDIA_SPECIALITIES_IN_CHEMISTRY_TOPIC_CLUSTERING = DatasetInfo(
        "BASF-AI/WikipediaMedium5Clustering", "WikipediaMedium5Clustering"
    )
    BIG_PATENT_CLUSTERING = DatasetInfo("jinaai/big-patent-clustering", "BigPatentClusteringP2P")
    """
    @article{shahinmoghadam2024benchmarking,
      author = {Shahinmoghadam, Mehrzad and Motamedi, Ali},
      journal = {arXiv preprint arXiv:2411.12056},
      title = {Benchmarking pre-trained text embedding models in aligning built asset information},
      year = {2024},
    }
    """
    BUILT_BENCH_CLUSTERING_P2P = DatasetInfo("mehrzad-shahin/BuiltBench-clustering-p2p", "BuiltBenchP2P")
    EMOTION = DatasetInfo("mteb/emotion", "Emotion")
    """
    @misc{lef23,
      author = {Lefebvre-Brossard, Antoine and Gazaille, Stephane and Desmarais, Michel C.},
      copyright = {Creative Commons Attribution Non Commercial Share Alike 4.0 International},
      doi = {10.48550/ARXIV.2302.07738},
      keywords = {Computation and Language (cs.CL), Information Retrieval (cs.IR), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
      publisher = {arXiv},
      title = {Alloprof: a new French question-answer education dataset and its use in an information retrieval case study},
      url = {https://arxiv.org/abs/2302.07738},
      year = {2023},
    }
    """
    ALLO_PROF_CLUSTERING_P2P = DatasetInfo("mteb/AlloProfClusteringP2P", "AlloProfClusteringP2P")
    TWENTY_NEWSGROUPS_V2 = DatasetInfo("mteb/llm-eval-twenty_newsgroups_v2", "20NewsgroupsV2")
    BANKING77 = DatasetInfo("mteb/llm-eval-banking77", "Banking77")
    BIG_PATENT = DatasetInfo("mteb/llm-eval-big_patent_clustering", "BigPatentClustering")
    DBPEDIA_14 = DatasetInfo("mteb/llm-eval-dbpedia_14", "DBPedia")
    """
    @online{wikidump2024,
      author = {Wikimedia Foundation},
      title = {Wikimedia Downloads},
      url = {https://dumps.wikimedia.org},
    }
    """
    WIKICITIES = DatasetInfo("mteb/WikiCitiesClustering", "Wikicities")
    TWEET_TOPIC_SINGLE = DatasetInfo("mteb/tweet_topic_single", "TweetTopicSingle")
    """
    @misc{twenty_newsgroups_113,
      author       = {Mitchell, Tom},
      title        = {{Twenty Newsgroups}},
      year         = {1997},
      howpublished = {UCI Machine Learning Repository},
      note         = {{DOI}: https://doi.org/10.24432/C5C323}
    }
    """
    TWENTY_NEWSGROUPS = DatasetInfo("sklearn/20newsgroups", "20Newsgroups")


custom_text_dataset_properties = {
    TextDatasets.CLUSTREC_COVID: {"text": "title", "label": "topic_id", "subset": "ClusTREC-Covid"},
}


class DatasetFactory:
    @staticmethod
    def from_string(name: str) -> TextDatasets:
        for dataset in TextDatasets:
            if dataset.value.technical_name == name:
                return dataset
        raise ValueError(f"Unknown dataset name: {name}")

    @staticmethod
    def get_dataset_name_map() -> dict[str, str]:
        return {ds.value.technical_name: ds.value.visual_name for ds in reversed(list(TextDatasets))}


@dataclass
class TextDataset:
    name: str
    encoding_model: str
    _embeddings: np.ndarray | None = None
    _labels: np.ndarray | None = None
    _texts: Sequence[str] | None = None

    @property
    def embeddings(self) -> np.ndarray:
        if self._embeddings is None:
            emb_dir = get_emb_dir(self.name, self.encoding_model)
            if os.path.exists(emb_dir):
                # mmap_mode='r': the OS maps the file into address space without
                # copying it.  Multiple processes opening the same .npy file share
                # the underlying physical pages (OS page cache), so N workers on the
                # same dataset do not multiply RAM usage by N.
                embeddings = np.load(os.path.join(emb_dir, "embeddings.npy"), mmap_mode="r")
                self._embeddings = embeddings
            else:
                raise ValueError(
                    f"Embeddings not found for dataset {self.name} with encoding model {self.encoding_model}. Please generate embeddings first."
                )
        assert self._embeddings is not None, "Embeddings should not be None after loading."
        return self._embeddings

    @embeddings.setter
    def embeddings(self, value: np.ndarray):
        self._embeddings = value

    @property
    def labels(self) -> np.ndarray:
        if self._labels is None:
            emb_dir = get_emb_dir(self.name, self.encoding_model)
            if os.path.exists(emb_dir):
                df = pd.read_parquet(os.path.join(emb_dir, "data.parquet"), columns=["label"])
                self._labels = np.asarray(df["label"].values)
            else:
                raise ValueError(
                    f"Labels not found for dataset {self.name} with encoding model {self.encoding_model}. Please generate embeddings first."
                )
        assert self._labels is not None, "Labels should not be None after loading."
        return self._labels

    @labels.setter
    def labels(self, value: np.ndarray):
        self._labels = value

    @property
    def texts(self) -> Sequence[str]:
        if self._texts is None:
            emb_dir = get_emb_dir(self.name, self.encoding_model)
            if os.path.exists(emb_dir):
                df = pd.read_parquet(os.path.join(emb_dir, "data.parquet"), columns=["text"])
                self._texts = df["text"].tolist()
            else:
                raise ValueError(
                    f"Texts not found for dataset {self.name} with encoding model {self.encoding_model}. Please generate embeddings first."
                )
        assert self._texts is not None, "Texts should not be None after loading."
        return self._texts

    @texts.setter
    def texts(self, value: Sequence[str]):
        self._texts = value

    @staticmethod
    def get(texts: Sequence[str], labels: np.ndarray, name: str, encoding_model: str) -> TextDataset:
        emb_dir = get_emb_dir(name, encoding_model)
        if os.path.exists(emb_dir):
            return TextDataset.from_dir(emb_dir)
        x = embed_texts(encoding_model, texts)
        data = TextDataset(
            name=name,
            encoding_model=encoding_model,
            _embeddings=x,
            _labels=labels,
            _texts=texts,
        )
        data.persist()
        return data

    def persist(self):
        emb_dir = get_emb_dir(self.name, self.encoding_model)
        if not os.path.exists(emb_dir):
            os.makedirs(emb_dir)
        np.save(os.path.join(emb_dir, "embeddings.npy"), self.embeddings)
        df = pd.DataFrame()
        df["text"] = self.texts
        df["label"] = self.labels
        df.to_parquet(os.path.join(emb_dir, "data.parquet"))
        with open(f"{emb_dir}/metadata.json", "w") as f:
            json_str = json.dumps(
                {
                    "name": self.name,
                    "encoding_model": self.encoding_model,
                    "len": len(self.texts),
                },
                indent=2,
            )
            f.write(json_str)

    @staticmethod
    def from_dir(emb_dir: str) -> TextDataset:
        with open(f"{emb_dir}/metadata.json") as f:
            metadata = json.load(f)
        return TextDataset(
            name=metadata["name"],
            encoding_model=metadata["encoding_model"],
        )

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray, str]:
        return self.embeddings[idx], self.labels[idx], self.texts[idx]

    def __repr__(self) -> str:
        return f"Dataset(name={self.name}, n_samples={len(self)}, embedding_dim={self.embeddings.shape[1]})"

    def __str__(self) -> str:
        return self.__repr__()


class DatasetManager:
    def __init__(self, encoding_model: str):
        self.encoding_model = encoding_model

    _cache: ClassVar[dict[tuple[TextDatasets, str], TextDataset]] = {}

    def get(self, dataset_id: TextDatasets, skip_cache: bool = False) -> TextDataset:
        if skip_cache:
            return self._get_dataset(dataset_id, self.encoding_model)
        cache_key = (dataset_id, self.encoding_model)
        if cache_key not in self._cache:
            dataset = self._get_dataset(dataset_id, self.encoding_model)
            self._cache[cache_key] = dataset
        return self._cache[cache_key]

    def clear_cache(self):
        self._cache.clear()

    @staticmethod
    def _get_dataset(dataset: TextDatasets, encoding_model: str) -> TextDataset:
        name = dataset.value.technical_name
        custom_datasets = set([i.value.technical_name for i in custom_text_dataset_properties.keys()])
        if name == "sklearn/20newsgroups":
            return DatasetManager.get_20newsgroups_embedding(encoding_model)
        elif name in custom_datasets:
            return DatasetManager.get_custom_dataset(dataset, encoding_model)
        else:
            return DatasetManager.get_mteb_clustering_data(name, encoding_model)

    @staticmethod
    def _get_cached_dataset(dataset_name: str, model_name: str) -> TextDataset | None:
        emb_dir = get_emb_dir(dataset_name, model_name)
        if os.path.exists(emb_dir):
            return TextDataset.from_dir(emb_dir)
        else:
            return None

    @staticmethod
    def get_custom_dataset(dataset: TextDatasets, model_name: str) -> TextDataset:
        text_col = custom_text_dataset_properties[dataset].get("text", None)
        label_col = custom_text_dataset_properties[dataset].get("label", None)
        split_names = cast(list[str], custom_text_dataset_properties[dataset].get("split_names", None))
        subset = custom_text_dataset_properties[dataset].get("subset", None)
        return DatasetManager.get_mteb_clustering_data(
            dataset.value.technical_name,
            model_name,
            custom_text_col=text_col,
            custom_label_col=label_col,
            custom_splits=split_names,
            subset=subset,
        )

    @staticmethod
    def get_20newsgroups_embedding(model: str):
        ng = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
        texts, y = ng.data, np.array(ng.target)

        return TextDataset.get(
            texts=texts,
            labels=y,
            name="20newsgroups",
            encoding_model=model,
        )

    @staticmethod
    def get_mteb_clustering_data(
        dataset_name: str,
        model: str,
        custom_text_col: str | None = None,
        custom_label_col: str | None = None,
        custom_splits: list[str] | None = None,
        subset: str | None = None,
    ) -> TextDataset:
        cached_dataset = DatasetManager._get_cached_dataset(dataset_name, model)
        if cached_dataset is not None:
            return cached_dataset
        if subset is not None:
            dataset = load_dataset(dataset_name, subset)
        else:
            dataset = load_dataset(dataset_name)

        all_sentences = []
        all_labels = []
        seen = set()

        split_names = custom_splits if custom_splits is not None else dataset.keys()

        for split_name in split_names:
            split_data = dataset[split_name]

            for row in split_data:
                if custom_text_col is not None:
                    text_col = custom_text_col
                elif "sentences" in row:
                    text_col = "sentences"
                elif "text" in row:
                    text_col = "text"
                else:
                    raise ValueError("Unexpected text col")

                if custom_label_col is not None:
                    row_col = custom_label_col
                elif "labels" in row:
                    row_col = "labels"
                elif "label" in row:
                    row_col = "label"
                else:
                    raise ValueError("Unexpected label col")
                if isinstance(row[row_col], str) or isinstance(row[row_col], int):
                    sentence = row[text_col]
                    label = row[row_col]
                    if sentence not in seen:
                        seen.add(sentence)
                        all_sentences.append(sentence)
                        all_labels.append(str(label))
                elif isinstance(row[row_col], Sequence):
                    for sentence, label in zip(row[text_col], row[row_col], strict=False):
                        if sentence not in seen:
                            seen.add(sentence)
                            all_sentences.append(sentence)
                            all_labels.append(label)
                else:
                    raise ValueError(f"Unexpected type: {type(row[row_col])}. Must be str, int or Sequence")

        if not all_sentences:
            raise ValueError(f"Dataset {dataset_name} is empty after preprocessing")

        label_mapping: dict[object, int] = {}
        encoded_labels = []
        for label in all_labels:
            if label not in label_mapping:
                label_mapping[label] = len(label_mapping)
            encoded_labels.append(label_mapping[label])

        if len(all_sentences) != len(encoded_labels):
            raise ValueError(
                f"Label/text length mismatch for {dataset_name}: "
                f"texts={len(all_sentences)}, labels={len(encoded_labels)}"
            )

        return TextDataset.get(
            all_sentences, np.array(encoded_labels, dtype=np.int64), name=dataset_name, encoding_model=model
        )
