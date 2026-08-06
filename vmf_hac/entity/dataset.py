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

from vmf_hac.functions.dataset import embed_texts, ensure_valid_embeddings, get_emb_dir


class TextDatasets(Enum):
    """
    @article{enevoldsen2025mmtebmassivemultilingualtext,
      title={MMTEB: Massive Multilingual Text Embedding Benchmark},
      author={Kenneth Enevoldsen and Isaac Chung and Imene Kerboua and Márton Kardos and Ashwin Mathur and David Stap and Jay Gala and Wissam Siblini and Dominik Krzemiński and Genta Indra Winata and Saba Sturua and Saiteja Utpala and Mathieu Ciancone and Marion Schaeffer and Gabriel Sequeira and Diganta Misra and Shreeya Dhakal and Jonathan Rystrøm and Roman Solomatin and Ömer Çağatan and Akash Kundu and Martin Bernstorff and Shitao Xiao and Akshita Sukhlecha and Bhavish Pahwa and Rafał Poświata and Kranthi Kiran GV and Shawon Ashraf and Daniel Auras and Björn Plüster and Jan Philipp Harries and Loïc Magne and Isabelle Mohr and Mariya Hendriksen and Dawei Zhu and Hippolyte Gisserot-Boukhlef and Tom Aarsen and Jan Kostkan and Konrad Wojtasik and Taemin Lee and Marek Šuppa and Crystina Zhang and Roberta Rocca and Mohammed Hamdy and Andrianos Michail and John Yang and Manuel Faysse and Aleksei Vatolin and Nandan Thakur and Manan Dey and Dipam Vasani and Pranjal Chitale and Simone Tedeschi and Nguyen Tai and Artem Snegirev and Michael Günther and Mengzhou Xia and Weijia Shi and Xing Han Lù and Jordan Clive and Gayatri Krishnakumar and Anna Maksimova and Silvan Wehrli and Maria Tikhonova and Henil Panchal and Aleksandr Abramov and Malte Ostendorff and Zheng Liu and Simon Clematide and Lester James Miranda and Alena Fenogenova and Guangyu Song and Ruqiya Bin Safi and Wen-Ding Li and Alessia Borghini and Federico Cassano and Hongjin Su and Jimmy Lin and Howard Yen and Lasse Hansen and Sara Hooker and Chenghao Xiao and Vaibhav Adlakha and Orion Weller and Siva Reddy and Niklas Muennighoff},
      publisher = {arXiv},
      journal={arXiv preprint arXiv:2502.13595},
      year={2025},
      url={https://arxiv.org/abs/2502.13595},
      doi = {10.48550/arXiv.2502.13595},
    }
    @article{muennighoff2022mteb,
      author = {Muennighoff, Niklas and Tazi, Nouamane and Magne, Lo{\"\\i}c and Reimers, Nils},
      title = {MTEB: Massive Text Embedding Benchmark},
      publisher = {arXiv},
      journal={arXiv preprint arXiv:2210.07316},
      year = {2022}
      url = {https://arxiv.org/abs/2210.07316},
      doi = {10.48550/ARXIV.2210.07316},
    }
    """

    ARXIV_CLUSTERING_S2S = "mteb/arxiv-clustering-s2s"
    BIORXIV_CLUSTERING_S2S = "mteb/biorxiv-clustering-s2s"
    MEDRXIV_CLUSTERING_S2S = "mteb/medrxiv-clustering-s2s"
    ARXIV_CLUSTERING_P2P = "mteb/arxiv-clustering-p2p"
    BIORXIV_CLUSTERING_P2P = "mteb/biorxiv-clustering-p2p"
    MEDRXIV_CLUSTERING_P2P = "mteb/medrxiv-clustering-p2p"
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
    REDDIT_CLUSTERING = "mteb/reddit-clustering"
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
    REDDIT_CLUSTERING_P2P = "mteb/reddit-clustering-p2p"
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
    STACKEXCHANGE_CLUSTERING = "mteb/stackexchange-clustering"
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
    STACKEXCHANGE_CLUSTERING_P2P = "mteb/stackexchange-clustering-p2p"
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
    STACKEXCHANGE_CLUSTERING_VN = "GreenNode/stackexchange-clustering-vn"
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
    STACKEXCHANGE_CLUSTERING_P2P_VN = "GreenNode/stackexchange-clustering-p2p-vn"
    ALLO_PROF_CLUSTERING_S2S = "mteb/AlloProfClusteringS2S"
    """
    @article{li2022csl,
      author = {Li, Yudong and Zhang, Yuqing and Zhao, Zhe and Shen, Linlin and Liu, Weijie and Mao, Weiquan and Zhang, Hui},
      journal = {arXiv preprint arXiv:2209.05034},
      title = {CSL: A large-scale Chinese scientific literature dataset},
      year = {2022},
    }
    """
    CLS_CLUSTERING_P2P = "mteb/CLSClusteringP2P"
    """
    @article{li2022csl,
      author = {Li, Yudong and Zhang, Yuqing and Zhao, Zhe and Shen, Linlin and Liu, Weijie and Mao, Weiquan and Zhang, Hui},
      journal = {arXiv preprint arXiv:2209.05034},
      title = {CSL: A large-scale Chinese scientific literature dataset},
      year = {2022},
    }
    """
    CLS_CLUSTERING_P2P_V2 = "mteb/CLSClusteringP2P.v2"
    """
    @article{li2022csl,
             author = {Li, Yudong and Zhang, Yuqing and Zhao, Zhe and Shen, Linlin and Liu, Weijie and Mao, Weiquan and Zhang, Hui},
    journal = {arXiv preprint arXiv:2209.05034},
    title = {CSL: A large-scale Chinese scientific literature dataset},
    year = {2022},
    }
    """
    CLS_CLUSTERING_S2S = "C-MTEB/CLSClusteringS2S"
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
    CLUSTREC_COVID = "Uri-ka/ClusTREC-Covid"
    DIGIKALAMAG_CLUSTERING = "mteb/DigikalamagClustering"
    """
    @misc{ciancone2024extending,
      archiveprefix = {arXiv},
      author = {Mathieu Ciancone and Imene Kerboua and Marion Schaeffer and Wissam Siblini},
      eprint = {2405.20468},
      primaryclass = {cs.CL},
      title = {Extending the Massive Text Embedding Benchmark to French},
      year = {2024},
    }
    """
    HAL_CLUSTERING_S2S_V2 = "mteb/HALClusteringS2S.v2"
    """
    @article{doddapaneni2022towards,
      author = {Sumanth Doddapaneni and Rahul Aralikatte and Gowtham Ramesh and Shreyansh Goyal and Mitesh M. Khapra and Anoop Kunchukuttan and Pratyush Kumar},
      doi = {10.18653/v1/2023.acl-long.693},
      journal = {Annual Meeting of the Association for Computational Linguistics},
      title = {Towards Leaving No Indic Language Behind: Building Monolingual Corpora, Benchmark and Models for Indic Languages},
      year = {2022},
    }
    """
    INDIC_REVIEWS_CLUSTERING_P2P = "mteb/IndicReviewsClusteringP2P"
    """
    @misc{park2021klue,
      archiveprefix = {arXiv},
      author = {Sungjoon Park and Jihyung Moon and Sungdong Kim and Won Ik Cho and Jiyoon Han and Jangwon Park and Chisung Song and Junseong Kim and Yongsook Song and Taehwan Oh and Joohong Lee and Juhyun Oh and Sungwon Lyu and Younghoon Jeong and Inkwon Lee and Sangwoo Seo and Dongjun Lee and Hyunwoo Kim and Myeonghwa Lee and Seongbo Jang and Seungwon Do and Sunkyoung Kim and Kyungtae Lim and Jongwon Lee and Kyumin Park and Jamin Shin and Seonghyun Kim and Lucy Park and Alice Oh and Jungwoo Ha and Kyunghyun Cho},
      eprint = {2105.09680},
      primaryclass = {cs.CL},
      title = {KLUE: Korean Language Understanding Evaluation},
      year = {2021},
    }
    """
    KLUE_MRC_DOMAIN_CLUSTERING = "mteb/KlueMrcDomainClustering"
    """
    @misc{park2021klue,
      archiveprefix = {arXiv},
      author = {Sungjoon Park and Jihyung Moon and Sungdong Kim and Won Ik Cho and Jiyoon Han and Jangwon Park and Chisung Song and Junseong Kim and Yongsook Song and Taehwan Oh and Joohong Lee and Juhyun Oh and Sungwon Lyu and Younghoon Jeong and Inkwon Lee and Sangwoo Seo and Dongjun Lee and Hyunwoo Kim and Myeonghwa Lee and Seongbo Jang and Seungwon Do and Sunkyoung Kim and Kyungtae Lim and Jongwon Lee and Kyumin Park and Jamin Shin and Seonghyun Kim and Lucy Park and Alice Oh and Jungwoo Ha and Kyunghyun Cho},
      eprint = {2105.09680},
      primaryclass = {cs.CL},
      title = {KLUE: Korean Language Understanding Evaluation},
      year = {2021},
    }
    """
    KLUE_YNAT_MRC_CATEGORY_CLUSTERING = "mteb/KlueYnatMrcCategoryClustering"
    LIVEDOOR_NEWS_CLUSTERING = "mteb/LivedoorNewsClustering"
    LIVEDOOR_NEWS_CLUSTERING_V2 = "mteb/LivedoorNewsClustering.v2"
    """
    @article{scialom2020mlsum,
      author = {Scialom, Thomas and Dray, Paul-Alexis and Lamprier, Sylvain and Piwowarski, Benjamin and Staiano, Jacopo},
      journal = {arXiv preprint arXiv:2004.14900},
      title = {MLSUM: The Multilingual Summarization Corpus},
      year = {2020},
    }
    """
    MLSUM_CLUSTERING_S2S_V2 = "mteb/mlsum"
    """
    @article{scialom2020mlsum,
      author = {Scialom, Thomas and Dray, Paul-Alexis and Lamprier, Sylvain and Piwowarski, Benjamin and Staiano, Jacopo},
      journal = {arXiv preprint arXiv:2004.14900},
      title = {MLSUM: The Multilingual Summarization Corpus},
      year = {2020},
    }
    """
    MLSUM_CLUSTERING_P2P_V2 = "mteb/mlsum"
    """
    @article{adelani2023masakhanews,
      author = {David Ifeoluwa Adelani and  Marek Masiak and  Israel Abebe Azime and  Jesujoba Oluwadara Alabi and  Atnafu Lambebo Tonja and  Christine Mwase and  Odunayo Ogundepo and  Bonaventure F. P. Dossou and  Akintunde Oladipo and  Doreen Nixdorf and  Chris Chinenye Emezue and  Sana Sabah al-azzawi and  Blessing K. Sibanda and  Davis David and  Lolwethu Ndolela and  Jonathan Mukiibi and  Tunde Oluwaseyi Ajayi and  Tatiana Moteu Ngoli and  Brian Odhiambo and  Abraham Toluwase Owodunni and  Nnaemeka C. Obiefuna and  Shamsuddeen Hassan Muhammad and  Saheed Salahudeen Abdullahi and  Mesay Gemeda Yigezu and  Tajuddeen Gwadabe and  Idris Abdulmumin and  Mahlet Taye Bame and  Oluwabusayo Olufunke Awoyomi and  Iyanuoluwa Shode and  Tolulope Anu Adelani and  Habiba Abdulganiy Kailani and  Abdul-Hakeem Omotayo and  Adetola Adeeko and  Afolabi Abeeb and  Anuoluwapo Aremu and  Olanrewaju Samuel and  Clemencia Siro and  Wangari Kimotho and  Onyekachi Raphael Ogbu and  Chinedu E. Mbonu and  Chiamaka I. Chukwuneke and  Samuel Fanijo and  Jessica Ojo and  Oyinkansola F. Awosan and  Tadesse Kebede Guge and  Sakayo Toadoum Sari and  Pamela Nyatsine and  Freedmore Sidume and  Oreen Yousuf and  Mardiyyah Oduwole and  Ussen Kimanuka and  Kanda Patrick Tshinu and  Thina Diko and  Siyanda Nxakama and   Abdulmejid Tuni Johar and  Sinodos Gebre and  Muhidin Mohamed and  Shafie Abdi Mohamed and  Fuad Mire Hassan and  Moges Ahmed Mehamed and  Evrard Ngabire and  and Pontus Stenetorp},
      journal = {ArXiv},
      title = {MasakhaNEWS: News Topic Classification for African languages},
      volume = {},
      year = {2023},
    }
    """
    MASAKHA_NEWS_CLUSTERING_P2P = "mteb/MasakhaNEWSClusteringP2P"
    """
    @article{adelani2023masakhanews,
      author = {David Ifeoluwa Adelani and  Marek Masiak and  Israel Abebe Azime and  Jesujoba Oluwadara Alabi and  Atnafu Lambebo Tonja and  Christine Mwase and  Odunayo Ogundepo and  Bonaventure F. P. Dossou and  Akintunde Oladipo and  Doreen Nixdorf and  Chris Chinenye Emezue and  Sana Sabah al-azzawi and  Blessing K. Sibanda and  Davis David and  Lolwethu Ndolela and  Jonathan Mukiibi and  Tunde Oluwaseyi Ajayi and  Tatiana Moteu Ngoli and  Brian Odhiambo and  Abraham Toluwase Owodunni and  Nnaemeka C. Obiefuna and  Shamsuddeen Hassan Muhammad and  Saheed Salahudeen Abdullahi and  Mesay Gemeda Yigezu and  Tajuddeen Gwadabe and  Idris Abdulmumin and  Mahlet Taye Bame and  Oluwabusayo Olufunke Awoyomi and  Iyanuoluwa Shode and  Tolulope Anu Adelani and  Habiba Abdulganiy Kailani and  Abdul-Hakeem Omotayo and  Adetola Adeeko and  Afolabi Abeeb and  Anuoluwapo Aremu and  Olanrewaju Samuel and  Clemencia Siro and  Wangari Kimotho and  Onyekachi Raphael Ogbu and  Chinedu E. Mbonu and  Chiamaka I. Chukwuneke and  Samuel Fanijo and  Jessica Ojo and  Oyinkansola F. Awosan and  Tadesse Kebede Guge and  Sakayo Toadoum Sari and  Pamela Nyatsine and  Freedmore Sidume and  Oreen Yousuf and  Mardiyyah Oduwole and  Ussen Kimanuka and  Kanda Patrick Tshinu and  Thina Diko and  Siyanda Nxakama and   Abdulmejid Tuni Johar and  Sinodos Gebre and  Muhidin Mohamed and  Shafie Abdi Mohamed and  Fuad Mire Hassan and  Moges Ahmed Mehamed and  Evrard Ngabire and  and Pontus Stenetorp},
      journal = {ArXiv},
      title = {MasakhaNEWS: News Topic Classification for African languages},
      volume = {},
      year = {2023},
    }
    """
    MASAKHA_NEWS_CLUSTERING_S2S = "mteb/MasakhaNEWSClusteringS2S"
    """
    @inproceedings{nishikawa-etal-2022-ease,
      address = {Seattle, United States},
      author = {Nishikawa, Sosuke  and
    Ri, Ryokan  and
    Yamada, Ikuya  and
    Tsuruoka, Yoshimasa  and
    Echizen, Isao},
      booktitle = {Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
      month = jul,
      pages = {3870--3885},
      publisher = {Association for Computational Linguistics},
      title = {{EASE}: Entity-Aware Contrastive Learning of Sentence Embedding},
      url = {https://aclanthology.org/2022.naacl-main.284},
      year = {2022},
    }
    """
    MEWS_C16_JA_CLUSTERING = "mteb/MewsC16JaClustering"
    NLP_TWITTER_ANALYSIS = "hamedhf/nlp_twitter_analysis"
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
    OPEN_TENTER_CLUSTERING_P2P = "clips/mteb-nl-opentender-cls-pr"
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
    OPEN_TENTER_CLUSTERING_S2S = "clips/mteb-nl-opentender-clst-s2s-pr"
    PLSC_CLUSTERING_P2P_V2 = "mteb/PlscClusteringP2P.v2"
    PLSC_CLUSTERING_S2S_V2 = "PL-MTEB/plsc-clustering-s2s"
    """
    @misc{pham2025vnmtebvietnamesemassivetext,
      archiveprefix = {arXiv},
      author = {Loc Pham and Tung Luu and Thu Vo and Minh Nguyen and Viet Hoang},
      eprint = {2507.21500},
      primaryclass = {cs.CL},
      title = {VN-MTEB: Vietnamese Massive Text Embedding Benchmark},
      url = {https://arxiv.org/abs/2507.21500},
      year = {2025},
    }
    """
    REDDIT_CLUSTERING_VN = "GreenNode/reddit-clustering-vn"
    """
    @misc{pham2025vnmtebvietnamesemassivetext,
      archiveprefix = {arXiv},
      author = {Loc Pham and Tung Luu and Thu Vo and Minh Nguyen and Viet Hoang},
      eprint = {2507.21500},
      primaryclass = {cs.CL},
      title = {VN-MTEB: Vietnamese Massive Text Embedding Benchmark},
      url = {https://arxiv.org/abs/2507.21500},
      year = {2025},
    }
    """
    REDDIT_CLUSTERING_P2P_VN = "GreenNode/reddit-clustering-p2p-vn"
    RU_SCI_BENCH_GRNTI_CLUSTERING_P2P = "ai-forever/ru-scibench-grnti-classification"
    RU_SCI_BENCH_OECD_CLUSTERING_P2P = "ai-forever/ru-scibench-oecd-classification"
    """
    @misc{pham2025vnmtebvietnamesemassivetext,
      archiveprefix = {arXiv},
      author = {Loc Pham and Tung Luu and Thu Vo and Minh Nguyen and Viet Hoang},
      eprint = {2507.21500},
      primaryclass = {cs.CL},
      title = {VN-MTEB: Vietnamese Massive Text Embedding Benchmark},
      url = {https://arxiv.org/abs/2507.21500},
      year = {2025},
    }
    """
    ROMANI_BIBLE_CLUSTERING = "mteb/RomaniBibleClustering"
    """
    @mastersthesis{navjord2023beyond,
      author = {Navjord, J{\\o}rgen Johnsen and Korsvik, Jon-Mikkel Ryen},
      school = {Norwegian University of Life Sciences, {\\AA}s},
      title = {Beyond extractive: advancing abstractive automatic text summarization in Norwegian with transformers},
      year = {2023},
    }
    """
    SNL_HIERARCHICAL_CLUSTERING_P2P = "mteb/SNLHierarchicalClusteringP2P"
    """
    @mastersthesis{navjord2023beyond,
      author = {Navjord, J{\\o}rgen Johnsen and Korsvik, Jon-Mikkel Ryen},
      school = {Norwegian University of Life Sciences, {\\AA}s},
      title = {Beyond extractive: advancing abstractive automatic text summarization in Norwegian with transformers},
      year = {2023},
    }
    """
    SNL_HIERARCHICAL_CLUSTERING_S2S = "mteb/SNLHierarchicalClusteringS2S"
    """
    @inproceedings{monsen2021method,
      author = {Monsen, Julius and J{\"o}nsson, Arne},
      booktitle = {Proceedings of CLARIN Annual Conference},
      title = {A method for building non-english corpora for abstractive text summarization},
      year = {2021},
    }
    """
    SWEDN_CLUSTERING = "mteb/SwednClustering"
    """
    @inproceedings{monsen2021method,
      author = {Monsen, Julius and J{\"o}nsson, Arne},
      booktitle = {Proceedings of CLARIN Annual Conference},
      title = {A method for building non-english corpora for abstractive text summarization},
      year = {2021},
    }
    """
    SWEDN_CLUSTERING_P2P = "mteb/SwednClusteringP2P"
    """
    @inproceedings{monsen2021method,
      author = {Monsen, Julius and J{\"o}nsson, Arne},
      booktitle = {Proceedings of CLARIN Annual Conference},
      title = {A method for building non-english corpora for abstractive text summarization},
      year = {2021},
    }
    """
    SWEDN_CLUSTERING_S2S = "mteb/SwednClusteringS2S"
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
    TEN_K_GNAD_CLUSTERING_P2P = "slvnwhrl/tenkgnad-clustering-p2p"
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
    TEN_K_GNAD_CLUSTERING_S2S = "slvnwhrl/tenkgnad-clustering-s2s"
    """
    @inproceedings{eisner2007proceedings,
      author = {Eisner, Jason},
      booktitle = {Proceedings of the 2007 Joint Conference on Empirical Methods in Natural Language Processing and Computational Natural Language Learning (EMNLP-CoNLL)},
      title = {Proceedings of the 2007 joint conference on empirical methods in natural language processing and computational natural language learning (EMNLP-CoNLL)},
      year = {2007},
    }
    @inproceedings{li2006comparison,
      author = {Li, Jingyang and Sun, Maosong and Zhang, Xian},
      booktitle = {proceedings of the 21st international conference on computational linguistics and 44th annual meeting of the association for computational linguistics},
      pages = {545--552},
      title = {A comparison and semi-quantitative analysis of words and character-bigrams as features in chinese text categorization},
      year = {2006},
    }
    """
    THU_NEWS_CLUSTERING_P2P = "C-MTEB/ThuNewsClusteringP2P"
    """
    @inproceedings{eisner2007proceedings,
      author = {Eisner, Jason},
      booktitle = {Proceedings of the 2007 Joint Conference on Empirical Methods in Natural Language Processing and Computational Natural Language Learning (EMNLP-CoNLL)},
      title = {Proceedings of the 2007 joint conference on empirical methods in natural language processing and computational natural language learning (EMNLP-CoNLL)},
      year = {2007},
    }
    @inproceedings{li2006comparison,
      author = {Li, Jingyang and Sun, Maosong and Zhang, Xian},
      booktitle = {proceedings of the 21st international conference on computational linguistics and 44th annual meeting of the association for computational linguistics},
      pages = {545--552},
      title = {A comparison and semi-quantitative analysis of words and character-bigrams as features in chinese text categorization},
      year = {2006},
    }
    """
    THU_NEWS_CLUSTERING_S2S = "C-MTEB/ThuNewsClusteringS2S"
    """
    @misc{pham2025vnmtebvietnamesemassivetext,
      archiveprefix = {arXiv},
      author = {Loc Pham and Tung Luu and Thu Vo and Minh Nguyen and Viet Hoang},
      eprint = {2507.21500},
      primaryclass = {cs.CL},
      title = {VN-MTEB: Vietnamese Massive Text Embedding Benchmark},
      url = {https://arxiv.org/abs/2507.21500},
      year = {2025},
    }
    """
    TWENTY_NEWSGROUPS_CLUSTERING_VN = "GreenNode/twentynewsgroups-clustering-vn"
    """
    @dataset{aspeslagh2024vabb,
      author = {Aspeslagh, Pieter and Guns, Raf and Engels, Tim C. E.},
      doi = {10.5281/zenodo.14214806},
      publisher = {Zenodo},
      title = {VABB-SHW: Dataset of Flemish Academic Bibliography for the Social Sciences and Humanities (edition 14)},
      url = {https://doi.org/10.5281/zenodo.14214806},
      year = {2024},
    }
    """
    VABB_CLUSTERING_P2P = "clips/mteb-nl-vabb-cls"
    """
    @dataset{aspeslagh2024vabb,
      author = {Aspeslagh, Pieter and Guns, Raf and Engels, Tim C. E.},
      doi = {10.5281/zenodo.14214806},
      publisher = {Zenodo},
      title = {VABB-SHW: Dataset of Flemish Academic Bibliography for the Social Sciences and Humanities (edition 14)},
      url = {https://doi.org/10.5281/zenodo.14214806},
      year = {2024},
    }
    """
    VABB_CLUSTERING_S2S = "clips/mteb-nl-vabb-cls"
    r"""
    @article{kasmaee2024chemteb,
      author = {Kasmaee, Ali Shiraee and Khodadad, Mohammad and Saloot, Mohammad Arshi and Sherck, Nick and Dokas, Stephen and Mahyar, Hamidreza and Samiee, Soheila},
      journal = {arXiv preprint arXiv:2412.00532},
      title = {ChemTEB: Chemical Text Embedding Benchmark, an Overview of Embedding Models Performance \& Efficiency on a Specific Domain},
      year = {2024},
    }
    """
    WIKIPEDIA_CHEMISTRY_TOPIC_CLUSTERING = "BASF-AI/WikipediaEasy10Clustering"
    r"""
    @article{kasmaee2024chemteb,
      author = {Kasmaee, Ali Shiraee and Khodadad, Mohammad and Saloot, Mohammad Arshi and Sherck, Nick and Dokas, Stephen and Mahyar, Hamidreza and Samiee, Soheila},
      journal = {arXiv preprint arXiv:2412.00532},
      title = {ChemTEB: Chemical Text Embedding Benchmark, an Overview of Embedding Models Performance \& Efficiency on a Specific Domain},
      year = {2024},
    }
    """
    WIKIPEDIA_SPECIALITIES_IN_CHEMISTRY_TOPIC_CLUSTERING = "BASF-AI/WikipediaMedium5Clustering"
    TWENTY_NEWSGROUPS_CLUSTERING = "mteb/twentynewsgroups-clustering"
    WIKI_CLUSTERING_P2P_V2 = "mteb/WikiClusteringP2P.v2"
    BEYTOOTE_CLUSTERING = "MCINext/beytoote-clustering"
    BIG_PATENT_CLUSTERING = "jinaai/big-patent-clustering"
    BIG_PATENT_CLUSTERING_V2 = "mteb/big-patent"
    """
    @inproceedings{Remus2019GermEval2T,
      author = {Steffen Remus and Rami Aly and Chris Biemann},
      booktitle = {Conference on Natural Language Processing},
      title = {GermEval 2019 Task 1: Hierarchical Classification of Blurbs},
      url = {https://api.semanticscholar.org/CorpusID:208334484},
      year = {2019},
    }
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
    BLURBS_CLUSTERING_P2P = "slvnwhrl/blurbs-clustering-p2p"
    """
    @inproceedings{Remus2019GermEval2T,
      author = {Steffen Remus and Rami Aly and Chris Biemann},
      booktitle = {Conference on Natural Language Processing},
      title = {GermEval 2019 Task 1: Hierarchical Classification of Blurbs},
      url = {https://api.semanticscholar.org/CorpusID:208334484},
      year = {2019},
    }
    """
    BLURBS_CLUSTERING_S2S = "slvnwhrl/blurbs-clustering-s2s"
    """
    @article{shahinmoghadam2024benchmarking,
      author = {Shahinmoghadam, Mehrzad and Motamedi, Ali},
      journal = {arXiv preprint arXiv:2411.12056},
      title = {Benchmarking pre-trained text embedding models in aligning built asset information},
      year = {2024},
    }
    """
    BUILT_BENCH_CLUSTERING_P2P = "mehrzad-shahin/BuiltBench-clustering-p2p"
    """
    @misc{banar2025mtebnle5nlembeddingbenchmark,
      title={MTEB-NL and E5-NL: Embedding Benchmark and Models for Dutch},
      author={Nikolay Banar and Ehsan Lotfi and Jens Van Nooten and Cristina Arhiliuc and Marija Kliocaite and Walter Daelemans},
      year={2025},
      eprint={2509.12340},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2509.12340},
    }
    @misc{max_scheijen_2022,
        title={Dutch News Articles},
        url={https://www.kaggle.com/ds/1013130},
        DOI={10.34740/KAGGLE/DS/1013130},
        publisher={Kaggle},
        author={Max Scheijen},
        year={2022}
    }
    """
    DUTCH_NEWS_ARTICLES_CLUSTERING_P2P = "clips/mteb-nl-news-articles-cls"
    """
    @misc{banar2025mtebnle5nlembeddingbenchmark,
      title={MTEB-NL and E5-NL: Embedding Benchmark and Models for Dutch},
      author={Nikolay Banar and Ehsan Lotfi and Jens Van Nooten and Cristina Arhiliuc and Marija Kliocaite and Walter Daelemans},
      year={2025},
      eprint={2509.12340},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2509.12340},
    }
    @misc{max_scheijen_2022,
        title={Dutch News Articles},
        url={https://www.kaggle.com/ds/1013130},
        DOI={10.34740/KAGGLE/DS/1013130},
        publisher={Kaggle},
        author={Max Scheijen},
        year={2022}
    }
    """
    DUTCH_NEWS_ARTICLES_CLUSTERING_S2S = "clips/mteb-nl-news-articles-cls"
    """
    @inproceedings{dadas-etal-2020-evaluation,
      address = {Marseille, France},
      author = {Dadas, Slawomir  and
    Pere{\\l}kiewicz, Micha{\\l}  and
    Po{\\'s}wiata, Rafa{\\l}},
      booktitle = {Proceedings of the Twelfth Language Resources and Evaluation Conference},
      editor = {Calzolari, Nicoletta  and
    B{\'e}chet, Fr{\'e}d{\'e}ric  and
    Blache, Philippe  and
    Choukri, Khalid  and
    Cieri, Christopher  and
    Declerck, Thierry  and
    Goggi, Sara  and
    Isahara, Hitoshi  and
    Maegaard, Bente  and
    Mariani, Joseph  and
    Mazo, H{\\'e}l{\\`e}ne  and
    Moreno, Asuncion  and
    Odijk, Jan  and
    Piperidis, Stelios},
      isbn = {979-10-95546-34-4},
      language = {English},
      month = may,
      pages = {1674--1680},
      publisher = {European Language Resources Association},
      title = {Evaluation of Sentence Representations in {P}olish},
      url = {https://aclanthology.org/2020.lrec-1.207},
      year = {2020},
    }
    """
    EIGHT_TAGS_CLUSTERING = "PL-MTEB/8tags-clustering"
    GEOREVIEW_CLUSTERING_P2P = "ai-forever/georeview-clustering-p2p"
    """
    @misc{ciancone2024extending,
      title={Extending the Massive Text Embedding Benchmark to French},
      author={Mathieu Ciancone and Imene Kerboua and Marion Schaeffer and Wissam Siblini},
      year={2024},
      eprint={2405.20468},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
    }
    """
    HAL_CLUSTERING_S2S = "lyon-nlp/clustering-hal-s2s"  # use mteb_eval subset
    """
    @misc{arxiv_org_submitters_2024,
      author = {arXiv.org submitters},
      doi = {10.34740/KAGGLE/DSV/7548853},
      publisher = {Kaggle},
      title = {arXiv Dataset},
      url = {https://www.kaggle.com/dsv/7548853},
      year = {2024},
    }
    """
    HUME_ARXIV_CLUSTERING_P2P = "mteb/mteb-human-arxiv-clustering"
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
    HUME_REDDIT_CLUSTERING_S2S = "mteb/mteb-human-reddit-clustering"
    """
    @inproceedings{adelani-etal-2023-sib,
      address = {Toronto, Canada},
      author = {Adelani, David Ifeoluwa  and
    Hedderich, Michael A.  and
    Zhu, Dawei  and
    van den Berg, Esther  and
    Klakow, Dietrich},
      booktitle = {Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
      doi = {10.18653/v1/2023.acl-long.660},
      month = jul,
      pages = {11784--11801},
      publisher = {Association for Computational Linguistics},
      title = {{SIB}-200: A Large-Scale News Classification Dataset for Over 200 Languages},
      url = {https://aclanthology.org/2023.acl-long.660},
      year = {2023},
    }
    """
    HUME_SIB200_CLUSTERING_S2S = "mteb/mteb-human-sib200-clustering"
    """
   @article{adelani2023sib,
      author = {Adelani, David Ifeoluwa and Liu, Hannah and Shen, Xiaoyu and Vassilyev, Nikita and Alabi, Jesujoba O and Mao, Yanke and Gao, Haonan and Lee, Annie En-Shiun},
      journal = {arXiv preprint arXiv:2309.07445},
      title = {SIB-200: A simple, inclusive, and big evaluation dataset for topic classification in 200+ languages and dialects},
      year = {2023},
    }
    """
    SIB_200_CLUSTERING_S2S = "mteb/sib200"
    SID_CLUSTERING = "MCINext/sid-clustering"
    HAMSHAHRI_CLUSTERING = "community-datasets/farsi_news"
    """
    @article{banar2023transfer,
      author = {Banar, Nikolay and Daelemans, Walter and Kestemont, Mike},
      journal = {ACM Journal on Computing and Cultural Heritage},
      number = {2},
      pages = {1--16},
      publisher = {ACM New York, NY},
      title = {Transfer learning for the visual arts: The multi-modal retrieval of iconclass codes},
      volume = {16},
      year = {2023},
    }
    """
    ICONCLASS_CLUSTERING_S2S = "clips/mteb-nl-iconclass-cls"
    """
    @inproceedings{saravia-etal-2018-carer,
      abstract = {Emotions are expressed in nuanced ways, which varies by collective or individual experiences, knowledge, and beliefs. Therefore, to understand emotion, as conveyed through text, a robust mechanism capable of capturing and modeling different linguistic nuances and phenomena is needed. We propose a semi-supervised, graph-based algorithm to produce rich structural descriptors which serve as the building blocks for constructing contextualized affect representations from text. The pattern-based representations are further enriched with word embeddings and evaluated through several emotion recognition tasks. Our experimental results demonstrate that the proposed method outperforms state-of-the-art techniques on emotion recognition tasks.},
      address = {Brussels, Belgium},
      author = {Saravia, Elvis  and
    Liu, Hsien-Chi Toby  and
    Huang, Yen-Hao  and
    Wu, Junlin  and
    Chen, Yi-Shin},
      booktitle = {Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing},
      doi = {10.18653/v1/D18-1404},
      editor = {Riloff, Ellen  and
    Chiang, David  and
    Hockenmaier, Julia  and
    Tsujii, Jun{'}ichi},
      month = oct # {-} # nov,
      pages = {3687--3697},
      publisher = {Association for Computational Linguistics},
      title = {{CARER}: Contextualized Affect Representations for Emotion Recognition},
      url = {https://aclanthology.org/D18-1404},
      year = {2018},
    }
    """
    """
    @misc{kevinmorgado2019spanish,
      author = {Kevin Morgado},
      howpublished = {Kaggle},
      title = {Spanish News Classification},
      url = {https://www.kaggle.com/datasets/kevinmorgado/spanish-news-classification},
      year = {2019},
    }
    """
    SPANISH_NEWS_CLUSTERING_P2P = "jinaai/spanish_news_clustering"
    EMOTION = "mteb/emotion"
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
    ALLO_PROF_CLUSTERING_P2P = "mteb/AlloProfClusteringP2P"
    TWENTY_NEWSGROUPS_V2 = "mteb/llm-eval-twenty_newsgroups_v2"
    BANKING77 = "mteb/llm-eval-banking77"
    BIG_PATENT = "mteb/llm-eval-big_patent_clustering"
    DBPEDIA_14 = "mteb/llm-eval-dbpedia_14"
    """
    @online{wikidump2024,
      author = {Wikimedia Foundation},
      title = {Wikimedia Downloads},
      url = {https://dumps.wikimedia.org},
    }
    """
    WIKICITIES = "mteb/WikiCitiesClustering"
    TWEET_TOPIC_SINGLE = "mteb/tweet_topic_single"
    """
    @misc{twenty_newsgroups_113,
      author       = {Mitchell, Tom},
      title        = {{Twenty Newsgroups}},
      year         = {1997},
      howpublished = {UCI Machine Learning Repository},
      note         = {{DOI}: https://doi.org/10.24432/C5C323}
    }
    """
    TWENTY_NEWSGROUPS = "sklearn/20newsgroups"


custom_text_dataset_properties = {
    TextDatasets.CLUSTREC_COVID: {"text": "title", "label": "topic_id", "subset": "ClusTREC-Covid"},
    TextDatasets.NLP_TWITTER_ANALYSIS: {"text": "tweet", "label": "label"},
    TextDatasets.HAL_CLUSTERING_S2S: {"text": "title", "label": "hal_id"},
    TextDatasets.HAMSHAHRI_CLUSTERING: {"text": "title", "label": "tags"},
    TextDatasets.INDIC_REVIEWS_CLUSTERING_P2P: {"subset": "hi"},
    TextDatasets.MLSUM_CLUSTERING_P2P_V2: {"text": "text", "label": "topic", "subset": "de"},
    TextDatasets.MLSUM_CLUSTERING_S2S_V2: {"text": "summary", "label": "topic", "subset": "de"},
    TextDatasets.MASAKHA_NEWS_CLUSTERING_P2P: {"subset": "eng"},
    TextDatasets.MASAKHA_NEWS_CLUSTERING_S2S: {"subset": "eng"},
    TextDatasets.VABB_CLUSTERING_P2P: {"text": "abstract", "label": "org_discipline"},
    TextDatasets.VABB_CLUSTERING_S2S: {"text": "title", "label": "org_discipline"},
    TextDatasets.WIKI_CLUSTERING_P2P_V2: {"subset": "da"},
    TextDatasets.SIB_200_CLUSTERING_S2S: {"text": "text", "label": "category"},
}


class DatasetFactory:
    @staticmethod
    def from_string(name: str) -> TextDatasets:
        for dataset in TextDatasets:
            if dataset.value == name:
                return dataset
        raise ValueError(f"Unknown dataset name: {name}")


@dataclass
class TextDataset:
    name: str
    encoding_model: str
    embeddings: np.ndarray
    labels: np.ndarray
    texts: Sequence[str]

    @staticmethod
    def get(texts: Sequence[str], labels: np.ndarray, name: str, encoding_model: str) -> TextDataset:
        emb_dir = get_emb_dir(name, encoding_model)
        if os.path.exists(emb_dir):
            return TextDataset.from_dir(emb_dir)
        x = embed_texts(encoding_model, texts)
        data = TextDataset(name=name, encoding_model=encoding_model, embeddings=x, labels=labels, texts=texts)
        data.persist()
        return data

    def persist(self):
        emb_dir = get_emb_dir(self.name, self.encoding_model)
        if not os.path.exists(emb_dir):
            os.makedirs(emb_dir)
        embeddings = ensure_valid_embeddings(self.embeddings, source=f"{self.name}/{self.encoding_model}")
        np.save(os.path.join(emb_dir, "embeddings.npy"), embeddings)
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
        embeddings = np.load(os.path.join(emb_dir, "embeddings.npy"))
        embeddings = ensure_valid_embeddings(embeddings, source=f"cached embeddings in {emb_dir}")
        df = pd.read_parquet(os.path.join(emb_dir, "data.parquet"))
        texts = df["text"].tolist()
        labels = df["label"].values
        return TextDataset(
            name=metadata["name"],
            encoding_model=metadata["encoding_model"],
            embeddings=embeddings,
            labels=labels,  # ty:ignore[invalid-argument-type]
            texts=texts,
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

    def get(self, dataset_id: TextDatasets) -> TextDataset:
        cache_key = (dataset_id, self.encoding_model)
        if cache_key not in self._cache:
            dataset = self._get_dataset(dataset_id, self.encoding_model)
            self._cache[cache_key] = dataset
        return self._cache[cache_key]

    def clear_cache(self):
        self._cache.clear()

    @staticmethod
    def _get_dataset(dataset: TextDatasets, encoding_model: str) -> TextDataset:
        name = dataset.value
        custom_datasets = set([i.value for i in custom_text_dataset_properties.keys()])
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
            dataset.value,
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
        if dataset := DatasetManager._get_cached_dataset(dataset_name, model):
            return dataset
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
