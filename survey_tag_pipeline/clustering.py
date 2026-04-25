from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np
import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP


@dataclass(slots=True)
class TopicModelArtifacts:
    assignments: pd.DataFrame
    topic_info: pd.DataFrame
    topic_words: dict[int, list[tuple[str, float]]]
    report: dict


def derive_topic_parameters(record_count: int) -> dict:
    n_neighbors = max(5, min(50, int(round(sqrt(max(record_count, 5))))))
    min_cluster_size = max(8, min(30, int(round(max(record_count, 30) * 0.01))))
    min_samples = max(3, min(10, min_cluster_size // 2))
    return {
        "n_neighbors": n_neighbors,
        "min_cluster_size": min_cluster_size,
        "min_samples": min_samples,
    }


def cluster_topics(
    responses: pd.DataFrame,
    embeddings: np.ndarray,
    top_n_words: int,
) -> tuple[BERTopic, TopicModelArtifacts]:
    params = derive_topic_parameters(len(responses))

    vectorizer_model = CountVectorizer(
        token_pattern=r"(?u)\b\w+\b",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
    )
    umap_model = UMAP(
        n_neighbors=params["n_neighbors"],
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )
    topic_model = BERTopic(
        language="multilingual",
        vectorizer_model=vectorizer_model,
        umap_model=umap_model,
        min_topic_size=params["min_cluster_size"],
        top_n_words=top_n_words,
        calculate_probabilities=False,
        verbose=False,
    )

    topics, _ = topic_model.fit_transform(responses["content"].tolist(), embeddings=embeddings)
    topic_info = topic_model.get_topic_info()
    assignments = responses.copy()
    assignments["topic_id"] = topics

    topic_words: dict[int, list[tuple[str, float]]] = {}
    for topic_id in topic_info["Topic"].tolist():
        if topic_id == -1:
            continue
        words = topic_model.get_topic(topic_id) or []
        topic_words[int(topic_id)] = [(word, float(score)) for word, score in words]

    report = {
        "record_count": int(len(responses)),
        "topic_count_including_outliers": int(topic_info.shape[0]),
        "non_outlier_topic_count": int((topic_info["Topic"] != -1).sum()),
        "outlier_count": int((assignments["topic_id"] == -1).sum()),
        "parameters": params,
    }
    artifacts = TopicModelArtifacts(
        assignments=assignments,
        topic_info=topic_info,
        topic_words=topic_words,
        report=report,
    )
    return topic_model, artifacts
