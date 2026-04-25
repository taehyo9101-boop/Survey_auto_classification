from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pandas as pd


def cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    safe = np.where(norms == 0.0, 1.0, norms)
    normalized = vectors / safe
    return normalized @ normalized.T


def select_representative_indices(centroid_distances: np.ndarray, count: int) -> list[int]:
    ordered = np.argsort(centroid_distances)
    return ordered[: min(count, len(ordered))].tolist()


def select_boundary_indices(centroid_distances: np.ndarray, count: int) -> list[int]:
    ordered = np.argsort(-centroid_distances)
    return ordered[: min(count, len(ordered))].tolist()


def select_frequent_indices(cluster_rows: pd.DataFrame, count: int) -> list[int]:
    ordered = cluster_rows.sort_values(
        by=["dup_count", "char_length"],
        ascending=[False, False],
        kind="stable",
    )
    return ordered.index[: min(count, len(ordered))].tolist()


def build_cluster_packages(
    assignments: pd.DataFrame,
    embeddings: np.ndarray,
    topic_words: dict[int, list[tuple[str, float]]],
) -> list[dict]:
    packages: list[dict] = []
    topic_ids = sorted([int(value) for value in assignments["topic_id"].unique() if int(value) != -1])
    similarity = cosine_similarity_matrix(embeddings)

    for topic_id in topic_ids:
        cluster_rows = assignments.loc[assignments["topic_id"] == topic_id].copy()
        cluster_positions = cluster_rows.index.to_numpy()
        cluster_vectors = embeddings[cluster_positions]
        centroid = cluster_vectors.mean(axis=0, keepdims=True)
        centroid_distances = np.linalg.norm(cluster_vectors - centroid, axis=1)

        center_idx = select_representative_indices(centroid_distances, count=3)
        boundary_idx = select_boundary_indices(centroid_distances, count=2)
        frequent_idx = select_frequent_indices(cluster_rows.reset_index(), count=3)

        nearest_other_topic = None
        overlap_terms: list[str] = []
        if len(topic_ids) > 1:
            cluster_similarities = similarity[np.ix_(cluster_positions, cluster_positions)]
            intra_score = float(cluster_similarities.mean())

            best_topic_score = -1.0
            for other_topic in topic_ids:
                if other_topic == topic_id:
                    continue
                other_positions = assignments.loc[assignments["topic_id"] == other_topic].index.to_numpy()
                inter_score = float(similarity[np.ix_(cluster_positions, other_positions)].mean())
                if inter_score > best_topic_score:
                    best_topic_score = inter_score
                    nearest_other_topic = int(other_topic)
            topic_terms = {word for word, _score in topic_words.get(topic_id, [])}
            other_terms = {word for word, _score in topic_words.get(nearest_other_topic or -999, [])}
            overlap_terms = sorted(topic_terms & other_terms)
        else:
            intra_score = float(similarity[np.ix_(cluster_positions, cluster_positions)].mean())

        cluster_rows_reset = cluster_rows.reset_index(drop=True)
        center_samples = cluster_rows_reset.iloc[center_idx][["content", "dup_count", "canonical_idx"]]
        boundary_samples = cluster_rows_reset.iloc[boundary_idx][["content", "dup_count", "canonical_idx"]]
        frequent_samples = cluster_rows_reset.iloc[frequent_idx][["content", "dup_count", "canonical_idx"]]

        short_distribution = Counter(cluster_rows_reset["short_response_group"].tolist())
        packages.append(
            {
                "topic_id": int(topic_id),
                "topic_size": int(len(cluster_rows_reset)),
                "topic_dup_count_sum": int(cluster_rows_reset["dup_count"].sum()),
                "cluster_density_score": round(intra_score, 6),
                "nearest_other_topic_id": nearest_other_topic,
                "topic_keywords": [word for word, _score in topic_words.get(topic_id, [])[:10]],
                "keyword_scores": [
                    {"term": word, "score": round(float(score), 6)}
                    for word, score in topic_words.get(topic_id, [])[:10]
                ],
                "center_examples": center_samples.to_dict(orient="records"),
                "frequent_examples": frequent_samples.to_dict(orient="records"),
                "boundary_examples": boundary_samples.to_dict(orient="records"),
                "overlap_terms_with_nearest_topic": overlap_terms,
                "short_response_distribution": dict(short_distribution),
                "source_indices": [
                    json.loads(value) for value in cluster_rows_reset["source_indices"].tolist()
                ],
            }
        )
    return packages
