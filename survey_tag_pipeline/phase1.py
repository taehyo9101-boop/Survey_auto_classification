from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

from survey_tag_pipeline.cluster_packages import build_cluster_packages
from survey_tag_pipeline.clustering import cluster_topics
from survey_tag_pipeline.config import PipelineConfig, create_run_dir
from survey_tag_pipeline.embeddings import embed_texts
from survey_tag_pipeline.io_utils import (
    load_project_env,
    write_csv,
    write_json,
    write_jsonl,
)
from survey_tag_pipeline.llm import generate_cluster_tag_candidates, merge_tag_candidates
from survey_tag_pipeline.preprocess import preprocess_responses


def _flatten_cluster_candidates(rows: list[dict]) -> pd.DataFrame:
    flattened: list[dict] = []
    for row in rows:
        topic_id = int(row.get("topic_id"))
        for candidate in row.get("tag_candidates", []):
            flattened.append(
                {
                    "topic_id": topic_id,
                    "tag_name": candidate.get("tag_name", ""),
                    "tag_definition": candidate.get("tag_definition", ""),
                    "include_rule": candidate.get("include_rule", ""),
                    "exclude_rule": candidate.get("exclude_rule", ""),
                    "example_responses": json.dumps(
                        candidate.get("example_responses", []),
                        ensure_ascii=False,
                    ),
                    "related_cluster_id": topic_id,
                }
            )
    return pd.DataFrame(flattened)


def _normalize_tag_name(tag_name: str) -> str:
    return "".join(tag_name.lower().split())


def _json_string_list(value: object) -> str:
    if isinstance(value, list):
        return json.dumps([str(item) for item in value if str(item).strip()], ensure_ascii=False)
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return json.dumps([], ensure_ascii=False)
        try:
            loaded = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return json.dumps([value], ensure_ascii=False)
        if isinstance(loaded, list):
            return json.dumps([str(item) for item in loaded if str(item).strip()], ensure_ascii=False)
    return json.dumps([], ensure_ascii=False)


def _parse_cluster_id_list(value: object) -> list[int]:
    if isinstance(value, list):
        raw = value
    elif isinstance(value, str) and value.strip():
        try:
            raw = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            raw = [value]
    else:
        raw = []
    cluster_ids: list[int] = []
    for item in raw:
        try:
            cluster_ids.append(int(item))
        except (TypeError, ValueError):
            continue
    return sorted(set(cluster_ids))


def _backfill_missing_topics(
    merged_frame: pd.DataFrame,
    candidate_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, list[int]]:
    source_topics = {int(value) for value in candidate_frame["topic_id"].dropna().tolist()}
    covered_topics: set[int] = set()
    if not merged_frame.empty and "related_cluster_ids" in merged_frame.columns:
        for value in merged_frame["related_cluster_ids"]:
            covered_topics.update(_parse_cluster_id_list(value))
    missing_topics = sorted(source_topics - covered_topics)
    if not missing_topics:
        return merged_frame, []

    fallback_rows: list[dict] = []
    for topic_id in missing_topics:
        topic_candidates = candidate_frame.loc[candidate_frame["topic_id"] == topic_id]
        for _, candidate in topic_candidates.iterrows():
            fallback_rows.append(
                {
                    "tag_name": candidate["tag_name"],
                    "tag_definition": candidate["tag_definition"],
                    "include_rule": candidate["include_rule"],
                    "exclude_rule": candidate["exclude_rule"],
                    "example_responses": candidate["example_responses"],
                    "related_cluster_ids": json.dumps([int(topic_id)], ensure_ascii=False),
                    "merge_rationale": "llm_merge_coverage_backfill",
                    "granularity_flag": "auxiliary",
                    "suggested_parent_tag": "",
                }
            )

    merged_frame = pd.concat([merged_frame, pd.DataFrame(fallback_rows)], ignore_index=True)
    return merged_frame, missing_topics


def _build_merged_frame(
    raw_records: list[dict],
    candidate_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    merged_frame = pd.DataFrame(raw_records)
    required_columns = [
        "tag_name",
        "tag_definition",
        "include_rule",
        "exclude_rule",
        "example_responses",
        "related_cluster_ids",
        "merge_rationale",
        "granularity_flag",
        "suggested_parent_tag",
    ]
    for column in required_columns:
        if column not in merged_frame.columns:
            merged_frame[column] = ""

    if not merged_frame.empty:
        merged_frame["tag_name"] = merged_frame["tag_name"].fillna("").astype(str).str.strip()
        merged_frame = merged_frame.loc[merged_frame["tag_name"] != ""].copy()
        merged_frame["tag_definition"] = merged_frame["tag_definition"].fillna("").astype(str).str.strip()
        merged_frame["include_rule"] = merged_frame["include_rule"].fillna("").astype(str).str.strip()
        merged_frame["exclude_rule"] = merged_frame["exclude_rule"].fillna("").astype(str).str.strip()
        merged_frame["merge_rationale"] = merged_frame["merge_rationale"].fillna("").astype(str).str.strip()
        merged_frame["granularity_flag"] = merged_frame["granularity_flag"].fillna("balanced").astype(str).str.strip()
        merged_frame["suggested_parent_tag"] = (
            merged_frame["suggested_parent_tag"].fillna("").astype(str).str.strip()
        )
        merged_frame["example_responses"] = merged_frame["example_responses"].map(_json_string_list)
        merged_frame["related_cluster_ids"] = merged_frame["related_cluster_ids"].map(
            lambda value: json.dumps(_parse_cluster_id_list(value), ensure_ascii=False)
        )

    merged_frame, missing_topics = _backfill_missing_topics(merged_frame, candidate_frame)
    merged_frame = merged_frame.reset_index(drop=True)
    merged_frame.insert(0, "tag_id", [f"tag_{index + 1:03d}" for index in range(len(merged_frame))])
    report = {
        "input_candidate_rows": int(len(candidate_frame)),
        "initial_merged_rows": int(len(raw_records)),
        "final_merged_rows": int(len(merged_frame)),
        "missing_topics_backfilled": missing_topics,
    }
    return merged_frame, report


def _rule_based_candidate_dedup(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    working = frame.copy()
    working["tag_name_key"] = working["tag_name"].map(_normalize_tag_name)
    working = working.sort_values(
        by=["tag_name_key", "tag_definition", "topic_id"],
        kind="stable",
    )
    deduped = working.drop_duplicates(subset=["tag_name_key", "tag_definition"], keep="first").copy()
    deduped["duplicate_candidate_count"] = (
        working.groupby(["tag_name_key", "tag_definition"])["tag_name"].transform("size")
    )
    return deduped.drop(columns=["tag_name_key"])


def run_phase1(config: PipelineConfig, *, limit: int | None = None) -> Path:
    load_project_env(config.env_path)
    run_dir = create_run_dir(config.output_root, "phase1", config.run_name)
    input_frame = pd.read_csv(config.input_path, encoding="utf-8-sig")
    if limit is not None:
        input_frame = input_frame.head(limit).copy()

    preprocess_artifacts = preprocess_responses(
        input_frame,
        short_response_max_chars=config.short_response_max_chars,
    )
    preprocess_dir = run_dir / "01_preprocess"
    preprocess_dir.mkdir(parents=True, exist_ok=True)
    write_csv(preprocess_dir / "valid_responses.csv", preprocess_artifacts.valid_responses)
    write_csv(preprocess_dir / "deduplicated_responses.csv", preprocess_artifacts.deduplicated_responses)
    write_csv(preprocess_dir / "excluded_rows.csv", preprocess_artifacts.excluded_rows)
    write_json(preprocess_dir / "report.json", preprocess_artifacts.report)

    embeddings_dir = run_dir / "02_embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    embedding_artifacts = embed_texts(
        texts=preprocess_artifacts.deduplicated_responses["content"].tolist(),
        model=config.models.embedding_model,
        batch_size=config.embedding_batch_size,
    )
    np.save(embeddings_dir / "embeddings.npy", embedding_artifacts.embeddings)
    write_csv(
        embeddings_dir / "embedding_metadata.csv",
        preprocess_artifacts.deduplicated_responses[["canonical_idx", "content", "dup_count"]],
    )
    write_json(embeddings_dir / "report.json", embedding_artifacts.report)

    clustering_dir = run_dir / "03_topics"
    clustering_dir.mkdir(parents=True, exist_ok=True)
    topic_model, topic_artifacts = cluster_topics(
        responses=preprocess_artifacts.deduplicated_responses,
        embeddings=embedding_artifacts.embeddings,
        top_n_words=config.top_n_topic_words,
    )
    write_csv(clustering_dir / "topic_assignments.csv", topic_artifacts.assignments)
    write_csv(clustering_dir / "topic_info.csv", topic_artifacts.topic_info)
    write_json(clustering_dir / "topic_words.json", topic_artifacts.topic_words)
    write_json(clustering_dir / "report.json", topic_artifacts.report)
    topic_model.save(clustering_dir / "bertopic_model", serialization="pickle")

    package_dir = run_dir / "04_cluster_packages"
    package_dir.mkdir(parents=True, exist_ok=True)
    cluster_packages = build_cluster_packages(
        assignments=topic_artifacts.assignments,
        embeddings=embedding_artifacts.embeddings,
        topic_words=topic_artifacts.topic_words,
    )
    write_jsonl(package_dir / "cluster_packages.jsonl", cluster_packages)
    write_json(package_dir / "cluster_packages.json", cluster_packages)

    llm_dir = run_dir / "05_cluster_tag_candidates"
    llm_dir.mkdir(parents=True, exist_ok=True)
    if cluster_packages:
        llm_artifacts = generate_cluster_tag_candidates(
            cluster_packages=cluster_packages,
            model=config.models.tag_generation_model,
            max_output_tokens=config.llm_max_output_tokens,
            temperature=config.llm_temperature,
            debug_dir=llm_dir / "debug",
        )
        cluster_candidate_frame = _flatten_cluster_candidates(llm_artifacts.records)
        write_jsonl(llm_dir / "cluster_tag_candidates.jsonl", llm_artifacts.records)
        write_csv(llm_dir / "cluster_tag_candidates.csv", cluster_candidate_frame)
        write_json(llm_dir / "usage_report.json", llm_artifacts.usage_report)
    else:
        llm_artifacts = None
        cluster_candidate_frame = pd.DataFrame(
            columns=[
                "topic_id",
                "tag_name",
                "tag_definition",
                "include_rule",
                "exclude_rule",
                "example_responses",
                "related_cluster_id",
            ]
        )
        write_jsonl(llm_dir / "cluster_tag_candidates.jsonl", [])
        write_csv(llm_dir / "cluster_tag_candidates.csv", cluster_candidate_frame)
        write_json(
            llm_dir / "usage_report.json",
            {
                "model": config.models.tag_generation_model,
                "requests": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            },
        )

    merge_dir = run_dir / "06_merged_tags"
    merge_dir.mkdir(parents=True, exist_ok=True)
    rule_based_frame = _rule_based_candidate_dedup(cluster_candidate_frame)
    write_csv(merge_dir / "rule_based_candidates.csv", rule_based_frame)
    if rule_based_frame.empty:
        merge_artifacts = None
        merged_frame = pd.DataFrame(
            columns=[
                "tag_id",
                "tag_name",
                "tag_definition",
                "include_rule",
                "exclude_rule",
                "example_responses",
                "related_cluster_ids",
                "merge_rationale",
                "granularity_flag",
                "suggested_parent_tag",
            ]
        )
        write_jsonl(merge_dir / "merged_tags.jsonl", [])
        write_csv(merge_dir / "merged_tags.csv", merged_frame)
        write_json(
            merge_dir / "usage_report.json",
            {
                "model": config.models.tag_merge_model,
                "requests": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            },
        )
        write_json(
            merge_dir / "merge_report.json",
            {
                "input_candidate_rows": 0,
                "initial_merged_rows": 0,
                "final_merged_rows": 0,
                "missing_topics_backfilled": [],
            },
        )
    else:
        merge_input_rows = rule_based_frame.fillna("").to_dict(orient="records")
        merge_artifacts = merge_tag_candidates(
            candidate_rows=merge_input_rows,
            model=config.models.tag_merge_model,
            max_output_tokens=config.llm_max_output_tokens,
            temperature=config.llm_temperature,
            debug_dir=merge_dir / "debug",
        )
        merged_frame, merge_report = _build_merged_frame(merge_artifacts.records, rule_based_frame)
        write_jsonl(merge_dir / "merged_tags.jsonl", merge_artifacts.records)
        write_csv(merge_dir / "merged_tags.csv", merged_frame)
        write_json(merge_dir / "usage_report.json", merge_artifacts.usage_report)
        write_json(merge_dir / "merge_report.json", merge_report)

    manifest = {
        "run_dir": str(run_dir),
        "input_path": str(config.input_path),
        "input_rows_used": int(len(input_frame)),
        "limit": limit,
        "files": {
            "preprocess_report": str(preprocess_dir / "report.json"),
            "deduplicated_responses": str(preprocess_dir / "deduplicated_responses.csv"),
            "embedding_report": str(embeddings_dir / "report.json"),
            "embeddings": str(embeddings_dir / "embeddings.npy"),
            "topic_info": str(clustering_dir / "topic_info.csv"),
            "cluster_packages": str(package_dir / "cluster_packages.jsonl"),
            "cluster_tag_candidates": str(llm_dir / "cluster_tag_candidates.csv"),
            "merged_tags": str(merge_dir / "merged_tags.csv"),
        },
        "config": config.to_dict(),
    }
    write_json(run_dir / "manifest.json", manifest)
    return run_dir
