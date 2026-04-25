from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from survey_tag_pipeline.config import PipelineConfig, create_run_dir
from survey_tag_pipeline.embeddings import EmbeddingArtifacts, embed_texts
from survey_tag_pipeline.io_utils import load_project_env, read_json, write_csv, write_json, write_jsonl
from survey_tag_pipeline.llm import map_responses_to_tags
from survey_tag_pipeline.preprocess import preprocess_responses

TAG_DICT_REQUIRED_COLUMNS = {"tag_id", "tag_name", "tag_definition", "include_rule", "exclude_rule"}


def _phase1_run_dir(config: PipelineConfig) -> Path:
    return config.output_root / "phase1" / config.run_name


def _load_phase1_preprocess_cache(
    config: PipelineConfig,
    *,
    limit: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, str | None]:
    if limit is not None:
        raise FileNotFoundError("phase1 preprocess cache is disabled when --limit is used.")
    phase1_dir = _phase1_run_dir(config)
    valid_path = phase1_dir / "01_preprocess" / "valid_responses.csv"
    excluded_path = phase1_dir / "01_preprocess" / "excluded_rows.csv"
    report_path = phase1_dir / "01_preprocess" / "report.json"
    if not (valid_path.exists() and excluded_path.exists() and report_path.exists()):
        raise FileNotFoundError("phase1 preprocess artifacts are missing.")
    valid = pd.read_csv(valid_path, encoding="utf-8-sig")
    excluded = pd.read_csv(excluded_path, encoding="utf-8-sig")
    report = read_json(report_path)
    return valid, excluded, report, str(phase1_dir)


def _load_phase1_response_embeddings_cache(
    config: PipelineConfig,
    *,
    responses: pd.DataFrame,
    limit: int | None,
) -> EmbeddingArtifacts:
    if limit is not None:
        raise FileNotFoundError("phase1 response embedding cache is disabled when --limit is used.")
    phase1_dir = _phase1_run_dir(config)
    embeddings_path = phase1_dir / "02_embeddings" / "embeddings.npy"
    metadata_path = phase1_dir / "02_embeddings" / "embedding_metadata.csv"
    if not (embeddings_path.exists() and metadata_path.exists()):
        raise FileNotFoundError("phase1 embedding artifacts are missing.")

    dedup_embeddings = np.load(embeddings_path)
    metadata = pd.read_csv(metadata_path, encoding="utf-8-sig")
    if "content" not in metadata.columns:
        raise ValueError(f"Missing 'content' column in {metadata_path}")
    if dedup_embeddings.shape[0] != len(metadata):
        raise ValueError(
            "phase1 embedding rows and metadata rows do not match: "
            f"{dedup_embeddings.shape[0]} vs {len(metadata)}"
        )

    content_to_index = {str(content): index for index, content in enumerate(metadata["content"].astype(str))}
    indices: list[int] = []
    missing_contents = 0
    for content in responses["content"].astype(str).tolist():
        index = content_to_index.get(content)
        if index is None:
            missing_contents += 1
            continue
        indices.append(index)
    if missing_contents:
        raise ValueError(f"Failed to map {missing_contents} response contents to phase1 deduplicated embeddings.")

    expanded = dedup_embeddings[np.asarray(indices, dtype=np.int64)]
    report = {
        "source": "phase1_cached_dedup_embeddings",
        "phase1_embeddings_path": str(embeddings_path),
        "phase1_metadata_path": str(metadata_path),
        "records_embedded": int(len(responses)),
        "embedding_dimension": int(expanded.shape[1]) if len(responses) else 0,
        "prompt_tokens": 0,
        "batch_size": None,
    }
    return EmbeddingArtifacts(embeddings=np.asarray(expanded, dtype=np.float32), report=report)


def _validate_tag_dictionary(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(TAG_DICT_REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required tag dictionary columns: {', '.join(missing)}")
    working = frame.copy()
    for column in TAG_DICT_REQUIRED_COLUMNS:
        working[column] = working[column].fillna("").astype(str).str.strip()
    working = working.loc[working["tag_id"] != ""].copy()
    if "tag_version" not in working.columns:
        working["tag_version"] = "v1"
    else:
        working["tag_version"] = working["tag_version"].fillna("").astype(str).str.strip().replace("", "v1")
    if working["tag_id"].duplicated().any():
        duplicated = sorted(set(working.loc[working["tag_id"].duplicated(keep=False), "tag_id"].tolist()))
        raise ValueError(f"Duplicate tag_id in tag dictionary: {', '.join(duplicated)}")
    return working.reset_index(drop=True)


def _tag_embedding_text(frame: pd.DataFrame) -> list[str]:
    return [
        f"{row.tag_name}\n{row.tag_definition}\n포함: {row.include_rule}\n제외: {row.exclude_rule}"
        for row in frame.itertuples(index=False)
    ]


def _cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    a_safe = a / np.clip(a_norm, 1e-12, None)
    b_safe = b / np.clip(b_norm, 1e-12, None)
    return np.matmul(a_safe, b_safe.T).astype(np.float32)


def _build_candidate_inputs(
    responses: pd.DataFrame,
    tags: pd.DataFrame,
    similarity_matrix: np.ndarray,
    top_k: int,
) -> tuple[pd.DataFrame, list[dict]]:
    shortlist_rows: list[dict] = []
    llm_inputs: list[dict] = []
    tag_ids = tags["tag_id"].tolist()
    k = max(1, min(top_k, len(tag_ids)))

    for row_index, row in enumerate(responses.itertuples(index=False)):
        sims = similarity_matrix[row_index]
        top_indices = np.argsort(sims)[::-1][:k]
        candidate_tags: list[dict] = []
        for rank, tag_index in enumerate(top_indices, start=1):
            candidate = {
                "tag_id": str(tags.iloc[tag_index]["tag_id"]),
                "tag_name": str(tags.iloc[tag_index]["tag_name"]),
                "tag_definition": str(tags.iloc[tag_index]["tag_definition"]),
                "include_rule": str(tags.iloc[tag_index]["include_rule"]),
                "exclude_rule": str(tags.iloc[tag_index]["exclude_rule"]),
                "similarity": float(sims[tag_index]),
            }
            candidate_tags.append(candidate)
            shortlist_rows.append(
                {
                    "idx": str(row.idx),
                    "content": str(row.content),
                    "tag_id": candidate["tag_id"],
                    "candidate_rank": rank,
                    "similarity": candidate["similarity"],
                }
            )

        llm_inputs.append(
            {
                "idx": str(row.idx),
                "content": str(row.content),
                "candidate_tags": candidate_tags,
            }
        )

    shortlist_frame = pd.DataFrame(shortlist_rows)
    return shortlist_frame, llm_inputs


def _coerce_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _build_output_frames(
    responses: pd.DataFrame,
    mapping_records: list[dict],
    shortlist_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mapping_by_idx = {str(row.get("idx", "")): row for row in mapping_records}
    sim_lookup = {
        (str(row.idx), str(row.tag_id)): float(row.similarity)
        for row in shortlist_frame.itertuples(index=False)
    }

    wide_rows: list[dict] = []
    long_rows: list[dict] = []

    for response in responses.itertuples(index=False):
        idx = str(response.idx)
        content = str(response.content)
        mapping = mapping_by_idx.get(
            idx,
            {
                "assigned_tags": [],
                "primary_tag_id": None,
                "sentiment_score": 0,
                "sentiment_label": "neutral",
            },
        )
        assigned = mapping.get("assigned_tags", [])
        if not isinstance(assigned, list):
            assigned = []
        assigned = sorted(
            [
                item
                for item in assigned
                if isinstance(item, dict) and str(item.get("tag_id", "")).strip()
            ],
            key=lambda item: int(item.get("tag_rank", 9999)),
        )

        assigned_tag_ids = [str(item["tag_id"]) for item in assigned]
        primary_tag_id = mapping.get("primary_tag_id")
        if primary_tag_id is not None:
            primary_tag_id = str(primary_tag_id).strip()
        if primary_tag_id not in assigned_tag_ids:
            primary_tag_id = assigned_tag_ids[0] if assigned_tag_ids else ""

        primary_conf = 0.0
        if primary_tag_id:
            for item in assigned:
                if str(item.get("tag_id")) == primary_tag_id:
                    primary_conf = _coerce_float(item.get("tag_confidence"))
                    break
        primary_sim = sim_lookup.get((idx, primary_tag_id), 0.0) if primary_tag_id else 0.0
        confidence = float(np.clip(0.7 * primary_conf + 0.3 * max(primary_sim, 0.0), 0.0, 1.0))
        review_flag = "review_required" if (not assigned_tag_ids or confidence < 0.55) else "auto_accept"

        sentiment_score = int(mapping.get("sentiment_score", 0))
        if sentiment_score not in {-1, 0, 1}:
            sentiment_score = 0
        sentiment_label = str(mapping.get("sentiment_label", "")).strip().lower()
        if sentiment_label not in {"negative", "neutral", "positive"}:
            sentiment_label = {-1: "negative", 0: "neutral", 1: "positive"}[sentiment_score]

        wide_rows.append(
            {
                "idx": idx,
                "content": content,
                "assigned_tag_ids": json.dumps(assigned_tag_ids, ensure_ascii=False),
                "primary_tag_id": primary_tag_id,
                "sentiment_score": sentiment_score,
                "sentiment_label": sentiment_label,
                "confidence": round(confidence, 4),
                "review_flag": review_flag,
            }
        )

        for rank, item in enumerate(assigned, start=1):
            tag_id = str(item.get("tag_id", "")).strip()
            if not tag_id:
                continue
            long_rows.append(
                {
                    "idx": idx,
                    "tag_id": tag_id,
                    "tag_rank": rank,
                    "is_primary": tag_id == primary_tag_id,
                    "tag_confidence": round(float(np.clip(_coerce_float(item.get("tag_confidence")), 0.0, 1.0)), 4),
                    "tagging_reason": str(item.get("tagging_reason", "")).strip(),
                    "evidence_span": str(item.get("evidence_span", "")).strip(),
                }
            )

    return pd.DataFrame(wide_rows), pd.DataFrame(long_rows)


def _build_summary_report(
    wide_frame: pd.DataFrame,
    long_frame: pd.DataFrame,
) -> dict:
    total = int(len(wide_frame))
    with_tags = int((wide_frame["assigned_tag_ids"] != "[]").sum()) if total else 0
    review_required = int((wide_frame["review_flag"] == "review_required").sum()) if total else 0

    sentiment_distribution = (
        wide_frame["sentiment_label"].value_counts().sort_index().to_dict() if total else {}
    )
    tag_counts = (
        long_frame.groupby("tag_id")["idx"].nunique().sort_values(ascending=False).to_dict()
        if not long_frame.empty
        else {}
    )
    tag_ratios = {tag_id: round(count / total, 6) for tag_id, count in tag_counts.items()} if total else {}

    pair_counts: dict[str, int] = {}
    if total:
        for value in wide_frame["assigned_tag_ids"].tolist():
            try:
                tags = json.loads(value)
            except json.JSONDecodeError:
                tags = []
            if not isinstance(tags, list):
                continue
            unique_tags = sorted({str(tag).strip() for tag in tags if str(tag).strip()})
            for left, right in itertools.combinations(unique_tags, 2):
                key = f"{left}|{right}"
                pair_counts[key] = pair_counts.get(key, 0) + 1

    top_cooccurrence = dict(sorted(pair_counts.items(), key=lambda item: item[1], reverse=True)[:20])
    return {
        "total_responses": total,
        "responses_with_tags": with_tags,
        "responses_without_tags": total - with_tags,
        "review_required_count": review_required,
        "auto_accept_count": total - review_required,
        "sentiment_distribution": sentiment_distribution,
        "tag_response_counts": tag_counts,
        "tag_response_ratios": tag_ratios,
        "top_tag_cooccurrence_pairs": top_cooccurrence,
    }


def run_phase3(
    config: PipelineConfig,
    *,
    tag_dictionary_path: Path,
    limit: int | None = None,
) -> Path:
    load_project_env(config.env_path)
    run_dir = create_run_dir(config.output_root, "phase3", config.run_name)
    input_frame = pd.read_csv(config.input_path, encoding="utf-8-sig")
    if limit is not None:
        input_frame = input_frame.head(limit).copy()

    tag_dictionary_raw = pd.read_csv(tag_dictionary_path, encoding="utf-8-sig")
    tag_dictionary = _validate_tag_dictionary(tag_dictionary_raw)

    preprocess_source = "phase3_recomputed"
    preprocess_source_run_dir: str | None = None
    try:
        valid_responses, excluded_rows, preprocess_report, preprocess_source_run_dir = _load_phase1_preprocess_cache(
            config,
            limit=limit,
        )
        preprocess_source = "phase1_cached"
    except (FileNotFoundError, ValueError):
        preprocess = preprocess_responses(
            frame=input_frame,
            short_response_max_chars=config.short_response_max_chars,
        )
        valid_responses = preprocess.valid_responses
        excluded_rows = preprocess.excluded_rows
        preprocess_report = preprocess.report

    preprocess_dir = run_dir / "01_preprocess"
    preprocess_dir.mkdir(parents=True, exist_ok=True)
    write_csv(preprocess_dir / "valid_responses.csv", valid_responses)
    write_csv(preprocess_dir / "excluded_rows.csv", excluded_rows)
    write_json(
        preprocess_dir / "report.json",
        {
            "source": preprocess_source,
            "source_run_dir": preprocess_source_run_dir,
            "report": preprocess_report,
        },
    )

    output_dir = run_dir / "05_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "final_tag_dictionary.csv", tag_dictionary)

    if valid_responses.empty:
        empty_wide = pd.DataFrame(
            columns=[
                "idx",
                "content",
                "assigned_tag_ids",
                "primary_tag_id",
                "sentiment_score",
                "sentiment_label",
                "confidence",
                "review_flag",
            ]
        )
        empty_long = pd.DataFrame(
            columns=[
                "idx",
                "tag_id",
                "tag_rank",
                "is_primary",
                "tag_confidence",
                "tagging_reason",
                "evidence_span",
            ]
        )
        write_csv(output_dir / "response_results_wide.csv", empty_wide)
        write_csv(output_dir / "response_tag_relations_long.csv", empty_long)
        write_json(output_dir / "summary_report.json", _build_summary_report(empty_wide, empty_long))
        write_json(
            run_dir / "manifest.json",
            {
                "run_dir": str(run_dir),
                "tag_dictionary_path": str(tag_dictionary_path),
                "input_rows_used": int(len(input_frame)),
                "limit": limit,
                "config": config.to_dict(),
                "preprocess_source": preprocess_source,
                "preprocess_source_run_dir": preprocess_source_run_dir,
                "files": {
                    "valid_responses": str(preprocess_dir / "valid_responses.csv"),
                    "excluded_rows": str(preprocess_dir / "excluded_rows.csv"),
                    "response_results_wide": str(output_dir / "response_results_wide.csv"),
                    "response_tag_relations_long": str(output_dir / "response_tag_relations_long.csv"),
                    "summary_report": str(output_dir / "summary_report.json"),
                },
            },
        )
        return run_dir

    embeddings_dir = run_dir / "02_embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    response_embedding_source = "phase3_recomputed"
    response_texts = valid_responses["content"].astype(str).tolist()
    try:
        response_embeddings = _load_phase1_response_embeddings_cache(
            config,
            responses=valid_responses,
            limit=limit,
        )
        response_embedding_source = "phase1_cached"
    except (FileNotFoundError, ValueError):
        response_embeddings = embed_texts(
            texts=response_texts,
            model=config.models.embedding_model,
            batch_size=config.embedding_batch_size,
        )
    tag_embeddings = embed_texts(
        texts=_tag_embedding_text(tag_dictionary),
        model=config.models.embedding_model,
        batch_size=config.embedding_batch_size,
    )
    np.save(embeddings_dir / "response_embeddings.npy", response_embeddings.embeddings)
    np.save(embeddings_dir / "tag_embeddings.npy", tag_embeddings.embeddings)
    write_json(
        embeddings_dir / "response_embedding_report.json",
        {
            "source": response_embedding_source,
            "report": response_embeddings.report,
        },
    )
    write_json(embeddings_dir / "tag_embedding_report.json", tag_embeddings.report)

    candidate_dir = run_dir / "03_candidate_shortlist"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    similarity_matrix = _cosine_similarity_matrix(response_embeddings.embeddings, tag_embeddings.embeddings)
    shortlist_frame, llm_inputs = _build_candidate_inputs(
        responses=valid_responses[["idx", "content"]].copy(),
        tags=tag_dictionary,
        similarity_matrix=similarity_matrix,
        top_k=config.candidate_tag_k,
    )
    write_csv(candidate_dir / "candidate_tag_shortlist.csv", shortlist_frame)
    write_jsonl(candidate_dir / "candidate_tag_shortlist.jsonl", llm_inputs)

    mapping_dir = run_dir / "04_tag_mapping"
    mapping_dir.mkdir(parents=True, exist_ok=True)
    mapping_artifacts = map_responses_to_tags(
        rows=llm_inputs,
        model=config.models.final_mapping_model,
        max_output_tokens=config.llm_max_output_tokens,
        temperature=config.llm_temperature,
        batch_size=config.llm_mapping_batch_size,
        debug_dir=mapping_dir / "debug",
    )
    write_jsonl(mapping_dir / "response_tag_mapping.jsonl", mapping_artifacts.records)
    write_json(mapping_dir / "usage_report.json", mapping_artifacts.usage_report)

    wide_frame, long_frame = _build_output_frames(
        responses=valid_responses[["idx", "content"]].copy(),
        mapping_records=mapping_artifacts.records,
        shortlist_frame=shortlist_frame,
    )
    write_csv(output_dir / "response_results_wide.csv", wide_frame)
    write_csv(output_dir / "response_tag_relations_long.csv", long_frame)
    write_json(output_dir / "summary_report.json", _build_summary_report(wide_frame, long_frame))

    write_json(
        run_dir / "manifest.json",
        {
            "run_dir": str(run_dir),
            "tag_dictionary_path": str(tag_dictionary_path),
            "input_rows_used": int(len(input_frame)),
            "limit": limit,
            "config": config.to_dict(),
            "preprocess_source": preprocess_source,
            "preprocess_source_run_dir": preprocess_source_run_dir,
            "response_embedding_source": response_embedding_source,
            "files": {
                "valid_responses": str(preprocess_dir / "valid_responses.csv"),
                "excluded_rows": str(preprocess_dir / "excluded_rows.csv"),
                "candidate_tag_shortlist": str(candidate_dir / "candidate_tag_shortlist.csv"),
                "response_tag_mapping": str(mapping_dir / "response_tag_mapping.jsonl"),
                "response_results_wide": str(output_dir / "response_results_wide.csv"),
                "response_tag_relations_long": str(output_dir / "response_tag_relations_long.csv"),
                "summary_report": str(output_dir / "summary_report.json"),
                "final_tag_dictionary": str(output_dir / "final_tag_dictionary.csv"),
            },
        },
    )
    return run_dir
