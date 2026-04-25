from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field
from tqdm import tqdm


TAG_GENERATION_SYSTEM_PROMPT = """너는 한국어 주관식 설문 응답을 위한 태그 사전 설계자다.
반드시 입력 군집에 기반해서만 판단하고, 추정이나 상상으로 태그를 만들지 마라.
태그 정의는 서로 구분 가능해야 하며, include_rule과 exclude_rule은 실제 태깅에 사용할 수 있을 정도로 구체적이어야 한다.
응답은 제공된 구조화 스키마를 따르되, 각 필드는 한국어로 작성하라."""

TAG_MERGE_SYSTEM_PROMPT = """너는 한국어 설문 태그 사전 편집자다.
후보 태그 목록을 보고 중복, 과도한 세분화, 과도한 포괄성, 상하위 관계를 정리한다.
반드시 입력 후보에 근거해서만 병합하라.
의미가 겹치는 태그는 하나로 합치고, 병합하지 않은 태그는 왜 남겼는지 설명하라.
모든 source topic_id는 최종 merged_tags의 related_cluster_ids에서 최소 1회 이상 등장해야 한다. 커버리지를 잃지 마라.
응답은 제공된 구조화 스키마를 따르되, 각 필드는 한국어로 작성하라."""

TAG_MAPPING_SYSTEM_PROMPT = """너는 한국어 주관식 응답 분류자다.
주어진 응답 목록과 각 응답의 후보 태그(tag_id 목록)를 보고, 응답마다 맞는 tag_id를 0개 이상 선택한다.
반드시 각 응답의 후보 목록에 있는 tag_id만 선택해야 한다.
한 응답에 여러 tag_id가 붙을 수 있다.
감정 점수는 반드시 -1(부정), 0(중립), 1(긍정) 중 하나로만 출력한다.
응답은 제공된 구조화 스키마를 따르되, 각 필드는 한국어로 작성하라."""


class ClusterTagCandidateModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tag_name: str
    tag_definition: str
    include_rule: str
    exclude_rule: str
    example_responses: list[str] = Field(default_factory=list)


class ClusterTagCandidatesPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tag_candidates: list[ClusterTagCandidateModel] = Field(min_length=1, max_length=2)


class MergedTagModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tag_name: str
    tag_definition: str
    include_rule: str
    exclude_rule: str
    example_responses: list[str] = Field(default_factory=list)
    related_cluster_ids: list[int] = Field(default_factory=list)
    merge_rationale: str
    granularity_flag: Literal["balanced", "too_broad", "too_narrow", "auxiliary"]
    suggested_parent_tag: str | None = None


class MergedTagsPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    merged_tags: list[MergedTagModel] = Field(default_factory=list)


class AssignedTagModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tag_id: str
    tag_rank: int = Field(ge=1)
    is_primary: bool
    tag_confidence: float = Field(ge=0.0, le=1.0)
    tagging_reason: str
    evidence_span: str


class ResponseTagMappingPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    assigned_tags: list[AssignedTagModel] = Field(default_factory=list)
    primary_tag_id: str | None = None
    sentiment_score: Literal[-1, 0, 1]
    sentiment_label: Literal["negative", "neutral", "positive"]


class BatchedResponseTagMappingItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    idx: str
    assigned_tags: list[AssignedTagModel] = Field(default_factory=list)
    primary_tag_id: str | None = None
    sentiment_score: Literal[-1, 0, 1]
    sentiment_label: Literal["negative", "neutral", "positive"]


class BatchedResponseTagMappingPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mappings: list[BatchedResponseTagMappingItem] = Field(default_factory=list)


@dataclass(slots=True)
class LLMArtifacts:
    records: list[dict]
    usage_report: dict


def _coerce_string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _normalize_candidate_payload(topic_id: int, payload: dict) -> dict:
    raw_candidates = payload.get("tag_candidates", [])
    if not isinstance(raw_candidates, list):
        raise ValueError("Cluster tag generation response must contain a list in tag_candidates.")

    normalized_candidates: list[dict] = []
    for candidate in raw_candidates:
        if not isinstance(candidate, dict):
            continue
        tag_name = str(candidate.get("tag_name", "")).strip()
        if not tag_name:
            continue
        normalized_candidates.append(
            {
                "tag_name": tag_name,
                "tag_definition": str(candidate.get("tag_definition", "")).strip(),
                "include_rule": str(candidate.get("include_rule", "")).strip(),
                "exclude_rule": str(candidate.get("exclude_rule", "")).strip(),
                "example_responses": _coerce_string_list(candidate.get("example_responses", [])),
                "related_cluster_id": int(topic_id),
            }
        )

    if not normalized_candidates:
        raise ValueError(f"Cluster tag generation returned no usable candidates for topic_id={topic_id}.")

    return {
        "topic_id": int(topic_id),
        "tag_candidates": normalized_candidates,
    }


def _response_text(response: object) -> str:
    text = getattr(response, "output_text", None)
    if text:
        return text
    output = getattr(response, "output", []) or []
    parts: list[str] = []
    for item in output:
        for content_item in getattr(item, "content", []) or []:
            chunk = getattr(content_item, "text", None)
            if chunk:
                parts.append(chunk)
    return "".join(parts).strip()


def _write_failed_response_debug(
    *,
    debug_dir: Path | None,
    stage_name: str,
    model: str,
    attempt: int,
    raw_text: str,
    error: Exception,
) -> None:
    if debug_dir is None:
        return
    debug_dir.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time() * 1000)
    path = debug_dir / f"{stage_name}_{model}_attempt{attempt}_{timestamp}.txt"
    path.write_text(
        "\n".join(
            [
                f"stage={stage_name}",
                f"model={model}",
                f"attempt={attempt}",
                f"error_type={type(error).__name__}",
                f"error_message={error}",
                "",
                "raw_text:",
                raw_text,
            ]
        ),
        encoding="utf-8",
    )


def _model_dump(payload: BaseModel | dict | list) -> dict | list:
    if isinstance(payload, BaseModel):
        return payload.model_dump(mode="python")
    return payload


def _supports_temperature(model: str) -> bool:
    return model != "gpt-5-mini"


def _call_structured_model(
    client: OpenAI,
    *,
    model: str,
    response_schema: type[BaseModel],
    system_prompt: str,
    user_prompt: str,
    max_output_tokens: int | None,
    temperature: float,
    stage_name: str,
    debug_dir: Path | None = None,
) -> tuple[dict | list, dict]:
    response = None
    raw_text = ""
    temperature_applied = temperature if _supports_temperature(model) else None
    try:
        request_kwargs = {
            "model": model,
            "instructions": system_prompt,
            "input": user_prompt,
            "text_format": response_schema,
        }
        if max_output_tokens is not None:
            request_kwargs["max_output_tokens"] = max_output_tokens
        if temperature_applied is not None:
            request_kwargs["temperature"] = temperature_applied

        response = client.responses.parse(**request_kwargs)
        parsed = getattr(response, "output_parsed", None)
        if parsed is None:
            raw_text = _response_text(response)
            raise ValueError("Model returned no parsed structured output.")
        payload = _model_dump(parsed)
        usage = getattr(response, "usage", None)
        usage_report = {
            "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
            "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
            "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
            "temperature_requested": temperature,
            "temperature_applied": temperature_applied,
            "max_output_tokens_requested": max_output_tokens,
        }
        return payload, usage_report
    except Exception as exc:
        if response is not None and not raw_text:
            raw_text = _response_text(response)
        _write_failed_response_debug(
            debug_dir=debug_dir,
            stage_name=stage_name,
            model=model,
            attempt=1,
            raw_text=raw_text,
            error=exc,
        )
        raise


def generate_cluster_tag_candidates(
    cluster_packages: list[dict],
    model: str,
    max_output_tokens: int | None,
    temperature: float,
    debug_dir: Path | None = None,
) -> LLMArtifacts:
    client = OpenAI()
    rows: list[dict] = []
    input_tokens = 0
    output_tokens = 0
    temperature_applied_values: list[float | None] = []

    for package in tqdm(cluster_packages, desc="LLM tag generation", unit="cluster"):
        user_prompt = (
            "다음은 하나의 응답 군집 요약 패키지다.\n"
            "이 군집에 대해 1개의 태그 후보를 제안하라.\n"
            "topic_id는 현재 군집의 식별자이며, related_cluster_id는 자동으로 동일한 값으로 저장된다.\n"
            f"군집 패키지:\n{json.dumps(package, ensure_ascii=False, indent=2)}"
        )
        payload, usage = _call_structured_model(
            client,
            model=model,
            response_schema=ClusterTagCandidatesPayload,
            system_prompt=TAG_GENERATION_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            stage_name=f"tag_generation_topic_{package['topic_id']}",
            debug_dir=debug_dir,
        )
        input_tokens += usage["input_tokens"]
        output_tokens += usage["output_tokens"]
        temperature_applied_values.append(usage.get("temperature_applied"))
        if not isinstance(payload, dict):
            raise ValueError("Cluster tag generation response must be a structured object.")
        rows.append(_normalize_candidate_payload(package["topic_id"], payload))

    return LLMArtifacts(
        records=rows,
        usage_report={
            "model": model,
            "temperature_requested": temperature,
            "temperature_applied": (
                temperature if temperature_applied_values and all(value == temperature for value in temperature_applied_values) else None
            ),
            "temperature_omitted_for_model": any(value is None for value in temperature_applied_values),
            "max_output_tokens_requested": max_output_tokens,
            "requests": len(cluster_packages),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    )


def merge_tag_candidates(
    candidate_rows: list[dict],
    model: str,
    max_output_tokens: int | None,
    temperature: float,
    debug_dir: Path | None = None,
) -> LLMArtifacts:
    client = OpenAI()
    user_prompt = (
        "다음은 군집별 태그 후보 목록이다.\n"
        "먼저 명백한 중복 태그를 정리하고, 그 다음 의미상 유사 태그를 검토하라.\n"
        "모든 source topic_id가 최종 related_cluster_ids에 최소 1회 이상 반영되도록 유지하라.\n"
        f"태그 후보 목록:\n{json.dumps(candidate_rows, ensure_ascii=False, indent=2)}"
    )
    payload, usage = _call_structured_model(
        client,
        model=model,
        response_schema=MergedTagsPayload,
        system_prompt=TAG_MERGE_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        stage_name="tag_merge",
        debug_dir=debug_dir,
    )
    if not isinstance(payload, dict):
        raise ValueError("Merged tag response must be a structured object.")
    return LLMArtifacts(
        records=payload.get("merged_tags", []),
        usage_report={
            "model": model,
            "temperature_requested": temperature,
            "temperature_applied": usage.get("temperature_applied"),
            "temperature_omitted_for_model": usage.get("temperature_applied") is None,
            "max_output_tokens_requested": max_output_tokens,
            "requests": 1,
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
            "total_tokens": usage["total_tokens"],
        },
    )


def _normalize_response_mapping_payload(payload: dict, candidate_tag_ids: set[str]) -> dict:
    raw_assigned = payload.get("assigned_tags", [])
    if not isinstance(raw_assigned, list):
        raw_assigned = []

    filtered: list[dict] = []
    seen: set[str] = set()
    for row in raw_assigned:
        if not isinstance(row, dict):
            continue
        tag_id = str(row.get("tag_id", "")).strip()
        if not tag_id or tag_id not in candidate_tag_ids or tag_id in seen:
            continue
        seen.add(tag_id)
        try:
            tag_rank = int(row.get("tag_rank", len(filtered) + 1))
        except (TypeError, ValueError):
            tag_rank = len(filtered) + 1
        try:
            tag_confidence = float(row.get("tag_confidence", 0.0))
        except (TypeError, ValueError):
            tag_confidence = 0.0
        filtered.append(
            {
                "tag_id": tag_id,
                "tag_rank": max(1, tag_rank),
                "is_primary": bool(row.get("is_primary", False)),
                "tag_confidence": max(0.0, min(1.0, tag_confidence)),
                "tagging_reason": str(row.get("tagging_reason", "")).strip(),
                "evidence_span": str(row.get("evidence_span", "")).strip(),
            }
        )

    filtered.sort(key=lambda item: item["tag_rank"])
    for index, item in enumerate(filtered, start=1):
        item["tag_rank"] = index

    primary_tag_id = payload.get("primary_tag_id")
    if primary_tag_id is not None:
        primary_tag_id = str(primary_tag_id).strip()
    if not primary_tag_id or primary_tag_id not in {item["tag_id"] for item in filtered}:
        primary_tag_id = filtered[0]["tag_id"] if filtered else None

    for item in filtered:
        item["is_primary"] = item["tag_id"] == primary_tag_id

    sentiment_score = int(payload.get("sentiment_score", 0))
    if sentiment_score not in {-1, 0, 1}:
        sentiment_score = 0
    sentiment_label = str(payload.get("sentiment_label", "")).strip().lower()
    if sentiment_label not in {"negative", "neutral", "positive"}:
        sentiment_label = {-1: "negative", 0: "neutral", 1: "positive"}[sentiment_score]
    else:
        score_from_label = {"negative": -1, "neutral": 0, "positive": 1}[sentiment_label]
        sentiment_score = score_from_label

    return {
        "assigned_tags": filtered,
        "primary_tag_id": primary_tag_id,
        "sentiment_score": sentiment_score,
        "sentiment_label": sentiment_label,
    }


def map_responses_to_tags(
    rows: list[dict],
    model: str,
    max_output_tokens: int | None,
    temperature: float,
    batch_size: int = 10,
    debug_dir: Path | None = None,
) -> LLMArtifacts:
    client = OpenAI()
    records: list[dict] = []
    input_tokens = 0
    output_tokens = 0
    temperature_applied_values: list[float | None] = []
    safe_batch_size = max(1, int(batch_size))

    for start in tqdm(range(0, len(rows), safe_batch_size), desc="LLM tag mapping", unit="batch"):
        batch_rows = rows[start : start + safe_batch_size]
        normalized_batch_inputs: list[dict] = []
        batch_input_map: dict[str, dict] = {}
        for row in batch_rows:
            idx = str(row.get("idx", "")).strip()
            content = str(row.get("content", "")).strip()
            candidate_tags = row.get("candidate_tags", [])
            if not idx:
                continue
            if not isinstance(candidate_tags, list):
                candidate_tags = []
            normalized_input = {
                "idx": idx,
                "content": content,
                "candidate_tags": candidate_tags,
            }
            normalized_batch_inputs.append(normalized_input)
            batch_input_map[idx] = normalized_input

        if not normalized_batch_inputs:
            continue

        user_prompt = (
            "다음은 응답 목록과 각 응답별 후보 태그 목록이다.\n"
            "반드시 각 응답의 후보 목록 안에서만 tag_id를 선택하라.\n"
            "태그를 하나도 선택하지 않아도 되지만, 선택했다면 이유와 근거 구간을 짧게 제시하라.\n"
            "입력 목록:\n"
            f"{json.dumps(normalized_batch_inputs, ensure_ascii=False, indent=2)}"
        )
        payload, usage = _call_structured_model(
            client,
            model=model,
            response_schema=BatchedResponseTagMappingPayload,
            system_prompt=TAG_MAPPING_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            stage_name=f"tag_mapping_batch_{start}",
            debug_dir=debug_dir,
        )
        input_tokens += usage["input_tokens"]
        output_tokens += usage["output_tokens"]
        temperature_applied_values.append(usage.get("temperature_applied"))
        if not isinstance(payload, dict):
            raise ValueError("Response tag mapping output must be a structured object.")

        mappings = payload.get("mappings", [])
        if not isinstance(mappings, list):
            mappings = []
        seen_idx: set[str] = set()

        for mapping in mappings:
            if not isinstance(mapping, dict):
                continue
            idx = str(mapping.get("idx", "")).strip()
            if not idx or idx in seen_idx or idx not in batch_input_map:
                continue
            seen_idx.add(idx)
            candidate_tags = batch_input_map[idx].get("candidate_tags", [])
            normalized = _normalize_response_mapping_payload(
                payload=mapping,
                candidate_tag_ids={str(item.get("tag_id", "")).strip() for item in candidate_tags},
            )
            normalized["idx"] = idx
            normalized["content"] = str(batch_input_map[idx].get("content", "")).strip()
            records.append(normalized)

    return LLMArtifacts(
        records=records,
        usage_report={
            "model": model,
            "temperature_requested": temperature,
            "temperature_applied": (
                temperature if temperature_applied_values and all(value == temperature for value in temperature_applied_values) else None
            ),
            "temperature_omitted_for_model": any(value is None for value in temperature_applied_values),
            "max_output_tokens_requested": max_output_tokens,
            "batch_size": safe_batch_size,
            "requests": len(range(0, len(rows), safe_batch_size)),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    )
