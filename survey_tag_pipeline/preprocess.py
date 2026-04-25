from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass

import pandas as pd

MISSING_TEXT_MARKERS = {
    "nan",
    "none",
    "null",
    "nat",
}

NO_ANSWER_TERMS = {
    "없음",
    "없다",
    "없어요",
    "없습니다",
    "x",
    "n",
    "na",
    "무",
    "무응답",
    "해당없음",
}
DEFERRED_TERMS = {
    "모름",
    "잘모름",
    "잘 모르겠음",
    "잘모르겠음",
    "모르겠다",
    "모르겠음",
    "잘 모르겠습니다",
    "모르겠습니다",
}
SHORT_OTHER_TERMS = {
    "그냥",
    "그냥요",
    "보통",
    "글쎄요",
    "네",
    "아니요",
    "좋아요",
}


@dataclass(slots=True)
class PreprocessArtifacts:
    valid_responses: pd.DataFrame
    deduplicated_responses: pd.DataFrame
    excluded_rows: pd.DataFrame
    report: dict


def normalize_text(text: object) -> str:
    if text is None or pd.isna(text):
        return ""
    value = unicodedata.normalize("NFKC", str(text))
    value = value.replace("\u200b", "")
    value = re.sub(r"[\r\n\t]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    if value.lower() in MISSING_TEXT_MARKERS:
        return ""
    return value


def compact_for_length(text: str) -> str:
    return re.sub(r"\s+", "", text)


def canonical_short_key(text: str) -> str:
    lowered = text.lower()
    lowered = re.sub(r"[^0-9a-zA-Z가-힣]+", "", lowered)
    return lowered


def is_symbol_only_without_korean_or_english(text: str) -> bool:
    compact = compact_for_length(text)
    if not compact:
        return False
    if re.search(r"[A-Za-z가-힣ㄱ-ㅎㅏ-ㅣ]", compact):
        return False
    return all(unicodedata.category(char).startswith(("P", "S")) for char in compact)


def classify_short_response(text: str, max_chars: int) -> tuple[bool, str]:
    compact = compact_for_length(text)
    if len(compact) > max_chars:
        return False, "not_short"
    key = canonical_short_key(text)
    if key in NO_ANSWER_TERMS:
        return True, "no_answer"
    if key in DEFERRED_TERMS:
        return True, "deferred_judgement"
    if key in SHORT_OTHER_TERMS:
        return True, "short_other"
    return True, "short_other"


def preprocess_responses(frame: pd.DataFrame, short_response_max_chars: int) -> PreprocessArtifacts:
    working = frame.copy()
    working["idx"] = working["idx"].astype(str).str.strip()
    working["raw_content"] = working["content"].where(~working["content"].isna(), "")
    working["normalized_content"] = working["content"].map(normalize_text)
    working["idx_duplicate_flag"] = working["idx"].duplicated(keep=False)

    blank_mask = working["normalized_content"].eq("")
    non_blank = working.loc[~blank_mask].copy()
    symbol_only_mask = non_blank["normalized_content"].map(is_symbol_only_without_korean_or_english)

    excluded_blank_rows = working.loc[blank_mask, ["idx", "raw_content"]].copy()
    excluded_blank_rows["exclusion_reason"] = "blank_after_normalization"

    excluded_symbol_rows = non_blank.loc[symbol_only_mask, ["idx", "raw_content"]].copy()
    excluded_symbol_rows["exclusion_reason"] = "symbol_only_without_korean_or_english"

    excluded_rows = pd.concat([excluded_blank_rows, excluded_symbol_rows], ignore_index=True)
    valid = non_blank.loc[~symbol_only_mask].copy()
    valid["char_length"] = valid["normalized_content"].map(lambda value: len(compact_for_length(value)))
    short_flags = valid["normalized_content"].map(
        lambda value: classify_short_response(value, short_response_max_chars)
    )
    valid["is_short_response"] = short_flags.map(lambda item: item[0])
    valid["short_response_group"] = short_flags.map(lambda item: item[1])

    deduplicated = (
        valid.groupby("normalized_content", dropna=False, sort=False)
        .agg(
            canonical_idx=("idx", "first"),
            representative_raw_content=("raw_content", "first"),
            dup_count=("idx", "size"),
            source_indices=("idx", lambda items: json.dumps(list(items), ensure_ascii=False)),
            char_length=("char_length", "first"),
            is_short_response=("is_short_response", "first"),
            short_response_group=("short_response_group", "first"),
            idx_duplicate_in_input=("idx_duplicate_flag", "max"),
        )
        .reset_index()
        .rename(columns={"normalized_content": "content"})
    )

    report = {
        "input_rows": int(len(frame)),
        "excluded_blank_rows": int(len(excluded_blank_rows)),
        "excluded_symbol_only_rows": int(len(excluded_symbol_rows)),
        "excluded_total_rows": int(len(excluded_rows)),
        "valid_rows": int(len(valid)),
        "unique_clean_responses": int(len(deduplicated)),
        "duplicate_response_rows_collapsed": int(len(valid) - len(deduplicated)),
        "duplicated_idx_rows": int(valid["idx_duplicate_flag"].sum()),
        "short_response_rows": int(valid["is_short_response"].fillna(False).astype(bool).sum()),
        "short_response_unique_groups": int(deduplicated["is_short_response"].fillna(False).astype(bool).sum()),
        "short_response_distribution": {
            key: int(value)
            for key, value in valid["short_response_group"].value_counts().sort_index().items()
        },
    }

    return PreprocessArtifacts(
        valid_responses=valid[
            [
                "idx",
                "raw_content",
                "normalized_content",
                "char_length",
                "is_short_response",
                "short_response_group",
                "idx_duplicate_flag",
            ]
        ].rename(columns={"normalized_content": "content"}),
        deduplicated_responses=deduplicated,
        excluded_rows=excluded_rows,
        report=report,
    )
