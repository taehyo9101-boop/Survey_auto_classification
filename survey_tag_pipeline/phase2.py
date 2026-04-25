from __future__ import annotations

from pathlib import Path

import pandas as pd

from survey_tag_pipeline.config import PipelineConfig, create_run_dir
from survey_tag_pipeline.io_utils import latest_run_dir, write_csv, write_json

FINAL_COLUMNS = [
    "tag_id",
    "tag_name",
    "tag_definition",
    "include_rule",
    "exclude_rule",
    "tag_version",
]

DIRECT_REQUIRED_COLUMNS = {
    "tag_id",
    "tag_name",
    "tag_definition",
    "include_rule",
    "exclude_rule",
}

REVIEW_DECISIONS = {"review_pending", "approved", "edited", "rejected"}
KEEP_DECISIONS = {"approved", "edited"}
REVIEW_REQUIRED_COLUMNS = {
    "tag_id",
    "researcher_decision",
    "approved_tag_name",
    "approved_tag_definition",
    "approved_include_rule",
    "approved_exclude_rule",
}


def _parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower()
    return normalized in {"1", "true", "t", "yes", "y"}


def _nonempty_text(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame[column].fillna("").astype(str).str.strip().ne("")


def _validate_no_duplicate_tag_id(frame: pd.DataFrame, *, context: str) -> None:
    duplicates = frame["tag_id"].astype(str)
    duplicates = duplicates[duplicates.duplicated(keep=False)]
    if duplicates.empty:
        return
    duplicate_ids = ", ".join(sorted(set(duplicates.tolist())))
    raise ValueError(f"Duplicate tag_id in {context}: {duplicate_ids}")


def _validate_required_columns(frame: pd.DataFrame, required: set[str], *, context: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns for {context}: {', '.join(missing)}")


def _prepare_direct_mode_frame(source_frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    _validate_required_columns(source_frame, DIRECT_REQUIRED_COLUMNS, context="edited merged_tags.csv")
    working = source_frame.copy()
    if "is_active" in working.columns:
        working["is_active"] = working["is_active"].map(_parse_bool)
    else:
        working["is_active"] = True
    if "tag_version" not in working.columns:
        working["tag_version"] = "v1"
    else:
        working["tag_version"] = working["tag_version"].fillna("").astype(str).str.strip().replace("", "v1")

    keep_mask = working["is_active"]
    final_frame = working.loc[keep_mask, FINAL_COLUMNS].copy()

    required_text = ["tag_id", "tag_name", "tag_definition", "include_rule", "exclude_rule", "tag_version"]
    for column in required_text:
        missing_mask = ~_nonempty_text(final_frame, column)
        if missing_mask.any():
            sample_ids = ", ".join(final_frame.loc[missing_mask, "tag_id"].astype(str).head(5).tolist())
            raise ValueError(f"Column '{column}' is empty in active rows. tag_id samples: {sample_ids}")

    _validate_no_duplicate_tag_id(final_frame, context="active edited merged_tags.csv")
    final_frame = final_frame.sort_values(by=["tag_id"], kind="stable").reset_index(drop=True)

    report = {
        "input_mode": "edited_merged_tags",
        "input_rows": int(len(working)),
        "active_rows": int(working["is_active"].sum()),
        "final_rows": int(len(final_frame)),
    }
    return final_frame, report


def _prepare_review_mode_frame(source_frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    _validate_required_columns(source_frame, REVIEW_REQUIRED_COLUMNS, context="legacy researcher review template")
    working = source_frame.copy()
    working["researcher_decision"] = (
        working["researcher_decision"]
        .fillna("review_pending")
        .astype(str)
        .str.strip()
        .str.lower()
    )
    invalid = working.loc[~working["researcher_decision"].isin(REVIEW_DECISIONS)]
    if not invalid.empty:
        bad_values = ", ".join(sorted(set(invalid["researcher_decision"].tolist())))
        raise ValueError(
            "Invalid researcher_decision values: "
            + bad_values
            + " (allowed: approved, edited, rejected, review_pending)"
        )

    if "is_active" in working.columns:
        working["is_active"] = working["is_active"].map(_parse_bool)
    else:
        working["is_active"] = True

    if "tag_version" not in working.columns:
        working["tag_version"] = "v1"
    else:
        working["tag_version"] = working["tag_version"].fillna("").astype(str).str.strip().replace("", "v1")

    keep_mask = working["researcher_decision"].isin(KEEP_DECISIONS) & working["is_active"]
    final_frame = working.loc[keep_mask].copy()
    final_frame["tag_name"] = final_frame["approved_tag_name"]
    final_frame["tag_definition"] = final_frame["approved_tag_definition"]
    final_frame["include_rule"] = final_frame["approved_include_rule"]
    final_frame["exclude_rule"] = final_frame["approved_exclude_rule"]
    final_frame = final_frame[FINAL_COLUMNS]

    required_text = ["tag_id", "tag_name", "tag_definition", "include_rule", "exclude_rule", "tag_version"]
    for column in required_text:
        missing_mask = ~_nonempty_text(final_frame, column)
        if missing_mask.any():
            sample_ids = ", ".join(final_frame.loc[missing_mask, "tag_id"].astype(str).head(5).tolist())
            raise ValueError(f"Column '{column}' is empty in approved/edited active rows. tag_id samples: {sample_ids}")

    _validate_no_duplicate_tag_id(final_frame, context="approved/edited active review rows")
    final_frame = final_frame.sort_values(by=["tag_id"], kind="stable").reset_index(drop=True)

    report = {
        "input_mode": "legacy_reviewer_template",
        "input_rows": int(len(working)),
        "active_rows": int(working["is_active"].sum()),
        "decision_counts": {
            decision: int((working["researcher_decision"] == decision).sum()) for decision in sorted(REVIEW_DECISIONS)
        },
        "final_rows": int(len(final_frame)),
    }
    return final_frame, report


def initialize_phase2_review(
    config: PipelineConfig,
    *,
    phase1_run_dir: Path | None = None,
) -> Path:
    source_run_dir = phase1_run_dir or latest_run_dir(
        config.output_root,
        "phase1",
        preferred_name=config.run_name,
    )
    merged_tags_path = source_run_dir / "06_merged_tags" / "merged_tags.csv"
    if not merged_tags_path.exists():
        raise FileNotFoundError(f"Merged tags file not found: {merged_tags_path}")

    merged_tags = pd.read_csv(merged_tags_path, encoding="utf-8-sig")
    _validate_required_columns(merged_tags, DIRECT_REQUIRED_COLUMNS, context="phase1 merged_tags.csv")
    editable = merged_tags.copy()
    if "tag_version" not in editable.columns:
        editable["tag_version"] = "v1"
    else:
        editable["tag_version"] = editable["tag_version"].fillna("").astype(str).str.strip().replace("", "v1")
    if "is_active" not in editable.columns:
        editable["is_active"] = True
    else:
        editable["is_active"] = editable["is_active"].map(_parse_bool)

    run_dir = create_run_dir(config.output_root, "phase2", config.run_name)
    template_path = run_dir / "editable_merged_tags.csv"
    write_csv(template_path, editable)

    write_json(
        run_dir / "manifest.json",
        {
            "phase1_run_dir": str(source_run_dir),
            "merged_tags_path": str(merged_tags_path),
            "editable_template": str(template_path),
            "finalize_input_recommendation": "Edit merged_tags.csv directly, then pass that file to phase2-finalize.",
        },
    )
    return run_dir


def build_final_tag_dictionary(review_csv_path: Path, output_path: Path) -> Path:
    source_frame = pd.read_csv(review_csv_path, encoding="utf-8-sig")
    review_mode = REVIEW_REQUIRED_COLUMNS.issubset(source_frame.columns)
    if review_mode:
        final_frame, report = _prepare_review_mode_frame(source_frame)
    else:
        final_frame, report = _prepare_direct_mode_frame(source_frame)

    write_csv(output_path, final_frame)
    report.update(
        {
            "input_path": str(review_csv_path),
            "output_path": str(output_path),
        }
    )
    write_json(output_path.with_suffix(".report.json"), report)
    return output_path
