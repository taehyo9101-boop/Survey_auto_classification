from __future__ import annotations

import argparse
from pathlib import Path

from survey_tag_pipeline.config import (
    PipelineConfig,
    default_final_tag_dictionary_path,
    derive_run_name,
)
from survey_tag_pipeline.io_utils import read_survey_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Survey free-text tagging pipeline",
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Project root directory. Defaults to the current directory.",
    )
    parser.add_argument(
        "--input",
        default="Survey_data.csv",
        help="Input CSV path relative to project root or absolute path.",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Path to .env file relative to project root or absolute path.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs",
        help="Output root directory relative to project root or absolute path.",
    )
    parser.add_argument(
        "--short-response-max-chars",
        type=int,
        default=5,
        help="Maximum non-space character count treated as a short response.",
    )
    parser.add_argument(
        "--candidate-tag-k",
        type=int,
        default=3,
        help="Reserved for phase 3 candidate tag pruning. Stored in the config and manifest.",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=128,
        help="Batch size for text embeddings.",
    )
    parser.add_argument(
        "--llm-mapping-batch-size",
        type=int,
        default=10,
        help="Batch size for phase3 LLM tag mapping.",
    )
    parser.add_argument(
        "--llm-max-output-tokens",
        type=int,
        default=None,
        help="Optional maximum output tokens for structured LLM responses. Defaults to no explicit limit.",
    )
    parser.add_argument(
        "--llm-temperature",
        type=float,
        default=0.0,
        help="Temperature for structured LLM responses. Defaults to 0.0.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    phase1_parser = subparsers.add_parser("phase1", help="Run phase 1 discovery pipeline.")
    phase1_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for verification runs on the real dataset.",
    )

    phase2_init_parser = subparsers.add_parser(
        "phase2-init",
        help="Optional: copy merged_tags.csv to an editable phase2 template.",
    )
    phase2_init_parser.add_argument(
        "--phase1-run-dir",
        default=None,
        help="Specific phase 1 run directory. Defaults to the latest phase 1 output.",
    )

    phase2_finalize_parser = subparsers.add_parser(
        "phase2-finalize",
        help="Build final tag dictionary from edited merged_tags.csv (legacy review template also supported).",
    )
    phase2_finalize_parser.add_argument(
        "--review-csv",
        default=None,
        help="Path to edited merged_tags.csv (or legacy researcher_review_template.csv). Defaults to the current input-file run directory.",
    )
    phase2_finalize_parser.add_argument(
        "--output",
        default=None,
        help="Path for the finalized tag dictionary CSV. Defaults to outputs/final_tag_dictionary/<input_stem>/final_tag_dictionary.csv.",
    )

    phase2_parser = subparsers.add_parser(
        "phase2",
        help="Shortcut for phase2-finalize with input-file-name-based default paths.",
    )
    phase2_parser.add_argument(
        "--review-csv",
        default=None,
        help="Path to edited merged_tags.csv. Defaults to outputs/phase1/<input_stem>/06_merged_tags/merged_tags.csv.",
    )
    phase2_parser.add_argument(
        "--output",
        default=None,
        help="Path for the finalized tag dictionary CSV. Defaults to outputs/final_tag_dictionary/<input_stem>/final_tag_dictionary.csv.",
    )

    phase3_parser = subparsers.add_parser(
        "phase3",
        help="Run response-level tag mapping with sentiment analysis.",
    )
    phase3_parser.add_argument(
        "--tag-dictionary",
        default=None,
        help="Path to finalized tag dictionary CSV. Defaults to outputs/final_tag_dictionary/<input_stem>/final_tag_dictionary.csv.",
    )
    phase3_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for verification runs on the real dataset.",
    )
    return parser


def _resolve_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return project_root / path


def build_config(args: argparse.Namespace) -> PipelineConfig:
    project_root = Path(args.project_root).resolve()
    input_path = _resolve_path(project_root, args.input)
    env_path = _resolve_path(project_root, args.env_file)
    output_root = _resolve_path(project_root, args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    read_survey_csv(input_path)
    return PipelineConfig(
        project_root=project_root,
        input_path=input_path,
        env_path=env_path,
        output_root=output_root,
        run_name=derive_run_name(input_path),
        short_response_max_chars=args.short_response_max_chars,
        candidate_tag_k=args.candidate_tag_k,
        embedding_batch_size=args.embedding_batch_size,
        llm_mapping_batch_size=args.llm_mapping_batch_size,
        llm_max_output_tokens=args.llm_max_output_tokens,
        llm_temperature=args.llm_temperature,
    )


def _default_phase1_merged_tags_path(config: PipelineConfig) -> Path:
    return config.output_root / "phase1" / config.run_name / "06_merged_tags" / "merged_tags.csv"


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    config = build_config(args)

    if args.command == "phase1":
        from survey_tag_pipeline.phase1 import run_phase1

        run_dir = run_phase1(config, limit=args.limit)
        print(run_dir)
        return 0

    if args.command == "phase2-init":
        from survey_tag_pipeline.phase2 import initialize_phase2_review

        phase1_run_dir = Path(args.phase1_run_dir).resolve() if args.phase1_run_dir else None
        run_dir = initialize_phase2_review(config, phase1_run_dir=phase1_run_dir)
        print(run_dir)
        return 0

    if args.command in {"phase2", "phase2-finalize"}:
        from survey_tag_pipeline.phase2 import build_final_tag_dictionary

        review_csv = (
            _resolve_path(config.project_root, args.review_csv)
            if args.review_csv
            else _default_phase1_merged_tags_path(config)
        )
        output_path = (
            _resolve_path(config.project_root, args.output)
            if args.output
            else default_final_tag_dictionary_path(config.output_root, config.run_name)
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        final_path = build_final_tag_dictionary(review_csv, output_path)
        print(final_path)
        return 0

    if args.command == "phase3":
        from survey_tag_pipeline.phase3 import run_phase3

        tag_dictionary_path = (
            _resolve_path(config.project_root, args.tag_dictionary)
            if args.tag_dictionary
            else default_final_tag_dictionary_path(config.output_root, config.run_name)
        )
        run_dir = run_phase3(
            config,
            tag_dictionary_path=tag_dictionary_path,
            limit=args.limit,
        )
        print(run_dir)
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2
