# Survey Tag

Survey Tag is a Python pipeline for building and applying a tag dictionary for
free-text survey responses. It combines preprocessing, embeddings, BERTopic
clustering, structured LLM outputs, researcher review, multi-label tag mapping,
and sentiment analysis.

The pipeline is designed for Korean open-ended survey responses, but the input
format is intentionally simple: each response only needs an `idx` and `content`.

## Features

- Cleans and filters free-text responses, including blank and symbol-only rows.
- Preserves duplicate response frequency during discovery.
- Groups semantically similar responses with embeddings and BERTopic.
- Generates cluster-level tag candidates with an LLM using structured outputs.
- Merges candidate tags into a researcher-editable draft tag dictionary.
- Finalizes a reviewed tag dictionary with validation.
- Assigns one or more `tag_id` values to each response.
- Runs sentiment analysis alongside tag mapping.
- Writes both wide response-level outputs and long response-tag relation tables.

## Requirements

- Python `>=3.13`
- [`uv`](https://docs.astral.sh/uv/)
- OpenAI API key

Install dependencies:

```bash
UV_PROJECT_ENVIRONMENT=venv uv sync
```

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=your_api_key_here
```

The `.env`, `outputs/`, `venv/`, and Python cache files should not be committed.

## Input Format

The input CSV must contain these columns:

| Column | Description |
| --- | --- |
| `idx` | Unique response ID |
| `content` | Free-text survey response |

Example:

```csv
idx,content
1,좋은 자료를 더 쉽게 찾을 수 있으면 좋겠습니다.
2,감사합니다.
```

By default, the CLI reads `Survey_data.csv` from the project root. You can use a
different file with `--input`.

```bash
UV_PROJECT_ENVIRONMENT=venv uv run main.py --input my_survey.csv phase1
```

## Quick Start

Run the full workflow:

```bash
UV_PROJECT_ENVIRONMENT=venv uv run main.py phase1
UV_PROJECT_ENVIRONMENT=venv uv run main.py phase2
UV_PROJECT_ENVIRONMENT=venv uv run main.py phase3
```

For a different input CSV:

```bash
UV_PROJECT_ENVIRONMENT=venv uv run main.py --input my_survey.csv phase1
UV_PROJECT_ENVIRONMENT=venv uv run main.py --input my_survey.csv phase2
UV_PROJECT_ENVIRONMENT=venv uv run main.py --input my_survey.csv phase3
```

Global options must appear before the phase command.

```bash
UV_PROJECT_ENVIRONMENT=venv uv run main.py --llm-mapping-batch-size 10 phase3
```

## Workflow

### Phase 1: Discover Tag Candidates

```bash
UV_PROJECT_ENVIRONMENT=venv uv run main.py phase1
```

Phase 1 performs:

- Response preprocessing
- Duplicate response aggregation
- Response embedding
- BERTopic clustering
- Cluster package generation
- LLM tag candidate generation
- Rule-based candidate deduplication
- LLM candidate merging

Main output:

```text
outputs/phase1/<input_stem>/06_merged_tags/merged_tags.csv
```

This file is the draft tag dictionary. Researchers should review and edit it
before running Phase 2.

### Phase 2: Finalize Reviewed Tag Dictionary

```bash
UV_PROJECT_ENVIRONMENT=venv uv run main.py phase2
```

By default, Phase 2 reads:

```text
outputs/phase1/<input_stem>/06_merged_tags/merged_tags.csv
```

It writes:

```text
outputs/final_tag_dictionary/<input_stem>/final_tag_dictionary.csv
outputs/final_tag_dictionary/<input_stem>/final_tag_dictionary.report.json
```

The finalized dictionary contains:

| Column | Description |
| --- | --- |
| `tag_id` | Stable tag identifier |
| `tag_name` | Human-readable tag name |
| `tag_definition` | Tag meaning |
| `include_rule` | Inclusion criteria |
| `exclude_rule` | Exclusion criteria |
| `tag_version` | Tag dictionary version, defaults to `v1` |

Rows with `is_active=False` are excluded if the source file has an `is_active`
column. Phase 2 validates required text fields and duplicate `tag_id` values.

You can also pass explicit paths:

```bash
UV_PROJECT_ENVIRONMENT=venv uv run main.py phase2-finalize \
  --review-csv path/to/edited_merged_tags.csv \
  --output path/to/final_tag_dictionary.csv
```

### Phase 3: Tag Responses and Analyze Sentiment

```bash
UV_PROJECT_ENVIRONMENT=venv uv run main.py phase3
```

By default, Phase 3 reads:

```text
outputs/final_tag_dictionary/<input_stem>/final_tag_dictionary.csv
```

Phase 3 performs:

- Reuses Phase 1 preprocessing and response embeddings when available.
- Embeds the finalized tag dictionary.
- Shortlists candidate tags per response with embedding similarity.
- Sends batched structured LLM requests for final multi-label tag mapping.
- Computes response-level confidence and review flags.
- Writes final response outputs and summary reports.

Main outputs:

```text
outputs/phase3/<input_stem>/05_outputs/response_results_wide.csv
outputs/phase3/<input_stem>/05_outputs/response_tag_relations_long.csv
outputs/phase3/<input_stem>/05_outputs/final_tag_dictionary.csv
outputs/phase3/<input_stem>/05_outputs/summary_report.json
```

`response_results_wide.csv` contains one row per response:

| Column | Description |
| --- | --- |
| `idx` | Response ID |
| `content` | Cleaned response text |
| `assigned_tag_ids` | JSON array of assigned `tag_id` values |
| `primary_tag_id` | Primary assigned tag |
| `sentiment_score` | `-1`, `0`, or `1` |
| `sentiment_label` | `negative`, `neutral`, or `positive` |
| `confidence` | Post-processed confidence score |
| `review_flag` | `auto_accept` or `review_required` |

`response_tag_relations_long.csv` contains one row per response-tag relation:

| Column | Description |
| --- | --- |
| `idx` | Response ID |
| `tag_id` | Assigned tag ID |
| `tag_rank` | Rank within the response |
| `is_primary` | Whether the tag is the primary tag |
| `tag_confidence` | LLM confidence for that tag |
| `tagging_reason` | LLM rationale |
| `evidence_span` | Supporting phrase from the response |

## Output Directory Layout

Output folders are derived from the input CSV filename without its extension.
For example, `Survey_data.csv` uses `Survey_data` as `<input_stem>`.

```text
outputs/
  phase1/<input_stem>/
  phase2/<input_stem>/
  phase3/<input_stem>/
  final_tag_dictionary/<input_stem>/
```

This makes repeated runs for the same input predictable. A new run for the same
input overwrites files in that input-specific output folder.

## CLI Options

Common options:

| Option | Default | Description |
| --- | --- | --- |
| `--input` | `Survey_data.csv` | Input CSV path |
| `--env-file` | `.env` | Environment file path |
| `--output-root` | `outputs` | Output root directory |
| `--short-response-max-chars` | `5` | Threshold for short-response grouping |
| `--candidate-tag-k` | `3` | Number of candidate tags passed to Phase 3 LLM mapping |
| `--embedding-batch-size` | `128` | Embedding API batch size |
| `--llm-mapping-batch-size` | `10` | Phase 3 LLM mapping batch size |
| `--llm-max-output-tokens` | none | Optional structured response output-token limit |
| `--llm-temperature` | `0.0` | Temperature for models that support it |

Phase-specific options:

```bash
UV_PROJECT_ENVIRONMENT=venv uv run main.py phase1 --limit 100
UV_PROJECT_ENVIRONMENT=venv uv run main.py phase3 --limit 100
UV_PROJECT_ENVIRONMENT=venv uv run main.py phase3 --tag-dictionary path/to/final_tag_dictionary.csv
```

## Model Defaults

The default model configuration is defined in
`survey_tag_pipeline/config.py`.

| Task | Default model |
| --- | --- |
| Embeddings | `text-embedding-3-small` |
| Tag candidate generation | `gpt-5-mini` |
| Tag candidate merge | `gpt-4.1-mini` |
| Final response mapping | `gpt-4.1-mini` |

Structured LLM calls use the OpenAI Responses API with parsed structured
outputs. `gpt-5-mini` is called without an explicit temperature because the
current code treats it as not supporting that parameter.

## Development

Compile-check the package:

```bash
./venv/bin/python -m compileall survey_tag_pipeline
```

Show CLI help:

```bash
UV_PROJECT_ENVIRONMENT=venv uv run main.py --help
```

## Notes

- Review `merged_tags.csv` before finalizing the tag dictionary. Phase 1 output
  is a draft, not a ground-truth taxonomy.
- Keep `tag_id` stable after Phase 2. Downstream response mappings depend on it.
- Phase 3 can assign multiple tags to one response.
- Generated output files can contain survey data and should be handled as
  sensitive research data.
