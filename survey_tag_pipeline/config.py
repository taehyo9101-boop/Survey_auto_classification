from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(slots=True)
class ModelConfig:
    embedding_model: str = "text-embedding-3-small"
    tag_generation_model: str = "gpt-5-mini"
    tag_merge_model: str = "gpt-4.1-mini"
    final_mapping_model: str = "gpt-4.1-mini"


@dataclass(slots=True)
class PipelineConfig:
    project_root: Path
    input_path: Path
    env_path: Path
    output_root: Path
    run_name: str
    short_response_max_chars: int = 5
    candidate_tag_k: int = 3
    embedding_batch_size: int = 128
    llm_mapping_batch_size: int = 10
    llm_max_output_tokens: int | None = None
    llm_temperature: float = 0.0
    top_n_topic_words: int = 10
    models: ModelConfig = field(default_factory=ModelConfig)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["project_root"] = str(self.project_root)
        payload["input_path"] = str(self.input_path)
        payload["env_path"] = str(self.env_path)
        payload["output_root"] = str(self.output_root)
        return payload


def derive_run_name(input_path: Path) -> str:
    stem = input_path.stem.strip()
    if not stem:
        return "input"
    normalized = "".join(char if (char.isalnum() or char in {"-", "_"}) else "_" for char in stem)
    normalized = normalized.strip("_")
    return normalized or "input"


def create_run_dir(base_dir: Path, phase_name: str, run_name: str) -> Path:
    run_dir = base_dir / phase_name / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def default_final_tag_dictionary_path(output_root: Path, run_name: str) -> Path:
    return output_root / "final_tag_dictionary" / run_name / "final_tag_dictionary.csv"
