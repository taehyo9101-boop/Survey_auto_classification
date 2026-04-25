from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from openai import OpenAI
from tqdm import tqdm


@dataclass(slots=True)
class EmbeddingArtifacts:
    embeddings: np.ndarray
    report: dict


def embed_texts(
    texts: list[str],
    model: str,
    batch_size: int,
    max_retries: int = 5,
) -> EmbeddingArtifacts:
    client = OpenAI()
    vectors: list[list[float]] = []
    prompt_tokens = 0

    for start in tqdm(range(0, len(texts), batch_size), desc="Embedding", unit="batch"):
        batch = texts[start : start + batch_size]
        for attempt in range(1, max_retries + 1):
            try:
                response = client.embeddings.create(
                    model=model,
                    input=batch,
                    encoding_format="float",
                )
                prompt_tokens += response.usage.prompt_tokens
                vectors.extend(item.embedding for item in response.data)
                break
            except Exception:
                if attempt == max_retries:
                    raise
                time.sleep(min(2**attempt, 20))

    embedding_array = np.asarray(vectors, dtype=np.float32)
    report = {
        "model": model,
        "records_embedded": int(len(texts)),
        "embedding_dimension": int(embedding_array.shape[1]) if len(texts) else 0,
        "prompt_tokens": int(prompt_tokens),
        "batch_size": int(batch_size),
    }
    return EmbeddingArtifacts(embeddings=embedding_array, report=report)
