import os
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
from typing import List

MODEL_NAME = "all-MiniLM-L6-v2"  # 384-dim, fast, solid baseline

DEFAULT_PREFER_LOCAL = os.getenv("EMBEDDING_MODEL_PREFER_LOCAL", "true").lower() in {
    "1",
    "true",
    "yes",
    "y",
}
DEFAULT_FORCE_LOCAL = os.getenv("EMBEDDING_MODEL_FORCE_LOCAL", "false").lower() in {
    "1",
    "true",
    "yes",
    "y",
}


def load_model(model_name: str = MODEL_NAME, local_files_only: bool | None = None) -> SentenceTransformer:
    """Load the sentence transformer model.

    By default this tries to use the local Hugging Face cache first and falls back to
    an online download only if the cache is missing. Set EMBEDDING_MODEL_FORCE_LOCAL=1
    to prevent any network access.
    """
    if local_files_only is None:
        if DEFAULT_FORCE_LOCAL:
            local_files_only = True
        elif DEFAULT_PREFER_LOCAL:
            local_files_only = True
        else:
            local_files_only = False

    try:
        return SentenceTransformer(model_name, local_files_only=local_files_only)
    except Exception as exc:
        if not local_files_only and DEFAULT_PREFER_LOCAL:
            raise

        if DEFAULT_FORCE_LOCAL:
            raise RuntimeError(
                f"Unable to load embedding model '{model_name}' from the local cache. "
                "Set EMBEDDING_MODEL_FORCE_LOCAL=0 or remove it to allow downloading, "
                "or download the model ahead of time when internet access is available."
            ) from exc

        try:
            return SentenceTransformer(model_name, local_files_only=False)
        except Exception:
            raise RuntimeError(
                f"Unable to load embedding model '{model_name}'. "
                "Please verify your network connection or download the model to the local Hugging Face cache."
            ) from exc

def embed_texts(
    texts: List[str],
    model: SentenceTransformer,
    batch_size: int = 64,
    show_progress: bool = True,
    normalize: bool = True
) -> np.ndarray:
    """Embed a list of texts and return normalized vectors."""
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=normalize,  # for cosine via dot product
    )
    return embeddings

def cache_embeddings(
    embeddings: np.ndarray,
    cache_path: str
) -> None:
    """Save embeddings to disk."""
    np.save(cache_path, embeddings)
    print(f"Cached {embeddings.shape[0]} embeddings -> {cache_path}")

def load_cached_embeddings(cache_path: str) -> np.ndarray:
    """Load embeddings from cache."""
    path = Path(cache_path)
    if not path.exists():
        raise FileNotFoundError(f"No cached embeddings at {cache_path}")
    return np.load(cache_path)