import os
import torch
from sentence_transformers import SentenceTransformer
from logger import get_logger

log = get_logger("embedder")

EMBED_MODEL = os.getenv("EMBED_MODEL", "Qwen/Qwen3-Embedding-0.6B")
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "64"))

_model = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info(f"Loading embedding model {EMBED_MODEL} on {device}")
        _model = SentenceTransformer(
            EMBED_MODEL,
            device=device,
            model_kwargs={"dtype": torch.float16, "attn_implementation": "sdpa"}
            if device == "cuda" else {},
        )
        _model.max_seq_length = 1024
        log.info(f"Embedding model loaded dim={_model.get_sentence_embedding_dimension()}")
    return _model


def embed(text: str, is_query: bool = False) -> list[float]:
    """Embed single text. Use is_query=True for search queries (adds prompt_name)."""
    log.debug(f"Embedding {'query' if is_query else 'doc'} ({len(text)} chars)")
    model = _get_model()
    kwargs = {"prompt_name": "query"} if is_query else {}
    vec = model.encode(text, normalize_embeddings=True, **kwargs).tolist()
    log.debug(f"Embed OK dim={len(vec)}")
    return vec


def embed_batch(texts: list[str], is_query: bool = False) -> list[list[float]]:
    """Embed batch of texts. Use is_query=True for search queries."""
    log.debug(f"Embedding {'query' if is_query else 'doc'} batch of {len(texts)} model={EMBED_MODEL}")
    model = _get_model()
    kwargs = {"prompt_name": "query"} if is_query else {}
    vecs = model.encode(
        texts,
        batch_size=EMBED_BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=False,
        **kwargs,
    ).tolist()
    log.debug(f"Batch embed OK count={len(vecs)}")
    return vecs
