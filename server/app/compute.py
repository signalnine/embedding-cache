# server/app/compute.py
import asyncio
from concurrent.futures import ProcessPoolExecutor
from typing import Optional
from app.config import settings

# Model cache (per-process)
_model_cache: dict = {}
_executor: Optional[ProcessPoolExecutor] = None


def _get_executor() -> ProcessPoolExecutor:
    """Get or create process pool executor."""
    global _executor
    if _executor is None:
        _executor = ProcessPoolExecutor(max_workers=2)
    return _executor


def _get_model(model_name: str):
    """Load model (cached per process)."""
    if model_name not in _model_cache:
        from sentence_transformers import SentenceTransformer

        # Map model names to HuggingFace IDs
        model_map = {
            "nomic-v1.5": "nomic-ai/nomic-embed-text-v1.5",
            "nomic-v2-moe": "nomic-ai/nomic-embed-text-v2-moe",
        }
        hf_name = model_map.get(model_name, model_name)
        _model_cache[model_name] = SentenceTransformer(
            hf_name,
            device=settings.gpu_device,
            trust_remote_code=True,
        )

    return _model_cache[model_name]


def _is_nomic_model(model: str) -> bool:
    # nomic-embed models are the only ones that consume 'search_document:'
    # / 'search_query:' style prefixes. Applying these prefixes to other
    # encoders (e.g. all-MiniLM-L6-v2) embeds the literal prefix text and
    # corrupts the vector.
    name = model.lower()
    return name.startswith("nomic-") or "nomic-embed" in name


def _apply_prefix(text: str, model: str, role: str) -> str:
    if not _is_nomic_model(model):
        return text
    if role == "query":
        return f"search_query: {text}"
    return f"search_document: {text}"


def compute_embedding_sync(text: str, model: str, role: str = "document") -> list[float]:
    """Compute embedding synchronously (runs in process pool).

    role: 'document' for stored items, 'query' for retrieval queries.
    Only affects nomic-* models; ignored for other encoders.
    """
    model_instance = _get_model(model)
    prefixed_text = _apply_prefix(text, model, role)
    embedding = model_instance.encode([prefixed_text])[0]
    return embedding.tolist()


def compute_batch_sync(texts: list[str], model: str, role: str = "document") -> list[list[float]]:
    """Compute batch embeddings synchronously."""
    model_instance = _get_model(model)
    prefixed_texts = [_apply_prefix(t, model, role) for t in texts]
    embeddings = model_instance.encode(prefixed_texts)
    return [e.tolist() for e in embeddings]


async def compute_embedding(text: str, model: str, role: str = "document") -> list[float]:
    """Compute embedding asynchronously using process pool."""
    loop = asyncio.get_event_loop()
    executor = _get_executor()
    return await loop.run_in_executor(executor, compute_embedding_sync, text, model, role)


async def compute_batch(texts: list[str], model: str, role: str = "document") -> list[list[float]]:
    """Compute batch embeddings asynchronously."""
    loop = asyncio.get_event_loop()
    executor = _get_executor()
    return await loop.run_in_executor(executor, compute_batch_sync, texts, model, role)
