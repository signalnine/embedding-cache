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
        _model_cache[model_name] = SentenceTransformer(hf_name, device=settings.gpu_device)

    return _model_cache[model_name]


def compute_embedding_sync(text: str, model: str) -> list[float]:
    """Compute embedding synchronously (runs in process pool)."""
    model_instance = _get_model(model)
    # nomic models expect 'search_query: ' or 'search_document: ' prefix
    prefixed_text = f"search_document: {text}"
    embedding = model_instance.encode([prefixed_text])[0]
    return embedding.tolist()


def compute_batch_sync(texts: list[str], model: str) -> list[list[float]]:
    """Compute batch embeddings synchronously."""
    model_instance = _get_model(model)
    prefixed_texts = [f"search_document: {t}" for t in texts]
    embeddings = model_instance.encode(prefixed_texts)
    return [e.tolist() for e in embeddings]


async def compute_embedding(text: str, model: str) -> list[float]:
    """Compute embedding asynchronously using process pool."""
    loop = asyncio.get_event_loop()
    executor = _get_executor()
    return await loop.run_in_executor(executor, compute_embedding_sync, text, model)


async def compute_batch(texts: list[str], model: str) -> list[list[float]]:
    """Compute batch embeddings asynchronously."""
    loop = asyncio.get_event_loop()
    executor = _get_executor()
    return await loop.run_in_executor(executor, compute_batch_sync, texts, model)
