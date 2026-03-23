"""
Embedding module — supports:
  - Multilingual embeddings (intfloat/multilingual-e5-large)
  - Domain-specific models (biomedical, legal, etc.)
  - Batch encoding with progress tracking
  - Query vs document prefixing (required by E5 models)
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from config import settings
from utils.logger import logger
from utils.language_detector import detect_language


# ── Domain → model mapping ────────────────────────────────────────────

DOMAIN_MODELS: dict[str, str] = {
    "biomedical": "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
    "legal":      "nlpaueb/legal-bert-base-uncased",
    "financial":  "ProsusAI/finbert",
    "code":       "microsoft/codebert-base",
    "general":    settings.DEFAULT_EMBEDDING_MODEL,
}

# E5 family requires these prefixes
_E5_MODELS = {"intfloat/multilingual-e5-large", "intfloat/e5-large-v2"}


@lru_cache(maxsize=4)
def _load_model(model_name: str) -> SentenceTransformer:
    logger.info(f"Loading embedding model: {model_name}")
    return SentenceTransformer(model_name)


# ── Public API ────────────────────────────────────────────────────────

def embed_texts(
    texts: list[str],
    mode: Literal["query", "document"] = "document",
    domain: str = "general",
    batch_size: int = 32,
) -> np.ndarray:
    """
    Embed a list of texts. Returns (N, D) float32 numpy array.

    mode    : "query" for retrieval queries, "document" for indexing
    domain  : selects the appropriate model
    """
    model_name = DOMAIN_MODELS.get(domain, settings.DEFAULT_EMBEDDING_MODEL)
    model = _load_model(model_name)

    # E5 models require a task prefix
    if model_name in _E5_MODELS:
        prefix = "query: " if mode == "query" else "passage: "
        texts = [prefix + t for t in texts]

    logger.debug(f"Embedding {len(texts)} texts with {model_name} (mode={mode})")

    embeddings: list[np.ndarray] = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding", leave=False):
        batch = texts[i : i + batch_size]
        vecs = model.encode(batch, convert_to_numpy=True, normalize_embeddings=True)
        embeddings.append(vecs)

    return np.vstack(embeddings).astype(np.float32)


def embed_query(query: str, domain: str = "general") -> np.ndarray:
    """Embed a single query string. Returns (D,) float32 array."""
    return embed_texts([query], mode="query", domain=domain)[0]


def embed_chunks(
    chunks: list[dict],
    domain: str = "general",
) -> tuple[list[dict], np.ndarray]:
    """
    Embed a list of chunk dicts.
    Uses the 'text' field (the embed_text from parsers).

    Returns (chunks, embeddings) where embeddings is (N, D).
    """
    texts = [c.get("text", "") for c in chunks]
    embeddings = embed_texts(texts, mode="document", domain=domain)
    return chunks, embeddings


def detect_domain(sample_texts: list[str]) -> str:
    """
    Heuristic domain detection from sample text.
    Returns a key from DOMAIN_MODELS.
    """
    combined = " ".join(sample_texts[:5]).lower()

    biomedical_kw = {"patient", "clinical", "diagnosis", "mrna", "protein", "genome"}
    legal_kw = {"plaintiff", "defendant", "jurisdiction", "statute", "clause", "liability"}
    financial_kw = {"revenue", "ebitda", "earnings per share", "portfolio", "dividend"}
    code_kw = {"def ", "function", "import ", "class ", "return ", "variable"}

    scores = {
        "biomedical": sum(1 for k in biomedical_kw if k in combined),
        "legal":      sum(1 for k in legal_kw if k in combined),
        "financial":  sum(1 for k in financial_kw if k in combined),
        "code":       sum(1 for k in code_kw if k in combined),
    }

    best = max(scores, key=scores.__getitem__)
    if scores[best] >= 2:
        logger.info(f"Domain detected: {best}")
        return best

    logger.info("Domain detected: general")
    return "general"