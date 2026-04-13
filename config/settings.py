from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional


class Settings(BaseSettings):
    # ── Unstructured.io API ──────────────────────────────────────────────
    UNSTRUCTURED_API_KEY: str = Field(..., description="Unstructured.io API key")
    UNSTRUCTURED_API_URL: str = "https://api.unstructuredapp.io/general/v0/general"

    # ── Groq Cloud ───────────────────────────────────────────────────────
    GROQ_API_KEY: str = Field(..., description="Groq Cloud API key")
    # Model options:
    #   llama-3.3-70b-versatile     ← best quality (default)
    #   llama-3.1-8b-instant        ← fastest / cheapest
    #   mixtral-8x7b-32768          ← long context window
    #   gemma2-9b-it                ← lightweight
    LLM_MODEL: str = "llama-3.3-70b-versatile"
    VISION_MODEL: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    MAX_TOKENS: int = 2048

    # ── Vector Store ─────────────────────────────────────────────────────
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    COLLECTION_NAME: str = "rag_collection"

    # ── Embedding Model ───────────────────────────────────────────────────
    DEFAULT_EMBEDDING_MODEL: str = "intfloat/multilingual-e5-large"
    AUTO_DETECT_EMBEDDING_DOMAIN: bool = False
    USE_FORMULA_VISION_FALLBACK: bool = True

    # ── Retrieval ─────────────────────────────────────────────────────────
    VECTOR_SEARCH_TOP_K: int = 30
    BM25_TOP_K: int = 30
    RERANK_TOP_K: int = 5
    RELEVANCE_THRESHOLD: float = 0.02
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ── Chunking ──────────────────────────────────────────────────────────
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64
    PARENT_CHUNK_SIZE: int = 1536

    # ── Context window ────────────────────────────────────────────────────
    # Groq's Llama 3.3 70B supports 128k context — we use 6k for retrieval
    MAX_CONTEXT_TOKENS: int = 6000

    # ── Recency decay ────────────────────────────────────────────────────
    RECENCY_DECAY_DAYS: int = 365

    # ── NLI hallucination guard ───────────────────────────────────────────
    NLI_MODEL: str = "cross-encoder/nli-deberta-v3-small"
    NLI_ENTAIL_THRESHOLD: float = -1.0

    # ── Prompt injection ─────────────────────────────────────────────────
    INJECTION_PATTERNS: list = [
        "ignore previous instructions",
        "ignore all instructions",
        "disregard",
        "forget your instructions",
        "new instructions",
        "system prompt",
        "jailbreak",
        "act as if",
        "pretend you are",
        "you are now",
    ]

    AGENT_MAX_STEPS: int = 5
    AGENT_MEMORY_DIR: str = "./agent_sessions"
    AUTO_ROUTE_QUERIES: bool = True
    TOOL_PROMPT_SUMMARIZE_DOCUMENT: str = "Summary of pdf"
    TOOL_PROMPT_EXPLAIN_FIGURE: str = "Kindly explain Figure {figure_number} properly"
    TOOL_PROMPT_COMPARE_FIGURES: str = (
        "What is the difference between Figure {left_figure} and Figure {right_figure}?"
    )
    TOOL_PROMPT_LOOKUP_TABLE: str = (
        "Explain Table {table_number} briefly and state its contents."
    )
    TOOL_PROMPT_COMPARE_TABLES: str = (
        "What is the difference between Table {left_table} and Table {right_table}?"
    )
    TOOL_PROMPT_LOOKUP_FORMULA: str = "mention the {formula_hint} equation"
    TOOL_PROMPT_LOOKUP_FORMULA_ALL: str = "Properly mention all the equations in the pdf"
    TOOL_PROMPT_LOOKUP_METRIC: str = (
        "What is the {metric_name} score of {model_name} model"
    )

    class Config:
        extra = "ignore"
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
