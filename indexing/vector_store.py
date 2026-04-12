"""
Vector store — ChromaDB wrapper with:
  - Soft-delete (mark deprecated, filter at query time)
  - Versioning support
  - Parent-child chunk storage
  - Metadata filtering
"""
from __future__ import annotations

from typing import Any

import chromadb
import numpy as np
from chromadb.config import Settings as ChromaSettings

from config import settings
from indexing.metadata import build_metadata, restore_metadata
from utils.logger import logger


class VectorStore:
    def __init__(self):
        self._client = chromadb.PersistentClient(
            path=settings.CHROMA_PERSIST_DIR,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        # Main collection — child chunks (used for retrieval)
        self._col = self._client.get_or_create_collection(
            name=settings.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        # Parent collection — full sections (fetched after retrieval)
        self._parent_col = self._client.get_or_create_collection(
            name=settings.COLLECTION_NAME + "_parents",
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"VectorStore ready. "
            f"Child chunks: {self._col.count()}, "
            f"Parent chunks: {self._parent_col.count()}"
        )

    # ── Indexing ──────────────────────────────────────────────────────

    def add_chunks(
        self,
        chunks: list[dict[str, Any]],
        embeddings: np.ndarray,
    ) -> None:
        if not chunks:
            return

        ids, docs, metas, embs = [], [], [], []
        for i, ch in enumerate(chunks):
            cid = ch.get("chunk_id") or str(i)
            ids.append(cid)
            docs.append(ch.get("text", ""))
            metas.append(build_metadata(ch))
            embs.append(embeddings[i].tolist())

        # Upsert so re-ingesting the same doc updates rather than duplicates
        self._col.upsert(
            ids=ids,
            documents=docs,
            metadatas=metas,
            embeddings=embs,
        )
        logger.info(f"Upserted {len(ids)} child chunks into vector store")

    def add_parent_chunks(
        self,
        parents: list[dict[str, Any]],
        embeddings: np.ndarray,
    ) -> None:
        if not parents:
            return
        ids, docs, metas, embs = [], [], [], []
        for i, p in enumerate(parents):
            pid = p.get("chunk_id") or f"parent_{i}"
            ids.append(pid)
            docs.append(p.get("text", ""))
            metas.append(build_metadata(p))
            embs.append(embeddings[i].tolist())
        self._parent_col.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
        logger.info(f"Upserted {len(ids)} parent chunks")

    def soft_delete_version(self, source_file: str, old_version: str) -> None:
        """Mark all chunks from a superseded document version as deprecated."""
        try:
            results = self._col.get(
                where={"$and": [
                    {"source_file": {"$eq": source_file}},
                    {"doc_version": {"$eq": old_version}},
                ]}
            )
            ids = results.get("ids", [])
            if not ids:
                return
            new_metas = []
            for m in results.get("metadatas", []):
                m = dict(m)
                m["is_deprecated"] = True
                new_metas.append(m)
            self._col.update(ids=ids, metadatas=new_metas)
            logger.info(f"Soft-deleted {len(ids)} chunks from {source_file} v{old_version[:8]}")
        except Exception as e:
            logger.warning(f"Soft-delete failed: {e}")

    # ── Retrieval ─────────────────────────────────────────────────────

    def query(
        self,
        query_embedding: np.ndarray,
        top_k: int = 30,
        where: dict | None = None,
        include_deprecated: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Vector search. Returns list of result dicts sorted by distance.
        Excludes deprecated chunks by default.
        """
        base_filter: dict = {"is_deprecated": {"$eq": False}}
        if not include_deprecated:
            if where:
                effective_where = {"$and": [base_filter, where]}
            else:
                effective_where = base_filter
        else:
            effective_where = where or {}

        try:
            results = self._col.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=min(top_k, max(1, self._col.count())),
                where=effective_where if effective_where else None,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            message = str(e)
            if "expecting embedding with dimension" in message:
                logger.warning(
                    "Vector query failed due to an embedding dimension mismatch. "
                    "The collection was indexed with a different embedding model; "
                    "clear the vector store and re-ingest with one consistent model. "
                    f"Details: {message}"
                )
            else:
                logger.warning(f"Vector query failed: {message}")
            return []

        return self._unpack_results(results)

    def get_parent(self, parent_id: str) -> dict[str, Any] | None:
        """Fetch a parent chunk by ID for context expansion."""
        if not parent_id:
            return None
        try:
            r = self._parent_col.get(ids=[parent_id], include=["documents", "metadatas"])
            if r["ids"]:
                return {
                    "chunk_id": r["ids"][0],
                    "text": r["documents"][0],
                    "metadata": restore_metadata(r["metadatas"][0]),
                }
        except Exception:
            pass
        return None

    def get_all_texts(self) -> list[str]:
        """Return all non-deprecated chunk texts for BM25 index building."""
        try:
            r = self._col.get(
                where={"is_deprecated": {"$eq": False}},
                include=["documents", "metadatas"],
            )
            return r.get("documents", [])
        except Exception:
            return []

    def get_all_chunks(self) -> list[dict[str, Any]]:
        """Return all non-deprecated chunks for BM25 index building."""
        try:
            r = self._col.get(
                where={"is_deprecated": {"$eq": False}},
                include=["documents", "metadatas"],
            )
            out = []
            for cid, doc, meta in zip(
                r.get("ids", []),
                r.get("documents", []),
                r.get("metadatas", []),
            ):
                out.append({
                    "chunk_id": cid,
                    "text": doc,
                    "metadata": restore_metadata(meta),
                    "score": 0.0,
                })
            return out
        except Exception as e:
            logger.warning(f"get_all_chunks failed: {e}")
            return []

    def get_all_parent_chunks(self) -> list[dict[str, Any]]:
        """Return all stored parent chunks for summary-style query fallbacks."""
        try:
            r = self._parent_col.get(
                include=["documents", "metadatas"],
            )
            out = []
            for cid, doc, meta in zip(
                r.get("ids", []),
                r.get("documents", []),
                r.get("metadatas", []),
            ):
                restored = restore_metadata(meta)
                if restored.get("is_deprecated", False):
                    continue
                out.append({
                    "chunk_id": cid,
                    "text": doc,
                    "metadata": restored,
                    "score": 0.0,
                })
            return out
        except Exception as e:
            logger.warning(f"get_all_parent_chunks failed: {e}")
            return []

    def count(self) -> int:
        return self._col.count()

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _unpack_results(raw: dict) -> list[dict[str, Any]]:
        ids       = raw.get("ids", [[]])[0]
        docs      = raw.get("documents", [[]])[0]
        metas     = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0]

        results = []
        for cid, doc, meta, dist in zip(ids, docs, metas, distances):
            # ChromaDB cosine distance → similarity score
            score = max(0.0, 1.0 - dist)
            results.append({
                "chunk_id": cid,
                "text": doc,
                "metadata": restore_metadata(meta),
                "score": round(score, 4),
            })
        return sorted(results, key=lambda x: x["score"], reverse=True)
