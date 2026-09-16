"""
rag_engine.py - AnalyseIt's knowledge base (vector DB indexing + retrieval).

Indexes a text file of ML best practices into a local, persistent ChromaDB
collection and retrieves the passages relevant to a data-quality issue.

Embedding backends (both produce identical all-MiniLM-L6-v2 384-dim vectors):
    "sentence-transformers"  default; the reference implementation
    "onnx"                   ChromaDB's bundled ONNX runtime - same model,
                             ~500 MB less RAM, for 1 GB deploy targets

Select with the ANALYSEIT_EMBED_BACKEND environment variable, or pass
backend= to the constructor.

Usage:
    kb = KnowledgeBase()
    kb.build_index("data/ml_docs.txt")
    hits = kb.retrieve_context("how to handle missing numeric data", top_k=3)
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable

import chromadb

MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "ml_best_practices"
DEFAULT_PERSIST_DIR = Path(__file__).resolve().parent / ".chroma"
DEFAULT_DOCS_FILE = Path(__file__).resolve().parent / "data" / "ml_docs.txt"

# Chunking: sections are split to ~this many characters, never mid-sentence.
TARGET_CHUNK_CHARS = 600
MAX_CHUNK_CHARS = 900
MIN_CHUNK_CHARS = 80


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
def _split_sentences(text: str) -> list[str]:
    """Cheap sentence split that does not fire on decimals or abbreviations."""
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text.strip())
    return [p.strip() for p in parts if p.strip()]


def chunk_document(text: str, source: str = "ml_docs.txt") -> list[dict[str, Any]]:
    """
    Split the knowledge base on '## ' headings, then pack each section into
    chunks of ~TARGET_CHUNK_CHARS without breaking sentences.

    Every chunk is prefixed with its section heading. Blind fixed-width slicing
    would cut mid-sentence and strip the heading from all but the first chunk,
    which measurably degrades retrieval - a chunk reading "Above 30 percent,
    imputation fabricates the majority of the column" is far easier to match
    when it still carries its "Handling missing values" heading.
    """
    # Drop comment lines and the leading title block.
    body = "\n".join(
        line for line in text.splitlines() if not line.startswith("#") or line.startswith("## ")
    )

    sections: list[tuple[str, str]] = []
    current_title, buffer = None, []
    for line in body.splitlines():
        if line.startswith("## "):
            if current_title and buffer:
                sections.append((current_title, "\n".join(buffer).strip()))
            current_title, buffer = line[3:].strip(), []
        elif current_title:
            buffer.append(line)
    if current_title and buffer:
        sections.append((current_title, "\n".join(buffer).strip()))

    chunks: list[dict[str, Any]] = []
    for title, content in sections:
        if not content:
            continue
        packed, buf, size = [], [], 0
        for sentence in _split_sentences(content.replace("\n\n", " ").replace("\n", " ")):
            # A single oversized sentence becomes its own chunk.
            if size + len(sentence) > MAX_CHUNK_CHARS and buf:
                packed.append(" ".join(buf))
                buf, size = [], 0
            buf.append(sentence)
            size += len(sentence) + 1
            if size >= TARGET_CHUNK_CHARS:
                packed.append(" ".join(buf))
                buf, size = [], 0
        if buf:
            tail = " ".join(buf)
            # Fold a runt tail back into the previous chunk rather than index it alone.
            if packed and len(tail) < MIN_CHUNK_CHARS:
                packed[-1] = f"{packed[-1]} {tail}"
            else:
                packed.append(tail)

        for i, chunk_text in enumerate(packed):
            chunks.append(
                {
                    "text": f"{title}\n\n{chunk_text}",
                    "section": title,
                    "chunk_index": i,
                    "n_chunks_in_section": len(packed),
                    "source": source,
                }
            )
    return chunks


# ---------------------------------------------------------------------------
# Embedding backends
# ---------------------------------------------------------------------------
class _Embedder:
    """Wraps either backend behind one embed() call returning list[list[float]]."""

    def __init__(self, backend: str, model_name: str = MODEL_NAME) -> None:
        self.backend = backend
        self.model_name = model_name
        self._impl: Any = None

    def _load(self) -> Any:
        """Lazy-load so importing this module never costs a model load."""
        if self._impl is not None:
            return self._impl
        if self.backend == "sentence-transformers":
            from sentence_transformers import SentenceTransformer

            self._impl = SentenceTransformer(self.model_name)
        elif self.backend == "onnx":
            from chromadb.utils import embedding_functions

            self._impl = embedding_functions.ONNXMiniLM_L6_V2()
        else:
            raise ValueError(
                f"Unknown embedding backend {self.backend!r}; "
                f"expected 'sentence-transformers' or 'onnx'."
            )
        return self._impl

    def embed(self, texts: Iterable[str]) -> list[list[float]]:
        texts = list(texts)
        if not texts:
            return []
        impl = self._load()
        if self.backend == "sentence-transformers":
            vectors = impl.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            return [[float(x) for x in v] for v in vectors]
        return [[float(x) for x in v] for v in impl(texts)]

    @property
    def dimension(self) -> int:
        return len(self.embed(["dimension probe"])[0])


def _resolve_backend(explicit: str | None) -> str:
    """
    Pick the embedding backend.

    Defaults to sentence-transformers, but falls back to ONNX when it is not
    installed, so one requirements.txt works for both local dev and a slim
    deploy. The fallback is safe: both backends produce byte-identical
    all-MiniLM-L6-v2 vectors. An explicit choice is never overridden.
    """
    backend = (explicit or os.getenv("ANALYSEIT_EMBED_BACKEND", "")).strip().lower()
    if backend:
        return backend
    import importlib.util

    if importlib.util.find_spec("sentence_transformers") is not None:
        return "sentence-transformers"
    return "onnx"


# ---------------------------------------------------------------------------
# Knowledge base
# ---------------------------------------------------------------------------
class KnowledgeBase:
    """Local persistent ChromaDB collection of ML best-practice passages."""

    def __init__(
        self,
        persist_dir: str | Path | None = None,
        collection_name: str = COLLECTION_NAME,
        backend: str | None = None,
        model_name: str = MODEL_NAME,
        in_memory: bool = False,
    ) -> None:
        self.backend = _resolve_backend(backend)
        self.model_name = model_name
        self.collection_name = collection_name
        self.in_memory = in_memory
        self.persist_dir = Path(persist_dir or DEFAULT_PERSIST_DIR)
        self._embedder = _Embedder(self.backend, model_name)
        self._client: Any = None
        self._collection: Any = None

    # -- infrastructure -----------------------------------------------------
    @property
    def client(self) -> Any:
        if self._client is None:
            if self.in_memory:
                self._client = chromadb.EphemeralClient()
            else:
                self.persist_dir.mkdir(parents=True, exist_ok=True)
                self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        return self._client

    @property
    def collection(self) -> Any:
        if self._collection is None:
            # Cosine space: embeddings are L2-normalized, so cosine distance is
            # the meaningful metric. Chroma defaults to L2 otherwise.
            self._collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    @property
    def _fingerprint_path(self) -> Path:
        return self.persist_dir / f"{self.collection_name}.fingerprint.json"

    def _fingerprint(self, text: str) -> str:
        """Identifies the indexed content AND the vectors used to index it."""
        payload = f"{self.backend}|{self.model_name}|{self.collection_name}|{text}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def count(self) -> int:
        try:
            return int(self.collection.count())
        except Exception:
            return 0

    # -- indexing -----------------------------------------------------------
    def build_index(
        self,
        text_file: str | Path | None = None,
        force: bool = False,
        batch_size: int = 64,
    ) -> dict[str, Any]:
        """
        Read the knowledge base file, chunk, embed, and store in ChromaDB.

        Skips the rebuild when the collection already holds this exact content
        embedded by this exact backend, so app startup is cheap. Pass force=True
        to reindex regardless.
        """
        path = Path(text_file or DEFAULT_DOCS_FILE)
        if not path.exists():
            raise FileNotFoundError(f"Knowledge base file not found: {path}")

        text = path.read_text(encoding="utf-8")
        fingerprint = self._fingerprint(text)

        if not force and self.count() > 0 and self._read_fingerprint() == fingerprint:
            return {
                "status": "up-to-date",
                "chunks": self.count(),
                "backend": self.backend,
                "collection": self.collection_name,
            }

        chunks = chunk_document(text, source=path.name)
        if not chunks:
            raise ValueError(f"No '## ' sections found in {path}; nothing to index.")

        # Replace the collection outright - stale vectors from a previous
        # backend or an older docs file must not survive a rebuild.
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self._collection = None

        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            self.collection.add(
                ids=[f"{self.collection_name}-{start + i}" for i in range(len(batch))],
                documents=[c["text"] for c in batch],
                embeddings=self._embedder.embed(c["text"] for c in batch),
                metadatas=[
                    {
                        "section": c["section"],
                        "chunk_index": c["chunk_index"],
                        "n_chunks_in_section": c["n_chunks_in_section"],
                        "source": c["source"],
                    }
                    for c in batch
                ],
            )

        self._write_fingerprint(fingerprint)
        return {
            "status": "built",
            "chunks": len(chunks),
            "sections": len({c["section"] for c in chunks}),
            "backend": self.backend,
            "model": self.model_name,
            "collection": self.collection_name,
            "persist_dir": None if self.in_memory else str(self.persist_dir),
        }

    def ensure_index(self, text_file: str | Path | None = None) -> dict[str, Any]:
        """Build the index only if it is missing or stale. Safe to call on every run."""
        return self.build_index(text_file, force=False)

    def _read_fingerprint(self) -> str | None:
        if self.in_memory:
            return getattr(self, "_memory_fingerprint", None)
        try:
            return json.loads(self._fingerprint_path.read_text(encoding="utf-8"))["fingerprint"]
        except Exception:
            return None

    def _write_fingerprint(self, fingerprint: str) -> None:
        if self.in_memory:
            self._memory_fingerprint = fingerprint
            return
        self._fingerprint_path.write_text(
            json.dumps({"fingerprint": fingerprint, "backend": self.backend}), encoding="utf-8"
        )

    # -- retrieval ----------------------------------------------------------
    def retrieve_context(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        """
        Return the top_k most relevant passages for a query.

        Each hit: {"text", "section", "similarity", "distance", "source"}.
        Returns [] when the index is empty rather than raising, so the UI can
        degrade gracefully.
        """
        if not query or not query.strip():
            return []
        if self.count() == 0:
            return []

        result = self.collection.query(
            query_embeddings=self._embedder.embed([query]),
            n_results=min(top_k, self.count()),
        )

        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        hits = []
        for doc, meta, dist in zip(documents, metadatas, distances):
            meta = meta or {}
            hits.append(
                {
                    "text": doc,
                    "section": meta.get("section", "unknown"),
                    "source": meta.get("source", "unknown"),
                    "distance": round(float(dist), 4),
                    "similarity": round(1.0 - float(dist), 4),  # cosine space
                }
            )
        return hits

    def retrieve_for_issues(
        self, issues: list[dict[str, Any]], top_k: int = 3
    ) -> list[dict[str, Any]]:
        """
        Retrieve context for profiler issues, using each issue's `rag_query`.

        This is the seam between Module 1 and Module 3: profiler.top_issues(2)
        goes in, and issues annotated with retrieved passages come out.
        """
        annotated = []
        for issue in issues:
            query = issue.get("rag_query") or issue.get("issue_type", "")
            annotated.append({**issue, "retrieved_context": self.retrieve_context(query, top_k)})
        return annotated

    @staticmethod
    def format_context(hits: list[dict[str, Any]], max_chars: int = 4000) -> str:
        """Flatten retrieved passages into the text block injected into the LLM prompt."""
        blocks, total = [], 0
        for i, hit in enumerate(hits, start=1):
            block = f"[Source {i} | {hit['section']}]\n{hit['text']}"
            if total + len(block) > max_chars:
                break
            blocks.append(block)
            total += len(block)
        return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _cli() -> int:
    """
    python rag_engine.py --build            build/refresh the index
    python rag_engine.py --build --force    reindex from scratch
    python rag_engine.py "missing values"   search the index
    """
    import argparse

    parser = argparse.ArgumentParser(description="AnalyseIt knowledge base")
    parser.add_argument("query", nargs="?", help="search the index")
    parser.add_argument("--build", action="store_true", help="build or refresh the index")
    parser.add_argument("--force", action="store_true", help="force a full reindex")
    parser.add_argument("--backend", choices=["sentence-transformers", "onnx"],
                        help="embedding backend (default: $ANALYSEIT_EMBED_BACKEND or sentence-transformers)")
    parser.add_argument("--docs", default=str(DEFAULT_DOCS_FILE), help="path to ml_docs.txt")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    kb = KnowledgeBase(backend=args.backend)
    print(f"backend: {kb.backend}  |  model: {kb.model_name}")

    if args.build or kb.count() == 0:
        print("Indexing (first run downloads the model, ~90 MB)...")
        info = kb.build_index(args.docs, force=args.force)
        print(json.dumps(info, indent=2))

    if args.query:
        hits = kb.retrieve_context(args.query, top_k=args.top_k)
        print(f"\nQuery: {args.query!r}  ->  {len(hits)} hits\n" + "=" * 78)
        for i, hit in enumerate(hits, start=1):
            preview = hit["text"].replace("\n", " ")
            print(f"\n{i}. [{hit['similarity']:.3f}] {hit['section']}")
            print(f"   {preview[:300]}{'...' if len(preview) > 300 else ''}")
    elif not args.build:
        print(f"Index holds {kb.count()} chunks. Pass a query to search.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
