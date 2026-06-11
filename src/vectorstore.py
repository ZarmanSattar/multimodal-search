"""
ChromaDB persistent vector store wrapper.

Provides a thin interface around ChromaDB for the multi-modal search system.
All embeddings (text, audio transcripts, video frames+transcripts) flow
through the same collection, distinguished by a `modality` field in metadata.

The store persists to ./chroma_db so the index survives between runs.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings

# Persistent location — relative to project root.
CHROMA_PATH = Path("./chroma_db")
COLLECTION_NAME = "multimodal"

_client: chromadb.ClientAPI | None = None
_collection: chromadb.Collection | None = None


def get_client() -> chromadb.ClientAPI:
    """Lazy-init the persistent ChromaDB client."""
    global _client
    if _client is None:
        CHROMA_PATH.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(
            path=str(CHROMA_PATH),
            settings=Settings(anonymized_telemetry=False),
        )
    return _client


def get_collection() -> chromadb.Collection:
    """Lazy-init the single multimodal collection."""
    global _collection
    if _collection is None:
        client = get_client()
        # cosine distance is the right choice for sentence-transformers embeddings
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def add_documents(
    ids: List[str],
    embeddings: List[List[float]],
    documents: List[str],
    metadatas: List[Dict[str, Any]],
) -> None:
    """
    Add documents to the collection.

    Args:
        ids: Unique string IDs for each document.
        embeddings: List of embedding vectors (each 384-dim for text/audio,
                    512-dim for CLIP video frames — kept separate by metadata).
        documents: Human-readable text snippets (transcript, caption, or raw text).
        metadatas: Per-doc metadata. Must include 'modality' ('text'|'audio'|'video')
                   and 'source' (filename or path).
    """
    collection = get_collection()
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )


def query(
    query_embedding: List[float],
    n_results: int = 5,
    where: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Find the top-N nearest documents to the given query embedding.

    Args:
        query_embedding: A single embedding vector (list of floats).
        n_results: How many matches to return.
        where: Optional metadata filter, e.g. {"modality": "audio"}.

    Returns:
        Dict with 'ids', 'documents', 'metadatas', 'distances' (each a list of lists).
    """
    collection = get_collection()
    return collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where,
    )


def count() -> int:
    """Return total number of documents in the collection."""
    return get_collection().count()


def reset() -> None:
    """Delete the entire collection. Useful for clean re-ingestion."""
    global _collection
    client = get_client()
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass
    _collection = None
