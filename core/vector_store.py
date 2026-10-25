import os
import json
import time
from typing import Optional, List, Dict, Any
from datetime import datetime

import streamlit as st
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

CHROMA_DIR = "vector_db"
COLLECTION_NAME = "meeting_transcript"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CATALOG_PATH = os.path.join(CHROMA_DIR, "catalog.json")


def _get_catalog() -> List[Dict[str, Any]]:
    if not os.path.exists(CATALOG_PATH):
        return []
    try:
        with open(CATALOG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _save_catalog(catalog: List[Dict[str, Any]]) -> None:
    os.makedirs(CHROMA_DIR, exist_ok=True)
    try:
        with open(CATALOG_PATH, "w", encoding="utf-8") as f:
            json.dump(catalog, f, indent=2)
    except Exception as e:
        print(f"Warning: Failed to save catalog: {e}")


def list_indexed_videos() -> List[Dict[str, Any]]:
    """Returns a list of all indexed videos in the vector store catalog."""
    return _get_catalog()


_cached_embeddings = None


def _init_embeddings():
    global _cached_embeddings
    if _cached_embeddings is None:
        _cached_embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"batch_size": 32, "normalize_embeddings": False},
        )
    return _cached_embeddings


try:
    @st.cache_resource(show_spinner=False)
    def get_embeddings():
        return _init_embeddings()
except Exception:
    def get_embeddings():
        return _init_embeddings()


def build_vector_store(
    transcript: Any,
    video_id: str = "default_session",
    title: str = "Meeting Recording",
) -> Chroma:
    """
    Builds or updates the vector store with multi-video metadata support
    and timestamp-aware document chunks using fast batched embeddings.
    """
    os.makedirs(CHROMA_DIR, exist_ok=True)
    embeddings = get_embeddings()

    # Use timestamped text if available on TranscriptResult
    text_to_index = getattr(transcript, "timestamped_text", str(transcript))

    # Optimized chunk size: 1200 chars gives rich semantic context and reduces embedding time by ~50%
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=120,
    )
    chunks = splitter.split_text(text_to_index)

    docs = []
    for i, chunk in enumerate(chunks):
        # Infer timestamp from the beginning of chunk if formatted as [MM:SS]
        ts = None
        if chunk.startswith("[") and "]" in chunk[:12]:
            ts = chunk[1:chunk.find("]")]

        meta = {
            "chunk_index": i,
            "video_id": video_id,
            "title": title[:100],
        }
        if ts:
            meta["timestamp"] = ts

        docs.append(Document(page_content=chunk, metadata=meta))

    # Connect to persistent Chroma collection
    vector_store = _safe_chroma_init(embeddings)

    # Delete existing entries for this video_id before re-inserting
    try:
        if hasattr(vector_store, "_collection") and vector_store._collection:
            vector_store.delete(where={"video_id": video_id})
    except Exception:
        pass

    if docs:
        vector_store.add_documents(docs)

    # Update catalog
    catalog = [v for v in _get_catalog() if v.get("video_id") != video_id]
    catalog.insert(0, {
        "video_id": video_id,
        "title": title,
        "chunks_indexed": len(docs),
        "indexed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })
    _save_catalog(catalog)

    return vector_store


def _safe_chroma_init(embeddings) -> Chroma:
    """
    Wraps Chroma instantiation with a retry that clears ChromaDB's stale
    SharedSystemClient cache — prevents KeyError crashes during Streamlit reruns.
    """
    try:
        return Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=CHROMA_DIR,
        )
    except KeyError:
        # ChromaDB's SharedSystemClient has a stale cached reference; clear it.
        try:
            import chromadb
            from chromadb.api.shared_system_client import SharedSystemClient
            SharedSystemClient._identifier_to_system.clear()
        except Exception:
            pass
        return Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=CHROMA_DIR,
        )


def load_vector_store() -> Chroma:
    embeddings = get_embeddings()
    return _safe_chroma_init(embeddings)


def vector_store_exists() -> bool:
    return os.path.exists(CHROMA_DIR) and bool(os.listdir(CHROMA_DIR))


def get_retriever(vector_store: Chroma, k: int = 4, video_id: Optional[str] = None):
    """
    Returns a similarity retriever. If video_id is given, filters strictly
    to that video. If None, retrieves across all indexed videos.
    """
    search_kwargs: Dict[str, Any] = {"k": k}
    if video_id and video_id != "all":
        search_kwargs["filter"] = {"video_id": video_id}

    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs,
    )