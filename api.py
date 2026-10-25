import os
import re
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List, Dict, Any
from pathlib import Path

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# ── Load .env BEFORE importing project modules ──────────────────────────────────
ENV_FILE = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_FILE, override=True)

# ── Core pipeline imports ────────────────────────────────────────────────────────
from utils.audio_processor import process_input
from core.transcriber import transcribe_all, TranscriptResult, get_youtube_transcript
from core.summarize import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions, extract_all_insights
from core.rag_engine import build_rag_chain, load_rag_chain, ask_question
from core.vector_store import load_vector_store, list_indexed_videos

# ── FastAPI App ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AI Video Assistant With RAG API",
    description="Backend REST API for video transcription, vector indexing, multi-video catalog and RAG Q&A",
    version="2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory state ──────────────────────────────────────────────────────────────
_state: Dict[str, Any] = {
    "rag_chain": None,
    "last_result": None,
    "active_video_id": None,
}


def extract_video_id(source: str) -> str:
    """Extracts a clean unique identifier from a YouTube URL or file path."""
    s = str(source).strip()
    if "youtu.be/" in s:
        return s.split("youtu.be/")[1].split("?")[0].split("&")[0]
    if "youtube.com/watch" in s:
        m = re.search(r"[?&]v=([^&]+)", s)
        if m:
            return m.group(1)
    if "youtube.com/shorts/" in s:
        return s.split("youtube.com/shorts/")[1].split("?")[0].split("&")[0]
    if "youtube.com/embed/" in s:
        return s.split("youtube.com/embed/")[1].split("?")[0].split("&")[0]
    base = os.path.splitext(os.path.basename(s))[0]
    return re.sub(r"[^a-zA-Z0-9_-]", "_", base)[:40] or "meeting_session"


# ── Request / Response Models ────────────────────────────────────────────────────

class ProcessVideoRequest(BaseModel):
    video_url: str = Field(
        ...,
        description="YouTube URL or local file path to the video/audio to ingest",
        examples=["https://www.youtube.com/watch?v=dQw4w9WgXcQ"],
    )
    language: str = Field(
        default="english",
        description="Transcription language: 'english' (Whisper) or 'hinglish' (Sarvam AI)",
    )


class ProcessVideoResponse(BaseModel):
    status: str
    message: str
    video_id: str
    title: Optional[str] = None
    summary: Optional[str] = None
    action_items: Optional[str] = None
    key_decisions: Optional[str] = None
    open_questions: Optional[str] = None
    transcript: Optional[str] = None
    timestamped_transcript: Optional[str] = None
    transcript_preview: Optional[str] = None
    chunks_indexed: Optional[int] = None
    elapsed_seconds: Optional[float] = None


class ChatRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=2,
        description="The question to ask about the video transcript",
    )
    video_id: Optional[str] = Field(
        default=None,
        description="Optional video_id to filter the search to. If omitted, searches the active video.",
    )


class ChatResponse(BaseModel):
    answer: str
    video_id: Optional[str] = None
    sources: Optional[List[Dict[str, Any]]] = Field(
        default=[],
        description="Retrieved transcript segments with timestamps used to generate the answer",
    )


# ── Core pipeline handlers ───────────────────────────────────────────────────────

def run_video_ingestion(video_url: str, language: str = "english") -> Dict[str, Any]:
    """Full pipeline: download → chunk → transcribe → summarise → extract → RAG index."""
    try:
        t0 = time.time()
        video_id = extract_video_id(video_url)

        transcript = None
        if video_url.startswith(("http://", "https://")) and ("youtube.com" in video_url or "youtu.be" in video_url):
            try:
                transcript = get_youtube_transcript(video_url.strip())
            except Exception as e:
                print(f"API direct transcript error: {e}")
                transcript = None

        if transcript is None:
            # Step 1 — Audio download & FFmpeg chunking
            chunks = process_input(video_url.strip(), language=language)

            # Step 2 — Parallel transcription with timestamps
            transcript = transcribe_all(chunks, language=language)
            if not transcript or not str(transcript).strip():
                raise RuntimeError("Transcription returned empty text.")

        timestamped_text = getattr(transcript, "timestamped_text", str(transcript))

        # Step 3 — High-speed single-pass Insights + Chroma Vector Store in parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            f_insights  = executor.submit(extract_all_insights, transcript)
            f_rag       = executor.submit(build_rag_chain, transcript, video_id=video_id)

            insights     = f_insights.result()
            title        = insights.get("title", "Meeting Brief")
            summary      = insights.get("summary", "")
            action_items = insights.get("action_items", "")
            decisions    = insights.get("key_decisions", "")
            questions    = insights.get("open_questions", "")
            rag_chain    = f_rag.result()

        elapsed = time.time() - t0

        # Retrieve indexed chunk count
        chunks_count = None
        try:
            vs = load_vector_store()
            if hasattr(vs, "_collection") and vs._collection:
                chunks_count = vs._collection.count()
        except Exception:
            pass

        result = {
            "video_id": video_id,
            "title": title,
            "transcript": str(transcript),
            "timestamped_transcript": timestamped_text,
            "summary": summary,
            "action_items": action_items,
            "key_decisions": decisions,
            "open_questions": questions,
            "rag_chain": rag_chain,
            "chunks_indexed": chunks_count,
            "elapsed": elapsed,
        }

        # Store in memory for subsequent /chat requests
        _state["rag_chain"] = rag_chain
        _state["last_result"] = result
        _state["active_video_id"] = video_id

        return result

    except Exception as e:
        raise RuntimeError(f"Video ingestion failed: {e}")


def run_rag_query(question: str, video_id: Optional[str] = None) -> Dict[str, Any]:
    """Queries the RAG chain and retrieves source chunks with timestamps."""
    target_vid = video_id or _state.get("active_video_id")

    # If targeting a specific video or no in-memory chain, load from ChromaDB
    if target_vid or _state.get("rag_chain") is None:
        try:
            chain = load_rag_chain(video_id=target_vid)
        except Exception:
            chain = _state.get("rag_chain")
    else:
        chain = _state.get("rag_chain")

    if chain is None:
        raise RuntimeError(
            "No video has been processed yet. Call /api/process-video first."
        )

    try:
        answer = ask_question(chain, question)
        sources = []
        try:
            vs = load_vector_store()
            search_kwargs = {"k": 3}
            if target_vid and target_vid != "all":
                search_kwargs["filter"] = {"video_id": target_vid}

            retriever = vs.as_retriever(search_type="similarity", search_kwargs=search_kwargs)
            docs = retriever.invoke(question)

            sources = [
                {
                    "content": doc.page_content,
                    "timestamp": doc.metadata.get("timestamp"),
                    "video_id": doc.metadata.get("video_id"),
                    "metadata": doc.metadata,
                }
                for doc in docs
            ]
        except Exception as e:
            print(f"Warning: Failed to fetch sources: {e}")

        return {
            "answer": str(answer),
            "video_id": target_vid,
            "sources": sources,
        }
    except Exception as e:
        raise RuntimeError(f"RAG query failed: {e}")


# ── API Endpoints ────────────────────────────────────────────────────────────────

@app.get("/")
def health_check():
    has_video = _state["last_result"] is not None
    return {
        "status": "online",
        "service": "AI Video Assistant With RAG",
        "video_loaded": has_video,
        "active_video_id": _state.get("active_video_id"),
    }


@app.get("/api/videos")
def get_videos_catalog():
    """Returns the catalog of all previously processed and indexed videos."""
    return {"videos": list_indexed_videos()}


@app.post("/api/process-video", response_model=ProcessVideoResponse)
async def process_video(req: ProcessVideoRequest):
    """Download, transcribe, summarise, extract insights, and index video."""
    try:
        result = await asyncio.to_thread(
            run_video_ingestion, req.video_url, req.language
        )
        return ProcessVideoResponse(
            status="completed",
            message="Video processed and indexed successfully.",
            video_id=result["video_id"],
            title=result["title"],
            summary=result["summary"],
            action_items=result["action_items"],
            key_decisions=result["key_decisions"],
            open_questions=result["open_questions"],
            transcript=result["transcript"],
            timestamped_transcript=result.get("timestamped_transcript"),
            transcript_preview=result["transcript"][:500],
            chunks_indexed=result.get("chunks_indexed"),
            elapsed_seconds=round(result["elapsed"], 2),
        )
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))


@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_video(req: ChatRequest):
    """Ask a question about a video transcript with timestamp-grounded RAG."""
    try:
        result = await asyncio.to_thread(run_rag_query, req.question, req.video_id)
        return ChatResponse(
            answer=result["answer"],
            video_id=result.get("video_id"),
            sources=result.get("sources", []),
        )
    except RuntimeError as err:
        raise HTTPException(status_code=400, detail=str(err))
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))


@app.get("/api/status")
def pipeline_status():
    """Returns the last pipeline result metadata."""
    if _state["last_result"] is None:
        return {"status": "idle", "message": "No video processed yet."}
    r = _state["last_result"]
    return {
        "status": "ready",
        "video_id": r.get("video_id"),
        "title": r.get("title"),
        "summary": r.get("summary"),
        "elapsed_seconds": round(r.get("elapsed", 0), 2),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
