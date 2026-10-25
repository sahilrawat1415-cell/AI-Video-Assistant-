# AI Video Assistant With RAG & Multilingual Intelligence

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ai-video-summary.streamlit.app/)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-181717?logo=github)](https://github.com/sahilrawat1415-cell/AI-Video-Assistant-)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)

🌐 **Live Demo Web App**: [https://ai-video-summary.streamlit.app/](https://ai-video-summary.streamlit.app/)

An enterprise-ready AI meeting and video intelligence system built with **FastAPI**, **Streamlit**, **LangChain**, **Groq**, **Sarvam AI**, and **ChromaDB**. It transcribes English and Hinglish videos, generates structured executive insights, and provides timestamp-grounded Retrieval-Augmented Generation (RAG) chat.

---

## Key Features

* **Intelligent Multilingual STT**:
  * **English**: Uses **Groq Whisper-large-v3-turbo** for near-instant, highly accurate transcription with segment timestamps.
  * **Hinglish**: Routes code-mixed Hindi-English meeting recordings to **Sarvam AI (saaras:v2.5)**.
* **Timestamp-Grounded RAG**: Answers cite exact timestamps (e.g., `[02:15]`) so you can jump directly to where a topic was spoken.
* **Multi-Video Session Library**: ChromaDB persists indexed sessions with a searchable video catalog; easily switch between meetings without re-processing.
* **High-Speed FFmpeg Audio Chunking**: Stream-segments audio into lightweight 16kHz mono MP3 chunks to minimize memory footprint and latency.
* **Map-Reduce Summarization**: Transcripts exceeding 10,000 characters automatically run through a concurrent map-reduce LLM pipeline to preserve all critical facts.
* **Structured Insights Extraction**: Automatically pulls **Action Items**, **Key Decisions**, and **Follow-up Questions**.
* **Dual-Mode Architecture**:
  * **REST API** powered by FastAPI (with OpenAPI / Swagger docs).
  * **Interactive Web App** powered by Streamlit with custom styling (*Syne* & *JetBrains Mono*).
  * **Terminal CLI** for automated scripting.

---

## Architecture

```
                          ┌──────────────────────────┐
                          │   YouTube / Local Media  │
                          └─────────────┬────────────┘
                                        │
                                        ▼
                          ┌──────────────────────────┐
                          │  FFmpeg Stream Chunking  │
                          └─────────────┬────────────┘
                                        │
                         ┌──────────────┴──────────────┐
                         ▼                             ▼
                 [ English Audio ]             [ Hinglish Audio ]
                 Groq Whisper v3-turbo        Sarvam AI (saaras:v2.5)
                         │                             │
                         └──────────────┬──────────────┘
                                        │
                                        ▼
                        ┌───────────────────────────────┐
                        │   Timestamped Transcript      │
                        │    & LLM Analysis Pipeline    │
                        └───────┬───────────────┬───────┘
                                │               │
         ┌──────────────────────┘               └──────────────────────┐
         ▼                                                             ▼
┌─────────────────────────────────┐                   ┌─────────────────────────────────┐
│     LLM Executive Insights      │                   │    ChromaDB Vector Store        │
│  - Title Generation             │                   │  - all-MiniLM-L6-v2 Embeddings  │
│  - Map-Reduce Summary           │                   │  - Video ID & Timestamp Meta    │
│  - Action Items Extraction      │                   │  - Multi-Video Catalog          │
│  - Key Decisions & Questions    │                   └────────────────┬────────────────┘
└─────────────────────────────────┘                                    │
                                                                       ▼
                                                      ┌─────────────────────────────────┐
                                                      │     Grounded RAG Engine         │
                                                      │    LangChain LCEL + Groq        │
                                                      └────────────────┬────────────────┘
                                                                       │
                                      ┌────────────────────────────────┴────────────────┐
                                      ▼                                                 ▼
                       ┌──────────────────────────────┐                  ┌──────────────────────────────┐
                       │   FastAPI REST Backend       │                  │     Streamlit Web UI         │
                       │   http://localhost:8000      │                  │   http://localhost:8501      │
                       └──────────────────────────────┘                  └──────────────────────────────┘
```

---

## Quickstart

### Prerequisites
* Python 3.10+
* [FFmpeg](https://ffmpeg.org/download.html) installed and in your system `PATH`
* API Keys for [Groq](https://console.groq.com) and [Sarvam AI](https://sarvam.ai) (optional, for Hinglish)

### 1. Clone & Set Up Environment

```bash
git clone https://github.com/sahilrawat1415-cell/AI-Video-Assistant-.git
cd AI-Video-Assistant-

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate       # On Linux/macOS
# or .venv\Scripts\Activate.ps1  # On Windows PowerShell

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create `.env` in the root folder based on `.env.example`:

```env
GROQ_API_KEY="your_groq_api_key_here"
SARVAM_API_KEY="your_sarvam_api_key_here"
SARVAM_STT_MODEL="saaras:v2.5"
WHISPER_MODEL="base"
USE_LOCAL_WHISPER="false"
```

---

## Running the Application

### Option A: Streamlit Web UI (Live Demo or Local)

* 🚀 **Live Hosted App**: [https://ai-video-summary.streamlit.app/](https://ai-video-summary.streamlit.app/)
* 🏠 **Run Locally**:
  ```powershell
  streamlit run app.py
  ```
  Open **`http://localhost:8501`** in your browser.

> [!TIP]
> `app.py` auto-detects whether the FastAPI server is running. If running, it delegates processing over HTTP; if not, it automatically runs in-process with zero configuration!

---

### Option B: FastAPI Backend + Streamlit UI (Full-Stack)

**Terminal 1 — API Server:**
```powershell
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```
* Interactive Swagger Docs: **`http://localhost:8000/docs`**

**Terminal 2 — Streamlit Frontend:**
```powershell
streamlit run app.py
```

---

### Option C: Terminal CLI

```powershell
python main.py
```

---

## REST API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check & active video state |
| `GET` | `/api/status` | Current processing status & session metadata |
| `GET` | `/api/videos` | Catalog of all indexed videos in ChromaDB |
| `POST` | `/api/process-video` | Downloads, transcribes, summarizes, and indexes a video |
| `POST` | `/api/chat` | Performs timestamp-grounded RAG Q&A |

### Example: Ingest Video (`POST /api/process-video`)
```json
{
  "video_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
  "language": "english"
}
```

### Example: Chat with Video (`POST /api/chat`)
```json
{
  "question": "What key deadlines were discussed?",
  "video_id": "optional_video_id"
}
```

**Response:**
```json
{
  "answer": "At [04:12], the speaker announced that the backend launch is scheduled for Friday.",
  "video_id": "dQw4w9WgXcQ",
  "sources": [
    {
      "content": "the backend launch is scheduled for Friday...",
      "timestamp": "04:12",
      "metadata": { "chunk_index": 3, "video_id": "dQw4w9WgXcQ" }
    }
  ]
}
```

---

## Project Structure

```
├── core/
│   ├── __init__.py
│   ├── extractor.py       # Action items, decisions & questions extraction
│   ├── rag_engine.py      # LangChain LCEL chain with timestamp grounding
│   ├── summarize.py       # Map-reduce & direct summarization
│   ├── transcriber.py     # Groq Whisper & Sarvam AI STT with timestamps
│   └── vector_store.py    # ChromaDB multi-video catalog & semantic retrieval
├── utils/
│   ├── __init__.py
│   └── audio_processor.py # YouTube downloading & FFmpeg stream segmentation
├── api.py                 # FastAPI REST API
├── app.py                 # Streamlit web application
├── main.py                # Command-line interface runner
├── requirements.txt       # Project dependencies
├── .gitignore             # Git exclusion rules
├── .env.example           # Example environment configuration
└── README.md              # Project documentation
```
