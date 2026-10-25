import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from dotenv import load_dotenv
from groq import Groq

# core/transcriber.py -> project root is the parent of the `core` directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"

load_dotenv(dotenv_path=ENV_FILE, override=True)

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
USE_LOCAL_WHISPER = os.getenv("USE_LOCAL_WHISPER", "false").lower() == "true"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "").strip()
SARVAM_STT_TRANSLATE_URL = "https://api.sarvam.ai/speech-to-text-translate"
SARVAM_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v2.5")

_model = None
_groq_client = None


def format_timestamp(seconds: float) -> str:
    """Converts seconds to HH:MM:SS or MM:SS."""
    secs = int(max(0, seconds))
    h = secs // 3600
    m = (secs % 3600) // 60
    s = secs % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


class TranscriptResult(str):
    """
    Subclasses str for 100% backward compatibility, while exposing
    structured segment timestamps and formatted timestamped text.
    """
    def __new__(cls, text: str, segments: list = None, timestamped_text: str = ""):
        inst = super().__new__(cls, text)
        inst.segments = segments or []
        inst.timestamped_text = timestamped_text or text
        return inst


def load_model():
    """Load local Whisper lazily, only when needed."""
    global _model
    if _model is None:
        import whisper

        print(f"Loading local Whisper model: {WHISPER_MODEL} ...")
        _model = whisper.load_model(WHISPER_MODEL)
    return _model


def get_groq_client():
    """Create the Groq client lazily."""
    global _groq_client
    if _groq_client is None:
        if not GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY is missing. Check the project's .env file."
            )
        _groq_client = Groq(api_key=GROQ_API_KEY)
    return _groq_client


def transcribe_chunk_whisper(chunk_path: str, chunk_offset_sec: float = 0.0) -> dict:
    """Transcribe one chunk with Groq Whisper (extracting timestamps) or local Whisper."""
    segments = []

    if not USE_LOCAL_WHISPER and GROQ_API_KEY:
        client = get_groq_client()
        with open(chunk_path, "rb") as audio_file:
            response = client.audio.transcriptions.create(
                file=(os.path.basename(chunk_path), audio_file.read()),
                model="whisper-large-v3-turbo",
                response_format="verbose_json",
            )

        raw_segs = getattr(response, "segments", None)
        if raw_segs is None and isinstance(response, dict):
            raw_segs = response.get("segments", [])
        raw_segs = raw_segs or []

        for seg in raw_segs:
            s_start = seg.get("start", 0) if isinstance(seg, dict) else getattr(seg, "start", 0)
            s_end = seg.get("end", 0) if isinstance(seg, dict) else getattr(seg, "end", 0)
            s_text = (seg.get("text", "") if isinstance(seg, dict) else getattr(seg, "text", "")).strip()
            if s_text:
                abs_start = chunk_offset_sec + s_start
                abs_end = chunk_offset_sec + s_end
                segments.append({
                    "start": abs_start,
                    "end": abs_end,
                    "timestamp": format_timestamp(abs_start),
                    "text": s_text,
                })

        text = getattr(response, "text", "")
        if not text and isinstance(response, dict):
            text = response.get("text", "")
        if not text and segments:
            text = " ".join(s["text"] for s in segments)

        return {"text": text.strip(), "segments": segments}

    model = load_model()
    result = model.transcribe(chunk_path, task="transcribe")
    raw_segs = result.get("segments", [])
    for seg in raw_segs:
        abs_start = chunk_offset_sec + seg.get("start", 0)
        abs_end = chunk_offset_sec + seg.get("end", 0)
        s_text = seg.get("text", "").strip()
        if s_text:
            segments.append({
                "start": abs_start,
                "end": abs_end,
                "timestamp": format_timestamp(abs_start),
                "text": s_text,
            })

    return {"text": result.get("text", "").strip(), "segments": segments}


def transcribe_chunk_sarvam(piece_path: str, chunk_offset_sec: float = 0.0) -> dict:
    """Transcribe one chunk with Sarvam AI."""
    if not SARVAM_API_KEY:
        raise RuntimeError(
            "SARVAM_API_KEY was not loaded. Checked: "
            f"{ENV_FILE}. Confirm the .env variable is named exactly "
            "SARVAM_API_KEY and restart Streamlit."
        )

    headers = {"api-subscription-key": SARVAM_API_KEY}
    with open(piece_path, "rb") as audio_file:
        files = {
            "file": (
                os.path.basename(piece_path),
                audio_file,
                "audio/mpeg",
            )
        }
        data = {
            "model": SARVAM_MODEL,
            "with_diarization": "false",
        }
        response = requests.post(
            SARVAM_STT_TRANSLATE_URL,
            headers=headers,
            files=files,
            data=data,
            timeout=120,
        )

    response.raise_for_status()
    transcript = response.json().get("transcript", "").strip()
    segments = []
    if transcript:
        segments.append({
            "start": chunk_offset_sec,
            "end": chunk_offset_sec + 25.0,
            "timestamp": format_timestamp(chunk_offset_sec),
            "text": transcript,
        })
    return {"text": transcript, "segments": segments}


def transcribe_chunk(chunk_path: str, language: str = "english", chunk_offset_sec: float = 0.0) -> dict:
    """Route Hinglish to Sarvam and English to Whisper."""
    if language.lower() == "hinglish":
        return transcribe_chunk_sarvam(chunk_path, chunk_offset_sec=chunk_offset_sec)
    return transcribe_chunk_whisper(chunk_path, chunk_offset_sec=chunk_offset_sec)


import json


def transcribe_all(chunks: list[str], language: str = "english", progress_callback=None) -> TranscriptResult:
    """Transcribe chunks concurrently while preserving their original order and timestamps, with smart disk caching."""
    if not chunks:
        return TranscriptResult("")

    # 1. Check if cached transcript already exists on disk for this chunk set
    chunk_dir = os.path.dirname(chunks[0])
    cache_path = os.path.join(chunk_dir, f"transcript_{language.lower()}.json")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            cached_text = cached.get("text", "")
            cached_segs = cached.get("segments", [])
            cached_ts = cached.get("timestamped_text", cached_text)
            if cached_text:
                print(f"⚡ Cache hit: reusing existing transcript from {cache_path}")
                if progress_callback:
                    progress_callback(len(chunks), len(chunks))
                return TranscriptResult(
                    text=cached_text,
                    segments=cached_segs,
                    timestamped_text=cached_ts,
                )
        except Exception as e:
            print(f"Warning: Failed to read transcript cache: {e}")

    is_local = language.lower() != "hinglish" and (
        USE_LOCAL_WHISPER or not GROQ_API_KEY
    )
    max_workers = 1 if is_local else 6
    results = [None] * len(chunks)
    completed = 0
    chunk_sec = 25.0 if language.lower() == "hinglish" else 600.0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(transcribe_chunk, chunk, language, index * chunk_sec): index
            for index, chunk in enumerate(chunks)
        }
        for future in as_completed(futures):
            index = futures[future]
            try:
                results[index] = future.result()
            except Exception as e:
                print(f"Error transcribing chunk {index}: {e}")
                results[index] = {"text": "", "segments": []}
            completed += 1
            if progress_callback:
                progress_callback(completed, len(chunks))

    # Aggregate plain text and detailed segments
    text_parts = [r["text"] for r in results if r and r.get("text")]
    full_text = " ".join(text_parts).strip()

    all_segments = []
    for r in results:
        if r and r.get("segments"):
            all_segments.extend(r["segments"])

    # Build clean timestamped transcript string
    timestamped_lines = [
        f"[{s['timestamp']}] {s['text']}"
        for s in all_segments
    ]
    timestamped_text = "\n".join(timestamped_lines) if timestamped_lines else full_text

    # 2. Persist transcript to cache
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({
                "text": full_text,
                "segments": all_segments,
                "timestamped_text": timestamped_text,
            }, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Warning: Failed to save transcript cache: {e}")

    return TranscriptResult(
        text=full_text,
        segments=all_segments,
        timestamped_text=timestamped_text,
    )


def extract_youtube_id(url: str) -> str:
    """Extracts YouTube 11-char video ID from various URL formats."""
    import re
    s = str(url).strip()
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
    return ""


def get_youtube_transcript(url: str) -> TranscriptResult | None:
    """
    Attempts to fetch timestamped transcript directly via YouTube's public subtitle API.
    Bypasses audio downloading, cloud IP restrictions, bot verification walls, and ffmpeg chunking.
    """
    vid = extract_youtube_id(url)
    if not vid:
        return None
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        api = YouTubeTranscriptApi()
        try:
            raw_segments = api.fetch(vid)
        except AttributeError:
            raw_segments = YouTubeTranscriptApi.get_transcript(vid)

        segments = []
        ts_lines = []
        plain_parts = []
        for item in raw_segments:
            text = getattr(item, 'text', None) if not isinstance(item, dict) else item.get('text', '')
            start = getattr(item, 'start', 0.0) if not isinstance(item, dict) else item.get('start', 0.0)
            dur = getattr(item, 'duration', 0.0) if not isinstance(item, dict) else item.get('duration', 0.0)
            if not text:
                continue
            text = text.replace('\n', ' ').strip()
            ts_str = format_timestamp(start)
            segments.append({
                "start": start,
                "end": start + dur,
                "timestamp": ts_str,
                "text": text,
            })
            ts_lines.append(f"[{ts_str}] {text}")
            plain_parts.append(text)

        if plain_parts:
            full_text = " ".join(plain_parts)
            timestamped_text = "\n".join(ts_lines)
            print(f"⚡ Successfully retrieved direct YouTube transcript ({len(segments)} segments) for '{vid}'")
            return TranscriptResult(
                text=full_text,
                segments=segments,
                timestamped_text=timestamped_text,
            )
    except Exception as e:
        print(f"Direct YouTube transcript fetch unavailable for '{vid}': {e}")
        return None
    return None
