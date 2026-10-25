import os
import glob
import subprocess
import yt_dlp

DOWNLOAD_DIR = "downloades"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


import re


def _extract_yt_id(url: str) -> str:
    if "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0].split("&")[0]
    if "youtube.com/watch" in url:
        m = re.search(r"[?&]v=([^&]+)", url)
        if m:
            return m.group(1)
    return ""


def download_youtube_audio(url: str) -> str:
    """Fast audio download: checks cache first, and strictly downloads audio-only streams to avoid 50MB+ video overhead."""
    vid = _extract_yt_id(url)
    if vid:
        existing = glob.glob(os.path.join(DOWNLOAD_DIR, f"{vid}.*"))
        for f in existing:
            if os.path.isfile(f) and os.path.getsize(f) > 50000 and not f.endswith(".part"):
                print(f"⚡ Cache hit: using already downloaded audio for video ID '{vid}': {f}")
                return f

    ydl_opts = {
        # Prioritize lightweight audio-only streams (3-5MB) instead of high-res video (50-100MB)
        "format": "ba[ext=m4a]/ba[ext=webm]/ba/b[height<=360]/b",
        "outtmpl": os.path.join(DOWNLOAD_DIR, "%(id)s.%(ext)s"),
        "extractor_args": {
            "youtube": {
                "player_client": ["android", "ios", "web"],
            }
        },
        "quiet": True,
        "no_warnings": True,
    }

    if os.path.exists("cookies.txt"):
        ydl_opts["cookiefile"] = "cookies.txt"

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        downloaded_file = ydl.prepare_filename(info)

        if not os.path.exists(downloaded_file):
            video_id = info.get("id")
            matches = glob.glob(os.path.join(DOWNLOAD_DIR, f"{video_id}.*"))
            if matches:
                downloaded_file = matches[0]

        return downloaded_file


def chunk_audio(input_path: str, chunk_seconds: int = 600) -> list[str]:
    """Fast FFmpeg stream segmentation directly into lightweight 16kHz mono MP3 chunks, with chunk caching."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Audio file not found: {input_path}")

    base_name = os.path.splitext(os.path.basename(input_path))[0]
    out_dir = os.path.join(DOWNLOAD_DIR, f"{base_name}_chunks")
    os.makedirs(out_dir, exist_ok=True)

    # If chunks already exist and are valid, reuse them
    existing_chunks = sorted(glob.glob(os.path.join(out_dir, "chunk_*.mp3")))
    if existing_chunks and all(os.path.getsize(c) > 1000 for c in existing_chunks):
        print(f"⚡ Cache hit: reusing {len(existing_chunks)} existing audio chunks from {out_dir}")
        return existing_chunks

    # Clean stale chunks if any
    for old in existing_chunks:
        try:
            os.remove(old)
        except Exception:
            pass

    pattern = os.path.join(out_dir, "chunk_%03d.mp3")
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-vn", "-ac", "1", "-ar", "16000", "-b:a", "64k",
        "-f", "segment", "-segment_time", str(chunk_seconds),
        pattern,
    ]

    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
    except FileNotFoundError:
        raise RuntimeError("FFmpeg is not installed or not in your system PATH.")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg segmentation failed: {e.stderr.decode()}")

    return sorted(glob.glob(os.path.join(out_dir, "chunk_*.mp3")))


def process_input(source: str, language: str = "english") -> list[str]:
    if source.startswith(("http://", "https://")):
        print(f"Detected YouTube URL. Downloading audio from: {source}")
        audio_path = download_youtube_audio(source)
    else:
        print("Detected local file.")
        audio_path = source

    # 25s pieces for Sarvam AI sync limit; 600s (10-min) chunks for Whisper
    chunk_sec = 25 if language.lower() == "hinglish" else 600
    print(f"Chunking audio into {chunk_sec}s segments...")
    chunks = chunk_audio(audio_path, chunk_seconds=chunk_sec)
    print(f"Audio ready — {len(chunks)} chunk(s) created.")
    return chunks