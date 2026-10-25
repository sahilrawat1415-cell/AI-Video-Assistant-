import os
import glob
import subprocess
import yt_dlp

DOWNLOAD_DIR = "downloades"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


import re


def _extract_yt_id(url: str) -> str:
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


def _get_cookie_file() -> str | None:
    """Detects cookie file from disk or environment variables."""
    for candidate in ["cookies.txt", os.path.join(DOWNLOAD_DIR, "cookies.txt")]:
        if os.path.exists(candidate) and os.path.getsize(candidate) > 10:
            return candidate

    cookie_env = os.getenv("YOUTUBE_COOKIES") or os.getenv("COOKIES_TXT")
    if not cookie_env:
        try:
            import streamlit as st
            if "YOUTUBE_COOKIES" in st.secrets:
                cookie_env = st.secrets["YOUTUBE_COOKIES"]
            elif "COOKIES_TXT" in st.secrets:
                cookie_env = st.secrets["COOKIES_TXT"]
            elif "youtube" in st.secrets and isinstance(st.secrets["youtube"], dict) and "cookies_text" in st.secrets["youtube"]:
                cookie_env = st.secrets["youtube"]["cookies_text"]
        except Exception:
            pass

    if cookie_env:
        try:
            target = "cookies.txt"
            with open(target, "w", encoding="utf-8") as f:
                f.write(cookie_env)
            return target
        except Exception:
            pass
    return None


def download_youtube_audio(url: str) -> str:
    """Fast audio download: checks cache first, and strictly downloads audio-only streams to avoid 50MB+ video overhead."""
    vid = _extract_yt_id(url)
    if vid:
        existing = glob.glob(os.path.join(DOWNLOAD_DIR, f"{vid}.*"))
        for f in existing:
            if os.path.isfile(f) and os.path.getsize(f) > 50000 and not f.endswith(".part"):
                print(f"⚡ Cache hit: using already downloaded audio for video ID '{vid}': {f}")
                return f

    base_opts = {
        # Prioritize lightweight audio-only streams (3-5MB) instead of heavy video
        "format": "ba[ext=m4a]/ba[ext=webm]/ba/b[height<=360]/b",
        "outtmpl": os.path.join(DOWNLOAD_DIR, "%(id)s.%(ext)s"),
        "geo_bypass": True,
        "nocheckcertificate": True,
        "http_headers": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        },
        "quiet": True,
        "no_warnings": True,
    }

    cookie_file = _get_cookie_file()
    if cookie_file:
        base_opts["cookiefile"] = cookie_file

    # Attempt 1: Standard extraction (uses default tv_samsung / web clients without GVS PO token requirements)
    try:
        with yt_dlp.YoutubeDL(base_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            downloaded_file = ydl.prepare_filename(info)

            if not os.path.exists(downloaded_file):
                video_id = info.get("id")
                matches = glob.glob(os.path.join(DOWNLOAD_DIR, f"{video_id}.*"))
                if matches:
                    downloaded_file = matches[0]

            if os.path.exists(downloaded_file) and os.path.getsize(downloaded_file) > 1000:
                return downloaded_file
    except Exception as err1:
        print(f"Standard download attempt failed ({err1}). Retrying with fallback options...")
        # Attempt 2: Fallback with tv client or alternate format if 403 or format error occurred
        fallback_opts = dict(base_opts)
        fallback_opts["format"] = "bestaudio/best"
        fallback_opts["extractor_args"] = {
            "youtube": {
                "player_client": ["tv", "web"],
            }
        }
        try:
            with yt_dlp.YoutubeDL(fallback_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                downloaded_file = ydl.prepare_filename(info)

                if not os.path.exists(downloaded_file):
                    video_id = info.get("id")
                    matches = glob.glob(os.path.join(DOWNLOAD_DIR, f"{video_id}.*"))
                    if matches:
                        downloaded_file = matches[0]

                if os.path.exists(downloaded_file) and os.path.getsize(downloaded_file) > 1000:
                    return downloaded_file
        except Exception as err2:
            raise RuntimeError(
                f"YouTube download error ({err2}). "
                "Cloud server IP addresses (like Streamlit Cloud / AWS) are frequently blocked by YouTube (HTTP 403 Forbidden). "
                "Please switch to the 'Upload Media File' option in the sidebar to upload your video/audio file directly!"
            ) from err2

    if vid:
        matches = glob.glob(os.path.join(DOWNLOAD_DIR, f"{vid}.*"))
        if matches and os.path.getsize(matches[0]) > 1000:
            return matches[0]

    raise RuntimeError("Downloaded audio file was not found after extraction.")


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