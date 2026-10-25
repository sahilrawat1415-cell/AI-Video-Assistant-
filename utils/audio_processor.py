import os
import glob
import subprocess
import yt_dlp

DOWNLOAD_DIR = "downloades"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


import re


def _extract_yt_id(url: str) -> str:
    s = str(url).strip()
    if len(s) == 11 and re.match(r"^[a-zA-Z0-9_-]{11}$", s):
        return s
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
    """Detects cookie file from disk, environment variables, or Streamlit secrets."""
    # Check Streamlit secrets or environment variables first
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

    if cookie_env and isinstance(cookie_env, str) and len(cookie_env.strip()) > 10:
        cleaned_cookies = cookie_env.strip()
        if (cleaned_cookies.startswith("'''") and cleaned_cookies.endswith("'''")) or (cleaned_cookies.startswith('"""') and cleaned_cookies.endswith('"""')):
            cleaned_cookies = cleaned_cookies[3:-3].strip()
        target = os.path.abspath("cookies.txt")
        try:
            with open(target, "w", encoding="utf-8") as f:
                f.write(cleaned_cookies + "\n")
            return target
        except Exception:
            pass

    # Check candidates on disk
    candidates = [
        "cookies.txt",
        "www.youtube.com_cookies.txt",
        "youtube.com_cookies.txt",
        os.path.join(DOWNLOAD_DIR, "cookies.txt"),
        os.path.join(DOWNLOAD_DIR, "www.youtube.com_cookies.txt"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate) and os.path.getsize(candidate) > 10:
            return os.path.abspath(candidate)

    return None


def download_youtube_audio(url: str) -> str:
    """Fast audio download: checks cache first, uses modern clients and JS challenge solving to avoid bot blocks."""
    vid = _extract_yt_id(url)
    target_url = url.strip()
    if vid and not target_url.startswith(("http://", "https://")):
        target_url = f"https://www.youtube.com/watch?v={vid}"

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
        "remote_components": ["ejs:github"],
        "socket_timeout": 15,
        "retries": 2,
        "quiet": True,
        "no_warnings": True,
    }

    cookie_file = _get_cookie_file()
    if cookie_file:
        base_opts["cookiefile"] = cookie_file

    # Attempt 1: Standard extraction with automated Deno JS challenge solving
    try:
        with yt_dlp.YoutubeDL(base_opts) as ydl:
            info = ydl.extract_info(target_url, download=True)
            downloaded_file = ydl.prepare_filename(info)

            if not os.path.exists(downloaded_file):
                video_id = info.get("id")
                matches = glob.glob(os.path.join(DOWNLOAD_DIR, f"{video_id}.*"))
                if matches:
                    downloaded_file = matches[0]

            if os.path.exists(downloaded_file) and os.path.getsize(downloaded_file) > 1000:
                return downloaded_file
    except Exception as err1:
        print(f"Standard download attempt failed ({err1}). Retrying with modern fallback clients...")

    # Attempt 2: Fallback with modern mobile/web clients (avoid deprecated 'tv' which triggers 'page needs to be reloaded')
    fallback_opts = dict(base_opts)
    fallback_opts["format"] = "bestaudio/best"
    fallback_opts["extractor_args"] = {
        "youtube": {
            "player_client": ["android", "ios", "mweb", "web_safari"],
        }
    }
    try:
        with yt_dlp.YoutubeDL(fallback_opts) as ydl:
            info = ydl.extract_info(target_url, download=True)
            downloaded_file = ydl.prepare_filename(info)

            if not os.path.exists(downloaded_file):
                video_id = info.get("id")
                matches = glob.glob(os.path.join(DOWNLOAD_DIR, f"{video_id}.*"))
                if matches:
                    downloaded_file = matches[0]

            if os.path.exists(downloaded_file) and os.path.getsize(downloaded_file) > 1000:
                return downloaded_file
    except Exception as err2:
        print(f"Modern fallback clients attempt failed ({err2}). Trying clean Android client...")

    # Attempt 3: Direct Android client extraction without cookies (bypasses datacenter 403 blocks & PO Token requirements)
    android_opts = {
        "format": "ba[ext=m4a]/ba[ext=webm]/ba/b[height<=360]/b",
        "outtmpl": os.path.join(DOWNLOAD_DIR, "%(id)s.%(ext)s"),
        "geo_bypass": True,
        "nocheckcertificate": True,
        "socket_timeout": 15,
        "retries": 2,
        "quiet": True,
        "no_warnings": True,
        "extractor_args": {
            "youtube": {
                "player_client": ["android"]
            }
        }
    }
    try:
        with yt_dlp.YoutubeDL(android_opts) as ydl:
            info = ydl.extract_info(target_url, download=True)
            downloaded_file = ydl.prepare_filename(info)

            if not os.path.exists(downloaded_file):
                video_id = info.get("id")
                matches = glob.glob(os.path.join(DOWNLOAD_DIR, f"{video_id}.*"))
                if matches:
                    downloaded_file = matches[0]

            if os.path.exists(downloaded_file) and os.path.getsize(downloaded_file) > 1000:
                print("⚡ Successfully downloaded using clean Android client!")
                return downloaded_file
    except Exception as err3:
        print(f"Android client download attempt failed ({err3}). Trying browser cookies fallback...")

    # Attempt 4: If running locally without cookies.txt, attempt reading cookies from installed browsers
    if not cookie_file:
        for browser in ["chrome", "edge", "firefox", "brave"]:
            try:
                b_opts = dict(fallback_opts)
                b_opts["cookiesfrombrowser"] = (browser,)
                with yt_dlp.YoutubeDL(b_opts) as ydl:
                    info = ydl.extract_info(target_url, download=True)
                    downloaded_file = ydl.prepare_filename(info)

                    if not os.path.exists(downloaded_file):
                        video_id = info.get("id")
                        matches = glob.glob(os.path.join(DOWNLOAD_DIR, f"{video_id}.*"))
                        if matches:
                            downloaded_file = matches[0]

                    if os.path.exists(downloaded_file) and os.path.getsize(downloaded_file) > 1000:
                        print(f"⚡ Successfully downloaded using {browser} cookies!")
                        return downloaded_file
            except Exception:
                continue

    if vid:
        matches = glob.glob(os.path.join(DOWNLOAD_DIR, f"{vid}.*"))
        if matches and os.path.getsize(matches[0]) > 1000:
            return matches[0]

    raise RuntimeError(
        "YouTube blocked media stream download requests (anti-bot challenge / 403 Forbidden). "
        "Please use the 'Upload Media File' option in the sidebar to upload your video/audio directly, "
        "or export cookies via 'Get cookies.txt LOCALLY' extension into cookies.txt!"
    )


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
    s = str(source).strip()
    is_yt = (
        s.startswith(("http://", "https://"))
        or "youtube.com" in s
        or "youtu.be" in s
        or bool(_extract_yt_id(s))
    )
    if is_yt and not os.path.exists(s):
        print(f"Detected YouTube source. Downloading audio from: {s}")
        audio_path = download_youtube_audio(s)
    else:
        print("Detected local file.")
        audio_path = s

    # 25s pieces for Sarvam AI sync limit; 600s (10-min) chunks for Whisper
    chunk_sec = 25 if language.lower() == "hinglish" else 600
    print(f"Chunking audio into {chunk_sec}s segments...")
    chunks = chunk_audio(audio_path, chunk_seconds=chunk_sec)
    print(f"Audio ready — {len(chunks)} chunk(s) created.")
    return chunks