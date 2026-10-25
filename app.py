import os
import re
import json
import html
import time
from pathlib import Path
from dotenv import load_dotenv

# ── Load .env BEFORE importing project modules ──────────────────────────────────
ENV_FILE = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_FILE, override=True)

import streamlit as st
import requests
from concurrent.futures import ThreadPoolExecutor

from utils.audio_processor import process_input, DOWNLOAD_DIR
from core.transcriber import transcribe_all, TranscriptResult
from core.summarize import summarize, generate_title
import importlib
import core.extractor
importlib.reload(core.extractor)
from core.extractor import extract_action_items, extract_key_decisions, extract_questions, extract_all_insights

try:
    from core.rag_engine import build_rag_chain, load_rag_chain, ask_question
    from core.vector_store import list_indexed_videos
except ImportError:
    import importlib
    import core.vector_store
    import core.rag_engine
    importlib.reload(core.vector_store)
    importlib.reload(core.rag_engine)
    from core.rag_engine import build_rag_chain, load_rag_chain, ask_question
    from core.vector_store import list_indexed_videos

API_SERVER_URL = os.getenv("API_SERVER_URL", "http://localhost:8000")


def check_api_server(url: str = API_SERVER_URL) -> bool:
    try:
        r = requests.get(url, timeout=0.8)
        return r.status_code == 200
    except Exception:
        return False


# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Video Assistant · RAG Intelligence",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS (Warm Beige #EDECE6 · Cobalt Blue #1546D9 · Sun Yellow #F9B318) ─
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200&display=swap');

:root {
    --bg: #EDECE6;              /* Warm Beige base */
    --surface: #FFFFFF;         /* Crisp Warm White */
    --surface-soft: #F7F5EE;    /* Soft Beige Card */
    --surface-sidebar: #F5F3EC; /* Sidebar Warm Base */
    --border: #DCD8CA;          /* Structural crisp warm border */
    --border-dark: #C9C4B3;     /* Darker separator */
    --accent: #1546D9;          /* Cobalt Blue */
    --accent-hover: #0E35AA;    /* Deep Cobalt */
    --accent-light: rgba(21, 70, 217, 0.08);
    --accent-2: #F9B318;        /* Sun Yellow */
    --accent-2-hover: #E39F0B;  /* Deep Sun Yellow */
    --accent-2-light: rgba(249, 179, 24, 0.18);
    --text: #121624;            /* Deep Ink Charcoal */
    --text-muted: #585E6E;      /* Slate Taupe */
}

/* Global Reset & Typography - do NOT override icon ligature fonts */
html, body, .stApp {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    color: var(--text);
}

/* Ensure Streamlit Material Symbols render as proper icons, not text ligatures */
[data-testid="stIconMaterial"],
.material-symbols-rounded,
.material-symbols-outlined,
[class*="material-symbols"] {
    font-family: 'Material Symbols Rounded', 'Material Symbols Outlined', 'Material Icons' !important;
    font-weight: normal !important;
    font-style: normal !important;
    letter-spacing: normal !important;
    text-transform: none !important;
    display: inline-block !important;
    white-space: nowrap !important;
    word-wrap: normal !important;
    direction: ltr !important;
    -webkit-font-smoothing: antialiased !important;
}

header[data-testid="stHeader"] {
    background: transparent !important;
}

/* Hide Streamlit top-right deploy button and unnecessary header clutter */
.stDeployButton {
    display: none !important;
}

/* Direct Ingestion Card Container */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #FFFFFF !important;
    border: 1.5px solid var(--border) !important;
    border-top: 4px solid var(--accent) !important;
    border-radius: 14px !important;
    padding: 1.3rem 1.6rem 1.1rem 1.6rem !important;
    box-shadow: 0 8px 24px rgba(21, 70, 217, 0.05) !important;
    margin: 1.2rem 0 !important;
}
div[data-testid="stVerticalBlockBorderWrapper"] > div {
    background: transparent !important;
    border: none !important;
}

/* Subtle Geometric Grid Background */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background-image:
        linear-gradient(rgba(21, 70, 217, 0.02) 1px, transparent 1px),
        linear-gradient(90deg, rgba(21, 70, 217, 0.02) 1px, transparent 1px);
    background-size: 36px 36px;
    pointer-events: none;
    z-index: 0;
}

/* ── Sidebar Styling ── */
[data-testid="stSidebar"] {
    background-color: var(--surface-sidebar) !important;
    border-right: 1.5px solid var(--border) !important;
}

[data-testid="stSidebar"] > div:first-child {
    padding: 1.5rem 1.25rem !important;
}

/* ── Headings ── */
h1, h2, h3, h4, .brand-title {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    font-weight: 800 !important;
    color: var(--text) !important;
    letter-spacing: -0.025em;
}

.hero-pill {
    background: var(--accent-2);
    color: var(--text);
    padding: 0.1rem 0.55rem;
    border-radius: 6px;
    display: inline-block;
    box-shadow: 0 2px 8px rgba(249, 179, 24, 0.3);
}

.hero-badge {
    background: var(--accent);
    color: #FFFFFF !important;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    font-weight: 700;
    padding: 0.28rem 0.75rem;
    border-radius: 999px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
}

.tag-yellow {
    background: var(--accent-2);
    color: var(--text);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    font-weight: 800;
    padding: 0.24rem 0.6rem;
    border-radius: 6px;
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
}

/* ── Clean Card Containers ── */
.glass-card {
    background: var(--surface);
    border: 1.5px solid var(--border);
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.25rem;
    box-shadow: 0 4px 18px rgba(18, 22, 36, 0.03);
    position: relative;
    overflow: hidden;
    transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
}

.glass-card:hover {
    border-color: var(--accent);
    box-shadow: 0 8px 24px rgba(21, 70, 217, 0.08);
}

.card-top-stripe::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 4px;
    background: linear-gradient(90deg, var(--accent) 0%, var(--accent-2) 100%);
}

.card-cobalt-stripe::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 5px; height: 100%;
    background: linear-gradient(180deg, var(--accent) 0%, var(--accent-2) 100%);
}

.card-header-label {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 0.72rem;
    font-weight: 800;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.45rem;
}

.card-text {
    font-size: 0.94rem;
    line-height: 1.85;
    color: var(--text);
    white-space: pre-line;
}

/* ── Structured Insights List Items ── */
.insight-item-row {
    display: flex;
    align-items: flex-start;
    gap: 0.85rem;
    padding: 0.75rem 0.5rem;
    border-bottom: 1px solid var(--border);
    transition: background 0.15s ease;
    border-radius: 8px;
}
.insight-item-row:last-child {
    border-bottom: none;
}
.insight-item-row:hover {
    background: var(--surface-soft);
}
.insight-num-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-width: 25px;
    height: 25px;
    border-radius: 6px;
    color: #FFFFFF;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.74rem;
    font-weight: 700;
    flex-shrink: 0;
    margin-top: 0.12rem;
}
.insight-item-text {
    font-size: 0.93rem;
    line-height: 1.65;
    color: var(--text);
    word-break: break-word;
    flex: 1;
}

/* ── Modern Tabs Navigation ── */
div[data-baseweb="tab-list"] {
    gap: 0.6rem;
    background: transparent !important;
    border-bottom: 2px solid var(--border) !important;
    padding-bottom: 0.25rem;
    margin-bottom: 1.5rem;
}

div[data-baseweb="tab-border"] {
    display: none !important;
}

button[data-baseweb="tab"] {
    background: #E8E5DA !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 10px 10px 0 0 !important;
    padding: 0.65rem 1.35rem !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.88rem !important;
    color: var(--text-muted) !important;
    transition: all 0.2s ease !important;
}

button[data-baseweb="tab"]:hover {
    color: var(--accent) !important;
    background: #FFFFFF !important;
    border-color: var(--accent) !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    background: var(--surface) !important;
    color: var(--accent) !important;
    border-color: var(--accent) !important;
    border-bottom: 3.5px solid var(--accent-2) !important;
    box-shadow: 0 -4px 14px rgba(21, 70, 217, 0.08);
}

/* ── Stat Pill Box ── */
.stat-pill {
    background: #FFFFFF;
    border: 1.5px solid var(--border);
    border-radius: 12px;
    padding: 0.95rem 1.25rem;
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
    box-shadow: 0 2px 10px rgba(18, 22, 36, 0.03);
    transition: transform 0.15s ease, border-color 0.15s ease;
}

.stat-pill:hover {
    border-color: var(--accent);
    transform: translateY(-2px);
}

.stat-pill-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--text-muted);
}

.stat-pill-value {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 1.35rem;
    font-weight: 800;
    color: var(--accent);
    letter-spacing: -0.02em;
}

/* ── Custom Buttons (Cobalt Blue Primary · Warm White Secondary) ── */
.stButton > button {
    border-radius: 10px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.01em !important;
    padding: 0.65rem 1.4rem !important;
    transition: all 0.2s ease !important;
}

/* Primary Button Styling */
.stButton > button[data-testid="baseButton-primary"],
.stButton > button:not([data-testid="baseButton-secondary"]) {
    background: var(--accent) !important;
    color: #FFFFFF !important;
    border: 1.5px solid var(--accent) !important;
    box-shadow: 0 4px 14px rgba(21, 70, 217, 0.25) !important;
}

.stButton > button[data-testid="baseButton-primary"] *,
.stButton > button:not([data-testid="baseButton-secondary"]) * {
    color: #FFFFFF !important;
}

.stButton > button[data-testid="baseButton-primary"]:hover,
.stButton > button:not([data-testid="baseButton-secondary"]):hover {
    background: var(--accent-hover) !important;
    border-color: var(--accent-2) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 22px rgba(21, 70, 217, 0.35) !important;
}

/* Secondary Button Styling */
.stButton > button[data-testid="baseButton-secondary"],
.stButton > button[kind="secondary"] {
    background: #FFFFFF !important;
    border: 1.5px solid var(--border) !important;
    color: var(--text) !important;
    box-shadow: 0 2px 6px rgba(18, 22, 36, 0.03) !important;
}

.stButton > button[data-testid="baseButton-secondary"] *,
.stButton > button[kind="secondary"] * {
    color: var(--text) !important;
}

.stButton > button[data-testid="baseButton-secondary"]:hover,
.stButton > button[kind="secondary"]:hover {
    border-color: var(--accent) !important;
    background: #FAF8F2 !important;
    color: var(--accent) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(21, 70, 217, 0.12) !important;
}

.stButton > button[data-testid="baseButton-secondary"]:hover *,
.stButton > button[kind="secondary"]:hover * {
    color: var(--accent) !important;
}

/* ── Form Submit Button (RAG Ask) ── */
.stFormSubmitButton > button {
    background: linear-gradient(135deg, var(--accent) 0%, #0E35AA 100%) !important;
    color: #FFFFFF !important;
    border: 1.5px solid var(--accent) !important;
    border-radius: 10px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.02em !important;
}

.stFormSubmitButton > button * {
    color: #FFFFFF !important;
}

.stFormSubmitButton > button:hover {
    border-color: var(--accent-2) !important;
    box-shadow: 0 6px 20px rgba(21, 70, 217, 0.35) !important;
    transform: translateY(-1px) !important;
}

/* ── Input Fields & Selects ── */
.stTextInput > div > div > input,
.stSelectbox > div > div {
    background: #FFFFFF !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 10px !important;
    font-size: 0.92rem !important;
    color: var(--text) !important;
    padding: 0.65rem 0.95rem !important;
}

.stTextInput > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(21, 70, 217, 0.15) !important;
}

/* ── Transcript Box with Glowing Timestamps ── */
.transcript-scroll {
    background: #FFFFFF;
    border: 1.5px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.84rem;
    line-height: 1.95;
    max-height: 520px;
    overflow-y: auto;
    color: var(--text);
    white-space: pre-wrap;
    word-break: break-word;
}

.ts-tag {
    display: inline-block;
    background: var(--accent-light);
    color: var(--accent);
    border: 1px solid rgba(21, 70, 217, 0.25);
    border-radius: 5px;
    padding: 0.1rem 0.45rem;
    font-weight: 700;
    margin-right: 0.45rem;
}

/* ── RAG Chat Interface ── */
.chat-wrapper {
    background: var(--surface-soft);
    border: 1.5px solid var(--border);
    border-radius: 14px;
    padding: 1.5rem;
    max-height: 480px;
    overflow-y: auto;
    margin-bottom: 1.25rem;
}

.chat-bubble-user {
    background: var(--accent);
    color: #FFFFFF !important;
    border-radius: 14px 14px 3px 14px;
    padding: 0.9rem 1.2rem;
    font-size: 0.92rem;
    line-height: 1.65;
    max-width: 82%;
    align-self: flex-end;
    box-shadow: 0 4px 12px rgba(21, 70, 217, 0.18);
}

.chat-bubble-assistant {
    background: #FFFFFF;
    border: 1.5px solid var(--border);
    border-left: 4.5px solid var(--accent-2);
    color: var(--text) !important;
    border-radius: 14px 14px 14px 3px;
    padding: 1rem 1.3rem;
    font-size: 0.92rem;
    line-height: 1.75;
    max-width: 88%;
    align-self: flex-start;
    box-shadow: 0 4px 14px rgba(18, 22, 36, 0.04);
}

.chat-avatar-user {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    font-weight: 700;
    color: var(--accent);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.25rem;
    align-self: flex-end;
}

.chat-avatar-bot {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    font-weight: 700;
    color: #A36F00;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.25rem;
    align-self: flex-start;
}

/* ── Sidebar Steps Indicator ── */
.status-step-row {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.6rem 0.85rem;
    background: #FFFFFF;
    border-radius: 8px;
    margin: 0.35rem 0;
    border: 1px solid var(--border);
    font-size: 0.76rem;
    font-weight: 600;
}

.dot-pulse {
    width: 9px; height: 9px;
    border-radius: 50%;
    flex-shrink: 0;
}
.dot-active { background: var(--accent-2); box-shadow: 0 0 0 4px rgba(249, 179, 24, 0.4); animation: pulseAnim 1.2s infinite; }
.dot-done   { background: var(--accent); }
.dot-idle   { background: #D5D1C2; }

@keyframes pulseAnim {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%      { opacity: 0.4; transform: scale(0.9); }
}

/* ── Progress Bar Customization ── */
div[data-testid="stProgress"] > div > div > div {
    background: linear-gradient(90deg, var(--accent) 0%, var(--accent-2) 100%) !important;
}

/* Scrollbars */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border-dark); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent); }
</style>
""", unsafe_allow_html=True)


# ─── Session State Initialization ───────────────────────────────────────────────
for key, default in {
    "result": None,
    "rag_chain": None,
    "pipeline_done": False,
    "pipeline_steps": {},
    "elapsed_time": None,
    "source_input_url": "",
    "front_source_input": "",
    "speech_lang_key": "english",
    "front_lang_key": "english",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


def sync_sidebar_url():
    st.session_state["front_source_input"] = st.session_state.get("source_input_url", "")


def sync_front_url():
    st.session_state["source_input_url"] = st.session_state.get("front_source_input", "")


def sync_sidebar_lang():
    st.session_state["front_lang_key"] = st.session_state.get("speech_lang_key", "english")


def sync_front_lang():
    st.session_state["speech_lang_key"] = st.session_state.get("front_lang_key", "english")


def set_demo_url(url: str, lang: str = "english"):
    st.session_state["source_input_url"] = url
    st.session_state["front_source_input"] = url
    st.session_state["speech_lang_key"] = lang
    st.session_state["front_lang_key"] = lang


PIPELINE_STEPS_LIST = [
    ("audio",      "FFmpeg Audio Chunking"),
    ("transcript", "Whisper / Sarvam STT"),
    ("title",      "Title Generation"),
    ("summary",    "Map-Reduce Summary"),
    ("extract",    "Action & Decision Extraction"),
    ("rag",        "Chroma Vector Store Index"),
]


def step_status(steps: dict, key: str) -> str:
    s = steps.get(key, "pending")
    if s == "active": return "dot-active"
    if s == "done":   return "dot-done"
    return "dot-idle"


def render_sidebar_steps(placeholder):
    with placeholder.container():
        st.markdown(
            '<div style="font-family:\'Plus Jakarta Sans\',sans-serif;font-size:0.75rem;font-weight:800;'
            'color:var(--accent);letter-spacing:0.08em;text-transform:uppercase;margin-bottom:0.5rem">'
            'Pipeline Status</div>',
            unsafe_allow_html=True,
        )
        for step, label in PIPELINE_STEPS_LIST:
            css = step_status(st.session_state.pipeline_steps, step)
            st.markdown(f"""
            <div class="status-step-row">
                <div class="dot-pulse {css}"></div>
                <span>{label}</span>
            </div>""", unsafe_allow_html=True)
        if st.session_state.elapsed_time:
            st.markdown(
                f'<div style="font-size:0.78rem;color:var(--accent);font-weight:800;margin-top:0.5rem;'
                f'background:#FFFFFF;padding:0.45rem 0.75rem;border-radius:6px;border:1px solid var(--border)">'
                f'⚡ Completed in {st.session_state.elapsed_time:.1f}s</div>',
                unsafe_allow_html=True,
            )


def format_clean_summary_html(summary_text: str) -> str:
    """Formats executive summary text into clean, modern HTML without raw ** asterisks."""
    if not summary_text or not summary_text.strip():
        return "<p style='color:var(--text-muted)'>No summary available.</p>"

    # 1. Strip raw markdown bold asterisks and convert to bold styling
    cleaned = re.sub(r"\*\*(.*?)\*\*", r'<strong style="color:var(--text);font-weight:700">\1</strong>', summary_text.strip())
    cleaned = cleaned.replace("**", "")

    lines = cleaned.split("\n")
    formatted_lines = []
    for line in lines:
        line_str = line.strip()
        if not line_str:
            formatted_lines.append("<div style='height:0.4rem'></div>")
            continue

        # Bullet point line
        if line_str.startswith(("- ", "* ", "• ")):
            bullet_body = line_str[2:].strip()
            # Replace timestamp tags
            bullet_body = re.sub(r'\[(\d{1,2}:\d{2}(?:-\d{1,2}:\d{2})?)\]', r'<span class="ts-tag">\1</span>', bullet_body)
            formatted_lines.append(
                f'<div style="display:flex;align-items:flex-start;gap:0.6rem;margin:0.3rem 0">'
                f'<span style="color:var(--accent);font-weight:800;font-size:1.1rem;line-height:1.2">•</span>'
                f'<div style="flex:1;font-size:0.93rem;line-height:1.65;color:var(--text)">{bullet_body}</div>'
                f'</div>'
            )
        # Standalone header or bold section lead
        elif line_str.startswith("<strong") and line_str.endswith("</strong>"):
            formatted_lines.append(
                f'<div style="font-family:\'Plus Jakarta Sans\',sans-serif;font-size:1.05rem;font-weight:800;'
                f'color:var(--accent);margin-top:0.85rem;margin-bottom:0.3rem;letter-spacing:-0.01em">{line_str}</div>'
            )
        else:
            par = re.sub(r'\[(\d{1,2}:\d{2}(?:-\d{1,2}:\d{2})?)\]', r'<span class="ts-tag">\1</span>', line_str)
            formatted_lines.append(f'<div style="font-size:0.93rem;line-height:1.65;color:var(--text);margin:0.25rem 0">{par}</div>')

    return "".join(formatted_lines)


def format_chat_bubble_html(content: str) -> str:
    """Formats assistant chat response into clean, styled HTML without raw ** asterisks."""
    if not content:
        return ""

    # Escape HTML to prevent injection
    text = html.escape(content)

    # Convert **bold** into bold strong tags
    text = re.sub(r"\*\*(.*?)\*\*", r'<strong style="color:var(--text);font-weight:700">\1</strong>', text)
    # Strip any stray or unmatched **
    text = text.replace("**", "")

    # Clean bullet stars: "* " -> "• "
    text = re.sub(r"^(\s*)\*\s+", r"\1<span style='color:var(--accent);font-weight:700'>• </span>", text, flags=re.MULTILINE)

    # Format timestamp tags like [04:34] or [05:15-05:24]
    text = re.sub(
        r'\[(\d{1,2}:\d{2}(?:-\d{1,2}:\d{2})?)\]',
        r'<span class="ts-tag">\1</span>',
        text,
    )

    # Convert newlines to <br>
    return text.replace("\n", "<br>")


def render_structured_list_html(raw_text: str, badge_color: str = "var(--accent)", max_items: int = 7) -> str:
    """Formats raw text into structured rows with numbers aligned in colored badges, 
    stripping asterisks, dropping fragments, and capping strictly at max_items (5-7)."""
    if not raw_text or not raw_text.strip():
        return "<p style='color:var(--text-muted);font-size:0.9rem;padding:0.5rem 0'>No items extracted.</p>"

    # 1. Clean up asterisks **
    cleaned = re.sub(r"\*\*(.*?)\*\*", r"\1", raw_text.strip())
    cleaned = cleaned.replace("**", "")

    # 2. Clean up disoriented newlines like "2.\n  The text..." into "2. The text..."
    cleaned = re.sub(r'(\d+)[\.\)]\s*\n+\s*', r'\1. ', cleaned)

    # 3. Extract non-empty lines
    lines = [line.strip() for line in cleaned.split("\n") if line.strip()]

    valid_contents = []
    for line in lines:
        m = re.match(r'^(\d+)[\.\)]\s*(.*)', line)
        if m:
            content = m.group(2).strip()
        elif line.startswith(("- ", "• ", "* ")):
            content = line[2:].strip()
        else:
            content = line.strip()

        # Strip accidental duplicate numbering at the start of content
        content = re.sub(r'^(\d+)[\.\)]\s*', '', content).strip()
        content = content.replace("**", "").replace("*", "")

        # Discard truncated fragments (like item "10. 1" or single digits)
        if len(content) < 4 or content.isdigit():
            continue

        valid_contents.append(content)
        if len(valid_contents) >= max_items:
            break

    if not valid_contents:
        return "<p style='color:var(--text-muted);font-size:0.9rem;padding:0.5rem 0'>No items extracted.</p>"

    html_items = []
    for idx, content in enumerate(valid_contents, start=1):
        # Escape HTML and format timestamp tags like [01:45]
        escaped_content = html.escape(content)
        escaped_content = re.sub(
            r'\[(\d{1,2}:\d{2}(?:-\d{1,2}:\d{2})?)\]',
            r'<span class="ts-tag">\1</span>',
            escaped_content,
        )

        html_items.append(
            f'<div class="insight-item-row">'
            f'<span class="insight-num-badge" style="background:{badge_color}">{idx}</span>'
            f'<div class="insight-item-text">{escaped_content}</div>'
            f'</div>'
        )

    return "".join(html_items)


def handle_rag_question(question_text: str):
    """Executes a RAG question using either the FastAPI backend or local rag_engine."""
    if st.session_state.get("use_api_backend", False):
        try:
            r = requests.post(
                f"{API_SERVER_URL}/api/chat",
                json={"question": question_text},
                timeout=90,
            )
            if r.status_code == 200:
                answer = r.json().get("answer", "")
                st.session_state.chat_history.append({"role": "user", "content": question_text})
                st.session_state.chat_history.append({"role": "assistant", "content": str(answer)})
                return
            else:
                st.error(f"FastAPI Server Error ({r.status_code}): {r.text}")
                return
        except Exception as e:
            st.warning(f"FastAPI request failed ({e}). Falling back to local RAG engine...")

    vid_id = st.session_state.result.get("video_id") if st.session_state.result else None
    try:
        active_chain = load_rag_chain(video_id=vid_id)
        if st.session_state.result:
            st.session_state.result["rag_chain"] = active_chain
        st.session_state.rag_chain = active_chain
    except Exception:
        active_chain = (
            st.session_state.result["rag_chain"]
            if st.session_state.result and "rag_chain" in st.session_state.result
            else st.session_state.rag_chain
        )

    if not active_chain:
        st.error("RAG Chain is not initialized yet. Process a video or load one from the library.")
        return

    answer = ask_question(active_chain, question_text)
    st.session_state.chat_history.append({"role": "user", "content": question_text})
    st.session_state.chat_history.append({"role": "assistant", "content": str(answer)})


# ─── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="margin-bottom:1.2rem">
        <div style="display:flex;align-items:center;gap:0.45rem;margin-bottom:0.35rem">
            <span class="hero-badge">AI Assistant</span>
            <span class="tag-yellow">v2.0 RAG</span>
        </div>
        <div style="font-family:'Plus Jakarta Sans',sans-serif;font-weight:800;font-size:1.45rem;color:var(--text);letter-spacing:-0.03em;line-height:1.2">
            Video <span style="color:var(--accent)">Intelligence</span>
        </div>
        <div style="font-size:0.75rem;color:var(--text-muted);font-weight:500;margin-top:0.25rem">
            Whisper · Groq · ChromaDB
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="card-header-label">01 · Input Source</div>', unsafe_allow_html=True)
    input_mode = st.radio(
        "Choose Input Method",
        ["YouTube / Local Path", "Upload Media File"],
        label_visibility="collapsed",
    )

    source_path = ""
    if input_mode == "YouTube / Local Path":
        source_path = st.text_input(
            "YouTube URL or File Path",
            key="source_input_url",
            placeholder="https://youtube.com/watch?v=... or path/to/video.mp4",
            on_change=sync_sidebar_url,
        )
    else:
        uploaded_file = st.file_uploader(
            "Upload MP4, MP3, WAV, or M4A",
            type=["mp4", "mp3", "wav", "m4a", "webm", "mkv"],
        )
        if uploaded_file is not None:
            os.makedirs(DOWNLOAD_DIR, exist_ok=True)
            saved_path = os.path.join(DOWNLOAD_DIR, uploaded_file.name)
            with open(saved_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            source_path = saved_path
            st.caption(f"📁 Saved: `{uploaded_file.name}`")

    st.markdown('<div class="card-header-label" style="margin-top:1rem">02 · Speech Model</div>', unsafe_allow_html=True)
    language = st.selectbox(
        "Speech Model Language",
        ["english", "hinglish"],
        key="speech_lang_key",
        help="'english' routes to Groq Whisper v3-turbo. 'hinglish' routes to Sarvam AI saaras:v2.5.",
        on_change=sync_sidebar_lang,
        label_visibility="collapsed",
    )

    # Auto-detect if FastAPI is running in the background
    st.session_state.use_api_backend = check_api_server()

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    run_btn = st.button("⚡ Analyse Video", use_container_width=True, type="primary")

    # ── Multi-Video Library (Sessions in ChromaDB) ──
    saved_videos = list_indexed_videos()
    if saved_videos:
        st.markdown("---")
        st.markdown('<div class="card-header-label">03 · Saved Video Library</div>', unsafe_allow_html=True)
        video_titles = [
            f"{v.get('title', 'Video')[:26]}..." if len(v.get('title', '')) > 26 else v.get('title', 'Video')
            for v in saved_videos
        ]
        selected_vid_idx = st.selectbox(
            "Select Session",
            range(len(saved_videos)),
            format_func=lambda i: f"{video_titles[i]}",
            key="library_video_select",
            label_visibility="collapsed",
        )
        if st.button("📂 Load Selected Session", use_container_width=True, type="secondary"):
            chosen = saved_videos[selected_vid_idx]
            vid_id = chosen.get("video_id", "default_session")
            loaded_chain = load_rag_chain(video_id=vid_id)
            st.session_state.result = {
                "title": chosen.get("title", "Indexed Video"),
                "summary": chosen.get("summary") or f"Indexed on {chosen.get('indexed_at', 'Previously')}. Ready for RAG chat queries.",
                "transcript": chosen.get("transcript") or f"Video ID: {vid_id}\nIndexed Chunks: {chosen.get('chunks_indexed', 'N/A')}\nIndexed At: {chosen.get('indexed_at', '')}",
                "plain_transcript": chosen.get("plain_transcript") or chosen.get("transcript") or "",
                "action_items": chosen.get("action_items") or "Ask the RAG assistant below to retrieve specific action items.",
                "key_decisions": chosen.get("key_decisions") or "Ask the RAG assistant below to retrieve key decisions.",
                "open_questions": chosen.get("open_questions") or "Ask anything about this meeting in the RAG chat below.",
                "rag_chain": loaded_chain,
            }
            st.session_state.rag_chain = loaded_chain
            st.session_state.elapsed_time = chosen.get("elapsed_time", 5.2)
            st.session_state.pipeline_done = True
            st.rerun()

    st.markdown("---")
    sidebar_status_box = st.empty()
    if st.session_state.pipeline_steps:
        render_sidebar_steps(sidebar_status_box)


# ─── Main Header ────────────────────────────────────────────────────────────────
header_col1, header_col2 = st.columns([4, 1.2])
with header_col1:
    st.markdown("""
    <div style="margin-bottom:1.5rem">
        <div style="display:flex;align-items:center;gap:0.55rem;margin-bottom:0.4rem;flex-wrap:wrap">
            <span class="hero-badge">Deep Video & Meeting Intelligence</span>
        </div>
        <div style="font-family:'Plus Jakarta Sans',sans-serif;font-weight:800;font-size:clamp(2rem, 3.2vw, 2.75rem);color:var(--text);letter-spacing:-0.03em;line-height:1.15;margin-top:0.25rem">
            AI Video <span style="color:var(--accent)">Assistant</span>
            <span class="tag-yellow" style="font-size:0.75rem;vertical-align:middle;margin-left:0.4rem">v2.0 RAG</span>
        </div>
        <div style="font-size:0.92rem;color:var(--text-muted);font-weight:500;margin-top:0.35rem">
            High-speed audio segmentation, multilingual transcription, executive briefs, and context-aware RAG Q&A.
        </div>
    </div>
    """, unsafe_allow_html=True)

with header_col2:
    if st.session_state.result:
        if st.button("🔄 New Analysis", use_container_width=True, type="secondary"):
            st.session_state.result = None
            st.session_state.chat_history = []
            st.session_state.pipeline_steps = {}
            st.session_state.elapsed_time = None
            st.rerun()


# ─── Pipeline Execution ─────────────────────────────────────────────────────────
def run_analysis_pipeline(target_source: str, target_lang: str):
    if not target_source or not target_source.strip():
        st.error("Please provide a YouTube URL, local file path, or upload a media file.")
        return

    t0 = time.time()
    st.session_state.pipeline_done = False
    st.session_state.result = None
    st.session_state.elapsed_time = None
    st.session_state.pipeline_steps = {k: "pending" for k, _ in PIPELINE_STEPS_LIST}

    banner = st.empty()
    prog_bar = st.progress(0)

    def mark_step(key: str, status: str):
        st.session_state.pipeline_steps[key] = status
        render_sidebar_steps(sidebar_status_box)

    try:
        # Check if FastAPI backend should be used
        if st.session_state.get("use_api_backend", False):
            banner.info("🚀 Processing video via FastAPI server (http://localhost:8000)...")
            mark_step("audio", "active")
            prog_bar.progress(20)

            resp = requests.post(
                f"{API_SERVER_URL}/api/process-video",
                json={"video_url": target_source.strip(), "language": target_lang},
                timeout=600,
            )
            if resp.status_code != 200:
                raise RuntimeError(f"FastAPI error ({resp.status_code}): {resp.text}")

            data = resp.json()
            for step, _ in PIPELINE_STEPS_LIST:
                mark_step(step, "done")
            prog_bar.progress(100)

            st.session_state.elapsed_time = data.get("elapsed_seconds") or (time.time() - t0)
            st.session_state.result = {
                "title":            data.get("title") or "Video Summary",
                "transcript":       data.get("timestamped_transcript") or data.get("transcript") or data.get("transcript_preview") or "",
                "plain_transcript": data.get("transcript") or "",
                "summary":          data.get("summary") or "",
                "action_items":     data.get("action_items") or "",
                "key_decisions":    data.get("key_decisions") or "",
                "open_questions":   data.get("open_questions") or "",
                "rag_chain":        None,
            }
            st.session_state.pipeline_done = True
            render_sidebar_steps(sidebar_status_box)

            banner.success(f"⚡ Completed via FastAPI in {st.session_state.elapsed_time:.1f}s!")
            time.sleep(0.4)
            banner.empty()
            prog_bar.empty()
            st.rerun()

        # ── Direct In-Process Pipeline ──
        # Step 1: Audio Processing & FFmpeg Chunking
        banner.info("Step 1/3 — Downloading & stream-segmenting audio with FFmpeg...")
        mark_step("audio", "active")
        chunks = process_input(target_source.strip(), language=target_lang)
        mark_step("audio", "done")
        prog_bar.progress(25)

        # Step 2: Parallel Transcription
        mark_step("transcript", "active")
        def on_chunk_done(done_count, total_count):
            banner.info(f"Step 2/3 — Transcribing segment {done_count}/{total_count}...")
            prog_bar.progress(25 + int(40 * (done_count / max(total_count, 1))))

        transcript = transcribe_all(chunks, language=target_lang, progress_callback=on_chunk_done)
        mark_step("transcript", "done")
        prog_bar.progress(65)

        # Step 3: High-speed single-pass Insights + Chroma Vector Store in parallel
        banner.info("Step 3/3 — Synthesizing Intelligence Dossier & Indexing Chroma Vectors in parallel...")
        for k in ["title", "summary", "extract", "rag"]:
            mark_step(k, "active")

        target_vid = extract_video_id(target_source.strip())

        with ThreadPoolExecutor(max_workers=2) as executor:
            f_insights = executor.submit(extract_all_insights, transcript)
            f_rag      = executor.submit(build_rag_chain, transcript, video_id=target_vid)

            insights   = f_insights.result()
            title        = insights.get("title", "Meeting Brief")
            summary      = insights.get("summary", "")
            action_items = insights.get("action_items", "")
            decisions    = insights.get("key_decisions", "")
            questions    = insights.get("open_questions", "")

            mark_step("title", "done")
            mark_step("summary", "done")
            mark_step("extract", "done")

            rag_chain    = f_rag.result()
            mark_step("rag", "done")

        prog_bar.progress(100)
        st.session_state.elapsed_time = time.time() - t0
        st.session_state.result = {
            "title":            title,
            "transcript":       getattr(transcript, "timestamped_text", str(transcript)),
            "plain_transcript": str(transcript),
            "summary":          summary,
            "action_items":     action_items,
            "key_decisions":    decisions,
            "open_questions":   questions,
            "rag_chain":        rag_chain,
        }
        st.session_state.rag_chain = rag_chain
        st.session_state.pipeline_done = True
        render_sidebar_steps(sidebar_status_box)

        banner.success(f"⚡ Completed in {st.session_state.elapsed_time:.1f}s!")
        time.sleep(0.4)
        banner.empty()
        prog_bar.empty()
        st.rerun()

    except Exception as e:
        for k in st.session_state.pipeline_steps:
            if st.session_state.pipeline_steps[k] == "active":
                st.session_state.pipeline_steps[k] = "pending"
        render_sidebar_steps(sidebar_status_box)
        prog_bar.empty()
        banner.error(f"Pipeline Error: {e}")


if run_btn:
    run_analysis_pipeline(source_path, language)


# ─── Analysis Results View (Segmented Tabs Layout) ───────────────────────────────
if st.session_state.result:
    r = st.session_state.result

    # ── Top Hero Stats Bar ──
    stat1, stat2, stat3, stat4 = st.columns(4)
    with stat1:
        st.markdown(f"""
        <div class="stat-pill">
            <span class="stat-pill-label">Processing Time</span>
            <span class="stat-pill-value">{st.session_state.elapsed_time:.1f}s</span>
        </div>""", unsafe_allow_html=True)
    with stat2:
        words = len(r.get("plain_transcript", r["transcript"]).split())
        st.markdown(f"""
        <div class="stat-pill">
            <span class="stat-pill-label">Transcript Words</span>
            <span class="stat-pill-value">{words:,}</span>
        </div>""", unsafe_allow_html=True)
    with stat3:
        st.markdown(f"""
        <div class="stat-pill">
            <span class="stat-pill-label">Language Mode</span>
            <span class="stat-pill-value" style="color:var(--text);font-size:1.05rem;text-transform:capitalize">
                {language}
            </span>
        </div>""", unsafe_allow_html=True)
    with stat4:
        st.markdown(f"""
        <div class="stat-pill">
            <span class="stat-pill-label">RAG Engine</span>
            <span class="stat-pill-value" style="font-size:1.05rem;color:#0e7a0d">ChromaDB ✅</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

    # ── Tab Navigation ──
    tab_overview, tab_insights, tab_transcript, tab_chat = st.tabs([
        "📌 Executive Overview",
        "⚡ Actions & Decisions",
        "⏱️ Timestamped Transcript",
        "💬 RAG Chat Studio",
    ])

    # ── TAB 1: EXECUTIVE OVERVIEW ──
    with tab_overview:
        st.markdown(f"""
        <div class="glass-card card-cobalt-stripe">
            <div class="card-header-label">Meeting / Video Title</div>
            <div style="font-family:'Plus Jakarta Sans',sans-serif;font-size:1.55rem;font-weight:800;color:var(--text);line-height:1.25;letter-spacing:-0.02em">
                {r['title']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        formatted_summary = format_clean_summary_html(r['summary'])
        st.html(f"""
        <div class="glass-card">
            <div class="card-header-label">Executive Summary</div>
            <div>{formatted_summary}</div>
        </div>
        """)

        exp_col1, exp_col2 = st.columns([1.2, 4])
        with exp_col1:
            clean_txt_summary = re.sub(r"\*\*(.*?)\*\*", r"\1", r['summary']).replace("**", "")
            st.download_button(
                "📥 Export Report (.txt)",
                data=f"TITLE: {r['title']}\n\nSUMMARY:\n{clean_txt_summary}\n\nACTION ITEMS:\n{r['action_items']}\n\nKEY DECISIONS:\n{r['key_decisions']}\n",
                file_name=f"{r['title'][:30]}_meeting_brief.txt",
                mime="text/plain",
                use_container_width=True,
                type="secondary",
            )

    # ── TAB 2: ACTIONS & DECISIONS ──
    with tab_insights:
        col_act, col_dec = st.columns(2, gap="large")
        with col_act:
            items_act = render_structured_list_html(r['action_items'], badge_color="var(--accent)", max_items=7)
            st.html(
                f'<div class="glass-card" style="border-top:4px solid var(--accent)">'
                f'<div class="card-header-label" style="color:var(--accent)">✅ Action Items & Next Steps (5-7 Key Points)</div>'
                f'{items_act}</div>'
            )

        with col_dec:
            items_dec = render_structured_list_html(r['key_decisions'], badge_color="var(--accent-2)", max_items=7)
            st.html(
                f'<div class="glass-card" style="border-top:4px solid var(--accent-2)">'
                f'<div class="card-header-label" style="color:#B87800">🔑 Key Finalized Decisions (5-7 Key Points)</div>'
                f'{items_dec}</div>'
            )

        items_q = render_structured_list_html(r['open_questions'], badge_color="#4F46E5", max_items=5)
        st.html(
            f'<div class="glass-card">'
            f'<div class="card-header-label">❓ Unresolved & Open Questions</div>'
            f'{items_q}</div>'
        )

    # ── TAB 3: TIMESTAMPED TRANSCRIPT ──
    with tab_transcript:
        st.markdown("""
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:0.75rem">
            <div class="card-header-label" style="margin:0">Complete Chronological Transcript</div>
            <span style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;color:var(--text-muted)">
                Numbered by segment offsets
            </span>
        </div>
        """, unsafe_allow_html=True)

        # Keyword Search inside Transcript
        search_query = st.text_input(
            "Filter Transcript",
            placeholder="Search keywords (e.g., 'framework', 'launch', 'budget')...",
            label_visibility="collapsed",
        )

        display_transcript = r["transcript"]
        if search_query.strip():
            matching_lines = [
                line for line in display_transcript.split("\n")
                if search_query.lower() in line.lower()
            ]
            display_transcript = "\n".join(matching_lines) if matching_lines else f"No lines matching: '{search_query}'"

        st.markdown(f'<div class="transcript-scroll">{html.escape(display_transcript)}</div>', unsafe_allow_html=True)

        st.download_button(
            "📥 Download Transcript (.txt)",
            data=r["transcript"],
            file_name=f"{r['title'][:30]}_transcript.txt",
            mime="text/plain",
            use_container_width=True,
            type="secondary",
        )

    # ── TAB 4: RAG CHAT STUDIO ──
    with tab_chat:
        st.markdown("""
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:0.75rem">
            <div style="font-family:'Plus Jakarta Sans',sans-serif;font-size:1.25rem;font-weight:800;color:var(--accent);letter-spacing:-0.02em">
                Context-Aware RAG Chat
            </div>
            <span class="hero-badge">ChromaDB + LCEL</span>
        </div>
        """, unsafe_allow_html=True)

        # 1-Click Prompt Chips
        st.caption("1-Click Quick Queries:")
        chips_cols = st.columns(4, gap="small")
        quick_queries = [
            "What are the main action items?",
            "What key decisions were agreed on?",
            "Were any dates or deadlines set?",
            "Summarize the speaker's main advice",
        ]
        for idx, q_text in enumerate(quick_queries):
            if chips_cols[idx].button(q_text, key=f"chip_{idx}", use_container_width=True, type="secondary"):
                with st.spinner("Retrieving video context & synthesizing response..."):
                    handle_rag_question(q_text)
                st.rerun()

        # Chat Message History
        if st.session_state.chat_history:
            st.markdown('<div class="chat-wrapper">', unsafe_allow_html=True)
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    st.html(f"""
                    <div style="display:flex;flex-direction:column;align-items:flex-end;margin-bottom:1rem">
                        <span class="chat-avatar-user">You</span>
                        <div class="chat-bubble-user">{html.escape(msg['content'])}</div>
                    </div>""")
                else:
                    formatted_ans = format_chat_bubble_html(msg['content'])
                    st.html(f"""
                    <div style="display:flex;flex-direction:column;align-items:flex-start;margin-bottom:1rem">
                        <span class="chat-avatar-bot">AI Assistant (RAG Grounded)</span>
                        <div class="chat-bubble-assistant">{formatted_ans}</div>
                    </div>""")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="glass-card" style="text-align:center;padding:2.5rem 1.5rem">
                <div style="font-family:'Plus Jakarta Sans',sans-serif;font-size:1.1rem;font-weight:700;color:var(--accent);margin-bottom:0.35rem">
                    Ask Anything About This Meeting
                </div>
                <div style="font-size:0.85rem;color:var(--text-muted)">
                    Ask who is assigned to what, cite timestamps, or ask for clarifications.
                </div>
            </div>""", unsafe_allow_html=True)

        # Chat Input Bar
        with st.form("chat_input_form", clear_on_submit=True):
            cin_col1, cin_col2 = st.columns([5, 1], gap="small")
            with cin_col1:
                user_q = st.text_input(
                    "Ask question",
                    placeholder="Ask a question about the video (e.g., 'What frameworks did the speaker recommend?')...",
                    label_visibility="collapsed",
                )
            with cin_col2:
                send_chat = st.form_submit_button("Ask RAG", use_container_width=True)

            if send_chat and user_q.strip():
                with st.spinner("Searching semantic vectors & citing timestamps..."):
                    handle_rag_question(user_q.strip())
                st.rerun()

        if st.session_state.chat_history:
            if st.button("🗑️ Clear Chat History", type="secondary"):
                st.session_state.chat_history = []
                st.rerun()

else:
    # ── Empty Landing Page (Interactive Hero + Front Ingestion Console + Demo Buttons) ──
    st.markdown("""
    <div class="glass-card card-cobalt-stripe" style="padding:2.8rem 2.2rem 2rem 2.2rem;text-align:center;margin-top:0.5rem">
        <div style="display:inline-block;margin-bottom:0.75rem">
            <span class="hero-badge">Ready to Analyze</span>
        </div>
        <div style="font-family:'Plus Jakarta Sans',sans-serif;font-size:clamp(1.75rem, 3.2vw, 2.4rem);font-weight:800;color:var(--text);letter-spacing:-0.03em;line-height:1.2;margin-bottom:0.75rem">
            Turn Any Meeting or Video into <span style="color:var(--accent)">Actionable Intelligence</span>
        </div>
        <div style="font-size:0.95rem;color:var(--text-muted);max-width:580px;margin:0 auto 1.2rem auto;line-height:1.7">
            Analyze any YouTube video or audio file right here or from the sidebar. Extract timestamps, executive briefs, decisions, and chat grounded in the transcript.
        </div>
        <div style="display:flex;gap:0.75rem;flex-wrap:wrap;justify-content:center;margin-bottom:0.5rem">
            <span style="background:var(--accent);color:#FFFFFF;padding:0.35rem 0.85rem;border-radius:8px;font-size:0.75rem;font-weight:700">
                ⚡ FFmpeg 16kHz Stream Chunking
            </span>
            <span style="background:var(--accent-2);color:var(--text);padding:0.35rem 0.85rem;border-radius:8px;font-size:0.75rem;font-weight:800">
                🎙️ Multilingual Whisper + Sarvam
            </span>
            <span style="background:#FFFFFF;border:1.5px solid var(--border);color:var(--text);padding:0.35rem 0.85rem;border-radius:8px;font-size:0.75rem;font-weight:700">
                ⏱️ Timestamp-Citing RAG
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Main Front Input Console (Direct Ingestion inside Card Container) ──
    with st.container(border=True):
        f_mode_col1, f_mode_col2 = st.columns([1.1, 4.1], gap="medium")
        with f_mode_col1:
            front_input_mode = st.radio(
                "Choose Front Method",
                ["YouTube / Local Path", "Upload Media File"],
                key="front_input_mode",
                label_visibility="collapsed",
            )

        front_source_val = ""
        front_run_btn = False
        with f_mode_col2:
            if front_input_mode == "YouTube / Local Path":
                f_col_url, f_col_lang, f_col_btn = st.columns([3.1, 1.35, 1.35], gap="small")
                with f_col_url:
                    front_source_val = st.text_input(
                        "YouTube URL or File Path",
                        key="front_source_input",
                        placeholder="https://youtube.com/watch?v=... or local path",
                        on_change=sync_front_url,
                        label_visibility="collapsed",
                    )
                with f_col_lang:
                    front_lang = st.selectbox(
                        "Speech Model",
                        ["english", "hinglish"],
                        key="front_lang_key",
                        format_func=lambda x: "🎙️ English" if x == "english" else "🎙️ Hinglish",
                        on_change=sync_front_lang,
                        label_visibility="collapsed",
                    )
                with f_col_btn:
                    front_run_btn = st.button("⚡ Analyse", key="front_run_yt_btn", type="primary", use_container_width=True)
            else:
                f_col_file, f_col_lang, f_col_btn = st.columns([3.1, 1.35, 1.35], gap="small")
                with f_col_file:
                    front_uploaded = st.file_uploader(
                        "Upload MP4, MP3, WAV, or M4A",
                        type=["mp4", "mp3", "wav", "m4a", "webm", "mkv"],
                        key="front_uploaded_file",
                        label_visibility="collapsed",
                    )
                    if front_uploaded is not None:
                        os.makedirs(DOWNLOAD_DIR, exist_ok=True)
                        saved_path = os.path.join(DOWNLOAD_DIR, front_uploaded.name)
                        with open(saved_path, "wb") as f:
                            f.write(front_uploaded.getbuffer())
                        front_source_val = saved_path
                        st.caption(f"📁 Ready to analyze: `{front_uploaded.name}`")
                with f_col_lang:
                    front_lang = st.selectbox(
                        "Speech Model",
                        ["english", "hinglish"],
                        key="front_lang_file_key",
                        format_func=lambda x: "🎙️ English" if x == "english" else "🎙️ Hinglish",
                        label_visibility="collapsed",
                    )
                with f_col_btn:
                    front_run_btn = st.button("⚡ Analyse", key="front_run_file_btn", type="primary", use_container_width=True)

    if front_run_btn:
        run_analysis_pipeline(front_source_val, front_lang)

    # ── Multi-Video Library Cards (If Sessions Exist) ──
    saved_videos = list_indexed_videos()
    if saved_videos:
        st.markdown("""
        <div style="font-family:'Plus Jakarta Sans',sans-serif;font-size:0.8rem;font-weight:800;letter-spacing:0.08em;text-transform:uppercase;color:var(--accent);margin:1.8rem 0 0.8rem 0">
            📂 Ready-to-Explore Library Sessions:
        </div>
        """, unsafe_allow_html=True)
        lib_cols = st.columns(min(len(saved_videos), 3), gap="medium")
        for idx, sv in enumerate(saved_videos[:3]):
            with lib_cols[idx]:
                st.markdown(f"""
                <div class="glass-card" style="margin-bottom:0.6rem;padding:1.2rem;border-top:3.5px solid var(--accent)">
                    <span class="tag-yellow" style="font-size:0.65rem">Indexed Session</span>
                    <div style="font-weight:800;font-size:0.95rem;margin:0.4rem 0 0.25rem 0;line-height:1.3;color:var(--text)">
                        {sv.get('title', 'Video')}
                    </div>
                    <div style="font-size:0.75rem;color:var(--text-muted);margin-bottom:0.75rem">
                        {sv.get('chunks_indexed', 'N/A')} vector chunks • {sv.get('indexed_at', '')[:10]}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                if st.button("⚡ Open Dashboard & RAG", key=f"main_lib_card_{idx}", use_container_width=True, type="primary"):
                    vid_id = sv.get("video_id", "default_session")
                    loaded_chain = load_rag_chain(video_id=vid_id)
                    st.session_state.result = {
                        "title": sv.get("title", "Indexed Video"),
                        "summary": sv.get("summary") or f"Indexed on {sv.get('indexed_at', 'Previously')}. Ready for RAG chat queries.",
                        "transcript": sv.get("transcript") or f"Video ID: {vid_id}\nIndexed Chunks: {sv.get('chunks_indexed', 'N/A')}\nIndexed At: {sv.get('indexed_at', '')}",
                        "plain_transcript": sv.get("plain_transcript") or sv.get("transcript") or "",
                        "action_items": sv.get("action_items") or "Ask the RAG assistant below to retrieve specific action items.",
                        "key_decisions": sv.get("key_decisions") or "Ask the RAG assistant below to retrieve key decisions.",
                        "open_questions": sv.get("open_questions") or "Ask anything about this meeting in the RAG chat below.",
                        "rag_chain": loaded_chain,
                    }
                    st.session_state.rag_chain = loaded_chain
                    st.session_state.elapsed_time = sv.get("elapsed_time", 5.2)
                    st.session_state.pipeline_done = True
                    st.rerun()

    # Quick Demo Trigger Cards
    st.markdown("""
    <div style="font-family:'Plus Jakarta Sans',sans-serif;font-size:0.8rem;font-weight:800;letter-spacing:0.08em;text-transform:uppercase;color:var(--accent);margin:1.8rem 0 0.8rem 0">
        Try Instant Demo Links:
    </div>
    """, unsafe_allow_html=True)

    dcol1, dcol2, dcol3 = st.columns(3, gap="medium")
    with dcol1:
        st.button(
            "▶️ Confidence & Habits\n(English · Groq Whisper)",
            use_container_width=True,
            type="secondary",
            on_click=set_demo_url,
            args=("https://youtu.be/vb8WJgRAKUU", "english"),
            key="main_demo_1",
        )

    with dcol2:
        st.button(
            "▶️ Productivity & Focus\n(English · Groq Whisper)",
            use_container_width=True,
            type="secondary",
            on_click=set_demo_url,
            args=("https://youtu.be/1wjFQsrXhK0", "english"),
            key="main_demo_2",
        )

    with dcol3:
        st.button(
            "▶️ Hinglish / Code-Mixed\n(Hinglish · Sarvam AI)",
            use_container_width=True,
            type="secondary",
            on_click=set_demo_url,
            args=("https://youtu.be/Bhn71mwOjsA", "hinglish"),
            key="main_demo_3",
        )