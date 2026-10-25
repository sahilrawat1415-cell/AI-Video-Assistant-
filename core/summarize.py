import os
import re
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter


def get_llm(max_tokens: int = 700):
    return ChatGroq(
        model="qwen/qwen3.8-27b",
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2,
        max_tokens=max_tokens,
    )


def split_transcript(transcript: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=5000,
        chunk_overlap=200,
    )
    return splitter.split_text(transcript)


def _clean_summary_text(text: str) -> str:
    """Removes raw markdown bold asterisks and formats clean bullet lines."""
    if not text:
        return ""
    # Strip all markdown bold asterisks like **Heading** -> Heading
    cleaned = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    cleaned = cleaned.replace("**", "")
    # Convert asterisk bullet points to clean hyphen bullets
    cleaned = re.sub(r"^(\s*)\*\s+", r"\1- ", cleaned, flags=re.MULTILINE)
    return cleaned.strip()


def summarize(transcript: str) -> str:
    if not transcript or not transcript.strip():
        return "No transcript available to summarize."

    llm = get_llm(max_tokens=700)

    # qwen/qwen3.8-27b on Groq has 32k context (~120k chars); 50k chars covers ~45 min meetings in 1 fast direct call
    if len(transcript) <= 50000:
        direct_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "Create a concise, professional meeting executive summary using clean bullet points and section titles. "
                "Preserve key decisions, action items, and important facts.\n"
                "RULES:\n"
                "- NEVER use markdown bold asterisks (**). Output clean plain text without asterisks.\n"
                "- Write section headers on their own line without asterisks.\n"
                "- Use clean hyphens (-) for bullet points, not asterisks (*)."
            ),
            ("human", "{text}"),
        ])
        raw_summary = (direct_prompt | llm | StrOutputParser()).invoke({"text": transcript})
        return _clean_summary_text(raw_summary)

    chunks = split_transcript(transcript)
    map_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Summarize this meeting transcript segment in at most 4 concise "
            "bullet points. Preserve important decisions, tasks, and facts.\n"
            "DO NOT use markdown bold asterisks (**). Use clean hyphens (-) for bullets."
        ),
        ("human", "{text}"),
    ])
    map_chain = map_prompt | llm | StrOutputParser()

    chunk_summaries = map_chain.batch([{"text": c} for c in chunks], config={"max_concurrency": 4})
    combined = "\n".join(chunk_summaries)[:12000]

    final_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Combine the supplied partial meeting summaries into one concise "
            "professional executive summary. Preserve key decisions and action items.\n"
            "RULES:\n"
            "- NEVER use markdown bold asterisks (**). Output clean text without asterisks.\n"
            "- Write section headers on their own line without asterisks.\n"
            "- Use clean hyphens (-) for bullet points, not asterisks (*)."
        ),
        ("human", "{text}"),
    ])
    final_chain = final_prompt | llm | StrOutputParser()
    raw_summary = final_chain.invoke({"text": combined})
    return _clean_summary_text(raw_summary)


def generate_title(transcript: str) -> str:
    if not transcript or not transcript.strip():
        return "Untitled Video"

    llm = get_llm(max_tokens=40)
    title_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You generate titles for YouTube videos and meeting recordings.
Rules:
- Identify the main topic of the transcript.
- Generate one clear, descriptive title (5-12 words).
- Return only the title. No HTML, Markdown, or quotes."""
        ),
        ("human", "Generate a suitable title for this video transcript:\n\n{text}"),
    ])

    try:
        title = (title_prompt | llm | StrOutputParser()).invoke({"text": transcript[:1500]}).strip()
        title = re.sub(r"<[^>]*>", "", title)
        title = re.sub(r"\*\*(.*?)\*\*", r"\1", title).strip().strip('"').strip("'")
        return title if title and title.lower() not in {"</div>", "<div>", "untitled"} else "Video Summary"
    except Exception as e:
        print(f"Title generation failed: {e}")
        return "Video Summary"