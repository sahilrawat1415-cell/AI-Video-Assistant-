import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough


def get_llm():
    return ChatGroq(
        model="qwen/qwen3.8-27b",
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.3,
        max_tokens=800,
    )


def build_chain(system_prompt: str):
    llm = get_llm()
    return (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{text}"),
        ])
        | llm
        | StrOutputParser()
    )


import re


def _clean_numbered_list(text: str, max_items: int = 7) -> str:
    """Fixes LLM disorientation, strips asterisks, filters truncated lines, and caps at max_items."""
    if not text:
        return ""
    # Strip all markdown bold asterisks
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = text.replace("**", "")

    # Merge "2.\n  Text" -> "2. Text"
    text = re.sub(r'(\d+)[\.\)]\s*\n+\s*', r'\1. ', text.strip())

    lines = [line.strip() for line in text.split("\n") if line.strip()]
    cleaned_items = []

    for line in lines:
        # Match numbered item: "1. Some text"
        m = re.match(r'^(\d+)[\.\)]\s*(.*)', line)
        if m:
            content = m.group(2).strip()
        elif line.startswith(("- ", "• ", "* ")):
            content = line[2:].strip()
        else:
            content = line.strip()

        # Clean any accidental nested numbers or remaining asterisks
        content = re.sub(r'^(\d+)[\.\)]\s*', '', content).strip()
        content = content.replace("**", "").replace("*", "")

        # Discard truncated fragments (like item "10. 1" or single word digits)
        if len(content) < 4 or content.isdigit():
            continue

        cleaned_items.append(f"{len(cleaned_items) + 1}. {content}")

        if len(cleaned_items) >= max_items:
            break

    return "\n".join(cleaned_items)


def extract_action_items(transcript: str) -> str:
    chain = build_chain(
        "You are an expert meeting and content analyst. From the transcript provided, "
        "extract strictly 5 to 7 of the most impactful, practical action items or next steps. "
        "Rules: "
        "- Provide strictly 5 to 7 items (minimum 5, maximum 7). Do NOT output more than 7. "
        "- Format as a clean numbered list: 1., 2., 3., 4., 5., etc. "
        "- Each item must have the number and complete text on the SAME line (e.g., '1. Next step text'). "
        "- DO NOT split the number and text across multiple lines. "
        "- NEVER use markdown bold asterisks (**). Output plain text only."
    )
    raw = chain.invoke(transcript)
    return _clean_numbered_list(raw, max_items=7)


def extract_key_decisions(transcript: str) -> str:
    chain = build_chain(
        "You are an expert meeting and content analyst. From the transcript provided, "
        "extract strictly 5 to 7 of the most critical decisions made, primary conclusions, or main takeaways. "
        "Rules: "
        "- Provide strictly 5 to 7 items (minimum 5, maximum 7). Do NOT output more than 7. "
        "- Format as a clean numbered list: 1., 2., 3., 4., 5., etc. "
        "- Each item must have the number and complete text on the SAME line (e.g., '1. Decision text'). "
        "- DO NOT split the number and text across multiple lines. "
        "- NEVER use markdown bold asterisks (**). Output plain text only."
    )
    raw = chain.invoke(transcript)
    return _clean_numbered_list(raw, max_items=7)


def extract_questions(transcript: str) -> str:
    chain = build_chain(
        "You are an expert meeting and content analyst. Based on the transcript provided, "
        "generate strictly 3 to 5 insightful questions that can be asked about this content. "
        "Rules: "
        "- Provide strictly 3 to 5 items. "
        "- Format as a clean numbered list: 1., 2., 3., etc. "
        "- Each item must have the number and complete text on the SAME line (e.g., '1. Question text'). "
        "- DO NOT split the number and text across multiple lines. "
        "- NEVER use markdown bold asterisks (**). Output plain text only."
    )
    raw = chain.invoke(transcript)
    return _clean_numbered_list(raw, max_items=5)


def extract_all_insights(transcript: str) -> dict:
    """
    Unified single-pass LLM extraction: generates Title, Executive Summary,
    Action Items (5-7), Key Decisions (5-7), and Open Questions (3-5)
    in a SINGLE fast Groq request (2-4s) instead of 5 separate requests.
    Prevents Groq 429 rate limit backoff (7,000 ITPM) and cuts pipeline time by >100 seconds!
    """
    if not transcript or not str(transcript).strip():
        return {
            "title": "Untitled Video",
            "summary": "No transcript available to analyze.",
            "action_items": "",
            "key_decisions": "",
            "open_questions": "",
        }

    # Sample up to 25k chars (~5k tokens) to stay well under Groq's 7k ITPM quota
    sample_text = str(transcript)[:25000]

    system_prompt = (
        "You are an elite video and executive meeting analyst. Analyze the transcript provided and generate a complete, "
        "structured intelligence briefing.\n\n"
        "RULES:\n"
        "- NEVER use markdown bold asterisks (**). Output clean plain text without asterisks.\n"
        "- Format strictly with the exact section delimiter tags shown below.\n\n"
        "### TITLE ###\n"
        "One clear, descriptive title (5-12 words). Plain text only.\n\n"
        "### SUMMARY ###\n"
        "A concise, professional executive brief with clean section lines and hyphen (-) bullets.\n\n"
        "### ACTION_ITEMS ###\n"
        "Strictly 5 to 7 high-impact action items. Format as a clean numbered list: 1., 2., 3., 4., 5., 6., 7. with full text on the same line.\n\n"
        "### KEY_DECISIONS ###\n"
        "Strictly 5 to 7 critical decisions, main conclusions, or key frameworks. Format as a clean numbered list: 1., 2., 3., 4., 5., 6., 7.\n\n"
        "### QUESTIONS ###\n"
        "Strictly 3 to 5 open, unresolved, or strategic follow-up questions. Format as a clean numbered list: 1., 2., 3., 4., 5."
    )

    llm = ChatGroq(
        model="qwen/qwen3.8-27b",
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2,
        max_tokens=1500,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Transcript content:\n\n{text}"),
    ])
    chain = prompt | llm | StrOutputParser()

    try:
        raw_output = chain.invoke({"text": sample_text})
    except Exception as e:
        print(f"Unified extraction failed, falling back to separate extractors: {e}")
        return {
            "title": "Meeting Brief",
            "summary": "Summary unavailable.",
            "action_items": extract_action_items(transcript),
            "key_decisions": extract_key_decisions(transcript),
            "open_questions": extract_questions(transcript),
        }

    # Parse sections
    def extract_section(tag: str, text: str) -> str:
        pattern = rf"### {tag} ###\s*(.*?)(?=(?:### [A-Z_]+ ###|$))"
        m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else ""

    title = extract_section("TITLE", raw_output).strip()
    title = re.sub(r"<[^>]*>", "", title).replace("**", "").strip('"').strip("'")
    if not title:
        title = "Meeting Intelligence Brief"

    raw_summary = extract_section("SUMMARY", raw_output)
    raw_summary = re.sub(r"\*\*(.*?)\*\*", r"\1", raw_summary).replace("**", "")

    raw_actions = extract_section("ACTION_ITEMS", raw_output)
    action_items = _clean_numbered_list(raw_actions, max_items=7)

    raw_decisions = extract_section("KEY_DECISIONS", raw_output)
    key_decisions = _clean_numbered_list(raw_decisions, max_items=7)

    raw_questions = extract_section("QUESTIONS", raw_output)
    open_questions = _clean_numbered_list(raw_questions, max_items=5)

    return {
        "title": title,
        "summary": raw_summary,
        "action_items": action_items,
        "key_decisions": key_decisions,
        "open_questions": open_questions,
    }