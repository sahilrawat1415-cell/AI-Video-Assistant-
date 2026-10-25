import os
import re
from typing import Optional, Any
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from core.vector_store import build_vector_store, load_vector_store, get_retriever

SYSTEM_PROMPT = """You are an expert video and meeting assistant. Answer the user's question 
based on the transcript context provided below.

Rules:
1. When answering, cite relevant timestamps (e.g., [02:15] or [12:52]) whenever available 
   so the user can easily locate that exact moment in the video.
2. If the user asks about an open question, strategic problem, or undecided topic from the meeting:
   - First, explain what the transcript DOES state about the topic (cite who raised it, timestamps, and any figures mentioned).
   - Second, clarify clearly that the meeting participants identified this issue but did NOT finalize or establish a long-term strategy or decision during this discussion.
   - Third, provide 2 to 3 concise, practical recommendations or next steps that the team could take to resolve this open issue.
3. If the topic was genuinely never mentioned or discussed in the transcript, say:
   "This topic was not discussed in the meeting transcript."
4. Always be concise, clear, and structured (use clean hyphens for bullet points).
5. NEVER use markdown bold asterisks (**) or bullet stars (*). Write clean, readable text without asterisks.
6. If quoting someone, mention it clearly.

Context from meeting transcript:
{context}"""


def get_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        try:
            import streamlit as st
            if "GROQ_API_KEY" in st.secrets:
                api_key = str(st.secrets["GROQ_API_KEY"]).strip()
        except Exception:
            pass
    return ChatGroq(
        model="qwen/qwen3.8-27b",
        groq_api_key=api_key,
        max_tokens=650,
        temperature=0.3,
    )


def format_docs(docs):
    formatted = []
    for doc in docs:
        ts = doc.metadata.get("timestamp")
        prefix = f"[{ts}] " if ts and not doc.page_content.startswith(f"[{ts}]") else ""
        formatted.append(f"{prefix}{doc.page_content}")
    return "\n\n".join(formatted)


def _create_lcel_chain(retriever):
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])
    return (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )


def build_rag_chain(
    transcript: Any,
    video_id: str = "default_session",
    title: str = "Meeting Recording",
):
    vector_store = build_vector_store(transcript, video_id=video_id, title=title)
    retriever = get_retriever(vector_store, k=4, video_id=video_id)
    return _create_lcel_chain(retriever)


def load_rag_chain(video_id: Optional[str] = None):
    vector_store = load_vector_store()
    retriever = get_retriever(vector_store, k=4, video_id=video_id)
    return _create_lcel_chain(retriever)


def ask_question(rag_chain, question: str) -> str:
    print(f"Question: {question}")
    answer = rag_chain.invoke(question)
    if isinstance(answer, str):
        # Strip all markdown bold asterisks
        answer = re.sub(r"\*\*(.*?)\*\*", r"\1", answer)
        answer = answer.replace("**", "")
    print(f"Answer: {answer}")
    return answer