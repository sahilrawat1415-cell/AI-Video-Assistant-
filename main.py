import time
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarize import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions, extract_all_insights
from core.rag_engine import build_rag_chain, ask_question

load_dotenv()


def run_pipeline(source: str, language: str = "english") -> dict:
    start_time = time.time()
    print(" Starting AI Video Assistant...")

    # Step 1: Fast FFmpeg Audio Processing & Chunking
    chunks = process_input(source, language=language)

    # Step 2: Parallel Transcription with Timestamps
    transcript = transcribe_all(chunks, language=language)
    timestamped = getattr(transcript, "timestamped_text", str(transcript))
    print(f"\n Timestamped Transcription Preview:\n{timestamped[:350]}...\n")

    # Step 3: High-speed single-pass Insights + Vector Store build in parallel
    print("⚡ Running Unified Insights and Vector Store build in parallel...")
    with ThreadPoolExecutor(max_workers=2) as executor:
        f_insights = executor.submit(extract_all_insights, transcript)
        f_rag      = executor.submit(build_rag_chain, transcript)

        insights = f_insights.result()
        rag_chain = f_rag.result()

        result = {
            "title": insights.get("title", "Meeting Brief"),
            "transcript": timestamped,
            "plain_transcript": str(transcript),
            "summary": insights.get("summary", ""),
            "action_items": insights.get("action_items", ""),
            "key_decisions": insights.get("key_decisions", ""),
            "open_questions": insights.get("open_questions", ""),
            "rag_chain": rag_chain,
        }

    elapsed = time.time() - start_time
    print(f"🎉 Pipeline finished in {elapsed:.2f} seconds!")
    return result


if __name__ == "__main__":
    source = input("Enter YouTube URL or local file path: ").strip()
    language = input("Language (english/hinglish): ").strip() or "english"
    result = run_pipeline(source, language)

    print("\n" + "=" * 60)
    print(f" Title: {result['title']}")
    print(f"\n Summary:\n{result['summary']}")
    print(f"\n Action Items:\n{result['action_items']}")
    print(f"\n Key Decisions:\n{result['key_decisions']}")
    print(f"\n Open Questions:\n{result['open_questions']}")
    print("=" * 60)

    print("\n Chat with your meeting (type 'exit' to quit)\n")
    rag_chain = result["rag_chain"]
    while True:
        question = input("You: ").strip()
        if question.lower() in ["exit", "quit", "q"]:
            print("Goodbye!")
            break
        if not question:
            continue
        answer = ask_question(rag_chain, question)
        print(f"\n Assistant: {answer}\n")