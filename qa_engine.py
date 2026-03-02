import logging

from clara_engine import answer_with_clara

logger = logging.getLogger(__name__)

def answer_query(question: str) -> str:
    """Compatibility wrapper that routes all questions to CLaRa."""
    try:
        return str(answer_with_clara(question, detailed_response=False))
    except Exception as exc:
        logger.error(f"Error answering query: {exc}")
        raise


def debug_retrieve(question: str):
    from processor import get_retriever

    return get_retriever().invoke(question)


def ask_query():
    print("\nAsk questions based on uploaded documents. Type 'exit' to stop.\n")
    while True:
        query = input(">> ")
        if query.lower() in ["exit", "quit"]:
            break
        try:
            answer = answer_query(query)
            print(answer)
        except Exception as exc:
            print(f"Error: {exc}")
