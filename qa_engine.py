import logging

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

try:
    from langchain_ollama import OllamaLLM
except ImportError:
    from langchain_community.llms import Ollama as OllamaLLM

from processor import get_retriever

logger = logging.getLogger(__name__)

_llm = None
_retriever = None
_rag_chain = None

PROMPT_TEMPLATE = """You are a detailed and thorough assistant answering questions based on provided documents.

CRITICAL INSTRUCTIONS:
1. Only answer using information from the provided context chunks.
2. Always provide complete information found in context.
3. If context does not contain the answer, say:
"This information is not available in the provided documents."
4. Do not infer or fabricate details.

Context:
{context}

Question: {question}

Answer:"""

PROMPT = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context", "question"],
)


def _get_llm():
    global _llm
    if _llm is None:
        _llm = OllamaLLM(model="llama3.2:latest")
    return _llm


def _get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = get_retriever()
    return _retriever


def _format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])


def _get_rag_chain():
    global _rag_chain
    if _rag_chain is None:
        retriever = _get_retriever()
        llm = _get_llm()
        _rag_chain = (
            {"context": retriever | _format_docs, "question": RunnablePassthrough()}
            | PROMPT
            | llm
            | StrOutputParser()
        )
    return _rag_chain


def answer_query(question: str) -> str:
    """Answer a question based on indexed documents."""
    try:
        retriever = _get_retriever()
        docs = retriever.invoke(question)
        if not docs:
            logger.warning(f"No source documents retrieved for: {question}")
            return "I couldn't find relevant information in the knowledge base to answer your question."

        rag_chain = _get_rag_chain()
        return rag_chain.invoke(question)
    except Exception as exc:
        logger.error(f"Error answering query: {exc}")
        raise


def debug_retrieve(question: str):
    retriever = _get_retriever()
    return retriever.invoke(question)


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
