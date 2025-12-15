import logging
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from processor import get_retriever

logger = logging.getLogger(__name__)

llm = Ollama(model="llama3.1:8b")
retriever = get_retriever()

# Custom prompt to guide the LLM better
PROMPT_TEMPLATE = """You are a detailed and thorough assistant answering questions based on provided documents.

CRITICAL INSTRUCTIONS:
1. ONLY answer using information from the provided context chunks
2. ALWAYS provide COMPLETE information - list ALL items mentioned in the context
3. For education: Include ALL schools/colleges, degrees, and date ranges
4. For experience: Include ALL job positions/internships with dates and key achievements
5. For projects: List ALL mentioned projects
6. If information exists in context, provide it fully - never say "only mention" when there's more
7. Use bullet points or numbering to organize information clearly
8. If the context does NOT contain information to answer the question, state clearly: "This information is not available in the provided documents"
9. Do NOT make up, assume, or infer information not explicitly stated in the context
10. Do NOT say "it can be inferred" or "it could imply" - only state what is explicitly written

Context:
{context}

Question: {question}

Answer:"""

PROMPT = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)


def answer_query(query: str) -> str:
    """Run a query against the document retriever and return the answer text."""
    try:
        result = qa_chain({"query": query})
        source_docs = result.get('source_documents', [])
        logger.info(f"Query: {query}")
        logger.info(f"Sources: {[doc.metadata.get('source_file', 'unknown') for doc in source_docs]}")
        
        # Check if any documents were actually retrieved
        if not source_docs or len(source_docs) == 0:
            return (
                "[ERROR] No relevant documents were retrieved for this question. "
                "Try rephrasing, or adjust retrieval settings (e.g., lower RETRIEVER_SCORE_THRESHOLD, increase RETRIEVER_K)."
            )
        
        return result.get("result", "No answer generated")
    except Exception as e:
        logger.error(f"Error answering query: {e}")
        raise


def debug_retrieve(query: str, k: int = 15) -> list:
    """Debug function to see what documents are being retrieved."""
    # Create a fresh retriever so debug can override k without restarting.
    debug_retriever = get_retriever({"k": k})
    docs = debug_retriever.get_relevant_documents(query)
    logger.info(f"Retrieved {len(docs)} documents for query: {query}")
    for i, doc in enumerate(docs[:k]):
        logger.info(f"  Doc {i+1}: {doc.metadata.get('source_file', 'unknown')} - {doc.page_content[:100]}...")
    return docs


def ask_query():
    print("\n💬 Ask questions based on uploaded documents. Type 'exit' to stop.\n")
    while True:
        query = input(">> ")
        if query.lower() in ["exit", "quit"]:
            break
        try:
            result = qa_chain.invoke({"query": query})
            answer = result.get("result") or result.get("output_text") or "No answer generated"
            print("🤖", answer)
        except Exception as e:
            print("❌ Error:", e)
            