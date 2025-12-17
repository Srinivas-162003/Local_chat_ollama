import logging
from langchain_community.llms import Ollama
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import PromptTemplate
from processor import get_retriever

logger = logging.getLogger(__name__)

llm = Ollama(model="llama3.2:latest")
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
    chain_type_kwargs={"prompt": PROMPT},
    return_source_documents=True
)

def answer_query(question: str) -> str:
    """Answer a question based on indexed documents."""
    try:
        result = qa_chain.invoke({"query": question})
        answer = result.get("result", "")
        
        # Safety check: if no source documents found, warn the user
        source_docs = result.get("source_documents", [])
        if not source_docs:
            logger.warning(f"No source documents retrieved for: {question}")
            answer = "I couldn't find relevant information in the knowledge base to answer your question."
        
        return answer
    except Exception as e:
        logger.error(f"Error answering query: {e}")
        raise

def debug_retrieve(question: str):
    """Retrieve documents for debugging without generating an answer."""
    return retriever.invoke(question)

def ask_query():
    print("\n💬 Ask questions based on uploaded documents. Type 'exit' to stop.\n")
    while True:
        query = input(">> ")
        if query.lower() in ["exit", "quit"]:
            break
        try:
            answer = answer_query(query)
            print("🤖", answer)
        except Exception as e:
            print("❌ Error:", e)
