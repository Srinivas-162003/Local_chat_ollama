import os
import logging
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from utils import update_file_index

logger = logging.getLogger(__name__)

UPLOAD_FOLDER = "uploads"
VECTOR_DB_DIR = "chroma_store"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VECTOR_DB_DIR, exist_ok=True)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning(f"Invalid {name}={raw!r}; using {default}")
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning(f"Invalid {name}={raw!r}; using {default}")
        return default


try:
    embedding_model = OllamaEmbeddings(model="llama3.2:latest")
    vectordb = Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=embedding_model)
    logger.info("Ollama embedding model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load Ollama: {e}")
    embedding_model = None
    vectordb = None

def process_file(file_path):
    file_name = os.path.basename(file_path)
    print(f"📄 Processing: {file_path}")

    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    else:
        print(f"⚠️ Unsupported file: {file_path}")
        return

    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    for i, chunk in enumerate(chunks):
        chunk.metadata["source_file"] = file_name

    vectordb.add_documents(chunks)
    update_file_index(file_name, len(chunks))
    print(f"✅ {file_name} added to vector store with {len(chunks)} chunks")

def get_retriever(overrides=None):
    """Get a retriever with optional configuration overrides.
    
    Config via environment variables:
    - RETRIEVER_SEARCH_TYPE: 'similarity' (default), 'mmr', or 'similarity_score_threshold'
    - RETRIEVER_K: number of chunks to retrieve (default: 15)
    - RETRIEVER_FETCH_K: fetch this many before filtering for MMR (default: 20)
    - RETRIEVER_LAMBDA_MULT: MMR diversity weight 0-1 (default: 0.5)
    - RETRIEVER_SCORE_THRESHOLD: min similarity score for threshold mode (default: 0.5)
    """
    if vectordb is None:
        raise RuntimeError("Vector database not initialized")
    
    search_type = os.getenv("RETRIEVER_SEARCH_TYPE", "similarity")
    k = _env_int("RETRIEVER_K", 15)
    fetch_k = _env_int("RETRIEVER_FETCH_K", 20)
    lambda_mult = _env_float("RETRIEVER_LAMBDA_MULT", 0.5)
    score_threshold = _env_float("RETRIEVER_SCORE_THRESHOLD", 0.5)
    
    if overrides:
        search_type = overrides.get("search_type", search_type)
        k = overrides.get("k", k)
        fetch_k = overrides.get("fetch_k", fetch_k)
        lambda_mult = overrides.get("lambda_mult", lambda_mult)
        score_threshold = overrides.get("score_threshold", score_threshold)
    
    logger.info(f"Retriever config: search_type={search_type}, k={k}, fetch_k={fetch_k}, lambda_mult={lambda_mult}, threshold={score_threshold}")
    
    if search_type == "mmr":
        return vectordb.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": lambda_mult}
        )
    elif search_type == "similarity_score_threshold":
        return vectordb.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": k, "score_threshold": score_threshold}
        )
    else:  # similarity
        return vectordb.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )
