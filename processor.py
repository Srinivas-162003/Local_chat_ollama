import os
import logging
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
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
    embedding_model = OllamaEmbeddings(model="llama3.1:8b")
    vectordb = Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=embedding_model)
    logger.info("Ollama embedding model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load Ollama: {e}")
    embedding_model = None
    vectordb = None


def process_file(file_path):
    if embedding_model is None or vectordb is None:
        logger.error("Ollama not available - cannot process file")
        return
    
    file_name = os.path.basename(file_path)
    logger.info(f"Processing: {file_path}")

    try:
        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        else:
            logger.warning(f"Unsupported file: {file_path}")
            return

        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_documents(docs)

        for i, chunk in enumerate(chunks):
            chunk.metadata["source_file"] = file_name

        vectordb.add_documents(chunks)
        update_file_index(file_name, len(chunks))
        logger.info(f"{file_name} added to vector store with {len(chunks)} chunks")
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")


def get_retriever(overrides: dict | None = None):
    if vectordb is None:
        raise RuntimeError("Vector database not initialized. Check Ollama connection.")

    overrides = overrides or {}

    # Retrieval configuration (restart server after changing env vars)
    # - RETRIEVER_SEARCH_TYPE: mmr | similarity | similarity_score_threshold
    # - RETRIEVER_K: number of chunks to return
    # - RETRIEVER_FETCH_K: MMR candidate pool
    # - RETRIEVER_LAMBDA_MULT: MMR relevance/diversity balance (1.0=relevance)
    # - RETRIEVER_SCORE_THRESHOLD: only for similarity_score_threshold
    search_type = (os.getenv("RETRIEVER_SEARCH_TYPE") or "mmr").strip().lower()
    k = overrides.get("k", _env_int("RETRIEVER_K", 15))

    if search_type == "mmr":
        fetch_k = overrides.get("fetch_k", _env_int("RETRIEVER_FETCH_K", 60))
        lambda_mult = overrides.get("lambda_mult", _env_float("RETRIEVER_LAMBDA_MULT", 0.8))
        search_kwargs = {"k": k, "fetch_k": fetch_k, "lambda_mult": lambda_mult}
    elif search_type == "similarity":
        search_kwargs = {"k": k}
    elif search_type == "similarity_score_threshold":
        score_threshold = overrides.get(
            "score_threshold",
            _env_float("RETRIEVER_SCORE_THRESHOLD", 0.3),
        )
        search_kwargs = {"k": k, "score_threshold": score_threshold}
    else:
        logger.warning(
            f"Unknown RETRIEVER_SEARCH_TYPE={search_type!r}; falling back to 'mmr'"
        )
        fetch_k = overrides.get("fetch_k", _env_int("RETRIEVER_FETCH_K", 60))
        lambda_mult = overrides.get("lambda_mult", _env_float("RETRIEVER_LAMBDA_MULT", 0.8))
        search_type = "mmr"
        search_kwargs = {"k": k, "fetch_k": fetch_k, "lambda_mult": lambda_mult}

    logger.info(f"Retriever: type={search_type} kwargs={search_kwargs}")
    return vectordb.as_retriever(search_type=search_type, search_kwargs=search_kwargs)
