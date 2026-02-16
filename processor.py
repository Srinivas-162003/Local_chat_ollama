import logging
import os

from dotenv import load_dotenv
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

from utils import update_file_index

load_dotenv()

logger = logging.getLogger(__name__)

UPLOAD_FOLDER = "uploads"
VECTOR_DB_DIR = "chroma_store"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VECTOR_DB_DIR, exist_ok=True)

_embedding_model = None
_vectordb = None
_init_error = None


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


def _detect_device() -> str:
    device = os.getenv("EMBEDDING_DEVICE", "auto")
    if device != "auto":
        return device

    try:
        import torch

        auto_device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Auto-detected embedding device: {auto_device}")
        return auto_device
    except Exception:
        return "cpu"


def _initialize_vector_store() -> None:
    global _embedding_model, _vectordb, _init_error

    if _vectordb is not None:
        return
    if _init_error is not None:
        raise RuntimeError(_init_error)

    device = _detect_device()
    try:
        _embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )
        _vectordb = Chroma(
            persist_directory=VECTOR_DB_DIR,
            embedding_function=_embedding_model,
        )
        logger.info(f"Embedding model loaded on {device.upper()}")
        logger.info(f"Vector database initialized at {VECTOR_DB_DIR}")
    except Exception as exc:
        _init_error = (
            "Embeddings/vector DB initialization failed. "
            "Install project dependencies in the active Python environment "
            "(for this repo: `pip install -r requirements.txt`). "
            f"Original error: {exc}"
        )
        logger.error(_init_error)
        raise RuntimeError(_init_error) from exc


def get_vector_store():
    _initialize_vector_store()
    return _vectordb


def process_file(file_path: str) -> None:
    file_name = os.path.basename(file_path)
    print(f"Processing: {file_path}")

    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    else:
        print(f"Unsupported file type: {file_path}")
        return

    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    for chunk in chunks:
        chunk.metadata["source_file"] = file_name

    vectordb = get_vector_store()
    vectordb.add_documents(chunks)
    update_file_index(file_name, len(chunks))
    print(f"{file_name} added to vector store with {len(chunks)} chunks")


def get_retriever(overrides=None):
    """Get a retriever with optional configuration overrides."""
    vectordb = get_vector_store()

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

    logger.info(
        f"Retriever config: search_type={search_type}, k={k}, "
        f"fetch_k={fetch_k}, lambda_mult={lambda_mult}, threshold={score_threshold}"
    )

    if search_type == "mmr":
        return vectordb.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": lambda_mult},
        )
    if search_type == "similarity_score_threshold":
        return vectordb.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": k, "score_threshold": score_threshold},
        )
    return vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
