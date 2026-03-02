import logging
import os
import shutil
import sys
from datetime import datetime

from dotenv import load_dotenv
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

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


def _get_embedding_dimension() -> int:
    if _embedding_model is None:
        raise RuntimeError("Embedding model is not initialized")
    sample_vec = _embedding_model.embed_query("dimension probe")
    return len(sample_vec)


def _get_collection_dimension(vectordb) -> int | None:
    try:
        collection = vectordb._collection
    except Exception:
        return None

    # Chroma internals can vary by version; try a few known places.
    try:
        model = getattr(collection, "_model", None)
        if model is not None:
            dim = getattr(model, "dimension", None)
            if isinstance(dim, int):
                return dim
    except Exception:
        pass

    try:
        metadata = getattr(collection, "metadata", None)
        if isinstance(metadata, dict):
            dim = metadata.get("dimension") or metadata.get("embedding_dimension")
            if isinstance(dim, int):
                return dim
            if isinstance(dim, str) and dim.isdigit():
                return int(dim)
    except Exception:
        pass

    return None


def _ensure_compatible_dimensions() -> None:
    global _vectordb

    if _vectordb is None:
        return

    expected_dim = _get_embedding_dimension()
    actual_dim = _get_collection_dimension(_vectordb)

    if actual_dim is not None and actual_dim != expected_dim:
        logger.warning(
            "Vector store dimension mismatch detected: "
            f"collection={actual_dim}, embedding={expected_dim}"
        )
        _rotate_vector_store()
        _vectordb = Chroma(
            persist_directory=VECTOR_DB_DIR,
            embedding_function=_embedding_model,
        )
        return

    # Force a tiny query via raw collection API so mismatch is detected early.
    try:
        _vectordb._collection.query(query_embeddings=[[0.0] * expected_dim], n_results=1)
    except Exception as exc:
        if _is_dimension_mismatch_error(exc):
            logger.warning(f"Vector store dimension mismatch detected: {exc}")
            _rotate_vector_store()
            _vectordb = Chroma(
                persist_directory=VECTOR_DB_DIR,
                embedding_function=_embedding_model,
            )
        else:
            raise


def _is_dimension_mismatch_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    keywords = [
        "dimension",
        "dimensions",
        "dimensionality",
        "embedding dimension",
    ]
    return any(token in msg for token in keywords)


def _reset_index_file() -> None:
    try:
        with open("file_index.json", "w", encoding="utf-8") as f:
            f.write("{}")
    except Exception as exc:
        logger.warning(f"Could not reset file_index.json: {exc}")


def _rotate_vector_store() -> None:
    global VECTOR_DB_DIR

    if not os.path.exists(VECTOR_DB_DIR):
        return

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_dir = f"{VECTOR_DB_DIR}_backup_{timestamp}"

    try:
        shutil.move(VECTOR_DB_DIR, backup_dir)
        os.makedirs(VECTOR_DB_DIR, exist_ok=True)
        _reset_index_file()
        logger.warning(
            "Detected incompatible embedding dimensions. "
            f"Rotated old vector store to '{backup_dir}' and created a fresh store. "
            "Please re-upload or re-process your documents."
        )
    except PermissionError:
        # If another process still holds the sqlite file, switch to a new directory
        # for this process so queries can proceed immediately.
        fresh_dir = f"{VECTOR_DB_DIR}_fresh_{timestamp}"
        os.makedirs(fresh_dir, exist_ok=True)
        VECTOR_DB_DIR = fresh_dir
        _reset_index_file()
        logger.warning(
            "Detected incompatible embedding dimensions, but existing vector "
            "store is locked by another process. "
            f"Switched to fresh vector store '{VECTOR_DB_DIR}'. "
            "Please restart other server instances and re-process your documents."
        )


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
    device = os.getenv("EMBEDDING_DEVICE", "cuda").strip().lower()
    if device == "auto":
        try:
            import torch

            auto_device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Auto-detected embedding device: {auto_device}")
            return auto_device
        except Exception:
            return "cpu"
    return device


def _initialize_vector_store() -> None:
    global _embedding_model, _vectordb, _init_error

    if _vectordb is not None:
        return
    if _init_error is not None:
        raise RuntimeError(_init_error)

    device = _detect_device()
    try:
        if device == "cuda":
            try:
                import torch

                if not torch.cuda.is_available():
                    logger.warning(
                        "EMBEDDING_DEVICE is set to CUDA but CUDA is not available. "
                        "Falling back to CPU embeddings."
                    )
                    device = "cpu"
            except Exception:
                logger.warning(
                    "Could not validate CUDA availability. Falling back to CPU embeddings."
                )
                device = "cpu"

        _embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )
        try:
            _vectordb = Chroma(
                persist_directory=VECTOR_DB_DIR,
                embedding_function=_embedding_model,
            )
            _ensure_compatible_dimensions()
        except Exception as chroma_exc:
            if _is_dimension_mismatch_error(chroma_exc):
                logger.warning(f"Vector store dimension mismatch detected: {chroma_exc}")
                _rotate_vector_store()
                _vectordb = Chroma(
                    persist_directory=VECTOR_DB_DIR,
                    embedding_function=_embedding_model,
                )
            else:
                raise
        logger.info(f"Embedding model loaded on {device.upper()}")
        logger.info(f"Vector database initialized at {VECTOR_DB_DIR}")
    except Exception as exc:
        if isinstance(exc, ImportError) or "sentence_transformers" in str(exc):
            env_hint = (
                f"Current Python executable: {sys.executable}. "
                "If this is not your project venv, run the server with: "
                "D:/Local-ollama/.venv/Scripts/python.exe server.py"
            )
        else:
            env_hint = ""

        _init_error = (
            "Embeddings/vector DB initialization failed. "
            "Install project dependencies in the active Python environment "
            "(for this repo: `pip install -r requirements.txt`). "
            f"Original error: {exc}. {env_hint}"
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
