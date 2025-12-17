import logging
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from utils import remove_from_index

logger = logging.getLogger(__name__)

VECTOR_DB_DIR = "chroma_store"
EMBED_MODEL = "llama3.2:latest"  # Must match processor.py


def delete_file(file_name):
    logger.info(f"Deleting vectors for: {file_name}")

    try:
        vectordb = Chroma(
            persist_directory=VECTOR_DB_DIR,
            embedding_function=OllamaEmbeddings(model=EMBED_MODEL),
        )
        docs = vectordb.get(include=["metadatas"])

        ids_to_delete = [
            doc_id for doc_id, meta in zip(docs['ids'], docs['metadatas'])
            if meta.get("source_file") == file_name
        ]

        if ids_to_delete:
            vectordb._collection.delete(ids=ids_to_delete)
            remove_from_index(file_name)
            logger.info(f"Removed {len(ids_to_delete)} chunks for {file_name}")
        else:
            logger.warning("No vectors found for this file")
    except Exception as e:
        logger.error(f"Error deleting file {file_name}: {e}")

