import logging

from processor import get_vector_store
from utils import remove_from_index

logger = logging.getLogger(__name__)


def delete_file(file_name: str) -> int:
    """Delete all vector chunks belonging to one source file.

    Returns number of deleted chunks.
    """
    logger.info(f"Deleting vectors for: {file_name}")
    vectordb = get_vector_store()

    docs = vectordb.get(include=["metadatas"])
    ids_to_delete = [
        doc_id
        for doc_id, meta in zip(docs.get("ids", []), docs.get("metadatas", []))
        if meta and meta.get("source_file") == file_name
    ]

    if not ids_to_delete:
        logger.warning(f"No vectors found for {file_name}")
        remove_from_index(file_name)
        return 0

    try:
        vectordb.delete(ids=ids_to_delete)
    except Exception:
        # Fallback for older Chroma wrappers.
        vectordb._collection.delete(ids=ids_to_delete)

    remove_from_index(file_name)
    logger.info(f"Removed {len(ids_to_delete)} chunks for {file_name}")
    return len(ids_to_delete)
