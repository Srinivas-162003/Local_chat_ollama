from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
from utils import remove_from_index

VECTOR_DB_DIR = "chroma_store"

def delete_file(file_name):
    print(f"🗑 Deleting vectors from: {file_name}")

    vectordb = Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=OllamaEmbeddings(model="mistral"))
    docs = vectordb.get(include=["metadatas"])

    ids_to_delete = [
        doc_id for doc_id, meta in zip(docs['ids'], docs['metadatas'])
        if meta.get("source_file") == file_name
    ]

    if ids_to_delete:
        vectordb._collection.delete(ids=ids_to_delete)
        remove_from_index(file_name)
        print(f"✅ Removed {len(ids_to_delete)} chunks for {file_name}")
    else:
        print("⚠️ No vectors found for this file")
