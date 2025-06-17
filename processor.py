import os
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
from utils import update_file_index

UPLOAD_FOLDER = "uploads"
VECTOR_DB_DIR = "chroma_store"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VECTOR_DB_DIR, exist_ok=True)

embedding_model = OllamaEmbeddings(model="llama3.1:8b")
vectordb = Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=embedding_model)

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

def get_retriever():
    return vectordb.as_retriever()
