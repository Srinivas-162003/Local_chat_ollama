import os
from pathlib import Path
from typing import List
import logging

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from processor import UPLOAD_FOLDER
from qa_engine import answer_query
from utils import load_file_index
from watcher import start_file_watcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR_ABS = BASE_DIR / UPLOAD_FOLDER

app = FastAPI(title="Local Chat Ollama", version="0.1.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

observer = None


class QueryRequest(BaseModel):
    question: str


@app.on_event("startup")
def on_startup():
    """Ensure folders exist and start the file watcher."""
    global observer
    UPLOAD_DIR_ABS.mkdir(parents=True, exist_ok=True)
    logger.info(f"Upload directory: {UPLOAD_DIR_ABS}")
    observer = start_file_watcher(str(UPLOAD_DIR_ABS))


@app.on_event("shutdown")
def on_shutdown():
    """Stop the file watcher when the server exits."""
    global observer
    if observer:
        observer.stop()
        observer.join()
        observer = None


@app.get("/")
def root():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/files")
def list_files():
    index = load_file_index()
    files: List[dict] = [
        {"name": name, "chunks": chunks}
        for name, chunks in sorted(index.items())
    ]
    return {"files": files}


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="A file is required")

    safe_name = os.path.basename(file.filename)
    if not safe_name.lower().endswith((".pdf", ".docx")):
        raise HTTPException(status_code=400, detail="Only PDF or DOCX files are supported")

    dest_path = UPLOAD_DIR_ABS / safe_name
    contents = await file.read()

    try:
        with open(dest_path, "wb") as f:
            f.write(contents)
        logger.info(f"File uploaded: {dest_path}")
        # The file watcher will pick this up and process asynchronously.
    except Exception as exc:
        logger.error(f"Upload error: {exc}")
        raise HTTPException(status_code=500, detail=f"Could not save file: {exc}")

    return {"file": safe_name, "status": "queued"}


@app.post("/api/query")
def query_documents(payload: QueryRequest):
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        answer = answer_query(question)
    except Exception as exc:  # pragma: no cover - propagate clean error
        raise HTTPException(status_code=500, detail=f"Could not generate answer: {exc}")

    return {"answer": answer}


@app.delete("/api/files/{file_name}")
def delete_document(file_name: str):
    """Delete a file from disk (uploads/), vector store, and index."""
    from delete_file import delete_file

    safe_name = os.path.basename(file_name)
    try:
        # 1) Remove vectors + index (idempotent)
        delete_file(safe_name)

        # 2) Remove the physical file from uploads/
        disk_path = UPLOAD_DIR_ABS / safe_name
        disk_deleted = False
        if disk_path.exists() and disk_path.is_file():
            disk_path.unlink()
            disk_deleted = True

        return {"file": safe_name, "status": "deleted", "disk_deleted": disk_deleted}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not delete file: {exc}")


@app.post("/api/process-uploads")
def process_uploads(file_name: str | None = None):
    """Manually trigger processing.

    - If file_name is provided: process only that file (useful as a fallback when watcher misses events).
    - If file_name is omitted: process all unindexed files in uploads/.
    """
    from processor import process_file
    import glob
    
    upload_path = UPLOAD_DIR_ABS

    safe_name = os.path.basename(file_name) if file_name else None
    if safe_name:
        candidate = upload_path / safe_name
        if not candidate.exists() or not candidate.is_file():
            raise HTTPException(status_code=404, detail=f"File not found in uploads: {safe_name}")
        all_files = [str(candidate)]
    else:
        pdf_files = glob.glob(str(upload_path / "*.pdf"))
        docx_files = glob.glob(str(upload_path / "*.docx"))
        all_files = pdf_files + docx_files
    
    processed = []
    errors = []
    
    index = load_file_index()
    for file_path in all_files:
        base_name = os.path.basename(file_path)
        # Check if already indexed
        if base_name not in index or index.get(base_name, 0) == 0:
            try:
                logger.info(f"Manually processing: {file_path}")
                process_file(file_path)
                processed.append(base_name)
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                errors.append({"file": base_name, "error": str(e)})
    
    return {
        "processed": processed,
        "errors": errors,
        "total": len(all_files),
    }


@app.post("/api/debug-query")
def debug_query(payload: QueryRequest):
    """Debug endpoint to see retrieved documents."""
    from qa_engine import debug_retrieve
    
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        docs = debug_retrieve(question)
        return {
            "query": question,
            "retrieved_count": len(docs),
            "documents": [
                {
                    "source": doc.metadata.get("source_file", "unknown"),
                    "content": doc.page_content[:200],
                }
                for doc in docs[:15]
            ],
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Debug error: {exc}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
