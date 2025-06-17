from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from processor import process_file, UPLOAD_FOLDER
from delete_file import delete_file
import os

class FileHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith((".pdf", ".docx")):
            process_file(event.src_path)

    def on_deleted(self, event):
        if not event.is_directory:
            delete_file(os.path.basename(event.src_path))

def start_file_watcher():
    observer = Observer()
    observer.schedule(FileHandler(), path=UPLOAD_FOLDER, recursive=False)
    observer.start()
    print(f"📂 Watching {UPLOAD_FOLDER} for changes...")
    return observer
