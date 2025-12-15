from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from processor import process_file
from delete_file import delete_file
import os
import logging
import time
import threading

logger = logging.getLogger(__name__)


class FileHandler(FileSystemEventHandler):
    def __init__(self):
        self.processing = set()  # Track files being processed to avoid duplicates
        
    def on_created(self, event):
        if event.is_directory:
            return
            
        if not event.src_path.lower().endswith((".pdf", ".docx")):
            return
        
        file_name = os.path.basename(event.src_path)
        
        # Avoid processing the same file multiple times
        if file_name in self.processing:
            return
        
        self.processing.add(file_name)
        
        # Small delay to ensure file is fully written
        def process_with_delay():
            try:
                time.sleep(1)
                logger.info(f"Detected new file: {event.src_path}")
                process_file(event.src_path)
                logger.info(f"Successfully processed: {file_name}")
            except Exception as e:
                logger.error(f"Error processing file {event.src_path}: {e}")
            finally:
                self.processing.discard(file_name)
        
        threading.Thread(target=process_with_delay, daemon=True).start()

    def on_deleted(self, event):
        if not event.is_directory:
            logger.info(f"Detected deleted file: {event.src_path}")
            try:
                delete_file(os.path.basename(event.src_path))
            except Exception as e:
                logger.error(f"Error deleting file {event.src_path}: {e}")


_handler = None

def start_file_watcher(watch_path=None):
    """Start watching a directory for file changes."""
    global _handler
    
    if watch_path is None:
        from processor import UPLOAD_FOLDER
        watch_path = UPLOAD_FOLDER
    
    _handler = FileHandler()
    observer = Observer()
    observer.schedule(_handler, path=watch_path, recursive=False)
    observer.start()
    logger.info(f"Watching {watch_path} for changes...")
    return observer

