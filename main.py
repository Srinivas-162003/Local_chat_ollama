from watcher import start_file_watcher
from qa_engine import ask_query

if __name__ == "__main__":
    observer = start_file_watcher()
    try:
        ask_query()
    finally:
        observer.stop()
        observer.join()