import os, json

INDEX_FILE = "file_index.json"

def load_file_index():
    if not os.path.exists(INDEX_FILE):
        return {}
    with open(INDEX_FILE, "r") as f:
        return json.load(f)

def update_file_index(file_name, chunk_count):
    index = load_file_index()
    index[file_name] = chunk_count
    with open(INDEX_FILE, "w") as f:
        json.dump(index, f)

def remove_from_index(file_name):
    index = load_file_index()
    if file_name in index:
        del index[file_name]
        with open(INDEX_FILE, "w") as f:
            json.dump(index, f)
