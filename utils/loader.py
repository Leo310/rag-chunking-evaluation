
import os
import json
from pathlib import Path
from typing import Dict, List
from langchain_core.documents import Document


def load_documents(data_dir) -> List[Document]:
    documents: List[Document] = []
    with open(data_dir + 'documents.jsonl', 'r') as jsonl_file:
        for line in jsonl_file:
            data = json.loads(line)
            obj = Document(**data)
            documents.append(obj)
    return documents

def save_documents(documents, data_dir):
    with open(data_dir + 'documents.jsonl', 'w') as jsonl_file:
        for document in documents:
            jsonl_file.write(json.dumps(document.dict()) + "\n")

def load_chunks(data_dir) -> Dict[str, List[Document]]:
    split_chunks: Dict[str, List[Document]] = {}
    for file_name in os.listdir(data_dir+"split_chunks/"):
        if file_name.endswith(".jsonl"):
            with open(f"{data_dir}split_chunks/{file_name}", 'r') as jsonl_file:
                experiment_name = Path(file_name).stem
                split_chunks[experiment_name] = []
                for line in jsonl_file:
                    data = json.loads(line)
                    obj = Document(**data)
                    split_chunks[experiment_name].append(obj)

    return split_chunks

def save_chunks(split_chunks, data_dir):
    for experiment_name, chunks in split_chunks.items():
        with open(f"{data_dir}split_chunks/{experiment_name}.jsonl", 'w') as jsonl_file:
            for chunk in chunks:
                jsonl_file.write(json.dumps(chunk.dict()) + "\n")