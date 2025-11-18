# main.py
import os
import json
import logging
import shutil
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.schema import Document

# load env
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG Index API")

STORAGE_DIR = os.getenv("STORAGE_DIR", "storage")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "llama3.2:3b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

class IndexRequest(BaseModel):
    folder_path: str
    index_name: str

class QueryRequest(BaseModel):
    index_name: str
    query: str

# helpers
def get_embedding_instance():
    from llama_index.embeddings.ollama import OllamaEmbedding
    return OllamaEmbedding(model_name=OLLAMA_EMBED_MODEL)


def get_llm_instance():
    from llama_index.llms.ollama import Ollama
    return Ollama(base_url=OLLAMA_BASE_URL, model=OLLAMA_LLM_MODEL, request_timeout=180.0)


def write_meta(index_path: str, meta: dict):
    os.makedirs(index_path, exist_ok=True)
    meta_file = os.path.join(index_path, "meta.json")
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def read_meta(index_path: str):
    meta_file = os.path.join(index_path, "meta.json")
    if not os.path.exists(meta_file):
        return None
    with open(meta_file, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_load_documents_from_folder(folder: str):
    skipped = []
    docs = []

    try:
        docs = SimpleDirectoryReader(folder).load_data()
        return docs, skipped
    except Exception as e:
        logger.warning("SimpleDirectoryReader failed: %s", e)

    for root, _, files in os.walk(folder):
        for fname in files:
            path = os.path.join(root, fname)
            if fname.lower().endswith(".txt"):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        text = f.read()
                    docs.append(Document(text, extra_info={"file_path": path}))
                except Exception as e2:
                    skipped.append({"file": path, "reason": str(e2)})
            else:
                skipped.append({"file": path, "reason": "unsupported extension"})
    return docs, skipped

# build
@app.post("/index/build")
def build_index(req: IndexRequest):
    folder = req.folder_path
    index_name = req.index_name

    if not os.path.isdir(folder):
        raise HTTPException(status_code=400, detail="folder_path is not a directory")

    index_path = os.path.join(STORAGE_DIR, index_name)

    # remove existing index to avoid leftovers
    if os.path.exists(index_path):
        shutil.rmtree(index_path)
        logger.info("Removed old index: %s", index_path)

    docs, skipped_files = safe_load_documents_from_folder(folder)
    if not docs:
        raise HTTPException(status_code=400, detail="No documents loaded.")

    embed_model = get_embedding_instance()
    index = VectorStoreIndex.from_documents(docs, embed_model=embed_model)
    index.storage_context.persist(index_path)

    meta = {
        "embedding": {"provider": "ollama", "model": OLLAMA_EMBED_MODEL},
        "llm_default": {"provider": "ollama", "model": OLLAMA_LLM_MODEL},
    }
    write_meta(index_path, meta)

    return {"status": "ok", "message": "Index created successfully.", "skipped": skipped_files}

# query
@app.post("/index/query")
def query_index(req: QueryRequest):
    index_path = os.path.join(STORAGE_DIR, req.index_name)
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="Index not found.")

    embed_model = get_embedding_instance()
    storage_context = StorageContext.from_defaults(persist_dir=index_path)
    index = load_index_from_storage(storage_context, embed_model=embed_model)
    logger.info("Index loaded: %s", index_path)

    meta = read_meta(index_path)
    if not meta:
        raise HTTPException(status_code=400, detail="meta.json missing")
    if meta["embedding"]["model"] != OLLAMA_EMBED_MODEL:
        raise HTTPException(status_code=400, detail="Embedding model mismatch. Rebuild required.")

    llm = get_llm_instance()
    query_engine = index.as_query_engine(llm=llm)
    response = query_engine.query(req.query)

    return {"status": "ok", "response": str(response)}

@app.get("/health")
def health():
    return {"status": "ok", "model": OLLAMA_LLM_MODEL}