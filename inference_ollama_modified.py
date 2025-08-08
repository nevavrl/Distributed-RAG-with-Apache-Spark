import os
import time
import json
import pickle
import numpy as np
import faiss
import torch
import pandas as pd
import requests
from transformers import AutoTokenizer, AutoModel
from pyspark.sql import SparkSession

# =========================================================
# Spark Session
# =========================================================
spark = (
    SparkSession.builder
        .appName("RAG_Spark_Parallel_Ollama_STRICT")
        .master("local[*]")
        .config("spark.driver.memory", "6g")
        .config("spark.python.worker.faulthandler.enabled", "true")
        .config("spark.sql.execution.pyspark.udf.faulthandler.enabled", "true")
        .config("spark.python.worker.reuse", "false")
        .config("spark.ui.enabled", "true")
        .config("spark.ui.port", "4040")
        .config("spark.ui.host", "0.0.0.0")
        .config("spark.sql.ui.retainedExecutions", "1000")
        .config("spark.ui.retainedJobs", "1000")
        .config("spark.ui.retainedStages", "1000")
        .config("spark.ui.retainedTasks", "100000")
        .getOrCreate()
)
sc = spark.sparkContext

# =========================================================
# Configs
# =========================================================
EMBEDDING_MODEL_NAME = "models/all-MiniLM-L6-v2"
INDEX_PATH           = "faiss_index.bin"
CHUNK_TEXTS_PATH     = "chunk_texts.pkl"
TOP_K                = 6

# Ollama (local)
OLLAMA_HOST  = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2:3b"
MAX_NEW_TOKENS = 128
NUM_CTX        = 4096
MAX_CONTEXT_CHARS = 12000

print("Running: CPU for embeddings, Ollama(chat) for generation (STRICT: context-only)")

# =========================================================
# Load & Broadcast (index -> BYTES!)
# =========================================================
"""
Load FAISS index and text chunks, serialize index bytes, and broadcast both.
"""
with open(CHUNK_TEXTS_PATH, "rb") as f:
    chunk_texts = pickle.load(f)

faiss_index = faiss.read_index(INDEX_PATH)
index_bytes = faiss.serialize_index(faiss_index)

bc_index_bytes = sc.broadcast(index_bytes)
bc_chunks      = sc.broadcast(chunk_texts)

# =========================================================
# Helpers
# =========================================================
_HTTP = None
_EMBED_TOK = None
_EMBED_MDL = None

def ensure_executor_init():
    """
    Initialize and return HTTP session, tokenizer, and embedding model on executor.
    """
    global _HTTP, _EMBED_TOK, _EMBED_MDL
    if _HTTP is None:
        _HTTP = requests.Session()
        _HTTP.headers.update({"Content-Type": "application/json"})
    if _EMBED_TOK is None or _EMBED_MDL is None:
        _EMBED_TOK = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME, local_files_only=True)
        _EMBED_MDL = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME, local_files_only=True).to("cpu").eval()
    return _HTTP, _EMBED_TOK, _EMBED_MDL

def call_ollama_chat(session, system_prompt, user_prompt, timeout=180, retries=3):
    """
    Call local Ollama chat API and return generated content.
    """
    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "options": {
            "num_predict": MAX_NEW_TOKENS,
            "num_ctx": NUM_CTX,
            "temperature": 0.1,
            "repeat_penalty": 1.05
        },
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
        ]
    }
    url = f"{OLLAMA_HOST}/api/chat"
    for attempt in range(retries):
        try:
            r = session.post(url, json=payload, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            return (data.get("message", {}) or {}).get("content", "").strip()
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(1.5 * (attempt + 1))

def build_messages(query, ctxs):
    """
    Construct system and user prompts using retrieved contexts.
    Ensures contexts are unique and within character limit.
    """
    ctxs = list(dict.fromkeys(ctxs))
    joined = "\n\n".join(ctxs)
    if len(joined) > MAX_CONTEXT_CHARS:
        joined = joined[:MAX_CONTEXT_CHARS]

    system = (
        "You are a concise assistant. Answer ONLY using the provided context. "
        "If the answer is not explicitly in the context, reply exactly: \"I don't know.\" "
        "Do not use outside knowledge. Do not guess. Do not add extra details."
    )
    user = (
        f"--- CONTEXT START ---\n{joined}\n--- CONTEXT END ---\n\n"
        f"Question: {query}\nAnswer:"
    )
    return system, user

# =========================================================
# Partition: Embed -> FAISS -> Prompt -> Ollama(chat)
# =========================================================
def make_answers(partition):
    local_index = faiss.deserialize_index(bc_index_bytes.value)
    chunks = bc_chunks.value

    print("===== embedding model load time======")
    start = time.time()

    http, embed_tok, embed_mdl = ensure_executor_init()
    end = time.time()
    print(f"Singleton init took {end - start:.2f} seconds")

    with torch.inference_mode():
        for query in partition:
            # 1) Embedding
            enc = embed_tok(query, return_tensors="pt", truncation=True, max_length=128)
            out = embed_mdl(**enc)
            vec = out.last_hidden_state.mean(dim=1).squeeze().cpu().numpy().astype(np.float32)

            # 2) FAISS
            D, I = local_index.search(np.expand_dims(vec, 0), TOP_K)
            dists = D[0].tolist()
            idxs  = I[0].tolist()
            ctxs  = [chunks[i] for i in idxs]

            # 3) Messages (STRICT)
            system_prompt, user_prompt = build_messages(query, ctxs)

            # 4) Ollama(chat)
            print("Answer generation time:")
            start = time.time()
            answer = call_ollama_chat(http, system_prompt, user_prompt)
            end = time.time()
            print(f"  Ollama(chat) took {end - start:.2f} seconds")

            # 5) Yield result
            yield (query, answer, ctxs, idxs, dists)

# =========================================================
# Example queries
# =========================================================
queries = [
    "When was Warfield Church built?",
    "What happened to the vicarage in 1843?",
    "Who designed the church restoration in 1874?",
    "What is the Swedish Civil Defence Board?",
    "Who is Shota Saito?",
    "What team does the Japanese footballer play for?",
    "When was the 2016-17 BIBL tournament held?",
    "Who won the 1968 BBC Farewell Spectacular?",
    "What disciplines are in figure skating championships?",
    "Where is the Japan Maritime Self Defence Force Museum?",
    "Who is Gérard Troupeau?",
    "When was the 1980 Cork Senior Hurling Championship?",
    "What position does Daisuke Sakai play?",
    "Who is Stephen Goss?",
    "What instrument does Stephen Goss play?",
    "When was St. Finbarr's championship victory?",
    "What is the Iron Whale Museum?",
    "How many teams participated in the 2016-17 BIBL?",
    "What happened during the Protestant reformation at Warfield?",
    "Who directed the Swedish Civil Defence Board?"
]

# =========================================================
# Run
# =========================================================
num_slices = min(len(queries), 10)
print(f"========= NUM SLICES: {num_slices} =========")
raw = sc.parallelize(queries, numSlices=num_slices) \
        .mapPartitions(make_answers) \
        .collect()

# =========================================================
# Save logs: JSONL + CSV
# =========================================================
os.makedirs("logs", exist_ok=True)

# JSONL
jsonl_path = "logs/retrieval_ollama.jsonl"
with open(jsonl_path, "w", encoding="utf-8") as f:
    for (q, a, ctxs, idxs, dists) in raw:
        rec = {"query": q, "answer": a, "contexts": ctxs, "indices": idxs, "distances": dists}
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

# CSV
rows = []
for (q, a, ctxs, idxs, dists) in raw:
    rows.append({
        "query": q,
        "answer": a,
        "topk": len(ctxs),
        "indices": "|".join(map(str, idxs)),
        "distances": "|".join(f"{d:.4f}" for d in dists),
        "context_concat": "\n\n---\n\n".join(ctxs)[:2000]  
    })
df = pd.DataFrame(rows)
csv_path = "logs/retrieval_ollama.csv"
df.to_csv(csv_path, index=False, encoding="utf-8")

# =========================================================
# Output results
# =========================================================
print(df[["query","topk"]].to_string(index=False))
print(f"\nSaved logs:\n  JSONL: {jsonl_path}\n  CSV  : {csv_path}")
