import os
import pickle
import numpy as np
import faiss
import torch
from transformers import AutoTokenizer, AutoModel
from concurrent.futures import ThreadPoolExecutor
import openai

# --- Config ---
INDEX_PATH = "faiss_index.bin"
CHUNK_TEXTS_PATH = "chunk_texts.pkl"
EMBEDDING_DIM = 384
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MAX_CONTEXT_TOKENS = 1024
MAX_COMPLETION_TOKENS = 256

openai.api_key = "EMPTY"
openai.api_base = "http://localhost:8000/v1"

# --- Load FAISS index and chunk texts ---
print("Loading FAISS index and chunks...")
index = faiss.read_index(INDEX_PATH)
with open(CHUNK_TEXTS_PATH, "rb") as f:
    chunk_texts = pickle.load(f)
print(f"Loaded index with {index.ntotal} vectors and {len(chunk_texts)} text chunks.")


# --- Load embedding model once ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Embedding model is running on: {device}")
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(device)
model.eval()


def embed_query(text: str) -> np.ndarray:
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return embedding.astype(np.float32)


def retrieve_chunks(query: str, k=5):
    emb = embed_query(query)
    D, I = index.search(np.expand_dims(emb, axis=0), k)
    return [chunk_texts[i] for i in I[0]]


def get_answer(query: str):
    try:
        chunks = retrieve_chunks(query)
        context = "\n\n".join(chunks)

        prompt = f"Answer the question based on the context below.\n\nQuestion: {query}\n\nContext:\n{context}\n\nAnswer:"

        response = openai.ChatCompletion.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=MAX_COMPLETION_TOKENS,
        )
        answer = response["choices"][0]["message"]["content"]
        return query, answer
    except Exception as e:
        return query, f"[Error] {str(e)}"


# --- Queries ---
queries = [
    "Who is Irene Marie Watler?",
    "Who is Harry Douglas?",
    "Who is the Elly's father?",
    "When was held United States House of Representatives elections in Florida?",
    "Who is Ahmet Refik Erduran?",
]


# --- Run in parallel ---
print("\nRunning multi-query test...\n")
with ThreadPoolExecutor(max_workers=5) as executor:
    futures = [executor.submit(get_answer, q) for q in queries]
    for future in futures:
        q, a = future.result()
        print(f"\nQ: {q}\nA: {a}\n{'-'*60}")
