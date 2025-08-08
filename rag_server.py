# rag_socket_server.py
import os
import time
import json
import pickle
import numpy as np
import faiss
import torch
import socket
import threading
import queue
from datetime import datetime
from transformers import AutoTokenizer, AutoModel
from pyspark.sql import SparkSession
import requests
import logging

# =========================
# Logging
# =========================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("rag_socket_server")

# =========================
# Server Config
# =========================
SERVER_HOST = "localhost"
SERVER_PORT = 8888
MAX_QUEUE_SIZE = 100

# =========================
# Spark Session
# =========================
spark = (
    SparkSession.builder
        .appName("RAG_Socket_Server")
        .master("local[*]")
        .config("spark.driver.memory", "6g")
        .config("spark.python.worker.reuse", "false")
        .config("spark.ui.enabled", "true")
        .config("spark.ui.port", "4040")
        .getOrCreate()
)
sc = spark.sparkContext

# =========================
# RAG Config
# =========================
EMBEDDING_MODEL_NAME = "../models/all-MiniLM-L6-v2"
INDEX_PATH = "../faiss_index.bin"
CHUNK_TEXTS_PATH = "../chunk_texts.pkl"
TOP_K = 6

# Ollama
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2:3b"
MAX_NEW_TOKENS = 128
NUM_CTX = 4096
MAX_CONTEXT_CHARS = 12000

# =========================
# Load & Broadcast
# =========================
logger.info("Loading FAISS index and chunks...")
with open(CHUNK_TEXTS_PATH, "rb") as f:
    chunk_texts = pickle.load(f)

faiss_index = faiss.read_index(INDEX_PATH)
index_bytes = faiss.serialize_index(faiss_index)

bc_index_bytes = sc.broadcast(index_bytes)
bc_chunks = sc.broadcast(chunk_texts)
logger.info(f"Loaded {len(chunk_texts)} chunks and FAISS index")

# =========================
# Worker Pool (single-dispatch)
# =========================
class QueryRequest:
    def __init__(self, query_id, query_text, client_socket):
        self.query_id = query_id
        self.query_text = query_text
        self.client_socket = client_socket
        self.timestamp = time.time()
        self.status = "queued"  # queued | processing | completed | failed

class WorkerPool:
    def __init__(self, max_workers=4):
        self.max_workers = max_workers
        self.active_workers = {}      # query_id -> QueryRequest
        self.query_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
        self.completed_queries = {}
        self.query_counter = 0
        self.lock = threading.Lock()

    def add_query(self, query_text, client_socket):
        with self.lock:
            self.query_counter += 1
            query_id = f"query_{self.query_counter}_{int(time.time())}"

            if self.query_queue.qsize() >= MAX_QUEUE_SIZE:
                return None, "Queue is full"

            request = QueryRequest(query_id, query_text, client_socket)
            self.query_queue.put(request)

            has_free_worker = (self.max_workers - len(self.active_workers)) > 0
            logger.info(f"Query {query_id} enqueued. qsize={self.query_queue.qsize()}, active={len(self.active_workers)}/{self.max_workers}")

        # Boş worker varsa anında tekli dispatch
        if has_free_worker:
            self.process_one_if_available()

        return query_id, "Queued successfully"

    def get_available_worker_count(self):
        with self.lock:
            return self.max_workers - len(self.active_workers)

    def get_queue_status(self):
        with self.lock:
            return {
                "active_workers": len(self.active_workers),
                "max_workers": self.max_workers,
                "queue_size": self.query_queue.qsize(),
                "total_completed": len(self.completed_queries),
            }

    def process_one_if_available(self):
        """Boş worker varsa kuyruktan 1 sorgu çek ve hemen işle."""
        if self.get_available_worker_count() <= 0:
            return

        try:
            request = self.query_queue.get_nowait()
        except queue.Empty:
            return

        with self.lock:
            request.status = "processing"
            self.active_workers[request.query_id] = request

        logger.info(f"Dispatching {request.query_id} to a worker.")
        self._process_one_async(request)

    def _process_one_async(self, request: QueryRequest):
        def run():
            try:
                # Tek sorguyu Spark ile işle
                results = (
                    sc.parallelize([request.query_text], numSlices=1)
                      .mapPartitions(make_answers)
                      .collect()
                )
                query, answer, contexts, indices, distances = results[0]

                response = {
                    "query_id": request.query_id,
                    "query": query,
                    "answer": answer,
                    "contexts": contexts,
                    "processing_time": time.time() - request.timestamp,
                    "status": "completed"
                }
                self._send_response_to_client(request.client_socket, response)

                with self.lock:
                    request.status = "completed"
                    self.completed_queries[request.query_id] = request
                    self.active_workers.pop(request.query_id, None)

                logger.info(f"Completed {request.query_id}. active={len(self.active_workers)}/{self.max_workers}")

            except Exception as e:
                logger.error(f"Processing failed for {request.query_id}: {e}")
                error_response = {
                    "query_id": request.query_id,
                    "query": request.query_text,
                    "error": str(e),
                    "status": "failed"
                }
                self._send_response_to_client(request.client_socket, error_response)
                with self.lock:
                    request.status = "failed"
                    self.active_workers.pop(request.query_id, None)
            finally:
                # Worker boşaldı; sıradaki işi hemen çek
                self.process_one_if_available()

        t = threading.Thread(target=run, name=f"SingleJob-{request.query_id}", daemon=True)
        t.start()

    def _send_response_to_client(self, client_socket, response):
        try:
            response_json = json.dumps(response, ensure_ascii=False) + "\n"
            client_socket.send(response_json.encode('utf-8'))
        except Exception as e:
            logger.error(f"Failed to send response to client: {e}")

# Global worker pool
worker_pool = WorkerPool(max_workers=4)

# =========================
# RAG Functions
# =========================
_HTTP = None
_EMBED_TOK = None
_EMBED_MDL = None

def ensure_executor_init():
    global _HTTP, _EMBED_TOK, _EMBED_MDL
    if _HTTP is None:
        _HTTP = requests.Session()
        _HTTP.headers.update({"Content-Type": "application/json"})
    if _EMBED_TOK is None or _EMBED_MDL is None:
        _EMBED_TOK = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME, local_files_only=True)
        _EMBED_MDL = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME, local_files_only=True).to("cpu").eval()
    return _HTTP, _EMBED_TOK, _EMBED_MDL

def call_ollama_chat(session, system_prompt, user_prompt, timeout=180, retries=3):
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
            {"role": "user", "content": user_prompt}
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

def make_answers(partition):
    # Spark worker-side RAG
    local_index = faiss.deserialize_index(bc_index_bytes.value)
    chunks = bc_chunks.value
    
    http, embed_tok, embed_mdl = ensure_executor_init()
    
    with torch.inference_mode():
        for query in partition:
            # 1) Embedding
            enc = embed_tok(query, return_tensors="pt", truncation=True, max_length=128)
            out = embed_mdl(**enc)
            vec = out.last_hidden_state.mean(dim=1).squeeze().cpu().numpy().astype(np.float32)
            
            # 2) FAISS
            D, I = local_index.search(np.expand_dims(vec, 0), TOP_K)
            dists = D[0].tolist()
            idxs = I[0].tolist()
            ctxs = [chunks[i] for i in idxs]
            
            # 3) Prompt
            system_prompt, user_prompt = build_messages(query, ctxs)
            
            # 4) LLM
            answer = call_ollama_chat(http, system_prompt, user_prompt)
            
            yield (query, answer, ctxs, idxs, dists)

# =========================
# Socket Server
# =========================
class RAGSocketServer:
    def __init__(self, host=SERVER_HOST, port=SERVER_PORT):
        self.host = host
        self.port = port
        self.server_socket = None
        self.is_running = False
        self.client_threads = []

    def start(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        try:
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(10)
            self.is_running = True

            logger.info(f"RAG Server started on {self.host}:{self.port}")
            logger.info(f"Max workers: {worker_pool.max_workers}")
            logger.info("Spark Web UI: http://localhost:4040")

            while self.is_running:
                try:
                    client_socket, address = self.server_socket.accept()
                    logger.info(f"Client connected: {address}")

                    client_thread = threading.Thread(
                        target=self._handle_client,
                        args=(client_socket, address),
                        name=f"ClientHandler-{address}"
                    )
                    client_thread.daemon = True
                    client_thread.start()
                    self.client_threads.append(client_thread)

                except socket.error as e:
                    if self.is_running:
                        logger.error(f"Socket error: {e}")
                    break

        except Exception as e:
            logger.error(f"Server error: {e}")
        finally:
            self.stop()

    def _handle_client(self, client_socket, address):
        try:
            welcome = {
                "type": "welcome",
                "message": "Connected to RAG Server",
                "server_status": worker_pool.get_queue_status()
            }
            client_socket.send((json.dumps(welcome) + "\n").encode('utf-8'))

            while self.is_running:
                try:
                    data = client_socket.recv(4096).decode('utf-8')
                    if not data:
                        break

                    data = data.strip()
                    low = data.lower()
                    if low in ['quit', 'exit', 'bye']:
                        break

                    if low == 'status':
                        status = worker_pool.get_queue_status()
                        response = {"type": "status", "data": status}
                        client_socket.send((json.dumps(response) + "\n").encode('utf-8'))
                        continue

                    # Sorgu ekle -> boş worker varsa anında ateşlenir
                    query_id, message = worker_pool.add_query(data, client_socket)

                    if query_id:
                        response = {
                            "type": "query_accepted",
                            "query_id": query_id,
                            "message": message,
                            "queue_status": worker_pool.get_queue_status()
                        }
                    else:
                        response = {
                            "type": "query_rejected",
                            "message": message
                        }

                    client_socket.send((json.dumps(response) + "\n").encode('utf-8'))

                except socket.timeout:
                    continue
                except Exception as e:
                    logger.error(f"Client handling error: {e}")
                    break

        except Exception as e:
            logger.error(f"Client connection error: {e}")
        finally:
            try:
                client_socket.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
            client_socket.close()
            logger.info(f"Client disconnected: {address}")

    def stop(self):
        self.is_running = False
        if self.server_socket:
            try:
                self.server_socket.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
            self.server_socket.close()
        logger.info("Server stopped")

if __name__ == "__main__":
    print("RAG Socket Server")
    print("=" * 50)
    print(f"Host: {SERVER_HOST}")
    print(f"Port: {SERVER_PORT}")
    print(f"Max Workers: {worker_pool.max_workers}")
    print(f"Queue Size: {MAX_QUEUE_SIZE}")
    print("=" * 50)

    # Ollama check
    try:
        response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        if response.status_code == 200:
            print("Ollama connection OK")
        else:
            print("Ollama connection issues")
    except Exception:
        print("Ollama not available")
        print("   Start Ollama: ollama serve")
        spark.stop()
        exit(1)

    print("\nStarting server...")

    server = RAGSocketServer()
    try:
        server.start()
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received")
    except Exception as e:
        print(f"Server error: {e}")
    finally:
        server.stop()
        spark.stop()
        print("Server shutdown complete")
