import os
import json
import glob
import numpy as np
import faiss
import pickle

# embedding_dir = "/Users/neva/Projects/DistributedRAG/Distributed-RAG-with-Apache-Spark/chunk_output_with_embeddings"
embedding_dir = "chunk_output_with_embeddings"  # JSONL dosyalarının bulunduğu klasör
embedding_dim = 384  # all-MiniLM-L6-v2 için sabit
index_path = "faiss_index.bin"
chunk_texts_path = "chunk_texts.pkl"

index = faiss.IndexFlatL2(embedding_dim)
chunk_texts = []

json_files = glob.glob(os.path.join(embedding_dir, "*.json"))
print(f"{len(json_files)} dosya bulundu. İşleniyor...")

for file_path in json_files:
    with open(file_path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            obj = json.loads(line.strip())
            if not obj.get("embedding") or not obj.get("chunk_text"):
                continue

            vec = np.array(obj["embedding"], dtype=np.float32)
            index.add(np.expand_dims(vec, axis=0))
            chunk_texts.append(obj["chunk_text"])

print(f"\nIndex oluşturuldu: {index.ntotal} vektör")

faiss.write_index(index, index_path)
print(f"FAISS index kaydedildi: {index_path}")

with open(chunk_texts_path, "wb") as f:
    pickle.dump(chunk_texts, f)
print(f"chunk_texts kaydedildi: {chunk_texts_path}")
