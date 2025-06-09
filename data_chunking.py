from datasets import load_dataset
import pandas as pd
from tqdm import tqdm

BATCH_SIZE = 100000

print("Wikipedia verisi indiriliyor...")
dataset = load_dataset("wikipedia", "20220301.en")["train"]

def chunk_text_overlap(text, chunk_size=512, overlap=64):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = ' '.join(words[i:i+chunk_size])
        if len(chunk.strip()) > 0:
            chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

total_records = len(dataset)
num_batches = (total_records + BATCH_SIZE - 1) // BATCH_SIZE

print(f"Toplam {num_batches} batch var. Her biri {BATCH_SIZE} kayıt içeriyor.\n")

for batch_idx in tqdm(range(num_batches), desc="Genel İlerleme", unit="batch"):
    start = batch_idx * BATCH_SIZE
    end = min((batch_idx + 1) * BATCH_SIZE, total_records)

    chunked_records = []
    for entry in tqdm(dataset.select(range(start, end)), desc=f"Chunklanıyor {start}-{end-1}", unit="makale", leave=False):
        title = entry['title']
        text = entry['text']
        chunks = chunk_text_overlap(text, chunk_size=512, overlap=64)
        for chunk in chunks:
            chunked_records.append({'title': title, 'chunk_text': chunk})

    df_chunks = pd.DataFrame(chunked_records)
    df_chunks.to_json(f"wikipedia_en_chunks_batch_{batch_idx}.jsonl", orient="records", lines=True)

print("Tüm batch’ler işlenip kaydedildi.")
