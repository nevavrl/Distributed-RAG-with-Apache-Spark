#!/bin/bash

set -euxo pipefail  # Hataları erkenden yakalamak için

# -----------------------------
# Torch, Transformers ve Sentence-Transformers kurulumu (offline wheel ile)
# -----------------------------

mkdir -p /opt/torch_cache
gsutil -m cp gs://my-wiki-bucket/torch_wheels/* /opt/torch_cache/

# Gerekli paketleri offline kur (internet gerekmeden)
pip install --no-index --find-links=/opt/torch_cache \
    torch==2.3.0 \
    transformers==4.41.1 \
    sentence-transformers==2.7.0 \
    pandas \
    pyarrow

# -----------------------------
# Model klasörü: all-MiniLM-L6-v2
# -----------------------------

mkdir -p /mnt/data/models
gsutil -m cp -r gs://my-wiki-bucket/models/all-MiniLM-L6-v2 /mnt/data/models/

# -----------------------------
# Bilgi mesajı
# -----------------------------

echo "✅ Init action completed successfully."
