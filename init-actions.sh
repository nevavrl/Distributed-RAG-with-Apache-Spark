#!/bin/bash
set -euxo pipefail

# -----------------------------
# Torch, Transformers ve Sentence-Transformers kurulumu (offline wheel ile)
# -----------------------------
mkdir -p /tmp/wheels
gsutil -m cp gs://my-wiki-bucket/torch_wheels/* /tmp/wheels/

# Gerekli paketleri offline kur (tf-keras hariç!)
pip install --no-index --find-links=/tmp/wheels \
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

echo "✅ Init action completed successfully."
