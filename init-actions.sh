#!/bin/bash

# -----------------------------
# Torch ve Transformers kurulumu (GCS'ten .whl ile)
# -----------------------------

# GCS'ten indirilecek dizin
mkdir -p /opt/torch_cache

# GCS'ten tüm wheel dosyalarını indir
gsutil -m cp gs://my-wiki-bucket/torch_wheels/* /opt/torch_cache/

# Offline pip kurulumu (internet yokken bile çalışır)
pip install --no-index --find-links=/opt/torch_cache torch==2.3.0 transformers==4.41.1

# -----------------------------
# Modeli GCS'ten indir (embedding için)
# -----------------------------

mkdir -p /mnt/data/models
gsutil -m cp -r gs://my-wiki-bucket/models/all-MiniLM-L6-v2 /mnt/data/models/
