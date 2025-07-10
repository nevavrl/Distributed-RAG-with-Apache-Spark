#!/usr/bin/env python3
import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, FloatType

# TF disable
os.environ["TRANSFORMERS_NO_TF"] = "1"


def embed_partition(pdf_iter):
    # Model path (init-action ile indirildi)
    LOCAL_MODEL_PATH = "/mnt/data/models/all-MiniLM-L6-v2"

    # Model ve tokenizer her partition'da yüklenir
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH, local_files_only=True)
    model = AutoModel.from_pretrained(LOCAL_MODEL_PATH, local_files_only=True)
    model.eval()

    for pdf in pdf_iter:
        texts = pdf["chunk_text"].tolist()

        # Tokenize
        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=128
        )

        # Embed
        with torch.no_grad():
            embs = model(**enc).last_hidden_state.mean(dim=1)

        # Result as list of float arrays
        pdf["embedding"] = embs.cpu().numpy().tolist()
        yield pdf


def main():
    # Spark session init
    spark = (
        SparkSession.builder
        .appName("DistributedRAG-Embeddings")
        .config("spark.network.timeout", "1000s")
        .config("spark.executor.heartbeatInterval", "300s")
        .config("spark.executor.memory", "4g")             # ↓ düşürüldü
        .config("spark.executor.memoryOverhead", "1g")     # ↓ düşürüldü
        .config("spark.executor.cores", "4")
        .getOrCreate()
    )

    # GCS paths
    input_path = "gs://my-wiki-bucket/chunks/"
    output_path = "gs://my-wiki-bucket/embeddings_parquet/"

    # Belirli schema (örnek)
    schema = StructType([
        StructField("chunk_text", StringType(), True),
        StructField("chunk_id", StringType(), True)  # varsa ek alanlar da tanımla
    ])

    # Worker sayısına göre partisyon ayarla
    num_partitions = spark.sparkContext.defaultParallelism  # Örn: 3 worker * 8 core = 24

    # Load data
    df = spark.read.schema(schema).json(input_path).repartition(num_partitions)

    # Add embedding
    df_emb = df.mapInPandas(embed_partition, schema=schema.add("embedding", ArrayType(FloatType())))

    # Write to GCS
    df_emb.write.mode("overwrite").parquet(output_path)

    spark.stop()


if __name__ == "__main__":
    main()
