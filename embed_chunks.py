#!/usr/bin/env python3
import os
# Disable TensorFlow components in transformers to avoid tf-keras issues
os.environ["TRANSFORMERS_NO_TF"] = "1"

from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import ArrayType, FloatType
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel


def main():
    # Initialize Spark with tuned memory and core settings
    spark = (
        SparkSession.builder
        .appName("DistributedRAG-Embeddings")
        .config("spark.network.timeout", "800s")
        .config("spark.executor.heartbeatInterval", "60s")
        .config("spark.executor.memory", "5g")            # Executor başına 5GB RAM
        .config("spark.yarn.executor.memoryOverhead", "512")  # Overhead 512MB
        .config("spark.executor.cores", "4")               # Executor başına 4 çekirdek
        .getOrCreate()
    )

    # Model path from init-action
    LOCAL_MODEL_PATH = "/mnt/data/models/all-MiniLM-L6-v2"

    # Load tokenizer and model locally
    tokenizer = AutoTokenizer.from_pretrained(
        LOCAL_MODEL_PATH,
        local_files_only=True
    )
    model = AutoModel.from_pretrained(
        LOCAL_MODEL_PATH,
        local_files_only=True
    )
    model.eval()

    @pandas_udf(ArrayType(FloatType()))
    def embed_batch(texts: pd.Series) -> pd.Series:
        enc = tokenizer(
            texts.tolist(),
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=128
        )
        with torch.no_grad():
            embs = model(**enc).last_hidden_state.mean(dim=1)
        return pd.Series(embs.cpu().numpy().tolist())

    # GCS paths
    input_path = "gs://my-wiki-bucket/chunks/"
    output_path = "gs://my-wiki-bucket/embeddings_parquet/"

    # Partition count = total vCPU in cluster
    num_partitions = spark.sparkContext.defaultParallelism

    # Read, embed, write as Parquet
    df = spark.read.json(input_path).repartition(num_partitions)
    df_emb = df.withColumn("embedding", embed_batch(df.chunk_text))
    df_emb.write.mode("overwrite").parquet(output_path)

    spark.stop()


if __name__ == "__main__":
    main()

