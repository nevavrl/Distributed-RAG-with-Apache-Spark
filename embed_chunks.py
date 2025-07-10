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
    # 1) Initialize Spark session with extended timeouts
    spark = (
        SparkSession.builder
        .appName("DistributedRAG-Embeddings")
        .config("spark.network.timeout", "800s")
        .config("spark.executor.heartbeatInterval", "60s")
        .getOrCreate()
    )

    # 2) Local model path (copied via init-action)
    LOCAL_MODEL_PATH = "/mnt/data/models/all-MiniLM-L6-v2"

    # 3) Load tokenizer & model from local directory, no remote calls
    tokenizer = AutoTokenizer.from_pretrained(
        LOCAL_MODEL_PATH,
        local_files_only=True
    )
    model = AutoModel.from_pretrained(
        LOCAL_MODEL_PATH,
        local_files_only=True
    )
    model.eval()

    # 4) Define pandas UDF for batch embedding
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

    # 5) GCS input/output paths
    input_path = "gs://my-wiki-bucket/chunks/"
    output_path = "gs://my-wiki-bucket/embeddings/"

    # 6) Determine number of partitions based on cluster cores
    num_partitions = spark.sparkContext.defaultParallelism

    # 7) Read JSONL, repartition, apply embedding, and write results
    df = spark.read.json(input_path).repartition(num_partitions)
    df_emb = df.withColumn("embedding", embed_batch(df.chunk_text))
    df_emb.write.mode("overwrite").json(output_path)

    # 8) Stop Spark
    spark.stop()

if __name__ == "__main__":
    main()
