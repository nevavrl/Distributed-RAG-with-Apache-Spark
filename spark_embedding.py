from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, FloatType
import os

from transformers import AutoTokenizer, AutoModel
import torch

# spark session baslatiyoruz
spark = SparkSession.builder \
    .appName("embedding_with_mappartitions_loop") \
    .config("spark.hadoop.mapreduce.fileoutputcommitter.marksuccessfuljobs", "false") \
    .config("spark.sql.sources.commitProtocolClass", "org.apache.spark.internal.io.HadoopMapReduceCommitProtocol") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .getOrCreate()


input_dir = "/Users/neva/Projects/DistributedRAG/Distributed-RAG-with-Apache-Spark/chunk_output"
output_dir = "/Users/neva/Projects/DistributedRAG/Distributed-RAG-with-Apache-Spark/chunk_output_with_embeddings"

# JSON verisinin beklenen semasi (chunk_id ekleyecegiz)
schema = StructType([
    StructField("chunk_id", StringType(), True),
    StructField("chunk_text", StringType(), True),
])

#artik worker basina model yukleyen embedding fonksiyonumuz var, oncesinde driver tarafinda tek bir kez model vardi
def embed_partition(partition):
    from transformers import AutoTokenizer, AutoModel
    import torch

    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model.eval()

    for row in partition:
        try:
            inputs = tokenizer(row['chunk_text'], return_tensors="pt", truncation=True, max_length=128)
            with torch.no_grad():
                outputs = model(**inputs)
                emb = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
            yield (row['chunk_id'], row['chunk_text'], emb)
        except Exception as e:
            print(f"Embedding hatasi: {e}")
            yield (row['chunk_id'], row['chunk_text'], [])

num_batches = 65


for i in reversed(range(num_batches)):
    input_path = os.path.join(input_dir, f"wikipedia_en_chunks_batch_{i}.jsonl")
    output_path = os.path.join(output_dir, f"batch_{i}_with_embeddings.json")
    print(f"[{i}] Batch isleniyor: {input_path}")

    try:
        df = spark.read.schema(schema).json(input_path)

        # RDD üzerinden mapPartitions ile model yüklemesini workera aktar
        rdd = df.rdd.mapPartitions(embed_partition)

        # RDD’yi tekrar DataFrame’e cevirir
        result_df = spark.createDataFrame(rdd, schema=StructType([
            StructField("chunk_id", StringType(), True),
            StructField("chunk_text", StringType(), True),
            StructField("embedding", ArrayType(FloatType()), True),
        ]))

        # JSON klasörü olarak yaz
        result_df.write.mode("overwrite").json(output_path)
        print(f"[{i}] Tamamlandi: {output_path}")

    except Exception as e:
        print(f"[{i}] CODE_ERROR: {e}")
