from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, FloatType

from transformers import AutoTokenizer, AutoModel
import torch

spark = SparkSession.builder.appName("embedding_udf").getOrCreate()

model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()
    return embeddings.tolist()

embedding_udf = udf(get_embedding, ArrayType(FloatType()))

#file_paths = [
#    "/Users/neva/Projects/DistributedRAG/Distributed-RAG-with-Apache-Spark/chunk_output/wikipedia_en_chunks_batch_0.jsonl",
#    "/Users/neva/Projects/DistributedRAG/Distributed-RAG-with-Apache-Spark/chunk_output/wikipedia_en_chunks_batch_1.jsonl",
#    "/Users/neva/Projects/DistributedRAG/Distributed-RAG-with-Apache-Spark/chunk_output/wikipedia_en_chunks_batch_2.jsonl"
#]
# 1. JSONL dosyasını oku
#df = spark.read.json(file_paths)
df = spark.read.json("/Users/neva/Projects/DistributedRAG/Distributed-RAG-with-Apache-Spark/chunk_output/")

# 2. Metin sütunu adını kontrol et, örn: 'text'
df.printSchema()
# Burada text sütunu varsa:
# 3. Embedding çıkar
df_with_emb = df.withColumn("embedding", embedding_udf(df.chunk_text))

#df_with_emb.show(truncate=False)
df_with_emb.write.mode("overwrite").json("output_with_embeddings_json")


