import os
import pickle
import numpy as np
import faiss
import torch
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM
)
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType, ArrayType, FloatType

# ---------------------------
# SPARK SESSION
# ---------------------------
spark = SparkSession.builder \
    .appName("RAG_SingleQuery_Spark") \
    .master("local[*]") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

# ---------------------------
# CONFIGS
# ---------------------------
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
INDEX_PATH = "faiss_index.bin"
CHUNK_TEXTS_PATH = "chunk_texts.pkl"
EMBEDDING_DIM = 384
TOP_K = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

# ---------------------------
# LOAD MODELS ON DRIVER
# ---------------------------
embed_tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
embed_model = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME).to(device).eval()

llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
llm_model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_NAME).to(device).eval()

# ---------------------------
# UDF-FRIENDLY FAISS LOADER
# ---------------------------
def load_faiss_and_chunks():
    global faiss_index, chunk_texts
    if "faiss_index" not in globals():
        faiss_index = faiss.read_index(INDEX_PATH)
        with open(CHUNK_TEXTS_PATH, "rb") as f:
            chunk_texts = pickle.load(f)

# ---------------------------
# EMBEDDING UDF
# ---------------------------
def embed_query(query_text):
    with torch.no_grad():
        inputs = embed_tokenizer(query_text, return_tensors="pt", truncation=True, max_length=128).to(device)
        outputs = embed_model(**inputs)
        query_emb = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return query_emb.astype(np.float32).tolist()

# ---------------------------
# RETRIEVAL UDF (FAISS)
# ---------------------------
def retrieve_chunks(query_emb):
    load_faiss_and_chunks()
    vec = np.expand_dims(np.array(query_emb, dtype=np.float32), axis=0)
    D, I = faiss_index.search(vec, TOP_K)
    return [chunk_texts[i] for i in I[0]]

# ---------------------------
# PROMPT BUILDER
# ---------------------------
def build_prompt(context_chunks, query):
    context = "\n\n".join(context_chunks)
    return (
        f"You are a helpful assistant. Use the following context to answer the question.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\nAnswer:"
    )

# ---------------------------
# GENERATE RESPONSE
# ---------------------------
def generate_answer(prompt):
    inputs = llm_tokenizer(prompt, return_tensors="pt").to(device)
    output = llm_model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.7,
        do_sample=True,
        pad_token_id=llm_tokenizer.eos_token_id
    )
    response = llm_tokenizer.decode(output[0], skip_special_tokens=True)
    return response[len(prompt):].strip()

# ---------------------------
# REGISTER UDFs
# ---------------------------
embed_udf = udf(embed_query, ArrayType(FloatType()))
retrieve_udf = udf(retrieve_chunks, ArrayType(StringType()))
prompt_udf = udf(build_prompt, StringType())
generate_udf = udf(generate_answer, StringType())

# ---------------------------
# TEST INPUTS
# ---------------------------
queries = [
    ("Who is Irene Marie Watler?",),
    ("What is the capital of France?",),
    ("Tell me about the Great Wall of China.",),
    ("When was the Declaration of Independence signed?",),
    ("What is the function of mitochondria?",)
]
df = spark.createDataFrame(queries, ["query"])

# ---------------------------
# PIPELINE
# ---------------------------
df_with_embeddings = df.withColumn("embedding", embed_udf("query"))
df_with_chunks = df_with_embeddings.withColumn("chunks", retrieve_udf("embedding"))
df_with_prompt = df_with_chunks.withColumn("prompt", prompt_udf("chunks", "query"))
df_with_response = df_with_prompt.withColumn("answer", generate_udf("prompt"))

# ---------------------------
# OUTPUT
# ---------------------------
df_with_response.select("query", "answer").show(truncate=False)
