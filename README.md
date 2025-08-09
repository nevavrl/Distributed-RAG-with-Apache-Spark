# Distributed RAG with Apache Spark 🚀

A high-performance, distributed Retrieval-Augmented Generation (RAG) system built with Apache Spark, FAISS, and Ollama. This project demonstrates how to build a scalable RAG pipeline that can handle large-scale document processing and real-time question answering.

## 🌟 Features

- **Distributed Processing**: Leverages Apache Spark for parallel data processing and embedding generation
- **Efficient Vector Search**: Uses FAISS for fast similarity search across millions of document chunks
- **Local LLM Integration**: Integrates with Ollama for privacy-preserving text generation
- **Scalable Architecture**: Handles large datasets (Wikipedia) with configurable batch processing
- **Real-time Serving**: Socket-based server for real-time query processing
- **Multi-client Support**: Concurrent query handling with worker pool management

## 🏗️ Architecture

The system consists of several key components:

1. **Data Preprocessing**: Wikipedia dataset chunking with overlap
2. **Distributed Embedding**: Parallel embedding generation using Spark
3. **Vector Index Creation**: FAISS index building for fast retrieval
4. **RAG Pipeline**: Query processing with retrieval and generation
5. **Socket Server**: Real-time serving infrastructure

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Wikipedia     │────│   Data Chunking │────│   Distributed   │
│   Dataset       │    │   & Processing  │    │   Embedding     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Socket Server │────│   RAG Pipeline  │────│   FAISS Index   │
│   (Real-time)   │    │   (Retrieval)   │    │   Creation      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                       ┌─────────────────┐
                       │   Ollama LLM    │
                       │   (Generation)  │
                       └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Apache Spark 3.x
- Java 8/11
- Ollama (for text generation)
- CUDA GPU (optional, for faster processing)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/Distributed-RAG-with-Apache-Spark.git
cd Distributed-RAG-with-Apache-Spark
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Install and start Ollama**
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
ollama serve

# Pull the model
ollama pull llama3.2:3b
```

4. **Download embedding model**
```bash
# The system will automatically download sentence-transformers/all-MiniLM-L6-v2
# Or manually place it in ./models/ directory
```

### Running the Pipeline

#### Step 1: Data Preparation and Chunking
```bash
python data_chunking.py
```
This script:
- Downloads the Wikipedia dataset (20220301.en)
- Chunks documents with 512-word chunks and 64-word overlap
- Processes ~6.5M articles in batches of 100K
- Saves chunked data as JSONL files

#### Step 2: Distributed Embedding Generation
```bash
python spark_embedding.py
```
This script:
- Uses Apache Spark to distribute embedding generation across workers
- Each worker loads the sentence-transformer model locally
- Processes chunks in parallel and generates 384-dimensional embeddings
- Saves embeddings with metadata as JSON files

#### Step 3: FAISS Index Creation
```bash
python create_faiss_index.py
```
This script:
- Combines all embedding files into a single FAISS index
- Creates IndexFlatL2 for exact similarity search
- Saves the index and chunk texts for retrieval

#### Step 4: Start the RAG Server
```bash
python rag_server.py
```
Features:
- Socket-based server on localhost:8888
- Worker pool for concurrent query processing
- Real-time embedding generation and retrieval
- Integrates with Ollama for answer generation

#### Step 5: Connect with Client
```bash
python client.py
```
Interactive client for asking questions and receiving answers in real-time.

## 📁 Project Structure

```
Distributed-RAG-with-Apache-Spark/
├── README.md
├── requirements.txt
├── data_chunking.py              # Wikipedia data processing and chunking
├── spark_embedding.py            # Distributed embedding generation
├── create_faiss_index.py         # FAISS index creation
├── rag_server.py                 # Socket-based RAG server
├── client.py                     # Interactive client
├── inference_ollama.py           # Batch inference script
├── install_data.py              # Data installation utilities
├── chunk_output_with_embeddings/ # Embedded chunks

```

## ⚙️ Configuration

### Key Parameters

**Data Processing:**
- `BATCH_SIZE`: 100,000 (articles per batch)
- `chunk_size`: 512 words per chunk
- `overlap`: 64 words overlap between chunks

**Embedding:**
- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Dimension: 384
- Max sequence length: 128 tokens

**Retrieval:**
- `TOP_K`: 6 (retrieved chunks per query)
- Index type: FAISS IndexFlatL2 (exact search)

**Generation:**
- LLM: `llama3.2:3b` via Ollama
- Max tokens: 128
- Context window: 4096 tokens
- Temperature: 0.1

**Server:**
- Host: localhost:8888
- Max workers: 4
- Queue size: 100

## 📊 Performance

**Dataset Scale:**
- ~6.5M Wikipedia articles
- ~50M text chunks after processing
- ~19GB embedding data (384-dim vectors)

**Throughput:**
- Embedding generation: ~1000 chunks/second (4-core setup)
- Query processing: ~2-3 seconds per query
- Concurrent users: Up to 4 simultaneous queries

**Resource Usage:**
- RAM: 8-16GB recommended
- Storage: ~25GB for full Wikipedia pipeline
- CPU: Multi-core recommended for Spark

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 🙏 Acknowledgments

- **Hugging Face** for the transformers library and datasets
- **Facebook Research** for FAISS vector search
- **Apache Spark** for distributed computing framework
- **Ollama** for local LLM serving
- **Wikimedia** for the Wikipedia dataset

---
