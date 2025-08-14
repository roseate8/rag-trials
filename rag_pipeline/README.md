# 🎯 Vector Storage & Chunking Tool

A clean, simple tool to test different chunking methods on documents using vector storage and LLM queries.

## 📁 Files Overview

### 🔧 **Essential Files (Use These)**
- **`setup_and_chunk.py`** - Run this ONCE to set up your vector database
- **`query.py`** - Use this to manually test LLM queries  
- **`conversation_log.json`** - Your query results are saved here

### ⚙️ **Core Implementation (Don't modify)**
- **`src/advanced_chunkers/layout_aware_chunker.py`** - Layout-aware chunking method
- **`src/llm_query.py`** - Streamlined LLM query system
- **`src/qdrant_store.py`** - Vector database interface
- **`src/embeddings.py`** - Embedding generation
- **`src/llm_table_classifier.py`** - Intelligent table title generation
- **`requirements.txt`** - Python dependencies

## 🚀 Quick Start

### 1. First Time Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Start Qdrant vector database (in background)
docker run -p 6333:6333 -d qdrant/qdrant

# Set up vector database with all chunking methods (run once)
python3 setup_and_chunk.py
```

### 2. Query the Database
```bash
# Navigate to the directory first
cd rag_pipeline

# Ask questions about the document
python3 query.py "Tell me about the EPS this year"
python3 query.py "What are the revenue figures?"  
python3 query.py "Show me information about iPhone sales"
```

## 🎯 Layout-Aware Chunking (Optimized Method)

**Single, Intelligent Chunking Method:**
- **Layout-Aware Chunking** - Respects document structure (titles, sections, tables)
- **JSON Enrichment** - Enhanced table processing with spatial context
- **LLM Table Classification** - Intelligent table titles using GPT-4o-mini
- **Multi-format Support** - Handles MD+JSON, MD-only, and JSON-only documents

## ✅ Performance Results

For the query **"Tell me about the EPS this year"**:

| Method | Result | Performance |
|--------|---------|-------------|
| **Layout-Aware Chunking** | ✅ **Found: Basic EPS $6.11, Diluted EPS $6.08** | 🚀 **Best accuracy, fastest, most cost-effective** |

## 💡 Why Layout-Aware Chunking?

- **🎯 Best Accuracy**: Respects document structure and table semantics
- **⚡ Fastest**: Single optimized method vs multiple methods
- **💰 Cost-Effective**: No redundant processing across multiple chunkers
- **🔧 Maintainable**: Clean, focused codebase

## 📈 Example Query Output

```bash
$ python3 query.py "Tell me about the EPS this year"

🔍 Initializing LLM Query System...
📝 Query: "Tell me about the EPS this year"
⏰ Time: 2025-08-13 15:30:12

🔸 LAYOUT-AWARE CHUNKING RESULTS
--------------------------------------------------
📊 Retrieved chunks: 5
📏 Context length: 2,341 characters
🎯 Tokens used: 487
🔍 Top similarity score: 0.8234
🎯 Top rerank score: 0.9121
📋 Chunk types found: {'table': 3, 'paragraph': 2}

💬 LLM Response:
For the fiscal year ended September 28, 2024, the earnings per share (EPS) figures are as follows:
- Basic EPS: $6.11
- Diluted EPS: $6.08
```

## 🗃️ File Structure
```
rag_pipeline/
├── README.md                 # This file
├── setup_and_chunk.py        # Setup script (run once)
├── query.py                  # Query interface (use this)
├── conversation_log.json     # Your query results
├── requirements.txt          # Dependencies
└── src/
    ├── chunking.py           # All chunking methods
    ├── llm_query.py          # LLM query system
    ├── qdrant_store.py       # Vector database
    └── embeddings.py         # Embeddings
```
