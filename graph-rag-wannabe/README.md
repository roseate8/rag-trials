# Graph-RAG Wannabe: 2-Hop Metadata-Driven Search

A lightweight implementation that simulates Graph-RAG behavior using vector search with intelligent metadata-driven expansions.

## Core Concept

Instead of building a graph database, we use **2-hop metadata-driven searches** within the vector store:

1. **Seed Search**: Initial vector search to find core relevant chunks
2. **Signal Extraction**: Collect topical signals from metadata (jira_ids, release_versions, metric_terms, doc_refs)
3. **Constrained Expansion**: Re-query with filters based on extracted signals
4. **Assembly**: Merge, dedupe, and rerank results to create a "trail"

## Architecture

```
Query Input
    ↓
Intent Classification (explain | numeric_evidence | lookup)
    ↓
Recipe Selection
    ↓ 
2-Hop Search Strategy
    ↓
[Pass 1: Seed Search] → [Signal Extraction] → [Pass 2: Filtered Expansion]
    ↓
Assembly & Reranking
    ↓
Response with Provenance Trail
```

## Features

### 🔍 **Query Types Supported**
- **Explain Queries**: "Why did revenue drop?" → 2-hop with causal signals
- **Numeric Evidence**: "What was Q4 revenue?" → Table-aware search
- **Relationship**: "Impact of release 24.7" → Version-filtered expansion

### 🎯 **Metadata Signals Used**
- `metric_terms`: Financial/business metrics
- `doc_refs`: Cross-references (Table X, Figure Y)
- `release_versions`: Software/product versions  
- `mentioned_dates`: Temporal context
- `entities`: Product/component names
- `jira_ids`: Issue tracking references

### 🔗 **2-Hop Methodology**
1. **Seed Search**: Vector similarity for core intent
2. **Signal Collection**: Extract metadata patterns from top results
3. **Expansion Search**: Re-query with metadata filters
4. **Trail Assembly**: Create coherent response path

## Quick Start

```python
from graph_rag_wannabe import GraphRAGWannabe

# Initialize with existing vector store
wannabe = GraphRAGWannabe(vector_store, api_key)

# Query with automatic 2-hop expansion
result = wannabe.query("Why did revenue drop in Q4?")

print(f"Answer: {result.answer}")
print(f"Trail: {result.provenance_trail}")
print(f"Hops: {len(result.hops)} expansions performed")
```

## Directory Structure

```
graph-rag-wannabe/
├── src/
│   ├── query_routing/          # Intent classification
│   ├── hop_recipes/           # 2-hop search strategies  
│   ├── metadata_extraction/   # Signal extraction logic
│   └── response_building/     # Trail assembly
├── tests/                     # Unit and integration tests
├── examples/                  # Usage examples
└── docs/                     # Documentation
```
