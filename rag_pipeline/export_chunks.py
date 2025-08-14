#!/usr/bin/env python3
"""
Export all chunks (text only) from each method into a single JSON file.
Output: rag_pipeline/output/chunks_by_method.json
"""
import os
import json
import logging

from src.qdrant_store import QdrantVectorStore


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    methods = [
        "character_chunking",
        "token_chunking",
        "semantic_chunking",
        "cluster_semantic_chunking",
        "llm_semantic_chunking",
        "layout_aware_chunking",
    ]

    store = QdrantVectorStore()
    out_simple = {}
    out_detailed = {}

    for method in methods:
        logging.info(f"Collecting chunks for method={method}")
        chunks = store.get_chunks_by_method(method)
        
        # Simple format (text only, backward compatibility)
        texts = [c["payload"].get("text", "") for c in chunks]
        out_simple[method] = texts
        
        # Detailed format with full metadata
        detailed_chunks = []
        for chunk in chunks:
            detailed_chunks.append({
                "id": chunk["id"],
                "method": chunk["payload"]["method"],
                "text": chunk["payload"]["text"],
                "chunk_type": chunk["payload"].get("chunk_type", "unknown"),
                "start_char": chunk["payload"].get("start_char", 0),
                "end_char": chunk["payload"].get("end_char", 0),
                "chunk_size": chunk["payload"].get("chunk_size", 0),
                "overlap": chunk["payload"].get("overlap", 0),
                "document": chunk["payload"].get("document", ""),
                "metadata": chunk["payload"].get("metadata", {})
            })
        out_detailed[method] = detailed_chunks
        
        logging.info(f"Collected {len(texts)} chunks for method={method}")

    # Resolve output under current module's directory (no nested rag_pipeline)
    here = os.path.abspath(os.path.dirname(__file__))
    out_dir = os.path.join(here, "output")
    os.makedirs(out_dir, exist_ok=True)
    
    # Write simple format
    out_path = os.path.join(out_dir, "chunks_by_method.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_simple, f, ensure_ascii=False, indent=2)
    print(f"Wrote simple chunks JSON to: {out_path}")
    
    # Write detailed format with metadata
    metadata_path = os.path.join(out_dir, "chunks_by_method_w_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(out_detailed, f, ensure_ascii=False, indent=2)
    print(f"Wrote detailed chunks JSON with metadata to: {metadata_path}")


if __name__ == "__main__":
    main()


