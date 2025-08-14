#!/usr/bin/env python3
"""
Layout-aware document chunker for multiple input files
Process all files in the input directory using layout-aware chunking only
"""

import logging
import os
import glob
from pathlib import Path
from typing import List, Dict, Any
from src.qdrant_store import QdrantVectorStore
from src.embeddings import EmbeddingGenerator
from src.advanced_chunkers import LayoutAwareChunker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def main():
    """Set up vector database with layout-aware chunking for all input files"""
    
    # Input directory containing files to process
    input_dir = "/Users/rudram.piplad/Documents/rag-trials/input"
    
    # Initialize components
    vector_store = QdrantVectorStore()
    embedding_generator = EmbeddingGenerator()
    
    # Clean up and create new collection
    logger.info("Setting up vector database...")
    vector_store.clean_database()
    vector_store.create_collection()
    
    # OpenAI API key for LLM table classification
    api_key = "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"
    
    # Initialize the original layout-aware chunker (handles all scenarios automatically)
    layout_chunker = LayoutAwareChunker(max_words=300, min_words=15, openai_api_key=api_key)
    
    # Find all files in input directory (both MD and JSON)
    md_files = glob.glob(os.path.join(input_dir, "*.md"))
    json_files = glob.glob(os.path.join(input_dir, "*.json"))
    
    # Create a mapping of base filenames to their file paths
    file_mapping = {}
    for md_path in md_files:
        base_name = Path(md_path).stem
        if base_name not in file_mapping:
            file_mapping[base_name] = {}
        file_mapping[base_name]['md'] = md_path
    
    for json_path in json_files:
        base_name = Path(json_path).stem
        if base_name not in file_mapping:
            file_mapping[base_name] = {}
        file_mapping[base_name]['json'] = json_path
    
    if not file_mapping:
        logger.error(f"No files found in {input_dir}")
        return
    
    all_chunks = []
    processed_files = set()  # Track processed files to avoid duplicates
    document_token_usage = {}  # Track tokens per document
    
    logger.info(f"Found {len(file_mapping)} unique documents in {input_dir}")
    for base_name, files in file_mapping.items():
        logger.info(f"  {base_name}: MD={files.get('md') is not None}, JSON={files.get('json') is not None}")
    
    # Process each unique document
    for base_name, files in file_mapping.items():
        # Skip if we've already processed this document
        if base_name in processed_files:
            logger.info(f"Skipping duplicate: {base_name}")
            continue
        
        # The original LayoutAwareChunker handles MD+JSON auto-detection automatically!
        # Just process any available MD files - JSON enrichment happens automatically
        if files.get('md'):
            primary_file = files['md']
            logger.info(f"Processing document '{base_name}' using MD file: {os.path.basename(primary_file)}")
            if files.get('json'):
                logger.info(f"  -> JSON file available for auto-enrichment: {os.path.basename(files['json'])}")
            else:
                logger.info(f"  -> MD-only processing (JSON auto-detection will show warning if needed)")
        else:
            # Skip JSON-only files for now (would need minimal modification to original chunker)
            logger.warning(f"Skipping JSON-only file (no markdown available): {base_name}")
            continue
        
        logger.info(f"Processing document: {base_name}")
        
        try:
            # Track tokens before processing this document
            tokens_before = layout_chunker.llm_classifier.total_tokens_used if layout_chunker.llm_classifier else 0
            
            # Use the original LayoutAwareChunker - it handles everything automatically!
            la_chunks, section_index = layout_chunker.chunk_document(file_path=primary_file, source_format="markdown")
            processing_method = "layout_aware_chunking"
            
            # Track tokens after processing this document
            tokens_after = layout_chunker.llm_classifier.total_tokens_used if layout_chunker.llm_classifier else 0
            tokens_used_for_doc = tokens_after - tokens_before
            document_token_usage[base_name] = tokens_used_for_doc
            
            # Add document field and source info to each chunk
            for chunk in la_chunks:
                chunk["document"] = base_name  # Use base name instead of filename
                # Ensure we have the method field
                if "method" not in chunk:
                    chunk["method"] = processing_method
                # Add information about which files were used
                chunk["source_files"] = {
                    "primary": os.path.basename(primary_file),
                    "json_enrichment": os.path.basename(files['json']) if files.get('json') else None
                }
            
            all_chunks.extend(la_chunks)
            processed_files.add(base_name)
            
            # Log with token usage info
            if tokens_used_for_doc > 0:
                logger.info(f"Processed {base_name}: {len(la_chunks)} chunks created, {tokens_used_for_doc} tokens used")
            else:
                logger.info(f"Processed {base_name}: {len(la_chunks)} chunks created")
            
        except Exception as e:
            logger.error(f"Error processing {base_name}: {e}")
            continue
    
    if not all_chunks:
        logger.error("No chunks were created. Exiting.")
        return
    
    # Generate embeddings and store
    logger.info("Generating embeddings...")
    texts = [chunk["text"] for chunk in all_chunks]
    embeddings = embedding_generator.generate_embeddings(texts)
    
    logger.info("Storing in vector database...")
    vector_store.store_chunks(all_chunks, embeddings)
    
    # Print summary
    print("\n" + "="*60)
    print("VECTOR DATABASE SETUP COMPLETE!")
    print("="*60)
    print(f"Total documents processed: {len(processed_files)}")
    print(f"Documents: {', '.join(processed_files)}")
    print(f"Total chunks stored: {len(all_chunks)}")
    print(f"Using specialized layout-aware chunkers based on available formats")
    
    # Show source file usage and processing modes
    print(f"\nSource file breakdown:")
    md_count = sum(1 for files in file_mapping.values() if files.get('md'))
    json_count = sum(1 for files in file_mapping.values() if files.get('json'))
    enriched_count = sum(1 for files in file_mapping.values() if files.get('md') and files.get('json'))
    md_only_count = sum(1 for files in file_mapping.values() if files.get('md') and not files.get('json'))
    json_only_count = sum(1 for files in file_mapping.values() if files.get('json') and not files.get('md'))
    
    print(f"  Total markdown files: {md_count}")
    print(f"  Total JSON files: {json_count}")
    print(f"  Documents with both MD+JSON (enriched): {enriched_count}")
    print(f"  MD-only documents: {md_only_count}")
    print(f"  JSON-only documents: {json_only_count}")
    
    # Show processing method breakdown
    method_counts = {}
    for chunk in all_chunks:
        method = chunk.get("method", "unknown")
        method_counts[method] = method_counts.get(method, 0) + 1
    
    print(f"\nProcessing method breakdown:")
    for method, count in sorted(method_counts.items()):
        print(f"  {method}: {count} chunks")
    
    # Count chunks by type
    by_type = {}
    for chunk in all_chunks:
        chunk_type = chunk.get("chunk_type", "unknown")
        by_type[chunk_type] = by_type.get(chunk_type, 0) + 1
    
    print("\nChunks by type:")
    for chunk_type, count in sorted(by_type.items()):
        print(f"  {chunk_type}: {count} chunks")
    
    # Show OpenAI API token usage
    if layout_chunker.llm_classifier:
        token_stats = layout_chunker.llm_classifier.get_token_usage_stats()
        print(f"\n📊 OpenAI API Usage:")
        print(f"  Total API calls made: {token_stats['total_calls_made']}")
        print(f"  Total tokens consumed: {token_stats['total_tokens_used']:,}")
        print(f"  Average tokens per call: {token_stats['average_tokens_per_call']}")
        
        # Show per-document breakdown
        if document_token_usage and any(tokens > 0 for tokens in document_token_usage.values()):
            print(f"\n  📋 Token usage by document:")
            for doc_name, tokens in document_token_usage.items():
                if tokens > 0:
                    print(f"    {doc_name}: {tokens:,} tokens")
                else:
                    print(f"    {doc_name}: 0 tokens (no table classification needed)")
        
        # Estimate cost (gpt-4o-mini pricing: $0.15 per 1M input tokens, $0.60 per 1M output tokens)
        # Rough estimate assuming 70% input, 30% output
        estimated_cost = (token_stats['total_tokens_used'] * 0.7 * 0.15 / 1000000) + \
                        (token_stats['total_tokens_used'] * 0.3 * 0.60 / 1000000)
        print(f"\n  💰 Estimated cost: ${estimated_cost:.4f} USD")
    else:
        print(f"\n📊 OpenAI API Usage: Not used (no LLM table classification)")
    
    print("\nNow you can use 'query.py' to test LLM queries!")
    print("="*60)

if __name__ == "__main__":
    main()
