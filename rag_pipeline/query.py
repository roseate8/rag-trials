#!/usr/bin/env python3
"""
LLM Query Interface with Chunk Type Selection
Your familiar interface, now enhanced with chunk type options.

Usage:
    python3 query.py "Your question here" [--chunk-type basic|graph_rag_not|both]
    python3 query.py "Your question here" [basic|graph_rag_not]  # Simple format
"""

import sys
import argparse
import json
from datetime import datetime
import logging
from src.llm_query import LLMQuerySystem

def parse_arguments():
    """Parse command line arguments with both simple and advanced formats"""
    # Handle simple format: python3 query.py "question" chunk-type
    if len(sys.argv) >= 3 and sys.argv[-1] in ["basic", "graph_rag_not"]:
        return {
            "query": " ".join(sys.argv[1:-1]),
            "chunk_type": sys.argv[-1],
            "verbose": False,
            "top_k": 10
        }
    
    # Handle advanced format with argparse
    parser = argparse.ArgumentParser(
        description="LLM Query Interface with chunk type selection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 query.py "Tell me about the EPS this year"
  python3 query.py "What are the revenue figures?" graph_rag_not
  python3 query.py "Show me iPhone sales" --chunk-type basic
  python3 query.py "Compare approaches" --chunk-type both
        """
    )
    
    parser.add_argument(
        "query",
        help="Your question to ask the LLM"
    )
    
    parser.add_argument(
        "--chunk-type", 
        choices=["basic", "graph_rag_not", "both"],
        default="basic",
        help="Search method: basic (layout-aware chunks), graph_rag_not (2-hop metadata search), or both (default: basic)"
    )
    
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top chunks to retrieve (default: 10)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    return {
        "query": args.query,
        "chunk_type": args.chunk_type,
        "verbose": args.verbose,
        "top_k": args.top_k
    }

def format_chunk_stats(chunks, chunk_type_name):
    """Format chunk statistics for display"""
    if not chunks:
        return f"No chunks found for {chunk_type_name}"
    
    # Count chunk types
    chunk_types = {}
    pages_covered = set()
    
    for chunk in chunks:
        # Count by chunk type
        chunk_type = chunk["payload"].get("chunk_type", "unknown")
        chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
        
        # Track pages
        if chunk["payload"].get("page"):
            pages_covered.add(chunk["payload"]["page"])
        elif chunk["payload"].get("metadata", {}).get("page_no"):
            pages_covered.add(chunk["payload"]["metadata"]["page_no"])
    
    stats = []
    stats.append(f"Retrieved {len(chunks)} chunks")
    if pages_covered:
        stats.append(f"covering {len(pages_covered)} pages")
    if chunk_types:
        stats.append(f"types: {dict(chunk_types)}")
    
    return ", ".join(stats)

def display_results(result, method_name, chunk_type_display):
    """Display query results in a formatted way"""
    chunks = result["retrieved_chunks"]
    context_len = result['context_length']
    llm_resp = result["llm_response"]
    tokens = llm_resp["tokens_used"]
    
    print(f"\n🔸 {chunk_type_display.upper()} RESULTS")
    print("-" * 60)
    print(f"📊 {format_chunk_stats(chunks, chunk_type_display)}")
    print(f"📏 Context length: {context_len:,} characters")
    print(f"🎯 Tokens used: {tokens}")
    
    if chunks:
        print(f"🔍 Top similarity score: {chunks[0]['score']:.4f}")
        if "rerank_score" in chunks[0]:
            print(f"🎯 Top rerank score: {chunks[0]['rerank_score']:.4f}")
    
    # Show sample enhanced metadata for layout-aware chunks
    if chunks and "layout_aware" in method_name:
        sample_chunk = chunks[0]["payload"]
        if sample_chunk.get("metadata", {}).get("enhanced_layout_aware"):
            print(f"\n🔬 Enhanced Metadata Sample:")
            print(f"   🆔 Doc ID: {sample_chunk.get('doc_id', 'N/A')}")
            print(f"   📄 Page: {sample_chunk.get('page', 'N/A')}")
            print(f"   📍 Section Path: {sample_chunk.get('headings_path', 'N/A')}")
            if sample_chunk.get('bbox'):
                print(f"   📐 Bbox: {sample_chunk['bbox']}")
    
    print(f"\n💬 LLM Response:")
    print(f"{llm_resp['answer']}")
    
    # Display timing metrics
    if "timing" in result:
        timing = result["timing"]
        print(f"\n⏱️  Performance Metrics:")
        print(f"   Total time: {timing['total_time']:.2f}s")
        print(f"   ├─ Vector search: {timing['vector_search_time']:.3f}s")
        print(f"   ├─ Reranking: {timing['rerank_time']:.3f}s")
        print(f"   ├─ Context assembly: {timing['context_time']:.3f}s")
        print(f"   └─ LLM response: {timing['llm_time']:.3f}s")

def main():
    # Handle old-style usage help
    if len(sys.argv) < 2:
        print("Usage: python3 query.py \"Your question here\" [--chunk-type layout-aware|graph-rag|both]")
        print("   or: python3 query.py \"Your question here\" [layout-aware|graph-rag]")
        print("\nExamples:")
        print("  python3 query.py \"Tell me about the EPS this year\"")
        print("  python3 query.py \"What are the revenue figures?\" graph_rag_not")
        print("  python3 query.py \"Show me iPhone sales\" --chunk-type basic")
        print("  python3 query.py \"Compare methods\" --chunk-type both")
        print("\nSearch Methods:")
        print("  basic: Uses layout-aware chunking with vector search (default)")
        print("  graph_rag_not: Uses 2-hop metadata-driven search (experimental)")
        print("  both: Compare both methods side-by-side")
        return
    
    try:
        args = parse_arguments()
    except SystemExit:
        return
    
    # Set up logging
    log_level = logging.DEBUG if args["verbose"] else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s:%(name)s:%(message)s")
    
    query = args["query"]
    chunk_type = args["chunk_type"]
    top_k = args["top_k"]
    
    # API key (in production, use environment variable)
    api_key = "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"
    
    print("🔍 Initializing LLM Query System...")
    print(f"📝 Query: \"{query}\"")
    print(f"🏷️  Chunk Type: {chunk_type}")
    print(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n" + "="*80)
    
    try:
        query_system = LLMQuerySystem(api_key)
        
        # Map user-friendly names to method names
        method_map = {
            "basic": "layout_aware_chunking",
            "graph_rag_not": "graph_rag_wannabe"
        }
        
        results = {}
        total_tokens = 0
        
        if chunk_type == "basic":
            method = method_map["basic"]
            result = query_system.full_query_pipeline(query, top_k, method)
            results[method] = result
            display_results(result, method, "Basic Search (Layout-Aware)")
            total_tokens += result["llm_response"]["tokens_used"]
            
        elif chunk_type == "graph_rag_not":
            method = method_map["graph_rag_not"]
            try:
                result = query_system.full_query_pipeline(query, top_k, method)
                results[method] = result
                display_results(result, method, "Graph_RAG_Not (2-Hop Search)")
                total_tokens += result["llm_response"]["tokens_used"]
            except Exception as e:
                print(f"❌ Graph_RAG_Not failed: {e}")
                print("💡 Please use 'basic' search method instead")
                return
            
        elif chunk_type == "both":
            # Compare both methods
            print("🔀 Comparing both chunking methods...")
            
            # Basic search results
            basic_method = method_map["basic"]
            basic_result = query_system.full_query_pipeline(query, top_k, basic_method)
            results[basic_method] = basic_result
            display_results(basic_result, basic_method, "Basic Search (Layout-Aware)")
            total_tokens += basic_result["llm_response"]["tokens_used"]
            
            print("\n" + "="*80)
            
            # Graph_RAG_Not results
            graph_method = method_map["graph_rag_not"]
            try:
                graph_result = query_system.full_query_pipeline(query, top_k, graph_method)
                results[graph_method] = graph_result
                display_results(graph_result, graph_method, "Graph_RAG_Not (2-Hop Search)")
                total_tokens += graph_result["llm_response"]["tokens_used"]
            except Exception as e:
                print(f"❌ Graph_RAG_Not failed: {e}")
                print("🔄 Comparison incomplete - Graph_RAG_Not system not available")
                return
            
            # Comparison summary
            print("\n" + "="*80)
            print("📊 COMPARISON SUMMARY")
            print("-" * 30)
            basic_chunks = len(basic_result["retrieved_chunks"])
            graph_chunks = len(graph_result["retrieved_chunks"])
            print(f"Basic Search: {basic_chunks} chunks, {basic_result['llm_response']['tokens_used']} tokens")
            print(f"Graph_RAG_Not: {graph_chunks} chunks, {graph_result['llm_response']['tokens_used']} tokens")
        
        print("\n" + "="*80)
        print(f"📈 Total tokens used: {total_tokens}")
        
        # Save results to log
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "chunk_type": chunk_type,
            "top_k": top_k,
            "results": results,
            "total_tokens": total_tokens
        }
        
        # Append to conversation log (keeping common file for UI compatibility)
        log_file = "conversation_log.json"
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                log_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            log_data = []
        
        log_data.append(log_entry)
        
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"💾 Results saved to {log_file}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        if args["verbose"]:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()