#!/usr/bin/env python3
"""
Simple LLM Query Interface - Use this to manually test queries on your vector database

Usage:
    python3 query.py "Your question here"
"""

import sys
import json
from datetime import datetime
import logging
from src.llm_query import LLMQuerySystem

def main():
    # Ensure INFO-level logs are visible (including reranker logs)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    if len(sys.argv) < 2:
        print("Usage: python3 query.py \"Your question here\"")
        print("\nExamples:")
        print("  python3 query.py \"Tell me about the EPS this year\"")
        print("  python3 query.py \"What are the revenue figures?\"")
        print("  python3 query.py \"Show me information about iPhone sales\"")
        return
    
    query = " ".join(sys.argv[1:])
    
    # Initialize LLM Query System
    print("🔍 Initializing LLM Query System...")
    api_key = "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"
    
    try:
        query_system = LLMQuerySystem(api_key)
        
        print(f"📝 Query: \"{query}\"")
        print(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\n" + "="*80)
        
        # Process query with layout-aware chunking
        result = query_system.full_query_pipeline(query)
        
        # Display results
        chunks = result["retrieved_chunks"]
        context_len = result['context_length']
        llm_resp = result["llm_response"]
        tokens = llm_resp["tokens_used"]
        
        print(f"\n🔸 LAYOUT-AWARE CHUNKING RESULTS")
        print("-" * 50)
        print(f"📊 Retrieved chunks: {len(chunks)}")
        print(f"📏 Context length: {context_len:,} characters")
        print(f"🎯 Tokens used: {tokens}")
        
        if chunks:
            print(f"🔍 Top similarity score: {chunks[0]['score']:.4f}")
            if "rerank_score" in chunks[0]:
                print(f"🎯 Top rerank score: {chunks[0]['rerank_score']:.4f}")
        
        # Show chunk breakdown
        chunk_types = {}
        for chunk in chunks:
            chunk_type = chunk["payload"].get("chunk_type", "unknown")
            chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
        
        if chunk_types:
            print(f"📋 Chunk types found: {dict(chunk_types)}")
        
        print(f"\n💬 LLM Response:")
        print(f"{llm_resp['answer']}")
        
        print("\n" + "="*80)
        
        # Display timing metrics
        if "timing" in result:
            timing = result["timing"]
            print(f"📈 Total tokens used: {tokens}")
            print(f"⏱️  Total query time: {timing['total_time']:.2f}s")
            print(f"   ├─ Vector search: {timing['vector_search_time']:.3f}s")
            print(f"   ├─ Reranking: {timing['rerank_time']:.3f}s")
            print(f"   ├─ Context assembly: {timing['context_time']:.3f}s")
            print(f"   └─ LLM response: {timing['llm_time']:.3f}s")
        else:
            print(f"📈 Total tokens used: {tokens}")
            print("⏱️  Timing metrics not available")
        
        # Display resource metrics
        if "resources" in result:
            resources = result["resources"]
            print(f"\n💻 System Resources:")
            print(f"   🧠 Memory: {resources['peak_memory_mb']:.1f}MB / {resources['final']['memory_total_mb']:.1f}MB ({resources['final']['memory_percent']:.1f}%)")
            print(f"   🔥 CPU: {resources['peak_cpu_percent']:.1f}% peak")
            
            # GPU info
            gpu = resources['final']['gpu']
            if gpu.get('gpu_available', False):
                print(f"   🎮 GPU: {gpu['gpu_utilization_percent']:.1f}% util, {gpu['gpu_memory_used_mb']:.1f}MB / {gpu['gpu_memory_total_mb']:.1f}MB")
                if 'gpu_temperature_c' in gpu:
                    print(f"      🌡️  Temperature: {gpu['gpu_temperature_c']:.1f}°C")
            elif gpu.get('gpuutil_not_installed', False):
                print(f"   🎮 GPU: GPUtil not installed (pip install gputil)")
            else:
                print(f"   🎮 GPU: Not available")
        else:
            print(f"\n💻 System Resources: Not available")
        
        # Save to log file
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "result": result,
            "tokens_used": tokens,
            "timing": result.get("timing", {}),
            "resources": result.get("resources", {})
        }
        
        # Append to conversation log
        try:
            with open("conversation_log.json", "r", encoding="utf-8") as f:
                log_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            log_data = []
        
        log_data.append(log_entry)
        
        with open("conversation_log.json", "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Results saved to conversation_log.json")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
