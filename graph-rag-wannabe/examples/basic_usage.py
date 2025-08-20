"""
Basic usage example for GraphRAGWannabe

Shows how to integrate with existing RAG pipeline to add 2-hop functionality.
"""

import sys
import os

# Add paths for imports
sys.path.append('/Users/rudram.piplad/Documents/rag-trials/rag_pipeline/src')
sys.path.append('/Users/rudram.piplad/Documents/rag-trials/graph-rag-wannabe/src')

def test_with_existing_rag_pipeline():
    """Test GraphRAGWannabe with the existing RAG pipeline components"""
    
    try:
        # Import existing RAG components
        from qdrant_store import QdrantVectorStore
        from embeddings import EmbeddingGenerator  
        from reranker import Reranker
        from graph_rag_wannabe import GraphRAGWannabe
        
        print("🚀 Testing GraphRAGWannabe with existing RAG pipeline")
        print("=" * 60)
        
        # Initialize existing components
        print("📦 Initializing components...")
        vector_store = QdrantVectorStore()
        embedding_generator = EmbeddingGenerator()
        reranker = Reranker()
        
        # API key for LLM calls
        api_key = "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"
        
        # Initialize GraphRAGWannabe
        print("🤖 Initializing GraphRAGWannabe...")
        wannabe = GraphRAGWannabe(
            vector_store=vector_store,
            embedding_generator=embedding_generator,
            reranker=reranker,
            openai_api_key=api_key
        )
        
        # Test queries
        test_queries = [
            "Why did revenue drop in Q4 2024?",
            "What was Apple's profit margin for FY2024?",
            "How did iPhone sales change between quarters?"
        ]
        
        print(f"\n🔍 Testing {len(test_queries)} queries...")
        
        for i, query in enumerate(test_queries):
            print(f"\n--- Query {i+1}: {query} ---")
            
            try:
                # Perform 2-hop search
                response = wannabe.query(query, verbose=True)
                
                print(f"\n💬 Answer: {response.answer[:200]}...")
                print(f"🔗 Trail: {response.hop_1_count} → {response.hop_2_count} → {len(response.final_chunks)} chunks")
                print(f"⏱️  Time: {response.total_time:.2f}s")
                
                # Show detailed trail for first query
                if i == 0:
                    print(f"\n🔍 Detailed trail explanation:")
                    wannabe.explain_trail(response)
                
            except Exception as e:
                print(f"❌ Error processing query: {e}")
        
        print(f"\n✅ GraphRAGWannabe testing complete!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're running from the correct directory and have all dependencies installed")
    except Exception as e:
        print(f"❌ Error: {e}")


def test_intent_classification_only():
    """Test just the LLM intent classification without full pipeline"""
    
    try:
        from query_routing.llm_intent_classifier import LLMIntentClassifier
        
        print("🧠 Testing LLM Intent Classification")
        print("=" * 40)
        
        api_key = "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"
        
        classifier = LLMIntentClassifier(api_key)
        
        test_queries = [
            "Why did revenue drop in Q4 2024?",
            "What was Apple's profit margin for FY2024?",
            "Compare iPhone sales between Q3 and Q4", 
            "What is EBITDA?",
            "Show me Table 5 data on quarterly expenses"
        ]
        
        for query in test_queries:
            print(f"\nQuery: {query}")
            intent = classifier.classify(query)
            print(f"  Intent: {intent.primary.value} ({intent.confidence:.2f})")
            print(f"  Strategy: {intent.expansion_strategy}")
            print(f"  Signals: {intent.signals}")
            print(f"  Explanation: {intent.explanation}")
        
        print(f"\n✅ Intent classification test complete!")
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    print("GraphRAGWannabe Test Suite")
    print("Choose test:")
    print("1. Full pipeline test (requires existing RAG components)")
    print("2. Intent classification only")
    
    choice = input("Enter 1 or 2: ").strip()
    
    if choice == "1":
        test_with_existing_rag_pipeline()
    elif choice == "2":
        test_intent_classification_only()
    else:
        print("Invalid choice. Running intent classification test...")
        test_intent_classification_only()
