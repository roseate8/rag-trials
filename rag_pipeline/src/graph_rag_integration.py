"""
GraphRAG Integration Module
Standalone wrapper to avoid import issues
"""

import sys
import os
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class GraphRAGIntegration:
    """Standalone GraphRAG integration that avoids relative import issues"""
    
    def __init__(self, vector_store, embedding_generator, reranker, openai_api_key: str):
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
        self.reranker = reranker
        self.openai_api_key = openai_api_key
        self.wannabe = None
        
        # Initialize on first use
        self._initialize_wannabe()
    
    def _initialize_wannabe(self):
        """Initialize the GraphRAG wannabe system"""
        try:
            # Add GraphRAG wannabe to path
            graph_rag_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'graph-rag-wannabe', 'src')
            if graph_rag_path not in sys.path:
                sys.path.insert(0, graph_rag_path)
            
            # Import and initialize
            from graph_rag_optimized import OptimizedGraphRAGWannabe
            self.wannabe = OptimizedGraphRAGWannabe(
                vector_store=self.vector_store,
                embedding_generator=self.embedding_generator,
                reranker=self.reranker,
                openai_api_key=self.openai_api_key,
                log_level="INFO"
            )
            logger.info("GraphRAG wannabe system initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize GraphRAG wannabe: {e}")
            self.wannabe = None
    
    def query(self, query: str, top_k: int = 10) -> Dict[str, Any]:
        """Execute GraphRAG wannabe query"""
        if not self.wannabe:
            raise RuntimeError("GraphRAG wannabe system not initialized")
        
        # Execute 2-hop search
        graph_response = self.wannabe.query(query, verbose=False)
        
        # Format chunks to match expected structure
        formatted_chunks = []
        for chunk in graph_response.final_chunks[:top_k]:
            formatted_chunks.append({
                "id": chunk.get("id", "unknown"),
                "score": chunk.get("score", 0.0),
                "rerank_score": chunk.get("rerank_score", chunk.get("score", 0.0)),
                "payload": chunk.get("payload", chunk)
            })
        
        return {
            "chunks": formatted_chunks,
            "response": graph_response,
            "answer": graph_response.answer,
            "metadata": {
                "intent_type": graph_response.intent_type,
                "intent_confidence": graph_response.intent_confidence,
                "hop_1_count": graph_response.hop_1_count,
                "hop_2_count": graph_response.hop_2_count,
                "expansion_signals": graph_response.expansion_signals,
                "expansion_strategy": graph_response.expansion_strategy,
                "total_time": graph_response.total_time
            }
        }
    
    def is_available(self) -> bool:
        """Check if GraphRAG wannabe is available"""
        return self.wannabe is not None
