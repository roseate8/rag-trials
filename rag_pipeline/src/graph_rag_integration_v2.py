"""
Improved GraphRAG Integration - Clean version with absolute imports
"""

import sys
import os
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class GraphRAGIntegrationV2:
    """Clean GraphRAG integration with absolute imports"""
    
    def __init__(self, vector_store, embedding_generator, reranker, openai_api_key: str):
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
        self.reranker = reranker
        self.openai_api_key = openai_api_key
        self.unified_system = None
        
        # Initialize on first use
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize the unified system with absolute imports"""
        try:
            # Add GraphRAG path with absolute approach
            current_dir = os.path.dirname(os.path.abspath(__file__))
            graph_rag_path = os.path.join(current_dir, '..', '..', 'graph-rag-wannabe', 'src')
            graph_rag_path = os.path.abspath(graph_rag_path)
            
            if graph_rag_path not in sys.path:
                sys.path.insert(0, graph_rag_path)
                logger.info(f"Added GraphRAG path: {graph_rag_path}")
            
            # Import with absolute path
            from unified_graph_rag import UnifiedGraphRAG
            
            # Initialize unified system
            self.unified_system = UnifiedGraphRAG(
                vector_store=self.vector_store,
                embedding_generator=self.embedding_generator,
                reranker=self.reranker,
                openai_api_key=self.openai_api_key
            )
            logger.info("✅ UnifiedGraphRAG system initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize UnifiedGraphRAG: {e}")
            logger.error(f"GraphRAG path attempted: {graph_rag_path}")
            logger.error(f"Path exists: {os.path.exists(graph_rag_path)}")
            
            # List what's actually in the directory
            if os.path.exists(graph_rag_path):
                files = os.listdir(graph_rag_path)
                logger.error(f"Files in GraphRAG src: {files}")
            
            self.unified_system = None
    
    def query(self, query: str, top_k: int = 10) -> Dict[str, Any]:
        """Execute GraphRAG query"""
        if not self.unified_system:
            raise RuntimeError("UnifiedGraphRAG system not initialized")
        
        # Execute 2-hop search
        graph_response = self.unified_system.query(query, verbose=False)
        
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
        """Check if system is available"""
        return self.unified_system is not None
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get detailed diagnostics"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        graph_rag_path = os.path.join(current_dir, '..', '..', 'graph-rag-wannabe', 'src')
        graph_rag_path = os.path.abspath(graph_rag_path)
        
        return {
            "system_available": self.is_available(),
            "graph_rag_path": graph_rag_path,
            "path_exists": os.path.exists(graph_rag_path),
            "files_in_path": os.listdir(graph_rag_path) if os.path.exists(graph_rag_path) else [],
            "python_path": sys.path[:5],  # First 5 entries
            "current_dir": current_dir
        }
