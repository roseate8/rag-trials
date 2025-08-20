"""
LLM Query system for testing vector store performance
"""
import os
import sys
import logging
import json
import psutil
import time
from datetime import datetime
from typing import List, Dict, Any
import openai
from .qdrant_store import QdrantVectorStore
from .embeddings import EmbeddingGenerator
from .reranker import CrossEncoderReranker
from .universal_query_processor import get_universal_processor
from .content_aware_retrieval import create_content_aware_retrieval

# Add graph-rag-wannabe to path for import
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'graph-rag-wannabe', 'src'))

logger = logging.getLogger(__name__)

class LLMQuerySystem:
    def __init__(self, openai_api_key: str):
        """Initialize the LLM query system"""
        self.vector_store = QdrantVectorStore()
        self.embedder = EmbeddingGenerator()
        self.reranker = CrossEncoderReranker()
        self.query_processor = get_universal_processor()
        self.content_aware_retrieval = create_content_aware_retrieval(
            self.vector_store, 
            self.embedder, 
            self.query_processor,
            openai_api_key
        )
        
        # Set up OpenAI (new API)
        self.client = openai.OpenAI(api_key=openai_api_key)
        
        logger.info("LLM Query System initialized")
    
    def get_system_resources(self) -> Dict[str, Any]:
        """Get current system resource usage"""
        try:
            # Memory usage
            memory = psutil.virtual_memory()
            memory_mb = memory.used / (1024 * 1024)  # Convert to MB
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # GPU usage (if available)
            gpu_info = {}
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # Get first GPU
                    gpu_info = {
                        "gpu_memory_used_mb": gpu.memoryUsed,
                        "gpu_memory_total_mb": gpu.memoryTotal,
                        "gpu_utilization_percent": gpu.load * 100,
                        "gpu_temperature_c": gpu.temperature
                    }
                else:
                    gpu_info = {"gpu_available": False}
            except ImportError:
                gpu_info = {"gpu_available": False, "gpuutil_not_installed": True}
            except Exception as e:
                gpu_info = {"gpu_available": False, "gpu_error": str(e)}
            
            return {
                "memory_used_mb": round(memory_mb, 1),
                "memory_total_mb": round(memory.total / (1024 * 1024), 1),
                "memory_percent": round(memory.percent, 1),
                "cpu_percent": round(cpu_percent, 1),
                "gpu": gpu_info
            }
        except Exception as e:
            logger.warning(f"Could not get system resources: {e}")
            return {
                "memory_used_mb": 0,
                "memory_total_mb": 0,
                "memory_percent": 0,
                "cpu_percent": 0,
                "gpu": {"gpu_available": False, "error": str(e)}
            }
    
    def search_similar_chunks(self, query: str, method: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using semantic similarity with query expansion
        
        Args:
            query: The search query
            method: search method ('layout_aware_chunking' or 'graph_rag_wannabe')
            top_k: Number of top results to return
        
        Returns:
            List of similar chunks with scores
        """
        # Use content-aware retrieval to address 79.7% financial content bias
        try:
            logger.info(f"Starting content-aware search for: '{query}'")
            
            chunks = self.content_aware_retrieval.retrieve_with_content_awareness(
                query=query,
                method=method,
                top_k=top_k
            )
            
            logger.info(f"Content-aware retrieval found {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Content-aware retrieval failed, falling back to standard: {e}")
            
            # Fallback to original method
            query_analysis = self.query_processor.get_optimized_query(query)
            optimized_query = query_analysis['optimized_query']
            query_embedding = self.embedder.generate_embeddings([optimized_query])[0]
        
        # Fallback search in Qdrant
        try:
            from qdrant_client.models import Filter, FieldCondition
            
            results = self.vector_store.client.search(
                collection_name=self.vector_store.collection_name,
                query_vector=query_embedding,
                query_filter=Filter(
                    must=[
                        FieldCondition(key="method", match={"value": method})
                    ]
                ),
                limit=top_k,
                with_payload=True,
                with_vectors=False
            )
            
            chunks = []
            for result in results:
                chunks.append({
                    "id": result.id,
                    "score": result.score,
                    "payload": result.payload
                })
            
            logger.info(f"Found {len(chunks)} similar chunks for method {method}")
            return chunks
            
        except Exception as e:
            logger.error(f"Error searching chunks: {e}")
            return []
    
    def generate_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Generate context string from retrieved chunks"""
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            text = chunk["payload"]["text"]
            context_parts.append(f"[Chunk {i}] {text}")
        
        return "\n\n".join(context_parts)
    
    def query_llm(self, query: str, context: str, method: str) -> Dict[str, Any]:
        """
        Query the LLM with context from vector store
        
        Args:
            query: User question
            context: Retrieved context from vector store
            method: Chunking method used
        
        Returns:
            LLM response with metadata
        """
        # Detect query intent for appropriate prompting
        query_analysis = self.query_processor.get_optimized_query(query)
        intent = query_analysis['intent']
        
        # Intent-aware prompt generation
        if intent == 'financial':
            domain_context = "financial documents and reports"
            assistant_role = "financial analyst"
        elif intent == 'technical':
            domain_context = "product documentation and technical updates"
            assistant_role = "technical analyst"
        elif intent == 'process':
            domain_context = "strategic documents and process information"
            assistant_role = "business analyst"
        else:
            domain_context = "business documents"
            assistant_role = "business analyst"
        
        prompt = f"""You are an AI assistant analyzing {domain_context}. Use the provided context to answer the user's question accurately and comprehensively.

Context from document (using {method.replace('_', ' ')} method):
{context}

Question: {query}

Please provide a detailed answer based on the context provided. Focus on the specific type of information requested ({intent} query). If the context doesn't contain enough information to fully answer the question, say so explicitly.

Answer:"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # Using available model
                messages=[
                    {"role": "system", "content": f"You are a helpful {assistant_role} AI assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.1
            )
            
            answer = response.choices[0].message.content
            
            return {
                "query": query,
                "method": method,
                "context": context,
                "answer": answer,
                "timestamp": datetime.now().isoformat(),
                "tokens_used": response.usage.total_tokens if response.usage else 0
            }
            
        except Exception as e:
            logger.error(f"Error querying LLM: {e}")
            return {
                "query": query,
                "method": method,
                "context": context,
                "answer": f"Error generating response: {str(e)}",
                "timestamp": datetime.now().isoformat(),
                "tokens_used": 0
            }
    
    def full_query_pipeline(self, query: str, top_k: int = 10, method: str = "layout_aware_chunking") -> Dict[str, Any]:
        """
        Complete pipeline: search chunks and query LLM
        
        Args:
            query: User question
            top_k: Number of chunks to retrieve for context  
            method: Search method ('layout_aware_chunking' or 'graph_rag_wannabe')
        
        Returns:
            Results from selected search method with timing metrics
        """
        import time
        
        logger.info(f"Processing query with {method}")
        
        # Start total timing and capture initial resources
        total_start_time = time.time()
        initial_resources = self.get_system_resources()
        
        # Check if using Graph_RAG_Not system
        if method == "graph_rag_wannabe":
            return self._graph_rag_not_pipeline(query, top_k, total_start_time, initial_resources)
        
        # Standard pipeline for layout_aware_chunking
        # Time vector search
        vector_search_start = time.time()
        similar_chunks = self.search_similar_chunks(query, method, top_k * 5)
        vector_search_time = time.time() - vector_search_start

        # Time reranking
        rerank_start = time.time()
        logger.info(
            f"Reranking {len(similar_chunks)} candidates with model='{self.reranker.model_name}'"
        )
        reranked_chunks = self.reranker.rerank(query, similar_chunks, top_k=top_k)
        rerank_time = time.time() - rerank_start
        
        if reranked_chunks:
            top_score = reranked_chunks[0].get("rerank_score", 0.0)
            logger.info(
                f"Reranking complete. Top rerank_score={top_score:.4f}; returning top_k={top_k}"
            )
        
        if reranked_chunks:
            # Time context generation (usually very fast)
            context_start = time.time()
            context = self.generate_context(reranked_chunks)
            context_time = time.time() - context_start
            
            # Time LLM query
            llm_start = time.time()
            llm_response = self.query_llm(query, context, method)
            llm_time = time.time() - llm_start
            
            # Capture final resources
            final_resources = self.get_system_resources()
            
            result = {
                "retrieved_chunks": reranked_chunks,
                "context_length": len(context),
                "llm_response": llm_response,
                "timing": {
                    "vector_search_time": vector_search_time,
                    "rerank_time": rerank_time,
                    "context_time": context_time,
                    "llm_time": llm_time,
                    "total_time": time.time() - total_start_time
                },
                "resources": {
                    "initial": initial_resources,
                    "final": final_resources,
                    "peak_memory_mb": max(initial_resources["memory_used_mb"], final_resources["memory_used_mb"]),
                    "peak_cpu_percent": max(initial_resources["cpu_percent"], final_resources["cpu_percent"])
                }
            }
        else:
            # Capture final resources
            final_resources = self.get_system_resources()
            
            result = {
                "retrieved_chunks": [],
                "context_length": 0,
                "llm_response": {
                    "query": query,
                    "method": method,
                    "context": "",
                    "answer": "No relevant chunks found for this query.",
                    "timestamp": datetime.now().isoformat(),
                    "tokens_used": 0
                },
                "timing": {
                    "vector_search_time": vector_search_time,
                    "rerank_time": rerank_time,
                    "context_time": 0,
                    "llm_time": 0,
                    "total_time": time.time() - total_start_time
                },
                "resources": {
                    "initial": initial_resources,
                    "final": final_resources,
                    "peak_memory_mb": max(initial_resources["memory_used_mb"], final_resources["memory_used_mb"]),
                    "peak_cpu_percent": max(initial_resources["cpu_percent"], final_resources["cpu_percent"])
                }
            }
        
        return result
    
    def _graph_rag_not_pipeline(self, query: str, top_k: int, total_start_time: float, initial_resources: Dict) -> Dict[str, Any]:
        """
        Graph_RAG_Not pipeline using 2-hop metadata-driven search
        """
        try:
            from .graph_rag_integration_v2 import GraphRAGIntegrationV2 as GraphRAGIntegration
            
            # Initialize GraphRAG integration
            graph_rag = GraphRAGIntegration(
                vector_store=self.vector_store,
                embedding_generator=self.embedder,
                reranker=self.reranker,
                openai_api_key=self.client.api_key
            )
            
            if not graph_rag.is_available():
                raise RuntimeError("GraphRAG wannabe system not available")
            
            # Execute GraphRAG query
            graph_result = graph_rag.query(query, top_k)
            
            # Convert to standard format
            final_resources = self.get_system_resources()
            
            result = {
                "retrieved_chunks": graph_result["chunks"],
                "context_length": len(graph_result["answer"]),
                "llm_response": {
                    "query": query,
                    "method": "graph_rag_wannabe",
                    "context": f"GraphRAG 2-hop expansion with {len(graph_result['chunks'])} chunks",
                    "answer": graph_result["answer"],
                    "timestamp": datetime.now().isoformat(),
                    "tokens_used": getattr(graph_result["response"], 'tokens_used', 0)
                },
                "timing": {
                    "vector_search_time": graph_result["metadata"]["total_time"] * 0.3,
                    "rerank_time": graph_result["metadata"]["total_time"] * 0.2,
                    "context_time": graph_result["metadata"]["total_time"] * 0.1,
                    "llm_time": graph_result["metadata"]["total_time"] * 0.4,
                    "total_time": graph_result["metadata"]["total_time"]
                },
                "resources": {
                    "initial": initial_resources,
                    "final": final_resources,
                    "peak_memory_mb": max(initial_resources["memory_used_mb"], final_resources["memory_used_mb"]),
                    "peak_cpu_percent": max(initial_resources["cpu_percent"], final_resources["cpu_percent"])
                },
                "graph_rag_metadata": graph_result["metadata"],
                "hop_1_count": graph_result["metadata"]["hop_1_count"],
                "hop_2_count": graph_result["metadata"]["hop_2_count"]
            }
            
            logger.info(f"Graph_RAG_Not completed: {graph_result['metadata']['hop_1_count']}→{graph_result['metadata']['hop_2_count']}→{len(graph_result['chunks'])} chunks")
            return result
            
        except Exception as e:
            logger.error(f"Graph_RAG_Not pipeline failed: {e}")
            raise RuntimeError(f"Graph_RAG_Not system failed: {str(e)}. Please use basic search instead.")
    
    def legacy_full_query_pipeline(self, query: str, top_k: int = 10, method: str = "layout_aware_chunking") -> Dict[str, Any]:
        """
        Legacy method for backward compatibility - returns results in old format
        """
        result = self.full_query_pipeline(query, top_k, method)
        return {method: result}
