"""
LLM Query system for testing vector store performance
"""
import os
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
from .semantic_query_variants import LightweightQueryVariants

logger = logging.getLogger(__name__)

class LLMQuerySystem:
    def __init__(self, openai_api_key: str):
        """Initialize the LLM query system"""
        self.vector_store = QdrantVectorStore()
        self.embedder = EmbeddingGenerator()
        self.reranker = CrossEncoderReranker()
        self.query_variants = LightweightQueryVariants()
        
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
            method: chunking method ('layout_aware_chunking')
            top_k: Number of top results to return
        
        Returns:
            List of similar chunks with scores
        """
        # Optimize query for better semantic matching (no tokens used)
        optimized_query = self.query_variants.get_best_query_for_search(query)
        
        # Generate embedding for the optimized query
        query_embedding = self.embedder.generate_embeddings([optimized_query])[0]
        
        # Search in Qdrant
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
        prompt = f"""You are an AI assistant analyzing financial documents. Use the provided context to answer the user's question accurately and comprehensively.

Context from document (using {method.replace('_', ' ')} method):
{context}

Question: {query}

Please provide a detailed answer based on the context provided. If the context doesn't contain enough information to fully answer the question, say so explicitly.

Answer:"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # Using available model
                messages=[
                    {"role": "system", "content": "You are a helpful financial analyst AI assistant."},
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
            method: Chunking method ('layout_aware_chunking')
        
        Returns:
            Results from selected chunking method with timing metrics
        """
        import time
        
        logger.info(f"Processing query with {method}")
        
        # Start total timing and capture initial resources
        total_start_time = time.time()
        initial_resources = self.get_system_resources()
        
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
    
    def legacy_full_query_pipeline(self, query: str, top_k: int = 10, method: str = "layout_aware_chunking") -> Dict[str, Any]:
        """
        Legacy method for backward compatibility - returns results in old format
        """
        result = self.full_query_pipeline(query, top_k, method)
        return {method: result}
