"""
Base classes for 2-hop search recipes in Graph-RAG Wannabe
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import sys
import os

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from config.config_manager import QueryIntent
from metadata_extraction.signal_extractor import ExtractedSignals


@dataclass 
class HopResult:
    """Result of a 2-hop search operation"""
    
    # Core results
    final_chunks: List[Dict[str, Any]] = field(default_factory=list)
    
    # Hop trail for provenance
    hop_1_chunks: List[Dict[str, Any]] = field(default_factory=list)
    hop_2_chunks: List[Dict[str, Any]] = field(default_factory=list)
    
    # Signal information
    extracted_signals: Optional[ExtractedSignals] = None
    applied_filters: List[str] = field(default_factory=list)
    
    # Performance metrics
    hop_1_count: int = 0
    hop_2_count: int = 0
    total_time: float = 0.0
    
    # Strategy information
    strategy_used: str = ""
    expansion_rationale: str = ""
    
    # Quality indicators
    signal_confidence: float = 0.0
    result_diversity: float = 0.0


class BaseRecipe(ABC):
    """
    Base class for all 2-hop search recipes.
    
    Each recipe implements a specific strategy for expanding search based on
    metadata signals extracted from Pass 1 results.
    """
    
    def __init__(self, vector_store, embedding_generator, reranker):
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator  
        self.reranker = reranker
    
    @abstractmethod
    def execute_hops(self, 
                     query: str, 
                     intent: QueryIntent) -> HopResult:
        """
        Execute the 2-hop search strategy.
        
        Args:
            query: Original user query
            intent: Classified query intent with signals and filters
            
        Returns:
            HopResult with final chunks and hop trail
        """
        pass
    
    def _perform_vector_search(self, 
                              query: str, 
                              filters: List[str] = None,
                              top_k: int = 50) -> List[Dict[str, Any]]:
        """
        Perform vector search with optional metadata filters.
        
        This is our basic "hop" operation - search the vector store
        with specific filters to simulate graph edge traversal.
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_generator.generate_embeddings([query])[0]
            
            # Build filter conditions for Qdrant
            filter_conditions = self._build_filter_conditions(filters)
            
            # Perform search
            search_results = self.vector_store.client.search(
                collection_name=self.vector_store.collection_name,
                query_vector=query_embedding,
                query_filter=filter_conditions,
                limit=top_k
            )
            
            # Convert to our format
            return [
                {
                    'id': result.id,
                    'score': result.score,
                    'payload': result.payload
                }
                for result in search_results
            ]
            
        except Exception as e:
            print(f"Vector search error: {e}")
            return []
    
    def _build_filter_conditions(self, filters: List[str]):
        """
        Build Qdrant filter conditions from filter strings.
        
        Converts filter strings like "chunk_type:table" into Qdrant filter objects.
        """
        if not filters:
            return None
        
        from qdrant_client.http.models import Filter, FieldCondition
        
        conditions = []
        for filter_str in filters:
            if ':' in filter_str:
                field, value = filter_str.split(':', 1)
                
                if value == '*':
                    # Exists filter
                    conditions.append(
                        FieldCondition(key=field, match={"any": []})
                    )
                else:
                    # Value filter
                    conditions.append(
                        FieldCondition(key=field, match={"value": value})
                    )
        
        if conditions:
            return Filter(must=conditions)
        return None
    
    def _rerank_results(self, 
                       query: str, 
                       results: List[Dict[str, Any]], 
                       top_k: int = 10) -> List[Dict[str, Any]]:
        """Rerank results using the reranker model"""
        
        if not results or len(results) <= top_k:
            return results
        
        try:
            # DEBUG: Check data format
            print(f"   🔧 Reranking {len(results)} results...")
            if results and 'payload' not in results[0]:
                print(f"   ⚠️  Warning: Results missing 'payload' field. Keys: {list(results[0].keys())}")
            
            # Extract texts for reranking - handle both formats
            texts = []
            for result in results:
                if 'payload' in result:
                    text = result['payload'].get('text', '')
                elif 'text' in result:
                    text = result['text']
                else:
                    text = str(result)
                texts.append(text)
            
            # Rerank using our reranker
            reranked_results = self.reranker.rerank(query, results, top_k=top_k)
            
            # Return reranked results (reranker already returns the reranked list)
            print(f"   ✅ Reranking successful: {len(reranked_results)} results")
            return reranked_results
            
        except Exception as e:
            print(f"Reranking error: {e}")
            return results[:top_k]
    
    def _calculate_result_diversity(self, results: List[Dict[str, Any]]) -> float:
        """Calculate diversity of results across different metadata dimensions"""
        
        if not results:
            return 0.0
        
        # Count unique values across different dimensions
        unique_sections = set()
        unique_pages = set()
        unique_chunk_types = set()
        
        for result in results:
            payload = result.get('payload', {})
            
            headings_path = payload.get('headings_path')
            if headings_path:
                unique_sections.add(headings_path)
            
            page = payload.get('page')
            if page:
                unique_pages.add(page)
            
            chunk_type = payload.get('chunk_type')
            if chunk_type:
                unique_chunk_types.add(chunk_type)
        
        # Diversity = average uniqueness across dimensions
        total_results = len(results)
        section_diversity = len(unique_sections) / total_results if total_results > 0 else 0
        page_diversity = len(unique_pages) / total_results if total_results > 0 else 0
        type_diversity = len(unique_chunk_types) / total_results if total_results > 0 else 0
        
        return (section_diversity + page_diversity + type_diversity) / 3
    
    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate results based on doc_id:page combination"""
        
        seen_keys = set()
        deduplicated = []
        
        for result in results:
            payload = result.get('payload', {})
            doc_id = payload.get('doc_id', '')
            page = payload.get('page', 0)
            
            key = f"{doc_id}:{page}"
            if key not in seen_keys:
                seen_keys.add(key)
                deduplicated.append(result)
        
        return deduplicated
