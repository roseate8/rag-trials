"""
Content-Aware Retrieval System

Addresses the critical issue where 79.7% financial content bias causes 
non-financial queries to fail. Implements intelligent retrieval strategies
based on query intent to ensure balanced, relevant results.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import re
import openai
from qdrant_client.models import Filter, FieldCondition, Range, MatchValue

logger = logging.getLogger(__name__)


class ContentType(Enum):
    """Content type classification for balanced retrieval."""
    FINANCIAL = "financial"
    PRODUCT = "product" 
    STRATEGY = "strategy"
    TECHNICAL = "technical"
    GENERAL = "general"


class ContentAwareRetrieval:
    """
    Intelligent retrieval system that addresses content bias.
    
    Problem: 79.7% financial content dominance causes non-financial queries to fail
    Solution: Content-type-aware retrieval with query expansion and balanced sampling
    """
    
    def __init__(self, vector_store, embedder, query_processor, openai_api_key: Optional[str] = None):
        self.vector_store = vector_store
        self.embedder = embedder
        self.query_processor = query_processor
        
        # Initialize OpenAI client for query classification
        self.client = None
        if openai_api_key:
            try:
                self.client = openai.OpenAI(api_key=openai_api_key)
                logger.info("OpenAI client initialized for query classification")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI client: {e}")
        else:
            logger.warning("No OpenAI API key provided - using fallback classification")
        
        # Content type detection patterns
        self.content_patterns = {
            ContentType.FINANCIAL: {
                'query_keywords': ['revenue', 'profit', 'earnings', 'financial', 'cost', 'income', 'cash', 'investment', 'fiscal', 'quarter', 'eps', 'ebitda'],
                'content_keywords': ['revenue', 'financial', 'earnings', 'profit', 'cost', 'fiscal', 'quarter', 'million', 'ebitda', 'margin', 'income', 'cash', 'investment'],
                'metadata_fields': ['metric_terms']
            },
            ContentType.PRODUCT: {
                'query_keywords': ['product', 'feature', 'release', 'launch', 'version', 'update', 'software', 'platform', 'service', 'solution', 'tool', 'application'],
                'content_keywords': ['product', 'feature', 'release', 'launch', 'version', 'update', 'software', 'platform', 'service', 'solution', 'tool', 'application'],
                'metadata_fields': ['entities', 'policy_tags']
            },
            ContentType.STRATEGY: {
                'query_keywords': ['strategy', 'initiative', 'vision', 'mission', 'goal', 'roadmap', 'direction', 'focus', 'priority', 'objective', 'plan', 'decision'],
                'content_keywords': ['strategy', 'initiative', 'vision', 'mission', 'goal', 'roadmap', 'direction', 'focus', 'priority', 'objective', 'plan', 'decision'],
                'metadata_fields': ['policy_tags']
            },
            ContentType.TECHNICAL: {
                'query_keywords': ['technical', 'system', 'technology', 'integration', 'api', 'development', 'engineering', 'implementation', 'architecture', 'infrastructure'],
                'content_keywords': ['technical', 'system', 'technology', 'integration', 'api', 'development', 'engineering', 'implementation', 'architecture', 'infrastructure'],
                'metadata_fields': ['policy_tags', 'entities']
            }
        }
        
        logger.info("Content-Aware Retrieval System initialized")
    
    def detect_query_content_type(self, query: str) -> Tuple[ContentType, float]:
        """
        Detect the content type that the query is looking for using LLM classification.
        
        Returns:
            (content_type, confidence_score)
        """
        if self.client:
            return self._llm_classify_query(query)
        else:
            return self._fallback_classify_query(query)
    
    def _llm_classify_query(self, query: str) -> Tuple[ContentType, float]:
        """Use LLM to classify query content type with minimal token usage."""
        try:
            # Minimal, efficient prompt for classification
            system_prompt = "Classify the query into: financial, product, strategy, technical, or general. Respond with only the category name."
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                max_tokens=10,  # Minimal tokens - just need one word
                temperature=0.1
            )
            
            classification = response.choices[0].message.content.strip().lower()
            
            # Map response to ContentType
            type_mapping = {
                'financial': ContentType.FINANCIAL,
                'product': ContentType.PRODUCT,
                'strategy': ContentType.STRATEGY,
                'technical': ContentType.TECHNICAL,
                'general': ContentType.GENERAL
            }
            
            content_type = type_mapping.get(classification, ContentType.GENERAL)
            confidence = 0.9  # High confidence in LLM classification
            
            logger.info(f"LLM classified '{query}' as '{content_type.value}' (confidence: {confidence})")
            return content_type, confidence
            
        except Exception as e:
            logger.warning(f"LLM classification failed: {e}, falling back to keyword matching")
            return self._fallback_classify_query(query)
    
    def _fallback_classify_query(self, query: str) -> Tuple[ContentType, float]:
        """Fallback keyword-based classification when LLM is unavailable."""
        query_lower = query.lower()
        
        # Simplified keyword detection for fallback
        if any(word in query_lower for word in ['revenue', 'profit', 'earnings', 'financial', 'quarter', 'fiscal']):
            return ContentType.FINANCIAL, 0.7
        elif any(word in query_lower for word in ['product', 'feature', 'release', 'launch', 'software']):
            return ContentType.PRODUCT, 0.7
        elif any(word in query_lower for word in ['strategy', 'initiative', 'vision', 'mission', 'goal']):
            return ContentType.STRATEGY, 0.7
        elif any(word in query_lower for word in ['technical', 'system', 'technology', 'api', 'development']):
            return ContentType.TECHNICAL, 0.7
        else:
            return ContentType.GENERAL, 0.5
    
    def expand_query_for_content_type(self, query: str, content_type: ContentType) -> List[str]:
        """
        Expand query with relevant terms for the target content type.
        
        This helps find relevant content even when the dataset is biased.
        """
        expansions = [query]  # Always include original
        
        if content_type == ContentType.GENERAL:
            return expansions
        
        patterns = self.content_patterns.get(content_type, {})
        keywords = patterns.get('query_keywords', [])
        
        # Create expanded queries
        if keywords:
            # Add individual important keywords
            key_terms = keywords[:3]  # Top 3 most relevant terms
            for term in key_terms:
                if term not in query.lower():
                    expansions.append(f"{query} {term}")
            
            # Create keyword-focused query
            query_words = query.lower().split()
            relevant_keywords = [kw for kw in keywords if kw not in query_words][:2]
            if relevant_keywords:
                expansions.append(" ".join(relevant_keywords))
        
        return expansions[:4]  # Limit to 4 variants for efficiency
    
    def retrieve_with_content_awareness(
        self, 
        query: str, 
        method: str = "layout_aware_chunking", 
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Retrieve chunks with content-type awareness to address bias.
        
        Strategy:
        1. Detect query content type
        2. Use appropriate retrieval strategy based on content type
        3. For non-financial queries, expand search and rebalance results
        """
        content_type, confidence = self.detect_query_content_type(query)
        
        logger.info(f"Query content type: {content_type.value} (confidence: {confidence:.2f})")
        
        if content_type == ContentType.FINANCIAL and confidence > 0.5:
            # Financial queries work well with standard retrieval
            return self._standard_retrieval(query, method, top_k)
        else:
            # Non-financial queries need special handling due to content bias
            return self._balanced_retrieval(query, content_type, method, top_k)
    
    def _standard_retrieval(
        self, 
        query: str, 
        method: str, 
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Standard retrieval for financial queries (works well)."""
        try:
            # Use existing query optimization
            query_analysis = self.query_processor.get_optimized_query(query)
            optimized_query = query_analysis['optimized_query']
            
            # Generate embedding
            query_embedding = self.embedder.generate_embeddings([optimized_query])[0]
            
            # Search
            from qdrant_client.models import Filter, FieldCondition
            results = self.vector_store.client.search(
                collection_name=self.vector_store.collection_name,
                query_vector=query_embedding,
                query_filter=Filter(
                    must=[FieldCondition(key="method", match={"value": method})]
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
            
            return chunks
            
        except Exception as e:
            logger.error(f"Standard retrieval failed: {e}")
            return []
    
    def _balanced_retrieval(
        self, 
        query: str, 
        content_type: ContentType,
        method: str, 
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Balanced retrieval for non-financial queries.
        
        Addresses the 79.7% financial content bias by:
        1. Query expansion
        2. Multi-strategy search  
        3. Content-type filtering
        4. Result rebalancing
        """
        try:
            # Expand query for better coverage
            expanded_queries = self.expand_query_for_content_type(query, content_type)
            
            all_results = []
            
            # Search with multiple query variants
            for i, expanded_query in enumerate(expanded_queries):
                logger.info(f"Searching with variant {i+1}: '{expanded_query}'")
                
                # Generate embedding for expanded query
                query_embedding = self.embedder.generate_embeddings([expanded_query])[0]
                
                # Search with different strategies
                results = self._multi_strategy_search(
                    query_embedding, 
                    content_type,
                    method, 
                    top_k // len(expanded_queries) + 5  # Get more results per variant
                )
                
                # Add variant info to results
                for result in results:
                    result['query_variant'] = i
                    result['expanded_query'] = expanded_query
                
                all_results.extend(results)
            
            # Deduplicate and rebalance
            unique_results = self._deduplicate_results(all_results)
            balanced_results = self._rebalance_by_content_type(unique_results, content_type, top_k)
            
            logger.info(f"Balanced retrieval: {len(balanced_results)} chunks for {content_type.value} query")
            
            return balanced_results
            
        except Exception as e:
            logger.error(f"Balanced retrieval failed: {e}")
            # Fallback to standard retrieval
            return self._standard_retrieval(query, method, top_k)
    
    def _multi_strategy_search(
        self,
        query_embedding: List[float],
        content_type: ContentType,
        method: str,
        limit: int
    ) -> List[Dict[str, Any]]:
        """
        Multi-strategy search to find relevant content despite bias.
        """
        from qdrant_client.models import Filter, FieldCondition, MatchAny
        
        results = []
        
        # Strategy 1: Standard semantic search
        try:
            standard_results = self.vector_store.client.search(
                collection_name=self.vector_store.collection_name,
                query_vector=query_embedding,
                query_filter=Filter(
                    must=[FieldCondition(key="method", match={"value": method})]
                ),
                limit=limit,
                with_payload=True,
                with_vectors=False
            )
            
            for result in standard_results:
                results.append({
                    "id": result.id,
                    "score": result.score,
                    "payload": result.payload,
                    "strategy": "semantic"
                })
        except Exception as e:
            logger.warning(f"Semantic search failed: {e}")
        
        # Strategy 2: Metadata-based search for non-financial content
        if content_type != ContentType.FINANCIAL:
            try:
                patterns = self.content_patterns.get(content_type, {})
                metadata_fields = patterns.get('metadata_fields', [])
                
                for field in metadata_fields:
                    # Search for chunks with relevant metadata
                    metadata_results = self.vector_store.client.search(
                        collection_name=self.vector_store.collection_name,
                        query_vector=query_embedding,
                        query_filter=Filter(
                            must=[
                                FieldCondition(key="method", match={"value": method}),
                                FieldCondition(key=field, match=MatchAny(any=[]))  # Has any metadata in this field
                            ]
                        ),
                        limit=limit // 2,
                        with_payload=True,
                        with_vectors=False
                    )
                    
                    for result in metadata_results:
                        results.append({
                            "id": result.id,
                            "score": result.score * 0.9,  # Slight penalty for metadata-based
                            "payload": result.payload,
                            "strategy": f"metadata_{field}"
                        })
            except Exception as e:
                logger.warning(f"Metadata search failed: {e}")
        
        return results
    
    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate results based on chunk ID."""
        seen_ids = set()
        unique_results = []
        
        # Sort by score to keep best results
        sorted_results = sorted(results, key=lambda x: x.get('score', 0), reverse=True)
        
        for result in sorted_results:
            chunk_id = result['id']
            if chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                unique_results.append(result)
        
        return unique_results
    
    def _rebalance_by_content_type(
        self, 
        results: List[Dict[str, Any]], 
        target_type: ContentType,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Rebalance results to prioritize target content type.
        
        This addresses the financial content bias by boosting relevant content.
        """
        if target_type == ContentType.FINANCIAL:
            # Financial queries don't need rebalancing
            return results[:top_k]
        
        # Categorize results by content type
        categorized = {
            'target': [],      # Matches target content type
            'relevant': [],    # Somewhat relevant
            'financial': [],   # Financial content
            'other': []        # Other content
        }
        
        patterns = self.content_patterns.get(target_type, {})
        target_keywords = patterns.get('content_keywords', [])
        financial_keywords = self.content_patterns[ContentType.FINANCIAL]['content_keywords']
        
        for result in results:
            text = result['payload'].get('text', '').lower()
            
            # Count keyword matches
            target_matches = sum(1 for kw in target_keywords if kw in text)
            financial_matches = sum(1 for kw in financial_keywords if kw in text)
            
            if target_matches >= 2:
                categorized['target'].append(result)
            elif target_matches >= 1:
                categorized['relevant'].append(result)
            elif financial_matches >= 2:
                categorized['financial'].append(result)
            else:
                categorized['other'].append(result)
        
        # Rebalance: prioritize target content
        balanced = []
        
        # Add target content first (up to 60% of results)
        target_quota = min(len(categorized['target']), int(top_k * 0.6))
        balanced.extend(categorized['target'][:target_quota])
        
        # Add relevant content (up to 30% of results)
        remaining = top_k - len(balanced)
        relevant_quota = min(len(categorized['relevant']), int(top_k * 0.3))
        balanced.extend(categorized['relevant'][:relevant_quota])
        
        # Fill remaining with best available content
        remaining = top_k - len(balanced)
        if remaining > 0:
            other_content = categorized['financial'] + categorized['other']
            other_content.sort(key=lambda x: x.get('score', 0), reverse=True)
            balanced.extend(other_content[:remaining])
        
        logger.info(f"Rebalanced results: {len(categorized['target'])} target, {len(categorized['relevant'])} relevant, {len(categorized['financial'])} financial")
        
        return balanced[:top_k]


def create_content_aware_retrieval(vector_store, embedder, query_processor, openai_api_key: Optional[str] = None) -> ContentAwareRetrieval:
    """Factory function to create content-aware retrieval system."""
    return ContentAwareRetrieval(vector_store, embedder, query_processor, openai_api_key)
