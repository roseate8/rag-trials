"""
Universal Query Processor - Domain Agnostic Query Optimization

Replaces the financially-biased semantic_query_variants.py with a flexible,
intent-aware query processing system that works for all query types.
"""

import re
import logging
from typing import List, Dict, Set, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class QueryIntent(Enum):
    """Query intent classification for better processing."""
    FINANCIAL = "financial"
    TECHNICAL = "technical" 
    PROCESS = "process"
    TEMPORAL = "temporal"
    COMPARATIVE = "comparative"
    FACTUAL = "factual"
    UNKNOWN = "unknown"


class UniversalQueryProcessor:
    """
    Domain-agnostic query processor that handles all types of queries equally.
    
    Key improvements:
    1. Intent-based processing instead of financial bias
    2. Preserves important context words
    3. Maintains semantic meaning
    4. Supports qualitative and quantitative queries equally
    """
    
    def __init__(self):
        # Intent detection patterns
        self.intent_patterns = {
            QueryIntent.FINANCIAL: {
                'patterns': [
                    r'\b(revenue|profit|earnings|cost|financial|fiscal|quarter|eps|ebitda)\b',
                    r'\b(margin|growth|decline|performance|budget|expense)\b'
                ],
                'weights': ['financial', 'quantitative', 'metrics']
            },
            QueryIntent.TECHNICAL: {
                'patterns': [
                    r'\b(feature|update|release|version|improvement|bug|fix|software)\b',
                    r'\b(technical|development|engineering|product|functionality)\b'
                ],
                'weights': ['technical', 'product', 'development']
            },
            QueryIntent.PROCESS: {
                'patterns': [
                    r'\b(decision|strategy|initiative|roadmap|plan|goal|process)\b',
                    r'\b(management|leadership|direction|vision|objective)\b'
                ],
                'weights': ['strategic', 'operational', 'planning']
            },
            QueryIntent.TEMPORAL: {
                'patterns': [
                    r'\b(latest|recent|new|current|this|last|next|future|upcoming)\b',
                    r'\b(year|quarter|month|period|time|timeline)\b'
                ],
                'weights': ['temporal', 'timeline', 'recency']
            },
            QueryIntent.COMPARATIVE: {
                'patterns': [
                    r'\b(compare|versus|vs|difference|better|worse|more|less)\b',
                    r'\b(change|improvement|decline|increase|decrease)\b'
                ],
                'weights': ['comparative', 'analysis', 'trends']
            }
        }
        
        # Important semantic preservers (NOT stopwords)
        self.semantic_preservers = {
            # Intent indicators
            'what', 'how', 'why', 'when', 'where', 'which',
            # Action words
            'show', 'tell', 'explain', 'describe', 'list', 'find',
            # Quality indicators  
            'major', 'key', 'important', 'significant', 'critical',
            # Temporal context
            'latest', 'recent', 'new', 'current', 'upcoming',
            # Process words
            'decision', 'strategy', 'initiative', 'update', 'change'
        }
        
        # True noise words (minimal list)
        self.noise_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'
        }
        
        logger.info("Universal Query Processor initialized - domain agnostic")
    
    def detect_query_intent(self, query: str) -> Tuple[QueryIntent, float]:
        """
        Detect the primary intent of a query.
        
        Returns:
            (intent, confidence_score)
        """
        query_lower = query.lower()
        intent_scores = {}
        
        for intent, config in self.intent_patterns.items():
            score = 0
            for pattern in config['patterns']:
                matches = len(re.findall(pattern, query_lower))
                score += matches
            
            if score > 0:
                intent_scores[intent] = score
        
        if not intent_scores:
            return QueryIntent.UNKNOWN, 0.0
        
        # Get highest scoring intent
        best_intent = max(intent_scores, key=intent_scores.get)
        max_score = intent_scores[best_intent]
        
        # Calculate confidence (normalize by query length)
        confidence = min(1.0, max_score / max(1, len(query.split())))
        
        return best_intent, confidence
    
    def extract_semantic_keywords(self, query: str, intent: QueryIntent) -> List[str]:
        """
        Extract semantically important keywords based on intent.
        
        Unlike the biased approach, this preserves context and meaning.
        """
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = []
        
        for word in words:
            # Always keep semantic preservers
            if word in self.semantic_preservers:
                keywords.append(word)
            # Skip true noise words
            elif word in self.noise_words:
                continue
            # Keep domain-specific terms based on intent
            elif self._is_relevant_for_intent(word, intent):
                keywords.append(word)
            # Keep meaningful words (length > 3, not pure numbers)
            elif len(word) > 3 and not word.isdigit():
                keywords.append(word)
        
        return keywords
    
    def _is_relevant_for_intent(self, word: str, intent: QueryIntent) -> bool:
        """Check if a word is relevant for the detected intent."""
        if intent == QueryIntent.UNKNOWN:
            return True  # Keep everything when intent is unclear
        
        intent_config = self.intent_patterns.get(intent, {})
        for pattern in intent_config.get('patterns', []):
            if re.search(pattern, word):
                return True
        
        return False
    
    def generate_search_variants(self, query: str) -> List[str]:
        """
        Generate search variants that preserve semantic meaning.
        
        This replaces the financially-biased variant generation.
        """
        intent, confidence = self.detect_query_intent(query)
        keywords = self.extract_semantic_keywords(query, intent)
        
        variants = [query]  # Always include original
        
        # Create intent-preserving variants
        if keywords:
            # Full keyword combination
            if len(keywords) > 1:
                variants.append(' '.join(keywords))
            
            # Intent-focused variant (keep most important terms)
            if intent != QueryIntent.UNKNOWN:
                important_words = self._get_intent_focused_terms(keywords, intent)
                if important_words and len(important_words) <= 4:
                    variants.append(' '.join(important_words))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_variants = []
        for variant in variants:
            if variant not in seen:
                seen.add(variant)
                unique_variants.append(variant)
        
        return unique_variants[:3]  # Limit to 3 for efficiency
    
    def _get_intent_focused_terms(self, keywords: List[str], intent: QueryIntent) -> List[str]:
        """Get the most important terms for the specific intent."""
        if intent == QueryIntent.FINANCIAL:
            financial_terms = ['revenue', 'profit', 'earnings', 'cost', 'financial', 'eps', 'ebitda', 'margin']
            return [w for w in keywords if w in financial_terms] + [w for w in keywords if w not in financial_terms][:2]
        
        elif intent == QueryIntent.TECHNICAL:
            technical_terms = ['feature', 'update', 'release', 'version', 'improvement', 'product', 'development']
            return [w for w in keywords if w in technical_terms] + [w for w in keywords if w not in technical_terms][:2]
        
        elif intent == QueryIntent.PROCESS:
            process_terms = ['decision', 'strategy', 'initiative', 'roadmap', 'plan', 'goal', 'management']
            return [w for w in keywords if w in process_terms] + [w for w in keywords if w not in process_terms][:2]
        
        elif intent == QueryIntent.TEMPORAL:
            temporal_terms = ['latest', 'recent', 'new', 'current', 'upcoming', 'future']
            return [w for w in keywords if w in temporal_terms] + [w for w in keywords if w not in temporal_terms][:3]
        
        else:
            # For unknown or comparative intents, keep most meaningful terms
            return keywords[:4]
    
    def get_optimized_query(self, query: str) -> Dict[str, any]:
        """
        Get optimized query with intent context.
        
        Returns comprehensive information instead of just a string.
        """
        intent, confidence = self.detect_query_intent(query)
        variants = self.generate_search_variants(query)
        keywords = self.extract_semantic_keywords(query, intent)
        
        # Choose best variant based on intent and confidence
        if confidence > 0.7 and len(variants) > 1:
            # High confidence - use intent-optimized variant
            best_query = variants[1] if len(variants) > 1 else variants[0]
        else:
            # Low confidence - stick with original to preserve context
            best_query = query
        
        return {
            'optimized_query': best_query,
            'original_query': query,
            'intent': intent.value,
            'confidence': confidence,
            'variants': variants,
            'keywords': keywords,
            'context_preserved': len(keywords) > 2  # Indicates if semantic context is maintained
        }
    
    def should_preserve_context(self, query: str) -> bool:
        """
        Determine if the full query context should be preserved.
        
        Used to avoid over-optimization that destroys meaning.
        """
        intent, confidence = self.detect_query_intent(query)
        
        # Always preserve context for low-confidence intent detection
        if confidence < 0.5:
            return True
        
        # Preserve context for complex qualitative queries
        if intent in [QueryIntent.PROCESS, QueryIntent.COMPARATIVE]:
            return True
        
        # Preserve context for temporal queries (recency matters)
        if intent == QueryIntent.TEMPORAL:
            return True
        
        return False


# Singleton instance for global use
universal_processor = UniversalQueryProcessor()


def get_universal_processor() -> UniversalQueryProcessor:
    """Get the global universal query processor instance."""
    return universal_processor
