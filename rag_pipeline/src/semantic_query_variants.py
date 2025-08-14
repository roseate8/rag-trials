"""
Lightweight Semantic Query Variants

Extracts key financial terms from conversational queries using simple
pattern matching and keyword extraction - no LLM calls, zero token usage.
"""

import re
from typing import List, Set
import logging

logger = logging.getLogger(__name__)

class LightweightQueryVariants:
    """
    Generate semantic query variants using lightweight pattern matching.
    
    No LLM calls, no tokens used - pure pattern matching and keyword extraction.
    """
    
    def __init__(self):
        """Initialize with common financial abbreviations and patterns."""
        
        # Common financial abbreviations (case-insensitive matching)
        self.financial_abbrevs = {
            'eps', 'roi', 'roe', 'roa', 'p/e', 'pe', 'ebitda', 'ebit',
            'fcf', 'capex', 'opex', 'cogs', 'sga', 'r&d', 'rd',
            'ltv', 'cac', 'arpu', 'mrr', 'arr', 'nps', 'ctr'
        }
        
        # Conversational stopwords to remove
        self.stopwords = {
            'tell', 'me', 'show', 'what', 'is', 'the', 'are', 'how', 'much',
            'give', 'provide', 'find', 'get', 'about', 'for', 'in', 'on',
            'of', 'to', 'from', 'with', 'by', 'at', 'this', 'that', 'these',
            'those', 'can', 'you', 'please', 'would', 'could', 'should'
        }
        
        # Temporal terms that add noise
        self.temporal_noise = {
            'year', 'quarter', 'month', 'period', 'time', 'current', 'latest',
            'recent', 'last', 'next', 'today', 'now', '2024', '2023', '2022'
        }
        
        logger.info("Lightweight Query Variants initialized")
    
    def extract_financial_terms(self, query: str) -> List[str]:
        """
        Extract financial abbreviations and key terms from query.
        
        Args:
            query: Original user query
            
        Returns:
            List of extracted financial terms
        """
        query_lower = query.lower()
        terms = []
        
        # Split into words and clean
        words = re.findall(r'\b\w+(?:/\w+)?\b', query_lower)
        
        for word in words:
            # Check if it's a known financial abbreviation
            if word in self.financial_abbrevs:
                terms.append(word.upper())  # Normalize to uppercase
            # Keep important financial keywords (length > 3, not stopwords)
            elif (len(word) > 3 and 
                  word not in self.stopwords and 
                  word not in self.temporal_noise and
                  any(c.isalpha() for c in word)):  # Contains letters
                terms.append(word)
        
        return list(dict.fromkeys(terms))  # Remove duplicates, preserve order
    
    def generate_query_variants(self, query: str) -> List[str]:
        """
        Generate multiple query variants for better semantic search.
        
        Args:
            query: Original user query
            
        Returns:
            List of query variants (including original)
        """
        variants = [query]  # Always include original
        
        # Extract key financial terms
        financial_terms = self.extract_financial_terms(query)
        
        if financial_terms:
            # Create variants with just the financial terms
            if len(financial_terms) == 1:
                variants.append(financial_terms[0])
            elif len(financial_terms) > 1:
                variants.append(' '.join(financial_terms))
                # Also try individual terms for broad search
                variants.extend(financial_terms[:2])  # Limit to top 2 terms
        
        # Remove duplicates while preserving order
        seen = set()
        unique_variants = []
        for variant in variants:
            if variant.lower() not in seen:
                seen.add(variant.lower())
                unique_variants.append(variant)
        
        # Log for debugging (lightweight)
        if len(unique_variants) > 1:
            logger.debug(f"Query variants: {unique_variants}")
        
        return unique_variants[:3]  # Limit to 3 variants max for efficiency
    
    def get_best_query_for_search(self, query: str) -> str:
        """
        Get the single best query variant for semantic search.
        
        Prioritizes extracted financial terms over conversational queries.
        
        Args:
            query: Original user query
            
        Returns:
            Best query variant for semantic search
        """
        variants = self.generate_query_variants(query)
        
        # If we extracted financial terms, use the most focused variant
        if len(variants) > 1:
            # Prefer single financial terms or short combinations
            for variant in variants[1:]:  # Skip original
                if len(variant.split()) <= 2 and variant != query:
                    logger.debug(f"Query optimization: '{query}' -> '{variant}'")
                    return variant
        
        # Fall back to original query
        return query
