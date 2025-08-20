"""
Optimized LLM Intent Classifier with minimal token usage and aggressive caching.
"""

import json
import logging
import hashlib
import time
from typing import Dict, List, Any, Optional
import openai
import sys
import os

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from config.config_manager import ConfigManager, IntentType, QueryIntent, AdaptiveThresholds

# Set up logging
logger = logging.getLogger(__name__)

class OptimizedIntentClassifier:
    """
    Ultra-efficient LLM intent classifier with configuration and adaptive learning.
    
    Optimizations:
    - Configuration-driven patterns and thresholds
    - Adaptive threshold learning from performance
    - Aggressive caching (avoid repeat LLM calls)
    - Minimal JSON response
    - Regex fallback for speed
    """
    
    def __init__(self, api_key: str, config_manager: Optional[ConfigManager] = None):
        self.client = openai.OpenAI(api_key=api_key)
        self.config = config_manager or ConfigManager()
        self.model = self.config.get('llm.model', 'gpt-4o-mini')
        self.cache: Dict[str, QueryIntent] = {}
        self.adaptive_thresholds = AdaptiveThresholds(self.config)
        
        # Build dynamic prompt from config
        self.prompt = self._build_prompt()
        
        logger.info(f"OptimizedIntentClassifier initialized with {self.model}")
        logger.info(f"Cache limit: {self.config.get('performance.cache_size_limit', 1000)}")
    
    def _build_prompt(self) -> str:
        """Build prompt from configuration"""
        return '''Classify this financial query intent and extract metadata signals:

INTENTS:
- explain: Why/how questions → causal_expansion  
- numeric_evidence: Numbers/tables/data → table_focused_expansion
- relationship: Compare/contrast → comparative_expansion
- lookup: What is/define → semantic_expansion

EXTRACT: temporal(Q4,2024), financial(revenue,profit), entities(Apple), references(Table 5)

JSON: {{"intent":"explain","confidence":0.9,"signals":{{"temporal":["Q4"],"financial":["revenue"]}},"strategy":"causal_expansion","explanation":"Brief reason"}}

Query: "{query}"'''
    
    def classify(self, query: str) -> QueryIntent:
        """Classify with caching and optimized prompt"""
        
        # Check cache first
        cache_key = hashlib.md5(query.encode()).hexdigest()
        if cache_key in self.cache:
            logger.debug(f"Cache hit for query: {query[:50]}...")
            return self.cache[cache_key]
        
        try:
            # Try regex-based fast classification first for simple cases
            if quick_intent := self._quick_classify(query):
                logger.debug(f"Quick classification: {quick_intent.primary.value}")
                self.cache[cache_key] = quick_intent
                return quick_intent
            
            # LLM classification for complex cases
            logger.debug(f"LLM classification for: {query[:50]}...")
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": self.prompt.format(query=query)}],
                temperature=0.1,
                max_tokens=100  # Much smaller response
            )
            
            result = json.loads(response.choices[0].message.content.strip())
            
            # Build intent object
            intent = QueryIntent(
                primary=IntentType(result["intent"]),
                confidence=result["confidence"],
                signals=self._flatten_signals(result["signals"]),
                suggested_filters=self._build_filters(result["signals"]),
                expansion_strategy=result["strategy"],
                explanation=result.get("explanation", "LLM classification")
            )
            
            # Cache result
            self.cache[cache_key] = intent
            logger.info(f"Classified '{query[:30]}...' as {intent.primary.value} ({intent.confidence:.2f})")
            
            return intent
            
        except Exception as e:
            logger.warning(f"LLM classification failed: {e}, using fallback")
            return self._fallback_classify(query)
    
    def _quick_classify(self, query: str) -> Optional[QueryIntent]:
        """Fast regex-based classification for simple queries"""
        
        q = query.lower()
        
        # Simple patterns for common cases
        if any(word in q for word in ["why", "how", "cause", "reason"]):
            return QueryIntent(
                primary=IntentType.EXPLAIN,
                confidence=0.8,
                signals=[],
                suggested_filters=[],
                expansion_strategy="causal_expansion",
                explanation="Quick regex: causal words detected"
            )
        
        if any(word in q for word in ["what is", "define", "definition"]) and len(q.split()) <= 5:
            return QueryIntent(
                primary=IntentType.LOOKUP,
                confidence=0.8,
                signals=[],
                suggested_filters=[],
                expansion_strategy="semantic_expansion",
                explanation="Quick regex: definition pattern"
            )
        
        # If not quickly classifiable, return None for LLM
        return None
    
    def _fallback_classify(self, query: str) -> QueryIntent:
        """Simple fallback when LLM fails"""
        
        q = query.lower()
        
        if any(word in q for word in ["why", "how", "cause"]):
            intent_type = IntentType.EXPLAIN
            strategy = "causal_expansion"
        elif any(word in q for word in ["compare", "vs", "versus", "between"]):
            intent_type = IntentType.RELATIONSHIP
            strategy = "comparative_expansion"
        elif any(word in q for word in ["table", "data", "number", "revenue", "profit"]):
            intent_type = IntentType.NUMERIC_EVIDENCE
            strategy = "table_focused_expansion"
        else:
            intent_type = IntentType.LOOKUP
            strategy = "semantic_expansion"
        
        return QueryIntent(
            primary=intent_type,
            confidence=0.6,
            signals=[],
            suggested_filters=[],
            expansion_strategy=strategy,
            explanation="Fallback classification"
        )
    
    def _flatten_signals(self, signals: Dict[str, List[str]]) -> List[str]:
        """Flatten nested signals to simple list"""
        flattened = []
        for signal_type, values in signals.items():
            for value in values:
                flattened.append(f"{signal_type}:{value}")
        return flattened
    
    def _build_filters(self, signals: Dict[str, List[str]]) -> List[str]:
        """Build metadata filters from signals"""
        filters = []
        
        for signal_type, values in signals.items():
            if signal_type == "temporal" and values:
                filters.extend([f"mentioned_dates:*{v}*" for v in values])
            elif signal_type == "financial" and values:
                filters.append("metric_terms:*")
            elif signal_type == "references" and values:
                filters.append("doc_refs:*")
                
        return filters
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get caching statistics"""
        return {
            "cache_size": len(self.cache),
            "cache_keys": list(self.cache.keys())[:5]  # Show first 5
        }
    
    def clear_cache(self):
        """Clear the classification cache"""
        self.cache.clear()
        logger.info("Classification cache cleared")
