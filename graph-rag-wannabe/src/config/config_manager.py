"""
Configuration Management System
Implements configuration-driven approach, dynamic threshold learning, and pluggable recipes
"""

import yaml
import os
import time
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

class IntentType(Enum):
    EXPLAIN = "explain"
    NUMERIC_EVIDENCE = "numeric_evidence"
    RELATIONSHIP = "relationship" 
    LOOKUP = "lookup"

@dataclass
class QueryIntent:
    primary: IntentType
    confidence: float
    signals: List[str]
    suggested_filters: List[str]
    expansion_strategy: str
    explanation: str

class ConfigManager:
    """Centralized configuration management"""
    
    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'config', 'default.yaml')
        
        self.config_path = config_path
        self.config = self._load_config()
        logger.info(f"Configuration loaded from {config_path}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {self.config_path} not found, using defaults")
            return self._get_default_config()
        except Exception as e:
            logger.error(f"Error loading config: {e}, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Fallback default configuration"""
        return {
            'intent_patterns': {
                'explain': ["why", "how", "cause", "reason"],
                'lookup': ["what is", "define", "definition"],
                'relationship': ["compare", "vs", "versus"],
                'numeric_evidence': ["table", "data", "number"]
            },
            'thresholds': {
                'quick_classify_confidence': 0.8,
                'fallback_confidence': 0.6,
                'performance_warning_time': 5.0
            },
            'search_limits': {
                'default_mode': {'hop_1_k': 50, 'hop_2_k': 30}
            },
            'llm': {
                'model': 'gpt-4o-mini',
                'temperature': 0.1,
                'max_tokens': 100
            }
        }
    
    def get(self, key: str, default=None):
        """Get configuration value with dot notation"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def update(self, key: str, value: Any):
        """Update configuration value"""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
        logger.info(f"Updated config {key} = {value}")

class AdaptiveThresholds:
    """Dynamic threshold learning system"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.performance_history: List[Dict[str, float]] = []
        self.accuracy_history: List[Dict[str, float]] = []
        
    def record_performance(self, intent_type: str, confidence: float, actual_relevance: float, response_time: float):
        """Record performance metrics for learning"""
        self.performance_history.append({
            'intent_type': intent_type,
            'confidence': confidence,
            'actual_relevance': actual_relevance,
            'response_time': response_time,
            'timestamp': time.time()
        })
        
        # Keep only recent history
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
    
    def learn_optimal_thresholds(self):
        """Analyze history and adjust thresholds"""
        if len(self.performance_history) < 50:
            return  # Need sufficient data
        
        # Analyze confidence vs actual relevance correlation
        for intent_type in IntentType:
            intent_data = [p for p in self.performance_history 
                          if p['intent_type'] == intent_type.value]
            
            if len(intent_data) >= 10:
                optimal_threshold = self._calculate_optimal_threshold(intent_data)
                current_key = f'thresholds.{intent_type.value}_confidence'
                self.config.update(current_key, optimal_threshold)
                logger.info(f"Learned optimal threshold for {intent_type.value}: {optimal_threshold:.3f}")
    
    def _calculate_optimal_threshold(self, data: List[Dict]) -> float:
        """Calculate optimal threshold based on precision/recall"""
        # Simple implementation - can be enhanced with more sophisticated ML
        accuracies = [d['actual_relevance'] for d in data if d['confidence'] > 0.7]
        if accuracies and len(accuracies) >= 5:
            avg_accuracy = sum(accuracies) / len(accuracies)
            return max(0.5, min(0.95, avg_accuracy - 0.1))  # Conservative adjustment
        return 0.8  # Default

class RecipeRegistry:
    """Pluggable recipe system"""
    
    def __init__(self):
        self.recipes: Dict[IntentType, Any] = {}
        self.default_recipe = None
        logger.info("Recipe registry initialized")
    
    def register_recipe(self, intent_type: IntentType, recipe_class: Any, **kwargs):
        """Register a recipe for an intent type"""
        self.recipes[intent_type] = {
            'class': recipe_class,
            'kwargs': kwargs,
            'instance': None
        }
        logger.info(f"Registered recipe for {intent_type.value}")
    
    def get_recipe(self, intent_type: IntentType, vector_store=None, embedding_generator=None, reranker=None):
        """Get recipe instance for intent type"""
        if intent_type not in self.recipes:
            if self.default_recipe:
                return self.default_recipe
            raise ValueError(f"No recipe registered for {intent_type.value}")
        
        recipe_info = self.recipes[intent_type]
        
        # Lazy initialization
        if recipe_info['instance'] is None:
            recipe_class = recipe_info['class']
            kwargs = recipe_info['kwargs']
            
            # Inject required dependencies
            if vector_store:
                kwargs['vector_store'] = vector_store
            if embedding_generator:
                kwargs['embedding_generator'] = embedding_generator  
            if reranker:
                kwargs['reranker'] = reranker
                
            recipe_info['instance'] = recipe_class(**kwargs)
            logger.info(f"Instantiated recipe for {intent_type.value}")
        
        return recipe_info['instance']
    
    def set_default_recipe(self, recipe_instance):
        """Set default recipe for unknown intent types"""
        self.default_recipe = recipe_instance
        logger.info("Default recipe set")
    
    def list_recipes(self) -> List[str]:
        """List all registered recipe types"""
        return [intent.value for intent in self.recipes.keys()]
