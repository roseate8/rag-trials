"""
Graph-RAG Wannabe: 2-Hop Metadata-Driven Search System

A lightweight alternative to Graph-RAG that uses intelligent metadata-driven
expansions within vector search to simulate graph-like behavior.
"""

from .unified_graph_rag import UnifiedGraphRAG
from .config.config_manager import ConfigManager, QueryIntent, IntentType
from .query_routing.optimized_intent_classifier import OptimizedIntentClassifier
from .hop_recipes.explain_recipe import ExplainRecipe
from .hop_recipes.numeric_evidence_recipe import NumericEvidenceRecipe
from .metadata_extraction.signal_extractor import MetadataSignalExtractor
from .response_building.trail_builder import ProvenanceTrailBuilder

__version__ = "1.0.0"
__all__ = [
    "GraphRAGWannabe",
    "QueryIntentClassifier", 
    "QueryIntent",
    "ExplainRecipe",
    "NumericEvidenceRecipe", 
    "MetadataSignalExtractor",
    "ProvenanceTrailBuilder"
]
