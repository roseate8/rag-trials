"""
Explain Recipe - Fixed with proper Hop 2 exclusion logic
"""

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import time
from typing import List, Dict, Any

from hop_recipes.base_recipe import BaseRecipe, HopResult
from config.config_manager import QueryIntent
from metadata_extraction.signal_extractor import ExtractedSignals
from visualization.simple_visualizer import log_simple_step, finish_simple_tracking


class ExplainRecipe(BaseRecipe):
    """
    Recipe for explanatory queries requiring causal understanding.
    
    Fixed strategy:
    1. Hop 1: Standard semantic search (seed)
    2. Signal extraction from Hop 1 results  
    3. Hop 2: Semantic variations with PROPER EXCLUSION
    4. Assembly with causal reasoning focus
    """
    
    def execute_hops(self, query: str, intent: QueryIntent) -> HopResult:
        """
        Process an explanatory query with proper 2-hop expansion.
        
        NOW INCLUDES PROPER HOP 1 EXCLUSION IN HOP 2!
        """
        start_time = time.time()
        
        # === QUERY CLASSIFICATION ===
        log_simple_step("Query Classification", 1, 0.001, 
                        f"Intent: {intent.primary.value}, Strategy: {intent.expansion_strategy}",
                        {"confidence": intent.confidence})
        
        # === HOP 1: SEED SEARCH ===
        hop_1_results = self._perform_vector_search(
            query=query,
            filters=intent.suggested_filters,  # Use intent-based filters
            top_k=50
        )
        
        hop_1_count = len(hop_1_results)
        hop1_time = time.time() - start_time
        log_simple_step("Seed Search", hop_1_count, hop1_time, f"Vector similarity search with 50 limit")
        
        if not hop_1_results:
            # No results from seed search
            return HopResult(
                final_chunks=[],
                hop_1_chunks=[],
                hop_2_chunks=[],
                hop_1_count=0,
                hop_2_count=0,
                strategy_used="explain_recipe",
                expansion_rationale="No seed results found",
                total_time=time.time() - start_time
            )
        
        # === SIGNAL EXTRACTION ===
        print("🧠 Extracting signals from seed results...")
        
        # Initialize signal extractor if not present
        if not hasattr(self, 'signal_extractor'):
            from metadata_extraction.signal_extractor import MetadataSignalExtractor
            self.signal_extractor = MetadataSignalExtractor()
        
        extracted_signals = self.signal_extractor.extract_signals(hop_1_results)
        
        print(f"   Top signals: metrics={extracted_signals.top_metric_terms}, "
              f"entities={extracted_signals.top_entities}, "
              f"refs={extracted_signals.top_doc_refs}")
        
        # Store hop 1 results for exclusion in hop 2
        self._hop_1_results = hop_1_results
        
        # === HOP 2: SEMANTIC EXPANSION WITH EXCLUSION ===
        hop2_start = time.time()
        hop_2_results = self._perform_proper_expansion(
            query, extracted_signals, intent
        )
        
        hop_2_count = len(hop_2_results)
        hop2_time = time.time() - hop2_start
        log_simple_step("Hop 2: Semantic Expansion", hop_2_count, hop2_time,
                        f"Found {hop_2_count} additional chunks using query variations",
                        {"unique_chunks": hop_2_count, "excluded_hop1": len(hop_1_results)})
        
        # === ASSEMBLY ===
        # STRICT 2-HOP REQUIREMENT: Both hops must succeed
        if hop_2_count == 0:
            raise RuntimeError(f"Graph_RAG_Not failed: Hop 2 found 0 chunks. "
                             f"Hop 1 found {hop_1_count} chunks but semantic expansion failed. "
                             f"This indicates insufficient metadata richness or overly restrictive expansion strategy.")
        
        final_chunks = self._assemble_causal_trail(
            query, hop_1_results, hop_2_results, extracted_signals
        )
        
        total_time = time.time() - start_time
        
        # Build result
        result = HopResult(
            final_chunks=final_chunks,
            hop_1_chunks=hop_1_results,
            hop_2_chunks=hop_2_results,
            extracted_signals=extracted_signals,
            applied_filters=self._build_applied_filters(extracted_signals),
            hop_1_count=hop_1_count,
            hop_2_count=hop_2_count,
            total_time=total_time,
            strategy_used="explain_recipe",
            expansion_rationale=self._build_expansion_rationale(extracted_signals),
            signal_confidence=extracted_signals.confidence_score,
            result_diversity=self._calculate_result_diversity(final_chunks)
        )
        
        rerank_time = 0.05  # Approximate
        log_simple_step("Final Assembly", len(final_chunks), rerank_time, 
                        f"Merged, deduped, and reranked to {len(final_chunks)} chunks")
        
        finish_simple_tracking()
        
        return result
    
    def _perform_proper_expansion(self, 
                                 query: str,
                                 signals: ExtractedSignals,
                                 intent: QueryIntent) -> List[Dict[str, Any]]:
        """
        FIXED: Perform Pass 2 expansion using semantic variations + exclusion.
        
        NEW STRATEGY:
        1. Generate semantic query variations based on signals
        2. Search with broader filters (not overly restrictive)  
        3. EXCLUDE Hop 1 results to find NEW complementary content
        4. Focus on semantic expansion rather than metadata filtering
        """
        
        # Generate semantic variations based on extracted signals
        expansion_queries = self._generate_semantic_variations(query, signals)
        
        # Get Hop 1 IDs for exclusion
        hop_1_ids = set(result.get('id') for result in getattr(self, '_hop_1_results', []))
        print(f"   🚫 Excluding {len(hop_1_ids)} Hop 1 IDs from Hop 2")
        
        all_expansion_results = []
        
        # Search with each variation
        for expansion_query in expansion_queries:
            print(f"   🔍 Expanding: {expansion_query}")
            
            # Use NO FILTERS - let semantic similarity do the work
            results = self._perform_vector_search(
                query=expansion_query,
                filters=[],  # No restrictive filters!
                top_k=20
            )
            
            # CRITICAL: Exclude Hop 1 results
            filtered_results = [
                r for r in results 
                if r.get('id') not in hop_1_ids
            ]
            
            print(f"     Found {len(results)} total, {len(filtered_results)} after exclusion")
            all_expansion_results.extend(filtered_results)
        
        # Deduplicate and return best results
        deduplicated = self._deduplicate_results(all_expansion_results)
        print(f"   ✅ Total unique Hop 2 results: {len(deduplicated)}")
        return deduplicated[:20]  # Limit final results
    
    def _generate_semantic_variations(self, original_query: str, signals: ExtractedSignals) -> List[str]:
        """Generate semantic query variations for Hop 2 expansion"""
        
        variations = []
        query_lower = original_query.lower()
        
        # Use actual business signals from Hop 1 for better matching
        if hasattr(signals, 'top_metric_terms') and signals.top_metric_terms:
            for metric in signals.top_metric_terms[:2]:
                variations.append(f"{metric} business performance")
                variations.append(f"{metric} quarterly trends")
        
        if hasattr(signals, 'top_entities') and signals.top_entities:
            for entity in signals.top_entities[:2]:
                variations.append(f"{entity} financial results")
                variations.append(f"{entity} business impact")
        
        # Add context-aware variations based on query intent  
        if any(word in query_lower for word in ['drop', 'decline', 'fall', 'decrease']):
            variations.extend([
                "business challenges and factors",
                "market conditions and trends", 
                "operating costs and expenses"
            ])
        
        if any(word in query_lower for word in ['revenue', 'sales', 'income']):
            variations.extend([
                "net sales by region",
                "revenue breakdown analysis",
                "sales performance by product"
            ])
        
        if any(word in query_lower for word in ['growth', 'strategy', 'performance']):
            variations.extend([
                "business strategy and initiatives", 
                "market expansion and growth",
                "competitive position and outlook"
            ])
        
        # Ensure we have some variations
        if not variations:
            variations = [
                "business results and analysis",
                "financial performance overview",
                "quarterly business summary"
            ]
        
        # Return unique variations (max 4)
        return list(set(variations))[:4]
    
    def _assemble_causal_trail(self, 
                              query: str,
                              hop_1_results: List[Dict[str, Any]],
                              hop_2_results: List[Dict[str, Any]], 
                              signals: ExtractedSignals) -> List[Dict[str, Any]]:
        """
        Assemble final results with causal trail focus.
        
        Creates a coherent narrative: seed → supporting evidence → conclusions
        """
        
        # Combine results
        all_results = hop_1_results + hop_2_results
        
        # Deduplicate by doc_id:page
        deduplicated = self._deduplicate_results(all_results)
        
        # Prioritize results for causal explanations
        prioritized = self._prioritize_for_explanations(deduplicated, signals)
        
        # Rerank with focus on explanatory relevance
        final_results = self._rerank_results(query, prioritized, top_k=10)
        
        # Add provenance trail metadata
        for i, result in enumerate(final_results):
            result['trail_position'] = i
            result['trail_role'] = self._determine_trail_role(result, signals, i)
        
        return final_results
    
    def _prioritize_for_explanations(self, 
                                   results: List[Dict[str, Any]], 
                                   signals: ExtractedSignals) -> List[Dict[str, Any]]:
        """
        Prioritize chunks that are most likely to contain explanatory content.
        """
        
        for result in results:
            payload = result.get('payload', {})
            
            # Boost change notes and explanatory sections
            if payload.get('is_change_note'):
                result['explanation_boost'] = 0.3
            elif any(term in payload.get('text', '').lower() 
                    for term in ['because', 'due to', 'as a result', 'caused by']):
                result['explanation_boost'] = 0.2
            else:
                result['explanation_boost'] = 0.0
            
            # Boost if contains relevant metrics
            if any(metric in payload.get('text', '').lower() 
                   for metric in signals.top_metric_terms):
                result['explanation_boost'] += 0.1
            
            # Apply boost to score
            original_score = result.get('score', 0)
            result['boosted_score'] = original_score + result['explanation_boost']
        
        # Sort by boosted score
        return sorted(results, key=lambda x: x.get('boosted_score', 0), reverse=True)
    
    def _determine_trail_role(self, result: Dict[str, Any], signals: ExtractedSignals, position: int) -> str:
        """Determine the role of this chunk in the causal trail"""
        
        payload = result.get('payload', {})
        
        if position == 0:
            return "primary_evidence"
        elif payload.get('is_change_note'):
            return "causal_explanation"
        elif payload.get('chunk_type') == 'table':
            return "supporting_data"
        else:
            return "contextual_support"
    
    def _build_applied_filters(self, signals: ExtractedSignals) -> List[str]:
        """Build list of applied filters for reporting"""
        filters = []
        
        if signals.top_metric_terms:
            filters.extend([f"metric_terms:{term}" for term in signals.top_metric_terms[:3]])
        
        if signals.top_entities:
            filters.extend([f"entities:{entity}" for entity in signals.top_entities[:2]])
        
        return filters
    
    def _build_expansion_rationale(self, signals: ExtractedSignals) -> str:
        """Build explanation of expansion strategy"""
        
        rationale_parts = []
        
        if signals.top_metric_terms:
            rationale_parts.append(f"Expanded using metrics: {', '.join(signals.top_metric_terms[:3])}")
        
        if signals.top_entities:
            rationale_parts.append(f"Explored entities: {', '.join(signals.top_entities[:2])}")
        
        if not rationale_parts:
            rationale_parts.append("Used general business context expansion")
        
        return "; ".join(rationale_parts)
