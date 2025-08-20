"""
Numeric Evidence Recipe: Table-Aware 2-Hop Search

Implements the "Numeric evidence (table-aware light)" strategy:
1. Detect if query needs numbers/tables (FY, Q, %, currency)
2. First pass vector → restrict to tables or chunks with table references
3. Boost chunks whose periods/units match query
4. Return table chunks with rich metadata (table_title, page, cell_samples)
"""

import time
import re
from typing import List, Dict, Any, Set
import sys
import os

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from hop_recipes.base_recipe import BaseRecipe, HopResult
from config.config_manager import QueryIntent
from metadata_extraction.signal_extractor import MetadataSignalExtractor


class NumericEvidenceRecipe(BaseRecipe):
    """
    2-hop search recipe for numeric/financial queries like "What was Q4 revenue?"
    
    The strategy:
    1. Prioritize table chunks and chunks with table references
    2. Match temporal periods (FY24, Q1-2025) and units (%, $, million)
    3. Extract supporting context around the tables
    4. Present with rich table metadata for citations
    """
    
    def __init__(self, vector_store, embedding_generator, reranker):
        super().__init__(vector_store, embedding_generator, reranker)
        self.signal_extractor = MetadataSignalExtractor(
            max_signals_per_type=3,  # More signals for numeric queries
            min_frequency_threshold=1,
            signal_boost_factors={
                "metric_terms": 2.5,     # Financial metrics are critical
                "doc_refs": 2.0,         # Table references are key
                "mentioned_dates": 1.8,  # Temporal context is crucial
                "entities": 1.2,         # Product context matters
                "release_versions": 1.0  # Less important for numeric queries
            }
        )
        
        self._setup_numeric_patterns()
    
    def _setup_numeric_patterns(self):
        """Setup patterns for detecting numeric/temporal signals in queries"""
        
        # Period patterns (FY, quarters, years)
        self.period_patterns = [
            re.compile(r'\b(FY\d{2,4})\b', re.IGNORECASE),
            re.compile(r'\b(Q[1-4])\s*-?\s*(\d{4})\b', re.IGNORECASE),
            re.compile(r'\b(Q[1-4])\b', re.IGNORECASE),
            re.compile(r'\b(\d{4})\b'),  # Years
            re.compile(r'\b(quarter|quarterly|annual|yearly|monthly)\b', re.IGNORECASE)
        ]
        
        # Unit patterns (currency, percentages, scales)
        self.unit_patterns = [
            re.compile(r'\b(\$|USD|EUR|GBP|CNY|JPY)\b', re.IGNORECASE),
            re.compile(r'\b(%|percent|percentage)\b', re.IGNORECASE),
            re.compile(r'\b(million|billion|thousand|MM|M|B|K)\b', re.IGNORECASE),
            re.compile(r'\b(ratio|margin|rate|growth|decline)\b', re.IGNORECASE)
        ]
        
        # Numeric value patterns
        self.numeric_patterns = [
            re.compile(r'\b\d+\.\d+\b'),  # Decimal numbers
            re.compile(r'\b\d+%\b'),      # Percentages
            re.compile(r'\$\d+'),         # Currency amounts
        ]
        
        # Table reference patterns
        self.table_ref_patterns = [
            re.compile(r'\b(table|figure|chart|exhibit)\s*(\d+(?:\.\d+)?)\b', re.IGNORECASE),
            re.compile(r'\b(show me|see|refer to|according to)\s+(table|figure)\b', re.IGNORECASE)
        ]
    
    def execute_hops(self, query: str, intent: QueryIntent) -> HopResult:
        """
        Execute 2-hop numeric evidence strategy.
        
        Focus on tables and temporal/unit matching for financial queries.
        """
        start_time = time.time()
        
        # Extract numeric signals from query
        query_signals = self._extract_query_signals(query)
        print(f"📊 Numeric query signals: {query_signals}")
        
        # === HOP 1: TABLE-FOCUSED SEARCH ===
        print(f"🔍 Hop 1: Table-focused search for '{query}'")
        
        hop_1_results = self._perform_table_focused_search(query, query_signals)
        hop_1_count = len(hop_1_results)
        print(f"   Found {hop_1_count} table-related chunks")
        
        if not hop_1_results:
            # Fallback to general search
            hop_1_results = self._perform_vector_search(
                query=query,
                filters=["metric_terms:*"],  # At least find financial content
                top_k=30
            )
            hop_1_count = len(hop_1_results)
        
        # === SIGNAL EXTRACTION ===
        print("🧠 Extracting signals from table results...")
        
        extracted_signals = self.signal_extractor.extract_signals(hop_1_results)
        print(f"   Key signals: metrics={extracted_signals.top_metric_terms}, "
              f"refs={extracted_signals.top_doc_refs}")
        
        # === HOP 2: CONTEXTUAL EXPANSION ===
        hop_2_results = self._perform_contextual_expansion(
            query, extracted_signals, query_signals
        )
        
        hop_2_count = len(hop_2_results)
        print(f"🔗 Hop 2: Found {hop_2_count} contextual chunks")
        
        # === ASSEMBLY ===
        # STRICT 2-HOP REQUIREMENT: Both hops must succeed
        if hop_2_count == 0:
            raise RuntimeError(f"Graph_RAG_Not failed: Hop 2 found 0 chunks. "
                             f"Hop 1 found {hop_1_count} chunks but contextual expansion failed. "
                             f"This indicates insufficient metadata richness or overly restrictive expansion strategy.")
        
        final_chunks = self._assemble_numeric_evidence(
            query, hop_1_results, hop_2_results, extracted_signals, query_signals
        )
        
        total_time = time.time() - start_time
        
        # Build result with table metadata
        result = HopResult(
            final_chunks=final_chunks,
            hop_1_chunks=hop_1_results,
            hop_2_chunks=hop_2_results,
            extracted_signals=extracted_signals,
            applied_filters=self._build_applied_filters(extracted_signals, query_signals),
            hop_1_count=hop_1_count,
            hop_2_count=hop_2_count,
            total_time=total_time,
            strategy_used="numeric_evidence_recipe",
            expansion_rationale=self._build_numeric_rationale(extracted_signals, query_signals),
            signal_confidence=extracted_signals.confidence_score,
            result_diversity=self._calculate_result_diversity(final_chunks)
        )
        
        print(f"✅ Numeric evidence recipe complete: {len(final_chunks)} final chunks in {total_time:.2f}s")
        
        return result
    
    def _extract_query_signals(self, query: str) -> Dict[str, List[str]]:
        """Extract numeric/temporal signals from the query itself"""
        
        signals = {
            'periods': [],
            'units': [],
            'numeric_values': [],
            'table_refs': []
        }
        
        # Extract periods
        for pattern in self.period_patterns:
            matches = pattern.findall(query)
            if matches:
                if isinstance(matches[0], tuple):
                    # Handle tuple matches (Q1, 2024)
                    signals['periods'].extend([' '.join(match) if isinstance(match, tuple) else match for match in matches])
                else:
                    signals['periods'].extend(matches)
        
        # Extract units
        for pattern in self.unit_patterns:
            signals['units'].extend(pattern.findall(query))
        
        # Extract numeric values
        for pattern in self.numeric_patterns:
            signals['numeric_values'].extend(pattern.findall(query))
        
        # Extract table references
        for pattern in self.table_ref_patterns:
            matches = pattern.findall(query)
            if matches:
                signals['table_refs'].extend([' '.join(match) if isinstance(match, tuple) else match for match in matches])
        
        return signals
    
    def _perform_table_focused_search(self, query: str, query_signals: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """
        Perform Pass 1 search focused on tables and numeric content.
        
        Strategy:
        1. Search for table chunks
        2. Search for chunks with table references  
        3. Boost matches with temporal/unit alignment
        """
        
        # Search 1: Direct table chunks
        table_results = self._perform_vector_search(
            query=query,
            filters=["chunk_type:table"],
            top_k=20
        )
        
        # Search 2: Chunks with table references
        ref_results = self._perform_vector_search(
            query=query,
            filters=["doc_refs:*"],
            top_k=15
        )
        
        # Search 3: Chunks with financial metrics
        metric_results = self._perform_vector_search(
            query=query,
            filters=["metric_terms:*"],
            top_k=15
        )
        
        # Combine and boost based on query signal alignment
        all_results = table_results + ref_results + metric_results
        
        # Boost results that match query signals
        boosted_results = self._boost_by_signal_alignment(all_results, query_signals)
        
        # Deduplicate and return top results
        deduplicated = self._deduplicate_results(boosted_results)
        return sorted(deduplicated, key=lambda x: x.get('boosted_score', x.get('score', 0)), reverse=True)[:30]
    
    def _boost_by_signal_alignment(self, results: List[Dict[str, Any]], query_signals: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """
        Boost results based on alignment with query signals.
        
        This implements the "boost chunks whose periods intersect query" logic.
        """
        
        for result in results:
            payload = result.get('payload', {})
            boost_factor = 1.0
            
            # Period alignment boost
            chunk_dates = payload.get('mentioned_dates', [])
            chunk_periods = payload.get('periods', [])
            
            if query_signals['periods']:
                for query_period in query_signals['periods']:
                    # Check if query period appears in chunk dates or periods
                    if any(query_period.lower() in date.lower() for date in chunk_dates):
                        boost_factor *= 1.5
                    if any(query_period.lower() in period.lower() for period in chunk_periods):
                        boost_factor *= 1.4
            
            # Unit alignment boost
            chunk_units = payload.get('units', [])
            if query_signals['units'] and chunk_units:
                unit_overlap = set(query_signals['units']) & set(chunk_units)
                if unit_overlap:
                    boost_factor *= 1.3
            
            # Table reference boost
            if query_signals['table_refs']:
                chunk_refs = payload.get('doc_refs', [])
                ref_overlap = any(ref.lower() in ' '.join(chunk_refs).lower() 
                                for ref in query_signals['table_refs'])
                if ref_overlap:
                    boost_factor *= 1.6
            
            # Table chunk boost (tables are primary for numeric queries)
            if payload.get('chunk_type') == 'table':
                boost_factor *= 1.4
            
            result['boosted_score'] = result.get('score', 0.0) * boost_factor
        
        return results
    
    def _perform_contextual_expansion(self, 
                                    query: str,
                                    signals: Any, 
                                    query_signals: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """
        Perform Pass 2 expansion to find supporting context around tables.
        
        Look for:
        1. Chunks that reference the same metrics
        2. Chunks from the same time periods
        3. Chunks that provide interpretation/analysis
        """
        
        expansion_filters = []
        
        # Add metric-based filters
        if signals.top_metric_terms:
            for metric in signals.top_metric_terms:
                expansion_filters.append(f"metric_terms:{metric}")
        
        # Add temporal filters based on extracted signals
        if signals.mentioned_dates:
            # Use the most common year from signals
            years = [date[:4] for date in signals.mentioned_dates if len(date) >= 4]
            if years:
                most_common_year = max(set(years), key=years.count)
                expansion_filters.append(f"mentioned_dates:*{most_common_year}*")
        
        # Add document reference filters
        if signals.top_doc_refs:
            # Don't just look for exact refs, look for other refs too
            expansion_filters.append("doc_refs:*")
        
        # Look for paragraph chunks (analysis/interpretation)
        expansion_filters.append("chunk_type:paragraph")
        
        if not expansion_filters:
            expansion_filters = ["metric_terms:*"]
        
        print(f"   Contextual expansion filters: {expansion_filters}")
        
        expansion_results = self._perform_vector_search(
            query=query,
            filters=expansion_filters,
            top_k=20
        )
        
        return expansion_results
    
    def _assemble_numeric_evidence(self, 
                                  query: str,
                                  hop_1_results: List[Dict[str, Any]],
                                  hop_2_results: List[Dict[str, Any]],
                                  signals: Any,
                                  query_signals: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """
        Assemble final results with focus on numeric evidence presentation.
        
        Prioritize:
        1. Table chunks with matching periods/units
        2. Chunks with supporting analysis
        3. Cross-references that provide additional context
        """
        
        # Combine results
        all_results = hop_1_results + hop_2_results
        
        # Deduplicate
        deduplicated = self._deduplicate_results(all_results)
        
        # Prioritize for numeric evidence
        prioritized = self._prioritize_for_numeric_evidence(deduplicated, signals, query_signals)
        
        # Rerank with numeric focus
        final_results = self._rerank_results(query, prioritized, top_k=10)
        
        # Enrich with table metadata for citations
        enriched_results = self._enrich_with_table_metadata(final_results)
        
        return enriched_results
    
    def _prioritize_for_numeric_evidence(self, 
                                       results: List[Dict[str, Any]], 
                                       signals: Any,
                                       query_signals: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """Prioritize chunks most relevant for numeric evidence"""
        
        for result in results:
            payload = result.get('payload', {})
            boost_score = result.get('boosted_score', result.get('score', 0.0))
            
            # Strong boost for table chunks
            if payload.get('chunk_type') == 'table':
                boost_score *= 1.5
                
                # Extra boost if table has cell samples (rich data)
                if payload.get('cell_samples'):
                    boost_score *= 1.2
            
            # Boost chunks with multiple metric matches
            chunk_metrics = payload.get('metric_terms', [])
            metric_matches = len(set(chunk_metrics) & set(signals.top_metric_terms))
            if metric_matches > 0:
                boost_score *= (1.1 ** metric_matches)
            
            # Boost chunks with temporal alignment
            chunk_dates = payload.get('mentioned_dates', [])
            if query_signals['periods'] and chunk_dates:
                temporal_matches = sum(1 for period in query_signals['periods'] 
                                     if any(period.lower() in date.lower() for date in chunk_dates))
                if temporal_matches > 0:
                    boost_score *= (1.15 ** temporal_matches)
            
            result['final_boost_score'] = boost_score
        
        return sorted(results, key=lambda x: x.get('final_boost_score', 0), reverse=True)
    
    def _enrich_with_table_metadata(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Enrich table chunks with additional metadata for rich citations.
        
        This implements the "return table chunks with table_title, page, cell_samples" requirement.
        """
        
        enriched = []
        
        for result in results:
            payload = result.get('payload', {})
            
            if payload.get('chunk_type') == 'table':
                # Add rich table metadata for citations
                table_metadata = {
                    'table_title': payload.get('table_title', 'Table'),
                    'page': payload.get('page'),
                    'cell_samples': payload.get('cell_samples', []),
                    'col_headers': payload.get('col_headers', []),
                    'row_headers': payload.get('row_headers', []),
                    'units': payload.get('units', []),
                    'periods': payload.get('periods', []),
                    'is_table': True
                }
                
                result['table_metadata'] = table_metadata
                result['citation_type'] = 'table'
            else:
                result['citation_type'] = 'text'
            
            enriched.append(result)
        
        return enriched
    
    def _build_applied_filters(self, signals: Any, query_signals: Dict[str, List[str]]) -> List[str]:
        """Build list of filters that were applied"""
        filters = ["chunk_type:table", "doc_refs:*", "metric_terms:*"]
        
        if signals.top_metric_terms:
            filters.extend([f"metric_terms:{m}" for m in signals.top_metric_terms])
        
        if query_signals['periods']:
            filters.extend([f"mentioned_dates:*{p}*" for p in query_signals['periods']])
        
        return filters
    
    def _build_numeric_rationale(self, signals: Any, query_signals: Dict[str, List[str]]) -> str:
        """Build explanation for numeric evidence strategy"""
        
        rationale_parts = []
        
        if query_signals['periods']:
            rationale_parts.append(f"temporal focus: {', '.join(query_signals['periods'])}")
        
        if query_signals['units']:
            rationale_parts.append(f"units: {', '.join(query_signals['units'])}")
        
        if signals.top_metric_terms:
            rationale_parts.append(f"metrics: {', '.join(signals.top_metric_terms)}")
        
        if signals.top_doc_refs:
            rationale_parts.append(f"table references: {', '.join(signals.top_doc_refs)}")
        
        if rationale_parts:
            return f"Table-focused search with {', '.join(rationale_parts)}"
        else:
            return "General table and metric-based search"


def test_numeric_evidence_recipe():
    """Test the numeric evidence recipe"""
    print("=== Numeric Evidence Recipe Test ===")
    print("This would test table-aware 2-hop logic with:")
    print("1. Table-focused initial search")
    print("2. Period/unit alignment boosting")
    print("3. Contextual expansion around tables")
    print("4. Rich table metadata for citations")


if __name__ == "__main__":
    test_numeric_evidence_recipe()
