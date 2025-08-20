"""
Metadata Signal Extractor for Graph-RAG Wannabe

Extracts topical signals from Pass 1 search results to guide Pass 2 expansion.
This is the key component that simulates graph traversal by following metadata trails.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Counter, Optional, Set
from collections import Counter
import re


@dataclass
class ExtractedSignals:
    """Container for extracted metadata signals from Pass 1 results"""
    
    # Core signals for Pass 2 expansion
    top_metric_terms: List[str] = field(default_factory=list)
    top_doc_refs: List[str] = field(default_factory=list)
    top_entities: List[str] = field(default_factory=list)

    
    # Temporal signals
    mentioned_dates: List[str] = field(default_factory=list)
    date_range: Optional[Dict[str, str]] = None  # {'start': '2024-01-01', 'end': '2024-12-31'}
    
    # Structural signals
    chunk_types: Counter = field(default_factory=Counter)
    sections: List[str] = field(default_factory=list)
    pages: Set[int] = field(default_factory=set)
    
    # Quality metrics
    confidence_score: float = 0.0
    signal_strength: Dict[str, float] = field(default_factory=dict)
    expansion_suggestions: List[str] = field(default_factory=list)


class MetadataSignalExtractor:
    """
    Extract topical signals from Pass 1 search results for Pass 2 expansion.
    
    The key insight: Instead of building a graph, we use metadata co-occurrence 
    patterns to simulate graph traversal. If chunks mention specific JIRA IDs,
    release versions, or cross-references together, we expand the search to
    find more content with those same signals.
    """
    
    def __init__(self, 
                 max_signals_per_type: int = 3,
                 min_frequency_threshold: int = 2,
                 signal_boost_factors: Optional[Dict[str, float]] = None):
        """
        Initialize signal extractor.
        
        Args:
            max_signals_per_type: Maximum signals to extract per type (e.g., top 3 versions)
            min_frequency_threshold: Minimum frequency for a signal to be considered
            signal_boost_factors: Boost factors for different signal types
        """
        self.max_signals_per_type = max_signals_per_type
        self.min_frequency_threshold = min_frequency_threshold
        
        # Default boost factors (higher = more important for expansion)
        self.signal_boost_factors = signal_boost_factors or {
            "metric_terms": 2.0,     # Financial metrics are very important
            "doc_refs": 1.5,         # Cross-references create linkages
            "entities": 1.2,         # Entities provide context
            "mentioned_dates": 1.0   # Dates are useful for temporal context
        }
        
        self._setup_patterns()
    
    def _setup_patterns(self):
        """Setup regex patterns for additional signal extraction"""
        
        # JIRA ticket patterns
        self.jira_pattern = re.compile(r'\b([A-Z]+-\d+)\b')
        
        # Version patterns (more comprehensive)
        self.version_patterns = [
            re.compile(r'\bv?(\d+\.\d+(?:\.\d+)?)\b', re.IGNORECASE),
            re.compile(r'\b(release|version|build)\s+([a-zA-Z0-9\.-]+)\b', re.IGNORECASE),
            re.compile(r'\b(\d{2,4}\.\d{1,2}(?:\.\d+)?)\b')  # Year.month versions
        ]
        
        # Date range extraction
        self.date_range_pattern = re.compile(r'\b(\d{4})-(\d{1,2})-(\d{1,2})\b')
        
        # Table/Figure reference patterns
        self.reference_patterns = [
            re.compile(r'\b(Table|Figure|Chart|Exhibit)\s*(\d+(?:\.\d+)?)\b', re.IGNORECASE),
            re.compile(r'\b(Section|Chapter)\s*(\d+(?:\.\d+)*)\b', re.IGNORECASE)
        ]
    
    def extract_signals(self, search_results: List[Dict[str, Any]]) -> ExtractedSignals:
        """
        Extract metadata signals from Pass 1 search results.
        
        This is the core logic that simulates graph traversal:
        1. Count frequency of metadata values across chunks
        2. Identify the most common signals (these are our "graph edges")
        3. Score signals by frequency + relevance
        4. Select top signals for Pass 2 expansion
        
        Args:
            search_results: List of chunks from Pass 1 vector search
            
        Returns:
            ExtractedSignals object with top signals for expansion
        """
        if not search_results:
            return ExtractedSignals()
        
        # Initialize counters for each signal type
        metric_terms_counter = Counter()
        doc_refs_counter = Counter()
        entities_counter = Counter()
        dates_list = []
        chunk_types_counter = Counter()
        sections_set = set()
        pages_set = set()
        
        # Process each chunk from Pass 1 results
        for result in search_results:
            payload = result.get('payload', {})
            
            # Extract metadata signals from chunk
            self._extract_from_chunk(
                payload, 
                metric_terms_counter,
                doc_refs_counter, 
                entities_counter,
                dates_list,
                chunk_types_counter,
                sections_set,
                pages_set
            )
        
        # Score and select top signals
        signals = ExtractedSignals()
        
        # Get top signals using frequency + boost factors
        signals.top_metric_terms = self._get_top_signals(
            metric_terms_counter, 
            "metric_terms"
        )
        
        signals.top_doc_refs = self._get_top_signals(
            doc_refs_counter,
            "doc_refs"
        )
        
        signals.top_entities = self._get_top_signals(
            entities_counter,
            "entities"
        )
        

        
        # Process dates and temporal signals
        signals.mentioned_dates = list(set(dates_list))
        signals.date_range = self._extract_date_range(dates_list)
        
        # Structural signals
        signals.chunk_types = chunk_types_counter
        signals.sections = list(sections_set)
        signals.pages = pages_set
        
        # Calculate quality metrics
        signals.confidence_score = self._calculate_confidence_score(
            [metric_terms_counter, doc_refs_counter, entities_counter]
        )
        
        signals.signal_strength = self._calculate_signal_strength(
            metric_terms_counter, doc_refs_counter, entities_counter
        )
        
        # Generate expansion suggestions
        signals.expansion_suggestions = self._generate_expansion_suggestions(signals)
        
        return signals
    
    def _extract_from_chunk(self, 
                           payload: Dict[str, Any],
                           metric_terms_counter: Counter,
                           doc_refs_counter: Counter,
                           entities_counter: Counter,
                           dates_list: List[str],
                           chunk_types_counter: Counter,
                           sections_set: set,
                           pages_set: set):
        """Extract signals from chunk metadata AND text content since metadata fields are mostly empty"""
        
        # Get text content for extraction since metadata fields are empty
        text = payload.get('text', '')
        
        # Extract business terms from TEXT since metadata fields are empty
        if text:
            self._extract_business_terms_from_text(text, metric_terms_counter, entities_counter, doc_refs_counter)
        
        # Use the few populated metadata fields
        mentioned_dates = payload.get('mentioned_dates', [])
        chunk_type = payload.get('chunk_type', 'unknown')
        headings_path = payload.get('headings_path', '')
        page = payload.get('page')
        
        # Add mentioned dates to list
        for date in mentioned_dates:
            if date:
                dates_list.append(date)
        
        # Structural metadata
        chunk_types_counter[chunk_type] += 1
        
        # Use headings_path as business context signals
        if headings_path and headings_path != 'Document':
            # Treat section headings as business entities
            entities_counter[headings_path] += 1
            sections_set.add(headings_path)
        
        if page:
            pages_set.add(page)
    
    def _extract_business_terms_from_text(self, text: str, metric_terms_counter: Counter, 
                                        entities_counter: Counter, doc_refs_counter: Counter):
        """Extract business-relevant terms using universal document-agnostic approach"""
        
        # Use the universal business extractor for any document type
        from .universal_business_extractor import UniversalBusinessExtractor
        
        universal_extractor = UniversalBusinessExtractor()
        universal_signals = universal_extractor.extract_universal_signals(text)
        
        # Convert to legacy counters format for backward compatibility
        legacy_signals = universal_extractor.consolidate_to_legacy_format(universal_signals)
        
        # Update counters with universal business signals
        for term in legacy_signals['metric_terms']:
            if term:
                metric_terms_counter[term] += 1
        
        for entity in legacy_signals['entities']:
            if entity:
                entities_counter[entity] += 1
        
        for ref in legacy_signals['doc_refs']:
            if ref:
                doc_refs_counter[ref] += 1
    
    def _extract_from_text(self, text: str, versions_counter: Counter):
        """Extract additional signals from chunk text using regex patterns"""
        
        # Extract JIRA tickets
        jira_matches = self.jira_pattern.findall(text)
        versions_counter.update(jira_matches)
        
        # Extract version numbers
        for pattern in self.version_patterns:
            version_matches = pattern.findall(text)
            if version_matches:
                # Handle tuple matches from capture groups
                if isinstance(version_matches[0], tuple):
                    # For patterns like r'(release|version)\s+([a-zA-Z0-9\.-]+)'
                    versions_counter.update([match[1] if len(match) > 1 else match[0] 
                                           for match in version_matches])
                else:
                    versions_counter.update(version_matches)
    
    def _get_top_signals(self, counter: Counter, signal_type: str) -> List[str]:
        """
        Get top signals for a given type, applying boost factors.
        
        This is where we simulate graph edge importance:
        - Higher frequency = stronger "edge" in our simulated graph
        - Boost factors emphasize certain signal types
        """
        if not counter:
            return []
        
        boost_factor = self.signal_boost_factors.get(signal_type, 1.0)
        
        # Apply frequency threshold and boost
        filtered_items = [
            (item, count * boost_factor) 
            for item, count in counter.items() 
            if count >= self.min_frequency_threshold
        ]
        
        # Sort by boosted score and take top N
        filtered_items.sort(key=lambda x: x[1], reverse=True)
        return [item for item, score in filtered_items[:self.max_signals_per_type]]
    
    def _extract_date_range(self, dates_list: List[str]) -> Optional[Dict[str, str]]:
        """Extract date range from mentioned dates for temporal filtering"""
        if not dates_list:
            return None
        
        # Parse dates and find range
        parsed_dates = []
        for date_str in dates_list:
            # Try to extract year at minimum
            year_match = re.search(r'\b(\d{4})\b', date_str)
            if year_match:
                parsed_dates.append(year_match.group(1))
        
        if parsed_dates:
            min_year = min(parsed_dates)
            max_year = max(parsed_dates)
            return {
                'start': f"{min_year}-01-01",
                'end': f"{max_year}-12-31"
            }
        
        return None
    
    def _calculate_confidence_score(self, counters: List[Counter]) -> float:
        """Calculate confidence score based on signal strength and consistency"""
        
        total_signals = sum(sum(counter.values()) for counter in counters)
        unique_signals = sum(len(counter) for counter in counters)
        
        if unique_signals == 0:
            return 0.0
        
        # Higher confidence when we have many repeated signals
        # (indicates strong thematic consistency)
        avg_frequency = total_signals / unique_signals if unique_signals > 0 else 0
        confidence = min(avg_frequency / 5.0, 1.0)  # Normalize to 0-1
        
        return confidence
    
    def _calculate_signal_strength(self, *counters: Counter) -> Dict[str, float]:
        """Calculate strength score for each signal type"""
        
        signal_types = ['metric_terms', 'doc_refs', 'entities']
        strength = {}
        
        for i, signal_type in enumerate(signal_types):
            counter = counters[i]
            if counter:
                # Strength = total occurrences * diversity
                total_count = sum(counter.values())
                diversity = len(counter)
                strength[signal_type] = (total_count * diversity) / 10.0  # Normalize
            else:
                strength[signal_type] = 0.0
        
        return strength
    
    def _generate_expansion_suggestions(self, signals: ExtractedSignals) -> List[str]:
        """Generate human-readable suggestions for Pass 2 expansion"""
        
        suggestions = []
        
        if signals.top_metric_terms:
            suggestions.append(f"Expand search for metrics: {', '.join(signals.top_metric_terms[:2])}")
        

        
        if signals.top_doc_refs:
            suggestions.append(f"Follow references: {', '.join(signals.top_doc_refs[:2])}")
        
        if signals.date_range:
            start_year = signals.date_range['start'][:4]
            end_year = signals.date_range['end'][:4]
            if start_year == end_year:
                suggestions.append(f"Temporal focus: {start_year}")
            else:
                suggestions.append(f"Temporal range: {start_year}-{end_year}")
        
        if 'table' in signals.chunk_types and signals.chunk_types['table'] > 1:
            suggestions.append("Prioritize table content for numeric evidence")
        
        return suggestions


def test_signal_extractor():
    """Test the signal extractor with mock search results"""
    
    # Mock search results that simulate what we'd get from Pass 1
    mock_results = [
        {
            'payload': {
                'text': 'Revenue declined 15% in Q4 2024 due to release v2.1 issues',
                'metric_terms': ['revenue', 'decline'],
                'entities': ['Q4'],
                'mentioned_dates': ['2024', 'Q4'],
                'doc_refs': ['Table 3'],
                'chunk_type': 'paragraph',
                'headings_path': 'Financial Results > Q4 Analysis',
                'page': 45
            }
        },
        {
            'payload': {
                'text': 'Table 3 shows revenue impact from PROJ-1234 implementation',
                'metric_terms': ['revenue', 'impact'],
                'entities': ['PROJ-1234'],
                'mentioned_dates': ['2024'],
                'doc_refs': ['Table 3', 'Figure 2'],
                'chunk_type': 'table',
                'headings_path': 'Financial Results > Q4 Analysis',
                'page': 46
            }
        },
        {
            'payload': {
                'text': 'User engagement dropped after v2.1 release',
                'metric_terms': ['engagement'],
                'entities': ['v2.1'],
                'mentioned_dates': ['2024'],
                'doc_refs': [],
                'chunk_type': 'paragraph',
                'headings_path': 'Product Analysis > Impact Assessment',
                'page': 52
            }
        }
    ]
    
    extractor = MetadataSignalExtractor()
    signals = extractor.extract_signals(mock_results)
    
    print("=== Signal Extraction Test ===")
    print(f"Top metric terms: {signals.top_metric_terms}")
    print(f"Top doc refs: {signals.top_doc_refs}")
    print(f"Top entities: {signals.top_entities}")

    print(f"Date range: {signals.date_range}")
    print(f"Chunk types: {dict(signals.chunk_types)}")
    print(f"Sections: {signals.sections}")
    print(f"Confidence: {signals.confidence_score:.2f}")
    print(f"Signal strength: {signals.signal_strength}")
    print(f"Expansion suggestions: {signals.expansion_suggestions}")


if __name__ == "__main__":
    test_signal_extractor()
