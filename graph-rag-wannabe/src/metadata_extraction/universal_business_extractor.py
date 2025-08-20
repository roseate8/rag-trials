"""
Universal Business Metadata Extractor
Extracts business-relevant signals from ANY document type, not just Apple/financial docs
"""

import re
from collections import Counter
from typing import Dict, List, Any, Set
from dataclasses import dataclass

@dataclass
class UniversalBusinessSignals:
    """Document-agnostic business signals"""
    
    # Financial & performance metrics (universal)
    financial_terms: List[str] = None
    performance_indicators: List[str] = None
    
    # Organizational entities (universal)
    business_entities: List[str] = None
    geographic_regions: List[str] = None
    time_periods: List[str] = None
    
    # Document structure (universal)
    document_references: List[str] = None
    sections_hierarchy: List[str] = None
    
    # Numerical insights (universal)
    percentages: List[str] = None
    monetary_values: List[str] = None
    quantities: List[str] = None
    
    def __post_init__(self):
        # Initialize empty lists if None
        for field_name, field_value in self.__dict__.items():
            if field_value is None:
                setattr(self, field_name, [])

class UniversalBusinessExtractor:
    """
    Document-agnostic business metadata extractor.
    
    Works for ANY business document:
    - Financial reports (Apple, Microsoft, Tesla, etc.)
    - Business proposals & strategies  
    - Marketing documents
    - Technical specifications with business context
    - Legal/compliance documents
    - Research reports
    """
    
    def __init__(self):
        self._setup_universal_patterns()
    
    def _setup_universal_patterns(self):
        """Setup patterns that work across ALL business document types"""
        
        # Universal financial terms (not company-specific)
        self.financial_terms = [
            # Revenue & Income
            'revenue', 'income', 'sales', 'earnings', 'profit', 'loss', 'margin',
            'gross', 'net', 'operating', 'ebitda', 'cash flow', 'dividend',
            
            # Costs & Expenses  
            'cost', 'expense', 'expenditure', 'investment', 'budget', 'spend',
            'capex', 'opex', 'overhead', 'liability', 'debt',
            
            # Performance Metrics
            'growth', 'decline', 'increase', 'decrease', 'performance', 'efficiency',
            'productivity', 'roi', 'return', 'yield', 'conversion', 'retention',
            
            # Market & Competition
            'market share', 'competition', 'competitive', 'advantage', 'strategy',
            'opportunity', 'risk', 'threat', 'demand', 'supply'
        ]
        
        # Universal business performance indicators
        self.performance_patterns = [
            r'\b(improved?|increased?|grew|growth|rose|rising|up|higher)\b',
            r'\b(declined?|decreased?|dropped?|fell|falling|down|lower)\b',
            r'\b(stable|maintained?|steady|consistent|flat)\b',
            r'\b(volatile|fluctuat\w+|variable|inconsistent)\b'
        ]
        
        # Universal organizational entities (scalable)
        self.org_entity_patterns = [
            # Departments & Functions
            r'\b(marketing|sales|finance|hr|human resources|it|technology|operations|legal|compliance)\b',
            r'\b(research|development|r&d|engineering|design|manufacturing|production)\b',
            
            # Business Units & Divisions
            r'\b(division|unit|department|team|group|subsidiary|affiliate)\b',
            r'\b(business unit|product line|service line|vertical)\b',
            
            # Stakeholders
            r'\b(customer|client|vendor|supplier|partner|investor|shareholder|stakeholder)\b',
            r'\b(management|executive|board|director|ceo|cfo|cto|cmo)\b'
        ]
        
        # Universal geographic patterns
        self.geographic_patterns = [
            # Regions
            r'\b(north america|south america|europe|asia|africa|oceania|middle east)\b',
            r'\b(americas|emea|apac|asia pacific|latin america)\b',
            
            # Countries (major business markets)
            r'\b(united states|usa|canada|mexico|brazil|uk|germany|france|china|japan|india|australia)\b',
            r'\b(singapore|hong kong|south korea|taiwan|thailand|vietnam|indonesia)\b',
            
            # Cities (major business hubs)
            r'\b(new york|london|tokyo|singapore|hong kong|shanghai|mumbai|berlin|paris|toronto)\b'
        ]
        
        # Universal time period patterns
        self.time_patterns = [
            # Fiscal periods
            r'\b(q[1-4]|quarter|fiscal year|fy|fiscal)\s*\d{2,4}\b',
            r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{4}\b',
            r'\b\d{4}\s*(q[1-4]|quarter)\b',
            
            # Business cycles
            r'\b(monthly|quarterly|annually|yearly|biannual|semi-annual)\b',
            r'\b(year over year|yoy|year-to-date|ytd|month over month|mom)\b'
        ]
        
        # Universal document reference patterns
        self.doc_ref_patterns = [
            r'\b(table|figure|chart|graph|exhibit|appendix|attachment)\s+[a-z0-9\-\.]+\b',
            r'\b(section|chapter|part|page|paragraph)\s+[a-z0-9\-\.]+\b',
            r'\b(reference|ref|see|refer to|as shown in)\s+[a-z0-9\-\.]+\b'
        ]
        
        # Universal numerical patterns
        self.numerical_patterns = {
            'percentages': r'\d+(?:\.\d+)?%',
            'monetary': r'[\$€£¥₹]\s*\d+(?:,\d{3})*(?:\.\d{2})?(?:\s*(?:million|billion|trillion|k|m|b))?',
            'quantities': r'\d+(?:,\d{3})*(?:\s*(?:units|pieces|items|customers|employees|users|subscribers))?'
        }
    
    def extract_universal_signals(self, text: str, headings_path: str = "", 
                                chunk_type: str = "", metadata: Dict = None) -> UniversalBusinessSignals:
        """
        Extract business signals from ANY business document.
        
        Args:
            text: The chunk text content
            headings_path: Document structure context
            chunk_type: Type of chunk (table, paragraph, etc.)
            metadata: Additional metadata if available
            
        Returns:
            UniversalBusinessSignals with extracted business context
        """
        
        signals = UniversalBusinessSignals()
        text_lower = text.lower()
        
        # 1. Extract Financial Terms (Universal)
        financial_counter = Counter()
        for term in self.financial_terms:
            count = text_lower.count(term)
            if count > 0:
                financial_counter[term] = count
        
        signals.financial_terms = [term for term, count in financial_counter.most_common(5)]
        
        # 2. Extract Performance Indicators (Universal)
        performance_terms = []
        for pattern in self.performance_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            performance_terms.extend(matches)
        
        signals.performance_indicators = list(set(performance_terms))[:5]
        
        # 3. Extract Business Entities (Universal)
        entity_matches = []
        for pattern in self.org_entity_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            entity_matches.extend(matches)
        
        signals.business_entities = list(set(entity_matches))[:5]
        
        # 4. Extract Geographic Regions (Universal)
        geo_matches = []
        for pattern in self.geographic_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            geo_matches.extend(matches)
        
        signals.geographic_regions = list(set(geo_matches))[:5]
        
        # 5. Extract Time Periods (Universal)
        time_matches = []
        for pattern in self.time_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            time_matches.extend(matches)
        
        signals.time_periods = list(set(time_matches))[:5]
        
        # 6. Extract Document References (Universal)
        doc_ref_matches = []
        for pattern in self.doc_ref_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            doc_ref_matches.extend(matches)
        
        signals.document_references = list(set(doc_ref_matches))[:5]
        
        # 7. Extract Numerical Insights (Universal)
        for num_type, pattern in self.numerical_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            
            if num_type == 'percentages':
                signals.percentages = list(set(matches))[:5]
            elif num_type == 'monetary':
                signals.monetary_values = list(set(matches))[:5]
            elif num_type == 'quantities':
                signals.quantities = list(set(matches))[:5]
        
        # 8. Use Headings Path as Section Hierarchy (Universal)
        if headings_path and headings_path != 'Document':
            # Parse hierarchical structure
            sections = headings_path.split(' > ') if ' > ' in headings_path else [headings_path]
            signals.sections_hierarchy = [s.strip() for s in sections if s.strip()]
        
        return signals
    
    def consolidate_to_legacy_format(self, universal_signals: UniversalBusinessSignals) -> Dict[str, List[str]]:
        """
        Convert universal signals to legacy format for backward compatibility.
        
        Maps universal business signals to the expected format:
        - metric_terms: Financial terms + performance indicators
        - entities: Business entities + geographic regions + time periods  
        - doc_refs: Document references
        """
        
        return {
            'metric_terms': (
                universal_signals.financial_terms + 
                universal_signals.performance_indicators +
                [f"amount_{v}" for v in universal_signals.monetary_values] +
                [f"percentage_{p}" for p in universal_signals.percentages]
            )[:10],  # Limit to top 10
            
            'entities': (
                universal_signals.business_entities +
                universal_signals.geographic_regions + 
                universal_signals.time_periods +
                universal_signals.sections_hierarchy
            )[:10],  # Limit to top 10
            
            'doc_refs': universal_signals.document_references[:5]
        }

def test_universal_extractor():
    """Test the universal extractor with different document types"""
    
    extractor = UniversalBusinessExtractor()
    
    # Test cases for different document types
    test_cases = [
        {
            'name': 'Financial Report (Apple)',
            'text': 'Revenue declined 8% in Q4 2024 due to lower iPhone sales in Greater China. Operating margin decreased to 25.2%.'
        },
        {
            'name': 'Tech Startup Report',
            'text': 'User growth increased 150% this quarter. Monthly recurring revenue reached $2.3 million. Customer retention improved significantly in Europe.'
        },
        {
            'name': 'Manufacturing Report',
            'text': 'Production efficiency increased by 12% in our Ohio facility. Cost reduction initiatives saved $5.8 million annually.'
        },
        {
            'name': 'Marketing Analysis',
            'text': 'Campaign conversion rates improved from 2.1% to 3.8%. ROI on digital marketing spend increased 45% year over year.'
        }
    ]
    
    print("🧪 UNIVERSAL BUSINESS EXTRACTOR TEST")
    print("="*60)
    
    for test_case in test_cases:
        print(f"\n📄 {test_case['name']}:")
        print(f"Text: {test_case['text']}")
        
        signals = extractor.extract_universal_signals(test_case['text'])
        legacy_format = extractor.consolidate_to_legacy_format(signals)
        
        print(f"✅ Financial Terms: {signals.financial_terms}")
        print(f"✅ Entities: {signals.business_entities + signals.geographic_regions}")
        print(f"✅ Performance: {signals.performance_indicators}")
        print(f"✅ Legacy Format: {legacy_format}")
        print("-" * 40)

if __name__ == "__main__":
    test_universal_extractor()
