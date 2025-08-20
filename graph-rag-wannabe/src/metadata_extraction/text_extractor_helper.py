"""
Helper functions for extracting business signals from text content
"""

import re
from collections import Counter
from typing import List

def extract_business_terms_from_text(text: str, metric_terms_counter: Counter, 
                                   entities_counter: Counter, doc_refs_counter: Counter):
    """Extract business-relevant terms directly from text content"""
    
    text_lower = text.lower()
    
    # Financial/business metric terms
    financial_terms = [
        'revenue', 'sales', 'profit', 'income', 'earnings', 'margin', 'cost', 'expense',
        'growth', 'decline', 'increase', 'decrease', 'drop', 'rise', 'performance',
        'ebitda', 'roi', 'cash flow', 'dividend', 'share', 'stock'
    ]
    
    for term in financial_terms:
        if term in text_lower:
            metric_terms_counter[term] += text_lower.count(term)
    
    # Extract table/figure references
    table_refs = re.findall(r'(table\s+\d+|figure\s+\d+|chart\s+\d+)', text_lower)
    for ref in table_refs:
        doc_refs_counter[ref] += 1
    
    # Extract business entities (quarters, regions, products)
    business_entities = [
        'q1', 'q2', 'q3', 'q4', 'quarter', 'fiscal year', 'fy',
        'greater china', 'americas', 'europe', 'asia pacific',
        'iphone', 'ipad', 'mac', 'apple watch', 'airpods', 'services'
    ]
    
    for entity in business_entities:
        if entity in text_lower:
            entities_counter[entity] += text_lower.count(entity)
    
    # Extract percentage/monetary values as signals  
    percentages = re.findall(r'\d+%', text)
    for pct in percentages:
        metric_terms_counter[f'percentage_{pct}'] += 1
    
    money_values = re.findall(r'\$\d+(?:\.\d+)?\s*(?:billion|million|thousand)?', text_lower)
    for money in money_values:
        metric_terms_counter[f'amount_{money}'] += 1
