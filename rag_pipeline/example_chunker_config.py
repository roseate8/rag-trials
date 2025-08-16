"""
Example configuration for Enhanced Layout-Aware Chunker

This demonstrates how to configure the chunker for different domains and document types
without hardcoded logic.
"""

# ============================================================================
# APPLE/TECH COMPANY CONFIGURATION
# ============================================================================

APPLE_CONFIG = {
    "controlled_vocab": {
        "products": ["iPhone", "iPad", "Mac", "Apple Watch", "AirPods", "Apple TV", "HomePod"],
        "metrics": ["net_retention", "gross_margin", "wau", "mau", "revenue", "ebitda", "nps"],
        "policy_tags": ["pii_redaction", "revenue_recognition", "gdpr", "sox_compliance"]
    },
    "table_sampling_config": {
        "max_cell_samples": 10,
        "sample_row_count": 5,
        "header_word_estimate": 3
    },
    "heading_config": {
        "max_heading_level": 3,
        "title_level": 1,
        "section_level": 2,
        "subsection_level": 3
    }
}

# ============================================================================
# PHARMACEUTICAL COMPANY CONFIGURATION
# ============================================================================

PHARMA_CONFIG = {
    "controlled_vocab": {
        "products": ["Advil", "Tylenol", "Lipitor", "Humira", "Ozempic"],
        "metrics": ["clinical_trial_success", "fda_approval", "patient_enrollment", "adverse_events"],
        "policy_tags": ["clinical_data", "regulatory_compliance", "patient_safety", "fda_submission"]
    },
    "table_sampling_config": {
        "max_cell_samples": 15,  # More samples for clinical data
        "sample_row_count": 7,   # More rows for patient data
        "header_word_estimate": 4  # Medical terms tend to be longer
    },
    "heading_config": {
        "max_heading_level": 4,  # Deeper hierarchy for protocols
        "title_level": 1,
        "section_level": 2,
        "subsection_level": 3
    }
}

# ============================================================================
# MANUFACTURING COMPANY CONFIGURATION  
# ============================================================================

MANUFACTURING_CONFIG = {
    "controlled_vocab": {
        "products": ["Model X", "Model Y", "Assembly Line A", "Production Unit B"],
        "metrics": ["oee", "yield_rate", "defect_rate", "throughput", "downtime"],
        "policy_tags": ["quality_control", "safety_compliance", "environmental_impact", "iso_certification"]
    },
    "table_sampling_config": {
        "max_cell_samples": 8,
        "sample_row_count": 4,
        "header_word_estimate": 2  # Technical terms are often shorter
    },
    "heading_config": {
        "max_heading_level": 3,
        "title_level": 1,
        "section_level": 2,
        "subsection_level": 3
    }
}

# ============================================================================
# FINANCIAL SERVICES CONFIGURATION
# ============================================================================

FINANCE_CONFIG = {
    "controlled_vocab": {
        "products": ["Credit Cards", "Mortgages", "Investment Accounts", "Business Loans"],
        "metrics": ["aum", "loan_portfolio", "credit_losses", "net_interest_margin", "roi"],
        "policy_tags": ["regulatory_capital", "stress_testing", "anti_money_laundering", "consumer_protection"]
    },
    "table_sampling_config": {
        "max_cell_samples": 12,
        "sample_row_count": 6,
        "header_word_estimate": 3
    },
    "heading_config": {
        "max_heading_level": 3,
        "title_level": 1,
        "section_level": 2,
        "subsection_level": 3
    }
}

# ============================================================================
# GENERIC/UNKNOWN DOMAIN CONFIGURATION
# ============================================================================

GENERIC_CONFIG = {
    "controlled_vocab": {
        "products": [],      # No assumptions about products
        "metrics": [],       # No assumptions about metrics
        "policy_tags": []    # No assumptions about policies
    },
    "table_sampling_config": {
        "max_cell_samples": 10,
        "sample_row_count": 5,
        "header_word_estimate": 3
    },
    "heading_config": {
        "max_heading_level": 3,
        "title_level": 1,
        "section_level": 2,
        "subsection_level": 3
    }
}

# ============================================================================
# USAGE EXAMPLES
# ============================================================================

def create_chunker_for_domain(domain: str):
    """Factory function to create domain-specific chunkers."""
    from rag_pipeline.src.advanced_chunkers.layout_aware_chunker import LayoutAwareChunker
    
    configs = {
        "apple": APPLE_CONFIG,
        "pharma": PHARMA_CONFIG,
        "manufacturing": MANUFACTURING_CONFIG,
        "finance": FINANCE_CONFIG,
        "generic": GENERIC_CONFIG
    }
    
    config = configs.get(domain.lower(), GENERIC_CONFIG)
    
    return LayoutAwareChunker(
        max_words=300,
        min_words=15,
        controlled_vocab=config["controlled_vocab"],
        table_sampling_config=config["table_sampling_config"],
        heading_config=config["heading_config"]
    )

# Example usage:
# apple_chunker = create_chunker_for_domain("apple")
# pharma_chunker = create_chunker_for_domain("pharma")
# generic_chunker = create_chunker_for_domain("generic")
