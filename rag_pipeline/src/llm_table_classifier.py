"""
LLM-based Table Classification

Uses LLM to generate intelligent table headings based on content analysis,
replacing hardcoded pattern matching.
"""

import logging
import openai
from typing import List, Dict, Any, Optional
import json

logger = logging.getLogger(__name__)

class LLMTableClassifier:
    """
    Generate intelligent table titles using LLM analysis of table content.
    
    Replaces hardcoded pattern matching with dynamic content understanding.
    """
    
    def __init__(self, openai_api_key: str):
        """Initialize with OpenAI API key."""
        self.client = openai.OpenAI(api_key=openai_api_key)
        self.total_tokens_used = 0
        self.total_calls_made = 0
        
        # Track tokens by method for detailed reporting
        self.tokens_by_method = {
            "table_classification": 0,
            "table_title_generation": 0
        }
        self.calls_by_method = {
            "table_classification": 0,
            "table_title_generation": 0
        }
        
        logger.info("LLM Table Classifier initialized")
    
    def generate_table_title(self, table_data: Dict[str, Any], table_number: int) -> str:
        """
        Generate an intelligent table title based on content analysis.
        
        Args:
            table_data: Table structure with cells, headers, etc.
            table_number: Table sequence number for fallback
            
        Returns:
            Generated table title
        """
        
        # Extract meaningful content for analysis
        content_sample = self._extract_content_sample(table_data)
        
        if not content_sample.strip():
            return f"Table {table_number}: Data Table"
        
        # Optimized LLM prompt for table classification (minimal tokens, high quality)
        classification_prompt = f"""Table content:
{content_sample}

Generate a descriptive title (max 5 words). Examples:
- "Revenue by Quarter"
- "Employee Count by Division" 
- "Financial Performance Summary"

Title:"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Generate concise table titles."},
                    {"role": "user", "content": classification_prompt}
                ],
                temperature=0.1,  # Low temperature for consistent results
                max_tokens=20     # Short titles only
            )
            
            title = response.choices[0].message.content.strip()
            
            # Track token usage by method
            tokens_used = response.usage.total_tokens
            self.total_tokens_used += tokens_used
            self.total_calls_made += 1
            
            # Track by specific method
            method_name = "table_classification"
            self.tokens_by_method[method_name] += tokens_used
            self.calls_by_method[method_name] += 1
            
            # Clean up the title
            title = title.replace('"', '').replace("'", '')
            
            # Validate title length and content
            if len(title.split()) > 6 or len(title) > 50:
                title = f"Table {table_number}: Financial Data"
            elif not title or title.lower() in ['table', 'data', 'unknown']:
                title = f"Table {table_number}: Business Data"
            
            logger.info(f"🤖 LLM table title: '{title}' ({tokens_used} tokens, total: {self.total_tokens_used})")
            return title
            
        except Exception as e:
            logger.warning(f"LLM table classification failed: {e}")
            return f"Table {table_number}: Data Table"
    
    def _extract_content_sample(self, table_data: Dict[str, Any], max_cells: int = 20) -> str:
        """
        Extract a representative sample of table content for LLM analysis.
        
        Args:
            table_data: Table structure
            max_cells: Maximum number of cells to include
            
        Returns:
            Formatted content sample
        """
        content_parts = []
        cells_added = 0
        
        # Get cells from the processed matrix
        cell_matrix = table_data.get("cell_matrix", {})
        if not cell_matrix:
            return ""
        
        # Extract column headers
        column_headers = cell_matrix.get("headers", {}).get("column", [])
        if column_headers:
            header_texts = [h.get("text", "").strip() for h in column_headers if h.get("text")]
            if header_texts:
                content_parts.append("Headers: " + " | ".join(header_texts[:5]))
                cells_added += len(header_texts[:5])
        
        # Extract some data rows
        rows = cell_matrix.get("rows", [])
        for row_idx, row in enumerate(rows):
            if cells_added >= max_cells:
                break
            if row_idx > 3:  # Limit to first few rows
                break
                
            row_texts = []
            for cell in row:
                if cell and cell.get("text"):
                    text = cell["text"].strip()
                    if text and len(text) < 50:  # Skip very long cells
                        row_texts.append(text)
                        cells_added += 1
                        if cells_added >= max_cells:
                            break
            
            if row_texts:
                content_parts.append("Row: " + " | ".join(row_texts))
        
        return "\n".join(content_parts)
    
    def get_token_usage_stats(self) -> Dict[str, Any]:
        """Get comprehensive token usage statistics for table classification."""
        avg_tokens = self.total_tokens_used / max(1, self.total_calls_made)
        return {
            "total_tokens_used": self.total_tokens_used,
            "total_calls_made": self.total_calls_made,
            "average_tokens_per_call": round(avg_tokens, 2),
            "tokens_by_method": self.tokens_by_method.copy(),
            "calls_by_method": self.calls_by_method.copy()
        }
    
    def reset_token_stats(self) -> None:
        """Reset token usage statistics for new chunking session."""
        self.total_tokens_used = 0
        self.total_calls_made = 0
        self.tokens_by_method = {
            "table_classification": 0,
            "table_title_generation": 0
        }
        self.calls_by_method = {
            "table_classification": 0,
            "table_title_generation": 0
        }


# Tiny POC test function
def test_llm_table_classifier():
    """Test the LLM table classifier with sample data."""
    
    # Sample table data (EPS table structure)
    sample_eps_table = {
        "cell_matrix": {
            "headers": {
                "column": [
                    {"text": "Years ended"},
                    {"text": "September 28, 2024"}, 
                    {"text": "September 30, 2023"},
                    {"text": "September 24, 2022"}
                ]
            },
            "rows": [
                [
                    {"text": "Earnings per share:"},
                    {"text": ""},
                    {"text": ""},
                    {"text": ""}
                ],
                [
                    {"text": "Basic"},
                    {"text": "$6.11"},
                    {"text": "$6.16"}, 
                    {"text": "$6.15"}
                ],
                [
                    {"text": "Diluted"},
                    {"text": "$6.08"},
                    {"text": "$6.13"},
                    {"text": "$6.11"}
                ]
            ]
        }
    }
    
    # Sample revenue table
    sample_revenue_table = {
        "cell_matrix": {
            "headers": {
                "column": [
                    {"text": "Product"},
                    {"text": "2024 Revenue"},
                    {"text": "2023 Revenue"}
                ]
            },
            "rows": [
                [
                    {"text": "iPhone"},
                    {"text": "$200,583"}, 
                    {"text": "$200,583"}
                ],
                [
                    {"text": "Mac"},
                    {"text": "$29,357"},
                    {"text": "$29,357"}
                ],
                [
                    {"text": "Services"},
                    {"text": "$96,169"},
                    {"text": "$85,200"}
                ]
            ]
        }
    }
    
    return sample_eps_table, sample_revenue_table
