"""
Enhanced Layout-aware document chunking with rich metadata and deterministic enrichment.

This implementation combines layout-aware chunking with GraphRAG-style metadata:
1. Hierarchical chunking: Title → Section → Subsection (never cross boundaries)
2. Element-specific handling: Tables, Lists, Paragraphs, Key-Value pairs
3. Canonical document identification and versioning
4. Deterministic metadata extraction (entities, dates, references)
5. Spatial context preservation (page numbers, bounding boxes)
6. Table-aware indexing for enhanced recall
"""
from __future__ import annotations

import json
import logging
import os
import re
import hashlib
import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime

logger = logging.getLogger(__name__)

# Import LLM classifier for intelligent table titles
try:
    from ..llm_table_classifier import LLMTableClassifier
except ImportError:
    LLMTableClassifier = None
    logger.warning("LLMTableClassifier not available - falling back to rule-based titles")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def count_alnum_words(text: str) -> int:
    """Count alphanumeric words in text."""
    tokens = re.findall(r"[A-Za-z0-9]+", text or "")
    return len(tokens)


def make_hash(payload: Any) -> str:
    """Generate hash for content deduplication."""
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def generate_stable_doc_id(file_path: str, doc_name: Optional[str] = None) -> str:
    """Generate stable document ID based on filename or provided name."""
    if doc_name:
        doc_id = re.sub(r'[^a-zA-Z0-9_-]', '_', doc_name.lower())
        return re.sub(r'_+', '_', doc_id).strip('_')
    
    basename = os.path.basename(file_path).replace('.md', '').replace('.json', '')
    doc_id = re.sub(r'[^a-zA-Z0-9_-]', '_', basename.lower())
    return re.sub(r'_+', '_', doc_id).strip('_')


def extract_folder_hierarchy(file_path: str) -> Dict[str, Any]:
    """
    Extract folder hierarchy from file path for any level of nesting.
    
    Examples:
        input/doc.html -> {"folder_path": [], "folder_hierarchy": "", "depth": 0}
        input/reports/doc.html -> {"folder_path": ["reports"], "folder_hierarchy": "reports", "depth": 1}
        input/a/b/c/d/doc.html -> {"folder_path": ["a","b","c","d"], "folder_hierarchy": "a/b/c/d", "depth": 4}
    """
    # Normalize path separators
    normalized_path = file_path.replace('\\', '/')
    
    # Extract file name
    file_name = os.path.basename(normalized_path)
    
    # Get directory path relative to input/
    dir_path = os.path.dirname(normalized_path)
    
    # Find the input directory and extract everything after it
    folder_path = []
    relative_path = normalized_path
    
    # Split path into components
    path_parts = [part for part in dir_path.split('/') if part]
    
    # Find 'input' in the path and take everything after it
    input_index = -1
    for i, part in enumerate(path_parts):
        if part == 'input':
            input_index = i
            break
    
    if input_index >= 0 and input_index < len(path_parts) - 1:
        # Extract folders after 'input/'
        folder_path = path_parts[input_index + 1:]
        relative_path = '/'.join(folder_path + [file_name]) if folder_path else file_name
    elif input_index >= 0:
        # File is directly in input/ folder
        folder_path = []
        relative_path = file_name
    else:
        # 'input' not found in path - use entire directory structure
        folder_path = path_parts
        relative_path = normalized_path
    
    # Create hierarchy string
    folder_hierarchy = '/'.join(folder_path) if folder_path else ""
    
    return {
        "folder_path": folder_path,
        "folder_hierarchy": folder_hierarchy,
        "file_name": file_name,
        "relative_path": relative_path,
        "folder_depth": len(folder_path)
    }


def create_headings_path(lineage: Dict[str, Any]) -> str:
    """Create hierarchical headings path like 'Title > Section > Subsection'."""
    path_parts = []
    if lineage.get("title_text"):
        path_parts.append(lineage["title_text"])
    if lineage.get("section_text"):
        path_parts.append(lineage["section_text"])
    if lineage.get("subsection_text"):
        path_parts.append(lineage["subsection_text"])
    return " > ".join(path_parts) if path_parts else "Document"


# ============================================================================
# METADATA EXTRACTION
# ============================================================================

class MetadataExtractor:
    """
    Deterministic metadata extraction using regex/heuristic patterns.
    Fast, cheap, and provides high-ROI metadata enrichment without LLM calls.
    """
    
    def __init__(
        self, 
        controlled_vocab: Optional[Dict[str, List[str]]] = None,
        table_config: Optional[Dict[str, int]] = None
    ):
        # Empty controlled vocabulary by default - must be provided externally
        # This ensures no domain-specific hardcoding in the chunker
        self.controlled_vocab = controlled_vocab or {
            "products": [],     # No default products - provide via config
            "metrics": [],      # No default metrics - provide via config  
            "policy_tags": []   # No default policy tags - provide via config
        }
        
        # Configurable table sampling parameters (no magic numbers)
        self.table_config = table_config or {
            "max_cell_samples": 10,      # Max key-value pairs to extract from tables
            "sample_row_count": 5,       # Number of rows to sample from
            "header_word_estimate": 3    # Average words per column header for estimation
        }
        
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for efficient matching."""
        # Date patterns
        self.date_patterns = [
            re.compile(r'\b\d{4}-\d{2}-\d{2}\b'),  # ISO dates
            re.compile(r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b'),
            re.compile(r'\bFY\d{2,4}\b'),
            re.compile(r'\bQ[1-4]-?\d{4}\b'),
        ]
        
        # Document reference patterns
        self.doc_ref_patterns = [
            re.compile(r'\b(?:Table|Figure|Chart|Exhibit)\s+\d+(?:\.\d+)?\b', re.IGNORECASE),
            re.compile(r'\b(?:Section|Chapter)\s+\d+(?:\.\d+)*\b', re.IGNORECASE),
            re.compile(r'\b(?:see|refer to|as shown in)\s+(?:Table|Figure|Section)\s+\d+', re.IGNORECASE)
        ]
        
        # Change note patterns
        self.change_patterns = [
            re.compile(r'\b(?:deprecated|migrated|renamed|changed|updated|modified|replaced|removed)\b', re.IGNORECASE)
        ]
        
        # Unit patterns for tables
        self.unit_patterns = [
            re.compile(r'\b(?:USD|EUR|GBP|CNY|JPY)\b'),
            re.compile(r'\b(?:%|percent|percentage)\b', re.IGNORECASE),
            re.compile(r'\b(?:MM|million|billion|thousand|K|M|B)\b', re.IGNORECASE),
        ]
        
        # Period patterns for tables
        self.period_patterns = [
            re.compile(r'\bFY\d{2,4}\b'),
            re.compile(r'\bQ[1-4]-?\d{4}\b'),
            re.compile(r'\b\d{4}\b')  # Simple year
        ]
    
    def extract_metadata(self, text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Extract comprehensive metadata from text using deterministic patterns."""
        metadata = {
            # References
            "metric_terms": self._extract_metric_terms(text),
            "doc_refs": self._extract_doc_refs(text),
            "entities": self._extract_entities(text),
            
            # Temporal signals
            "mentioned_dates": self._extract_dates(text),
            "is_change_note": self._is_change_note(text),
            
            # Governance
            "policy_tags": self._extract_policy_tags(text)
        }
        
        return metadata
    
    def _extract_metric_terms(self, text: str) -> List[str]:
        """Extract metric terms from controlled vocabulary."""
        text_lower = text.lower()
        return [metric for metric in self.controlled_vocab.get("metrics", []) 
                if metric.lower() in text_lower]
    
    def _extract_doc_refs(self, text: str) -> List[str]:
        """Extract document references."""
        refs = []
        for pattern in self.doc_ref_patterns:
            refs.extend(pattern.findall(text))
        return list(set(refs))
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract entities semantically and from controlled vocabulary."""
        entities = []
        
        # 1. Semantic extraction of product names using patterns
        semantic_entities = self._extract_semantic_entities(text)
        entities.extend(semantic_entities)
        
        # 2. Controlled vocabulary (if provided)
        text_lower = text.lower()
        for product in self.controlled_vocab.get("products", []):
            if product.lower() in text_lower and product not in entities:
                entities.append(product)
        
        return list(set(entities))  # Remove duplicates
    
    def _extract_semantic_entities(self, text: str) -> List[str]:
        """Extract product names and entities semantically from context."""
        entities = []
        
        # Product name patterns (semantic detection)
        product_patterns = [
            # Product version patterns: "ProductName v1.2", "Product 2.0"  
            r'\b([A-Z][a-zA-Z]*(?:[A-Z][a-zA-Z]*)*)\s+(?:v\d+\.\d+|\d+\.\d+|version\s+\d+)\b',
            # Branded products: "DataAnalytics solution", "CloudServices platform"
            r'\b([A-Z][a-zA-Z]*(?:[A-Z][a-zA-Z]*)*)\s+(?:solution|platform|product|service|system|suite)\b',
            # Product launches: "introducing SecuritySuite", "announcing DataPlatform"  
            r'\b(?:introducing|announcing|launching|releasing)\s+(?:the\s+)?(?:new\s+)?([A-Z][a-zA-Z]*(?:[A-Z][a-zA-Z]*)*)\b',
            # Product with features: "SearchEngine offers", "Analytics provides"
            r'\b([A-Z][a-zA-Z]*(?:[A-Z][a-zA-Z]*)*)\s+(?:offers|provides|enables|supports|includes|delivers)\b',
            # Capitalized compound words: "DataAnalytics", "CloudServices"  
            r'\b([A-Z][a-zA-Z]*[A-Z][a-zA-Z]*)\b',
            # Our/The product pattern: "our SearchPlatform", "the new SecurityTool"
            r'\b(?:our|the)\s+(?:new\s+)?([A-Z][a-zA-Z]*(?:[A-Z][a-zA-Z]*)*)\b'
        ]
        
        for pattern in product_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                match = match.strip()
                # Filter out common words, single letters, and generic terms
                common_words = {
                    'the', 'our', 'this', 'that', 'new', 'old', 'first', 'last', 'are', 'and', 'for', 'all',
                    'system', 'product', 'solution', 'platform', 'service', 'customer', 'customers', 'enterprise',
                    'business', 'data', 'user', 'users', 'team', 'company', 'market', 'markets', 'time', 'now',
                    'version', 'latest', 'recent', 'enhanced', 'improved', 'advanced', 'comprehensive', 'better'
                }
                if (len(match) > 3 and 
                    match.lower() not in common_words and
                    len(match.split()) <= 2 and  # Max 2 words for product names
                    match[0].isupper() and  # Must start with capital
                    not match.lower().endswith('ing') and  # Filter out gerunds
                    not match.lower().endswith('ed')):  # Filter out past participles
                    entities.append(match)
        
        return entities
    
    def _extract_dates(self, text: str) -> List[str]:
        """Extract date mentions."""
        dates = []
        for pattern in self.date_patterns:
            dates.extend(pattern.findall(text))
        return sorted(list(set(dates)))
    
    def _is_change_note(self, text: str) -> bool:
        """Check if text indicates a change/update."""
        return any(pattern.search(text) for pattern in self.change_patterns)
    
    def _extract_policy_tags(self, text: str) -> List[str]:
        """Extract policy/compliance tags and semantic action indicators."""
        text_lower = text.lower()
        tags = []
        
        # 1. Semantic detection of strategic elements
        semantic_tags = self._extract_semantic_policy_tags(text)
        tags.extend(semantic_tags)
        
        # 2. Controlled vocabulary (if provided)
        for tag in self.controlled_vocab.get("policy_tags", []):
            if tag.lower() in text_lower and tag not in tags:
                tags.append(tag)
        
        return list(set(tags))
    
    def _extract_semantic_policy_tags(self, text: str) -> List[str]:
        """Extract policy and strategic elements semantically."""
        tags = []
        text_lower = text.lower()
        
        # Decision indicators
        decision_patterns = [
            r'\b(?:decided|decision|determined|concluded|resolved|chose|selected)\b',
            r'\b(?:board\s+(?:decided|approved)|management\s+(?:decided|approved))\b',
            r'\b(?:we\s+(?:decided|chose|determined|resolved))\b'
        ]
        
        # Update/Change indicators  
        update_patterns = [
            r'\b(?:updated|upgraded|improved|enhanced|modified|changed|revised)\b',
            r'\b(?:new\s+(?:version|release|update|feature))\b',
            r'\b(?:latest\s+(?:version|release|update))\b'
        ]
        
        # Feature/Product indicators
        feature_patterns = [
            r'\b(?:feature|functionality|capability|enhancement|improvement)\b',
            r'\b(?:introduced|launched|released|announced|unveiled)\b',
            r'\b(?:new\s+(?:feature|product|service|offering))\b'
        ]
        
        # Initiative/Strategy indicators
        initiative_patterns = [
            r'\b(?:initiative|strategy|strategic|roadmap|plan|goal|objective)\b',
            r'\b(?:investing\s+in|focusing\s+on|prioritizing|emphasizing)\b',
            r'\b(?:direction|vision|mission|commitment|approach)\b'
        ]
        
        # Factor/Impact indicators
        factor_patterns = [
            r'\b(?:factors?\s+(?:affecting|influencing|impacting))\b',
            r'\b(?:due\s+to|because\s+of|as\s+a\s+result\s+of|driven\s+by)\b',
            r'\b(?:impact|effect|influence|outcome|consequence)\b'
        ]
        
        # Check each pattern category
        pattern_categories = [
            (decision_patterns, "decision"),
            (update_patterns, "update"),  
            (feature_patterns, "feature"),
            (initiative_patterns, "initiative"),
            (factor_patterns, "factor")
        ]
        
        for patterns, tag_name in pattern_categories:
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    tags.append(tag_name)
                    break  # Only add tag once per category
        
        # Additional contextual tags
        if re.search(r'\b(?:quarterly|annual|monthly)\s+(?:results|report|review)\b', text_lower):
            tags.append("periodic_review")
        
        if re.search(r'\b(?:risk|challenge|issue|problem|concern)\b', text_lower):
            tags.append("risk_factor")
        
        if re.search(r'\b(?:opportunity|growth|expansion|market)\b', text_lower):
            tags.append("growth_opportunity")
        
        if re.search(r'\b(?:customer|client|user)\s+(?:feedback|satisfaction|experience)\b', text_lower):
            tags.append("customer_insight")
        
        return tags
    
    def extract_table_metadata(self, table_data: Dict[str, Any], table_text: str) -> Dict[str, Any]:
        """Extract table-specific metadata for enhanced indexing."""
        table_meta = {
            "table_id": table_data.get("table_id") or str(uuid.uuid4())[:8],
            "table_title": table_data.get("table_title", "Table"),
            "row_headers": [],
            "col_headers": [],
            "units": [],
            "periods": [],
            "cell_samples": []
        }
        
        # Extract headers from cell matrix
        cell_matrix = table_data.get("cell_matrix", {})
        if cell_matrix:
            col_headers = cell_matrix.get("headers", {}).get("column", [])
            table_meta["col_headers"] = [h.get("text", "").strip() for h in col_headers if h.get("text")]
            
            row_headers = cell_matrix.get("headers", {}).get("row", [])
            table_meta["row_headers"] = [h.get("text", "").strip() for h in row_headers if h.get("text")]
        
        # Extract units and periods
        for pattern in self.unit_patterns:
            table_meta["units"].extend(pattern.findall(table_text))
        for pattern in self.period_patterns:
            table_meta["periods"].extend(pattern.findall(table_text))
        
        table_meta["units"] = list(set(table_meta["units"]))
        table_meta["periods"] = list(set(table_meta["periods"]))
        
        # Extract cell samples for recall
        table_meta["cell_samples"] = self._extract_cell_samples(
            cell_matrix, 
            max_samples=self.table_config["max_cell_samples"]
        )
        
        return table_meta
    
    def _extract_cell_samples(self, cell_matrix: Dict[str, Any], max_samples: int = 10) -> List[str]:
        """Extract representative cell samples as key-value pairs."""
        samples = []
        if not cell_matrix or not cell_matrix.get("rows"):
            return samples
        
        rows = cell_matrix["rows"]
        col_headers = cell_matrix.get("headers", {}).get("column", [])
        header_texts = [h.get("text", "") for h in col_headers]
        
        # Sample from configurable number of rows
        sample_rows = self.table_config.get("sample_row_count", 5)
        for row_idx, row in enumerate(rows[:sample_rows]):
            if not row or len(samples) >= max_samples:
                continue
            
            for col_idx, cell in enumerate(row):
                if not cell or not cell.get("text") or len(samples) >= max_samples:
                    continue
                
                cell_text = cell["text"].strip()
                if col_idx < len(header_texts) and header_texts[col_idx]:
                    key = header_texts[col_idx].strip()
                    if key and cell_text:
                        samples.append(f"{key}: {cell_text}")
                else:
                    samples.append(cell_text)
        
        return samples[:max_samples]


# ============================================================================
# ENHANCED CHUNK DATACLASS
# ============================================================================

@dataclass
class EnhancedChunk:
    """
    Enhanced chunk with comprehensive metadata for rich retrieval.
    Combines layout-aware chunking with GraphRAG-style metadata.
    """
    # Core content
    text: str
    
    # Canonical identification
    doc_id: str = ""
    doc_version: str = "1"
    source_type: str = "document"
    product_component: str = ""
    confidentiality: str = "public"
    
    # Temporal metadata
    ingested_at: str = ""
    effective_from: Optional[str] = None
    effective_to: Optional[str] = None
    
    # Chunk identification  
    chunk_id: str = ""  # UUID for internal tracking
    chunk_type: str = "paragraph"  # paragraph | table | list (removed kv)
    method: str = "layout_aware_chunking"
    source_format: str = ""  # html, md, json
    
    # Spatial and structural metadata
    page: Optional[int] = None
    bbox: Optional[List[float]] = None
    section_h1: Optional[str] = None
    section_h2: Optional[str] = None
    section_h3: Optional[str] = None
    headings_path: str = ""
    
    # Folder hierarchy metadata (supports arbitrary nesting depth)
    folder_path: List[str] = field(default_factory=list)  # ["folder 2", "folder 3", "investor reports"]
    folder_hierarchy: str = ""  # "folder 2/folder 3/investor reports"
    file_name: str = ""  # "document.html"
    relative_path: str = ""  # "folder 2/folder 3/investor reports/document.html"
    folder_depth: int = 0  # Number of nested folders (3 in above example)
    
    # Deterministic metadata enrichment
    metric_terms: List[str] = field(default_factory=list)
    doc_refs: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    mentioned_dates: List[str] = field(default_factory=list)
    is_change_note: bool = False
    policy_tags: List[str] = field(default_factory=list)
    
    # Table-aware metadata
    table_id: Optional[str] = None
    table_title: Optional[str] = None
    row_headers: List[str] = field(default_factory=list)
    col_headers: List[str] = field(default_factory=list)
    units: List[str] = field(default_factory=list)
    periods: List[str] = field(default_factory=list)
    cell_samples: List[str] = field(default_factory=list)
    
    # Legacy compatibility fields
    lineage: Dict[str, Any] = field(default_factory=dict)
    headers: Dict[str, Any] = field(default_factory=dict)
    table_meta: Dict[str, Any] = field(default_factory=dict)
    counts: Dict[str, Any] = field(default_factory=dict)
    source: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# ENHANCED LAYOUT-AWARE CHUNKER
# ============================================================================

class LayoutAwareChunker:
    """
    Enhanced layout-aware chunker with rich metadata and deterministic enrichment.
    
    Key features:
    1. Respects document hierarchy (never crosses subsection boundaries)
    2. Canonical document identification and versioning
    3. Deterministic metadata extraction (no LLM calls)
    4. Spatial context preservation from JSON
    5. Table-aware indexing for enhanced recall
    6. Table of Contents detection and semantic structure extraction
    7. Clean, maintainable code architecture
    """
    
    def __init__(
        self,
        max_words: int = 300,
        min_words: int = 15,
        external_table_dir: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        # Enhanced parameters
        doc_name: Optional[str] = None,
        doc_version: str = "1",
        source_type: str = "document",
        product_component: str = "",
        confidentiality: str = "public",
        effective_from: Optional[str] = None,
        effective_to: Optional[str] = None,
        controlled_vocab: Optional[Dict[str, List[str]]] = None,
        table_sampling_config: Optional[Dict[str, int]] = None,
        heading_config: Optional[Dict[str, int]] = None,
    ) -> None:
        # Core parameters
        self.max_words = max_words
        self.min_words = min_words
        
        # Document metadata
        self.doc_name = doc_name
        self.doc_version = doc_version
        self.doc_id = f"{doc_name}_{doc_version}" if doc_name else "document_v1"
        self.source_type = source_type
        self.product_component = product_component
        self.confidentiality = confidentiality
        self.effective_from = effective_from
        self.effective_to = effective_to
        self.ingested_at = datetime.now().isoformat()
        
        # Initialize LLM classifier for table titles
        self.llm_classifier = None
        if LLMTableClassifier and openai_api_key:
            try:
                self.llm_classifier = LLMTableClassifier(openai_api_key)
                logger.info("LLM table classifier enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM classifier: {e}")
        
        # Comprehensive metrics tracking
        self.metrics = {
            "timing": {
                "chunking_start": None,
                "chunking_end": None,
                "embedding_start": None,
                "embedding_end": None,
                "storage_start": None,
                "storage_end": None
            },
            "resources": {
                "memory_start": None,
                "memory_peak": None,
                "memory_end": None,
                "cpu_time_start": None,
                "cpu_time_end": None
            },
            "tokens": {
                "by_method": {},
                "total_used": 0,
                "total_calls": 0
            },
            "chunks": {
                "total_created": 0,
                "by_type": {},
                "average_size": 0
            },
            "file_info": {
                "size_bytes": 0,
                "format": "",
                "name": ""
            }
        }
        
        # Set up table storage directory
        if external_table_dir is None:
            pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            self.external_table_dir = os.path.join(pkg_root, "data", "tables")
        else:
            self.external_table_dir = external_table_dir
        os.makedirs(self.external_table_dir, exist_ok=True)
        
        # Configure table sampling
        self.table_config = table_sampling_config or {
            "max_cell_samples": 10,
            "sample_row_count": 5,
            "header_word_estimate": 3
        }
        
        # Configure heading levels (no hardcoded 1,2,3 assumption)
        self.heading_config = heading_config or {
            "max_heading_level": 3,     # Support H1, H2, H3 by default
            "title_level": 1,           # H1 = title
            "section_level": 2,         # H2 = section  
            "subsection_level": 3       # H3 = subsection
        }
        
        # Initialize metadata extractor with table config
        self.metadata_extractor = MetadataExtractor(controlled_vocab, self.table_config)
        
        # Document structure extracted from Table of Contents
        self.document_structure = None
        self.toc_section_mapping = {}
    
    def chunk_document(
        self,
        file_path: str,
        source_format: str = "markdown",
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Main entry point for enhanced layout-aware document chunking.
        
        Supports: markdown, html, json
        
        File Format Behavior:
        - markdown: Process .md file, optionally load corresponding .json for enrichment
        - html: Process .html file independently (no JSON dependency)  
        - json: Process .json file directly (extract tables and structured content)
            
        Returns:
            (chunks, section_index) - enhanced chunks with rich metadata
        """
        if source_format not in ["markdown", "html", "json"]:
            raise ValueError("Only 'markdown', 'html', and 'json' source formats are supported")
            
        # Start comprehensive metrics tracking
        self._start_metrics_tracking(file_path, source_format)
            
        # Generate stable document ID
        self.doc_id = generate_stable_doc_id(file_path, self.doc_name)
        
        if source_format == "html":
            # Import and use HTML processor module
            from .html_processor import HTMLProcessor
            html_processor = HTMLProcessor(self)
            result = html_processor.process_html_document(file_path)
        elif source_format == "json":
            # Process JSON file directly
            result = self._process_json_only_document(file_path)
        else:  # markdown
            # Load JSON data for enrichment (optional)
            json_data = self._load_json_data(file_path)
            result = self._process_document(file_path, json_data)
        
        # Apply post-processing fixes to improve chunk quality
        original_count = len(result[0])
        try:
            from .chunker_fixes import apply_all_fixes
            result = (apply_all_fixes(result[0]), result[1])
            logger.info(f"Applied post-processing fixes: {original_count} -> {len(result[0])} chunks")
        except Exception as e:
            logger.warning(f"Post-processing fixes failed: {e}")
        
        # Complete metrics tracking and log to unified logger
        self._complete_chunking_metrics(result[0])
        self._log_to_unified_logger(file_path, result[0])
        
        return result
    
    def _process_json_only_document(self, json_path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Process JSON file directly without markdown companion.
        Extracts tables and structured content from JSON document structure.
        """
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                json_data = json.load(f)
        except Exception as e:
            logger.error(f"Could not load JSON file {json_path}: {e}")
            return [], {}
        
        # Build JSON index for content extraction
        json_index = self._build_json_index(json_data)
        
        # Extract table chunks from JSON structure
        table_chunks = self._create_table_chunks(json_index, json_path)
        
        # Extract document structure from Table of Contents if present
        self._extract_document_structure_from_toc(table_chunks)
        
        # For JSON-only files, we primarily extract structured content (tables)
        # Text content would need additional processing if available in JSON structure
        all_chunks = table_chunks
        
        # Build section index
        section_index = self._build_section_index(all_chunks)
        
        # Convert to output format and add sequential chunk_id for compatibility
        output_chunks = []
        for i, chunk in enumerate(all_chunks):
            chunk_dict = chunk.__dict__.copy()
            chunk_dict["chunk_id"] = i
            chunk_dict["uuid"] = chunk.chunk_id
            output_chunks.append(chunk_dict)
        
        logger.info(f"JSON-only chunking complete: {len(output_chunks)} chunks")
        logger.info(f"Document: {self.doc_id} v{self.doc_version} ({self.source_type})")
        
        return output_chunks, section_index
    
    def _load_json_data(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Load corresponding JSON data for enrichment."""
        json_path = None
        
        if "/markdown/" in file_path:
            json_path = file_path.replace("/markdown/", "/json/").replace(".md", ".json")
        else:
            json_path = file_path.replace(".md", ".json")
        
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load JSON data from {json_path}: {e}")
            return None
    
    def _process_document(
        self, 
        md_path: str, 
        json_data: Optional[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Process markdown document with JSON enrichment."""
        
        with open(md_path, "r", encoding="utf-8") as f:
            md_content = f.read()
        
        # Build JSON index for enrichment
        json_index = self._build_json_index(json_data) if json_data else {}
        
        # Create table chunks from JSON first
        table_chunks = self._create_table_chunks(json_index, md_path)
        
        # Extract document structure from Table of Contents if present
        self._extract_document_structure_from_toc(table_chunks)
        
        # Process markdown content
        text_chunks = self._process_markdown_content(md_content, md_path, json_index)
        
        # Combine and finalize chunks
        all_chunks = table_chunks + text_chunks
        
        # Build section index
        section_index = self._build_section_index(all_chunks)
        
        # Convert to output format and add sequential chunk_id for compatibility
        output_chunks = []
        for i, chunk in enumerate(all_chunks):
            chunk_dict = chunk.__dict__.copy()
            # Add integer chunk_id for legacy compatibility (while keeping UUID for internal tracking)
            chunk_dict["chunk_id"] = i
            chunk_dict["uuid"] = chunk.chunk_id  # Preserve UUID separately
            output_chunks.append(chunk_dict)
        
        logger.info(f"Enhanced chunking complete: {len(output_chunks)} chunks "
                   f"({len(table_chunks)} table + {len(text_chunks)} text)")
        logger.info(f"Document: {self.doc_id} v{self.doc_version} ({self.source_type})")
        
        return output_chunks, section_index
    
    def _build_json_index(self, json_data: Dict[str, Any]) -> Dict[str, Any]:
        """Build searchable index from JSON data."""
        index = {
            "tables": {},
            "text_by_content": {},
            "page_map": {}
        }
        
        # Index tables
        if "tables" in json_data:
            for table in json_data["tables"]:
                if "data" in table and "table_cells" in table["data"]:
                    table_id = table["self_ref"]
                    cells = table["data"]["table_cells"]
                    
                    cell_matrix = self._build_cell_matrix(cells)
                    page_no = None
                    if table.get("prov") and table["prov"]:
                        page_no = table["prov"][0].get("page_no")
                    
                    index["tables"][table_id] = {
                        "cells": cells,
                        "cell_matrix": cell_matrix,
                        "page_no": page_no,
                        "label": table.get("label", "table")
                    }
        
        # Index text elements
        if "texts" in json_data:
            for text_elem in json_data["texts"]:
                content = text_elem.get("text", "").strip()
                if content:
                    page_no = None
                    bbox = None
                    if text_elem.get("prov") and text_elem["prov"]:
                        prov = text_elem["prov"][0]
                        page_no = prov.get("page_no")
                        bbox = prov.get("bbox")
                    
                    index["text_by_content"][content] = {
                        "page_no": page_no,
                        "bbox": bbox,
                        "label": text_elem.get("label", "text")
                    }
                    
                    if page_no:
                        index["page_map"][content] = page_no
        
        return index
    
    def _build_cell_matrix(self, cells: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build structured cell matrix from table cells."""
        if not cells:
            return {"rows": [], "headers": {"column": [], "row": []}, "sections": []}
        
        max_row = max(cell["end_row_offset_idx"] for cell in cells)
        max_col = max(cell["end_col_offset_idx"] for cell in cells)
        
        matrix = [[None for _ in range(max_col)] for _ in range(max_row)]
        headers = {"column": [], "row": []}
        sections = []
        
        for cell in cells:
            start_row = cell["start_row_offset_idx"]
            end_row = cell["end_row_offset_idx"]
            start_col = cell["start_col_offset_idx"]
            end_col = cell["end_col_offset_idx"]
            
            cell_data = {
                "text": cell.get("text", ""),
                "is_column_header": cell.get("column_header", False),
                "is_row_header": cell.get("row_header", False),
                "is_row_section": cell.get("row_section", False)
            }
            
            # Fill matrix
            for r in range(start_row, end_row):
                for c in range(start_col, end_col):
                    if r < len(matrix) and c < len(matrix[r]):
                        matrix[r][c] = cell_data
            
            # Collect headers and sections
            if cell_data["is_column_header"]:
                headers["column"].append({
                    "text": cell_data["text"],
                    "row": start_row,
                    "col_range": (start_col, end_col)
                })
            
            if cell_data["is_row_header"]:
                headers["row"].append({
                    "text": cell_data["text"],
                    "col": start_col,
                    "row_range": (start_row, end_row)
                })
            
            if cell_data["is_row_section"]:
                sections.append({
                    "text": cell_data["text"],
                    "row": start_row,
                    "col_range": (start_col, end_col)
                })
        
        return {
            "rows": matrix,
            "headers": headers,
            "sections": sections,
            "dimensions": (max_row, max_col)
        }
    
    def _create_table_chunks(self, json_index: Dict[str, Any], md_path: str) -> List[EnhancedChunk]:
        """Create enhanced table chunks from JSON data."""
        table_chunks = []
        table_counter = 1
        
        for table_id, table_data in json_index.get("tables", {}).items():
            cell_matrix = table_data["cell_matrix"]
            if not cell_matrix or not cell_matrix["rows"]:
                continue
                
            matrix = cell_matrix["rows"]
            column_headers = cell_matrix["headers"]["column"]
            sections = cell_matrix["sections"]
            
            # Generate table title
            table_title = self._generate_table_title(table_data, json_index, table_counter)
            table_counter += 1
            
            # Group rows semantically
            row_groups = self._group_table_rows(matrix, column_headers, sections)
            
            for group_idx, row_group in enumerate(row_groups):
                chunk_text = self._build_table_chunk_text(table_title, column_headers, matrix, row_group)
                
                if count_alnum_words(chunk_text) < self.min_words:
                    continue
                
                # Create enhanced chunk
                chunk = self._create_enhanced_chunk(
                    text=chunk_text,
                    chunk_type="table",
                    md_path=md_path,
                    json_index=json_index,
                    lineage={"table_id": table_id, "table_title": table_title},
                    table_data={
                        "table_id": table_id,
                        "table_title": table_title,
                        "cell_matrix": cell_matrix,
                        "page_no": table_data.get("page_no")
                    }
                )
                
                table_chunks.append(chunk)
                
        return table_chunks
    
    def _process_markdown_content(
        self, 
        content: str, 
        md_path: str, 
        json_index: Dict[str, Any]
    ) -> List[EnhancedChunk]:
        """Process markdown content into enhanced chunks."""
        lines = content.splitlines()
        chunks = []
        
        # Hierarchy tracking
        current_hierarchy = {"title": None, "section": None, "subsection": None}
        paragraph_buffer = []
        
        def flush_paragraph():
            if not paragraph_buffer:
                return
            
            text = " ".join(line.strip() for line in paragraph_buffer if line.strip())
            if not text or count_alnum_words(text) == 0:
                paragraph_buffer.clear()
                return
            
            lineage = self._get_current_lineage(current_hierarchy)
            
            # Add subsection context
            if lineage.get("subsection_text"):
                text = f"{lineage['subsection_text']}: {text}"
            
            # Handle minimum word requirement
            word_count = count_alnum_words(text)
            if (word_count < self.min_words and chunks and 
                chunks[-1].chunk_type == "paragraph" and
                chunks[-1].lineage.get("subsection_id") == lineage.get("subsection_id")):
                # Merge with previous chunk
                chunks[-1].text = chunks[-1].text + " " + text
                chunks[-1].counts["alnum_words"] = count_alnum_words(chunks[-1].text)
                paragraph_buffer.clear()
                return
            
            chunk = self._create_enhanced_chunk(
                text=text,
                chunk_type="paragraph",
                md_path=md_path,
                json_index=json_index,
                lineage=lineage
            )
            chunks.append(chunk)
            paragraph_buffer.clear()
        
        # Process lines
        for line in lines:
            if self._is_heading(line):
                flush_paragraph()
                level, heading_text = self._parse_heading(line)
                self._update_hierarchy(current_hierarchy, level, heading_text)
            elif self._is_table_line(line):
                continue  # Skip - handled by JSON processing
            elif self._is_key_value_line(line):
                flush_paragraph()
                lineage = self._get_current_lineage(current_hierarchy)
                
                text = line
                if lineage.get("subsection_text"):
                    text = f"{lineage['subsection_text']}: {text}"
                
                chunk = self._create_enhanced_chunk(
                    text=text,
                    chunk_type="kv",
                    md_path=md_path,
                    json_index=json_index,
                    lineage=lineage
                )
                chunks.append(chunk)
            elif line.strip() == "":
                flush_paragraph()
            elif count_alnum_words(line) > 0 and len(line.strip()) > 1:
                paragraph_buffer.append(line)
        
        # Final flush
        flush_paragraph()
        
        return chunks
    
    def _create_enhanced_chunk(
        self, 
        text: str,
        chunk_type: str,
        md_path: str,
        json_index: Dict[str, Any],
        lineage: Dict[str, Any] = None,
        table_data: Dict[str, Any] = None
    ) -> EnhancedChunk:
        """Create an enhanced chunk with comprehensive metadata."""
        
        lineage = lineage or {}
        
        # Extract deterministic metadata
        context = {"source_type": self.source_type, "doc_id": self.doc_id}
        enriched_metadata = self.metadata_extractor.extract_metadata(text, context)
        
        # Create chunk
        chunk = EnhancedChunk(
            # Core content
            text=text,
            
            # Identification
            doc_id=self.doc_id,
            doc_version=self.doc_version,
            source_type=self.source_type,
            product_component=self.product_component,
            confidentiality=self.confidentiality,
            
            # Temporal
            ingested_at=self.ingested_at,
            effective_from=self.effective_from,
            effective_to=self.effective_to,
            
            # Chunk metadata
            chunk_id=str(uuid.uuid4()),
            chunk_type=chunk_type,
            
            # Hierarchical context
            section_h1=lineage.get("title_text"),
            section_h2=lineage.get("section_text"),
            section_h3=lineage.get("subsection_text"),
            headings_path=create_headings_path(lineage),
            
            # Deterministic metadata
            metric_terms=enriched_metadata.get("metric_terms", []),
            doc_refs=enriched_metadata.get("doc_refs", []),
            entities=enriched_metadata.get("entities", []),
            mentioned_dates=enriched_metadata.get("mentioned_dates", []),
            is_change_note=enriched_metadata.get("is_change_note", False),
            policy_tags=enriched_metadata.get("policy_tags", []),
            
            # Legacy compatibility
            lineage=lineage,
            counts={"alnum_words": count_alnum_words(text)},
            source={"file_path": md_path, "source_format": "markdown"},
            metadata={"enhanced_layout_aware": True}
        )
        
        # Add folder hierarchy metadata
        self._enrich_folder_hierarchy(chunk, md_path)
        
        # Add spatial metadata
        self._enrich_spatial_metadata(chunk, json_index)
        
        # Add table metadata if applicable
        if chunk_type == "table" and table_data:
            self._enrich_table_metadata(chunk, table_data, text)
        
        # Enrich with Table of Contents context if available
        self._enrich_chunk_with_toc_context(chunk)
        
        return chunk
    
    def _enrich_folder_hierarchy(self, chunk: EnhancedChunk, file_path: str) -> None:
        """
        Enrich chunk with folder hierarchy metadata for arbitrary nesting depth.
        Extracts and preserves the complete folder structure from the file path.
        """
        hierarchy_info = extract_folder_hierarchy(file_path)
        
        # Set folder hierarchy fields
        chunk.folder_path = hierarchy_info["folder_path"]
        chunk.folder_hierarchy = hierarchy_info["folder_hierarchy"]
        chunk.file_name = hierarchy_info["file_name"]
        chunk.relative_path = hierarchy_info["relative_path"]
        chunk.folder_depth = hierarchy_info["folder_depth"]
    
    def _start_metrics_tracking(self, file_path: str, source_format: str) -> None:
        """Initialize comprehensive metrics tracking for chunking process."""
        import time
        import psutil
        import os
        
        # Timing metrics
        self.metrics["timing"]["chunking_start"] = time.time()
        
        # Resource metrics
        process = psutil.Process()
        self.metrics["resources"]["memory_start"] = process.memory_info().rss / (1024 * 1024)  # MB
        self.metrics["resources"]["cpu_time_start"] = process.cpu_times().user + process.cpu_times().system
        
        # File info metrics
        if os.path.exists(file_path):
            self.metrics["file_info"]["size_bytes"] = os.path.getsize(file_path)
            self.metrics["file_info"]["size_mb"] = self.metrics["file_info"]["size_bytes"] / (1024 * 1024)
        
        self.metrics["file_info"]["format"] = source_format
        self.metrics["file_info"]["name"] = os.path.basename(file_path)
        self.metrics["file_info"]["path"] = file_path
        
        # Reset LLM token tracking
        if self.llm_classifier:
            self.llm_classifier.reset_token_stats()
        
        logger.info(f"📊 Started metrics tracking for: {self.metrics['file_info']['name']}")
    
    def _complete_chunking_metrics(self, chunks: List[Dict[str, Any]]) -> None:
        """Complete chunking metrics tracking and calculate final statistics."""
        import time
        import psutil
        
        # Complete timing
        self.metrics["timing"]["chunking_end"] = time.time()
        
        # Complete resource tracking
        process = psutil.Process()
        self.metrics["resources"]["memory_end"] = process.memory_info().rss / (1024 * 1024)  # MB
        self.metrics["resources"]["memory_peak"] = process.memory_info().peak_wss / (1024 * 1024) if hasattr(process.memory_info(), 'peak_wss') else self.metrics["resources"]["memory_end"]  # MB
        self.metrics["resources"]["cpu_time_end"] = process.cpu_times().user + process.cpu_times().system
        
        # Calculate derived metrics
        self.metrics["timing"]["chunking_duration"] = self.metrics["timing"]["chunking_end"] - self.metrics["timing"]["chunking_start"]
        self.metrics["resources"]["memory_used"] = self.metrics["resources"]["memory_end"] - self.metrics["resources"]["memory_start"]
        self.metrics["resources"]["cpu_time_used"] = self.metrics["resources"]["cpu_time_end"] - self.metrics["resources"]["cpu_time_start"]
        
        # Chunk metrics
        self.metrics["chunks"]["total_created"] = len(chunks)
        
        # Count by type
        for chunk in chunks:
            chunk_type = chunk.get('chunk_type', 'unknown')
            self.metrics["chunks"]["by_type"][chunk_type] = self.metrics["chunks"]["by_type"].get(chunk_type, 0) + 1
        
        # Calculate average chunk size
        total_text_length = sum(len(chunk.get('text', '')) for chunk in chunks)
        self.metrics["chunks"]["average_size"] = total_text_length / len(chunks) if chunks else 0
        
        # Token metrics
        if self.llm_classifier:
            token_stats = self.llm_classifier.get_token_usage_stats()
            self.metrics["tokens"]["total_used"] = token_stats.get("total_tokens_used", 0)
            self.metrics["tokens"]["total_calls"] = token_stats.get("total_calls_made", 0)
            self.metrics["tokens"]["average_per_call"] = token_stats.get("average_tokens_per_call", 0)
            self.metrics["tokens"]["by_method"]["table_classification"] = token_stats.get("total_tokens_used", 0)
        else:
            self.metrics["tokens"]["total_used"] = 0
            self.metrics["tokens"]["total_calls"] = 0
            self.metrics["tokens"]["average_per_call"] = 0
        
        # Performance metrics
        if self.metrics["timing"]["chunking_duration"] > 0:
            self.metrics["performance"] = {
                "chunks_per_second": len(chunks) / self.metrics["timing"]["chunking_duration"],
                "mb_per_second": self.metrics["file_info"]["size_mb"] / self.metrics["timing"]["chunking_duration"],
                "tokens_per_chunk": self.metrics["tokens"]["total_used"] / len(chunks) if chunks else 0
            }
    
    def _log_to_unified_logger(self, file_path: str, chunks: List[Dict[str, Any]]) -> None:
        """Log chunking metrics to unified logger."""
        from .unified_logger import get_unified_logger
        
        # Get unified logger instance
        unified_logger = get_unified_logger()
        
        # Calculate file metrics
        file_size = self.metrics.get("file_info", {}).get("size_bytes", 0)
        processing_time = self.metrics.get("timing", {}).get("chunking_duration", 0)
        tokens_used = self.metrics.get("tokens", {}).get("total_used", 0)
        tokens_by_method = self.metrics.get("tokens", {}).get("by_method", {})
        source_format = self.metrics.get("file_info", {}).get("format", "unknown")
        
        # Log to unified logger
        unified_logger.log_file_processing(
            file_path=file_path,
            chunks_created=len(chunks),
            processing_time=processing_time,
            tokens_used=tokens_used,
            tokens_by_method=tokens_by_method,
            file_size_bytes=file_size,
            source_format=source_format
        )
    
    # Removed: _generate_markdown_summary() - now using unified logger
    
    def _log_chunking_summary(self, file_path: str, source_format: str, chunks: List[Dict[str, Any]]) -> None:
        """
        Log comprehensive summary of chunking process including token usage.
        """
        import time
        import os
        
        # Calculate processing time
        processing_time = time.time() - self.chunking_start_time if self.chunking_start_time else 0
        
        # Count chunk types
        chunk_types = {}
        for chunk in chunks:
            chunk_type = chunk.get('chunk_type', 'unknown')
            chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
        
        # Get LLM token usage stats
        llm_stats = {"total_tokens_used": 0, "total_calls_made": 0, "average_tokens_per_call": 0}
        if self.llm_classifier:
            llm_stats = self.llm_classifier.get_token_usage_stats()
        
        # Calculate file size
        file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
        file_size_mb = file_size / (1024 * 1024)
        
        # Log comprehensive summary
        logger.info("🎯 CHUNKING PROCESS COMPLETE")
        logger.info("=" * 60)
        logger.info(f"📄 File: {os.path.basename(file_path)} ({file_size_mb:.2f} MB)")
        logger.info(f"🔧 Format: {source_format}")
        logger.info(f"⏱️  Processing time: {processing_time:.2f} seconds")
        logger.info(f"📊 Total chunks created: {len(chunks)}")
        
        # Chunk type breakdown
        for chunk_type, count in sorted(chunk_types.items()):
            logger.info(f"   └─ {chunk_type}: {count} chunks")
        
        # LLM Token Usage Summary
        logger.info(f"🤖 LLM TOKEN USAGE:")
        logger.info(f"   └─ Total tokens used: {llm_stats['total_tokens_used']} tokens")
        logger.info(f"   └─ LLM calls made: {llm_stats['total_calls_made']} calls")
        logger.info(f"   └─ Average tokens per call: {llm_stats['average_tokens_per_call']}")
        
        # Efficiency metrics
        if processing_time > 0:
            chunks_per_second = len(chunks) / processing_time
            logger.info(f"⚡ Performance: {chunks_per_second:.1f} chunks/second")
        
        if llm_stats['total_tokens_used'] > 0:
            tokens_per_chunk = llm_stats['total_tokens_used'] / len(chunks)
            logger.info(f"💰 Token efficiency: {tokens_per_chunk:.2f} tokens/chunk")
        
        logger.info("=" * 60)
    
    def _enrich_spatial_metadata(self, chunk: EnhancedChunk, json_index: Dict[str, Any]):
        """Add spatial metadata (page, bbox) from JSON index."""
        # Find page number
        page_no = self._find_page_number(chunk.text, json_index)
        if page_no:
            chunk.page = page_no
            chunk.metadata["page_no"] = page_no
        
        # Find bounding box
        text_clean = chunk.text.split(":", 1)[1].strip() if ":" in chunk.text else chunk.text
        if text_clean in json_index.get("text_by_content", {}):
            text_meta = json_index["text_by_content"][text_clean]
            if text_meta.get("bbox"):
                chunk.bbox = text_meta["bbox"]
    
    def _enrich_table_metadata(self, chunk: EnhancedChunk, table_data: Dict[str, Any], table_text: str):
        """Add table-specific metadata."""
        table_meta = self.metadata_extractor.extract_table_metadata(table_data, table_text)
        
        chunk.table_id = table_meta.get("table_id")
        chunk.table_title = table_meta.get("table_title")
        chunk.row_headers = table_meta.get("row_headers", [])
        chunk.col_headers = table_meta.get("col_headers", [])
        chunk.units = table_meta.get("units", [])
        chunk.periods = table_meta.get("periods", [])
        chunk.cell_samples = table_meta.get("cell_samples", [])
        
        # Legacy compatibility
        chunk.table_meta.update(table_meta)
    
    def _find_page_number(self, text: str, json_index: Dict[str, Any]) -> Optional[int]:
        """Find page number for text using JSON index."""
        if text in json_index.get("page_map", {}):
            return json_index["page_map"][text]
        
        text_clean = text.split(":", 1)[1].strip() if ":" in text else text
        if text_clean in json_index.get("page_map", {}):
            return json_index["page_map"][text_clean]
        
        # Fuzzy matching
        for json_text, metadata in json_index.get("text_by_content", {}).items():
            if json_text in text or text in json_text:
                return metadata.get("page_no")
        
        return None
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def _is_heading(self, line: str) -> bool:
        max_level = self.heading_config["max_heading_level"]
        pattern = f"^#{{{1},{max_level}}}\\s+"
        return bool(re.match(pattern, line))
    
    def _parse_heading(self, line: str) -> Tuple[int, str]:
        max_level = self.heading_config["max_heading_level"]
        pattern = f"^(#{{{1},{max_level}}})\\s+(.+)"
        match = re.match(pattern, line)
        if match:
            return len(match.group(1)), match.group(2).strip()
        return 1, line.strip()
    
    def _is_key_value_line(self, line: str) -> bool:
        patterns = [r"^[^:]+:\s*[^:]+$", r"^[^-]+-\s*[^-]+$", r"^[^=]+=\s*[^=]+$"]
        return any(re.match(pattern, line.strip()) for pattern in patterns)
    
    def _is_table_line(self, line: str) -> bool:
        stripped = line.strip()
        return (stripped and "|" in stripped and 
                stripped != "|" and not stripped.startswith("#"))
    
    def _update_hierarchy(self, hierarchy: Dict, level: int, text: str):
        """Update hierarchical tracking using configurable levels."""
        title_level = self.heading_config["title_level"]
        section_level = self.heading_config["section_level"]
        subsection_level = self.heading_config["subsection_level"]
        
        if level == title_level:
            hierarchy["title"] = {"text": text}
            hierarchy["section"] = None
            hierarchy["subsection"] = None
        elif level == section_level:
            hierarchy["section"] = {"text": text}
            hierarchy["subsection"] = None
        elif level == subsection_level:
            hierarchy["subsection"] = {"text": text}
    
    def _get_current_lineage(self, hierarchy: Dict) -> Dict[str, Any]:
        """Get current hierarchical lineage."""
        return {
            "title_text": hierarchy["title"]["text"] if hierarchy["title"] else None,
            "section_text": hierarchy["section"]["text"] if hierarchy["section"] else None,
            "subsection_text": hierarchy["subsection"]["text"] if hierarchy["subsection"] else None,
        }
    
    def _generate_table_title(self, table_data: Dict, json_index: Dict, table_number: int) -> str:
        """Generate meaningful table title with Table of Contents detection."""
        # Try JSON label first
        json_label = table_data.get("label", "")
        if json_label and json_label != "table":
            # Clean up label format
            cleaned = json_label.replace("_", " ").replace("-", " ").title()
            
            # Generic Table of Contents detection (domain-agnostic)
            if self._is_table_of_contents_label(cleaned):
                return "Table of Contents"
            
            return cleaned
        
        # Try LLM classification
        if self.llm_classifier:
            try:
                llm_title = self.llm_classifier.generate_table_title(table_data, table_number)
                if not llm_title.startswith(f"Table {table_number}"):
                    return llm_title
            except Exception as e:
                logger.warning(f"LLM table classification failed: {e}")
        
        # Generic fallback without domain assumptions
        return f"Table {table_number}"
    
    def _group_table_rows(self, matrix: List, column_headers: List, sections: List) -> List[Dict]:
        """Group table rows semantically while respecting word limits."""
        if not matrix:
            return []
        
        groups = []
        current_group = {"rows": [], "context": "", "word_count": 0}
        # Use configurable header word estimate instead of magic number
        header_word_estimate = self.metadata_extractor.table_config.get("header_word_estimate", 3)
        base_words = len(column_headers) * header_word_estimate
        
        for row_idx, row in enumerate(matrix):
            if not row:
                continue
                
            row_text = " ".join(cell.get("text", "") for cell in row if cell)
            row_words = count_alnum_words(row_text)
            
            is_section_header = any(section["row"] == row_idx for section in sections)
            projected_words = current_group["word_count"] + base_words + row_words
            
            # Start new group if needed
            if ((projected_words > self.max_words and current_group["rows"]) or
                (is_section_header and current_group["rows"])):
                if current_group["rows"]:
                    groups.append(current_group)
                current_group = {"rows": [], "context": "", "word_count": 0}
            
            current_group["rows"].append(row_idx)
            current_group["word_count"] += row_words
            
            if is_section_header:
                current_group["context"] = row_text
        
        if current_group["rows"]:
            groups.append(current_group)
        
        return groups if groups else [{"rows": list(range(len(matrix))), "context": "", "word_count": 0}]
    
    def _build_table_chunk_text(self, title: str, headers: List, matrix: List, row_group: Dict) -> str:
        """Build comprehensive table chunk text."""
        lines = []
        
        if title:
            lines.append(f"Table: {title}")
        
        if headers:
            header_texts = [h["text"] for h in headers]
            lines.append(" | ".join(header_texts))
            lines.append(" | ".join(["---"] * len(header_texts)))
        
        for row_idx in row_group["rows"]:
            if row_idx < len(matrix) and matrix[row_idx]:
                cell_texts = []
                for cell in matrix[row_idx]:
                    if cell and cell.get("text"):
                        cell_texts.append(cell["text"])
                    else:
                        cell_texts.append("")
            lines.append(" | ".join(cell_texts))
        
        return "\n".join(lines)
    
    def _build_section_index(self, chunks: List[EnhancedChunk]) -> Dict[str, Any]:
        """Build hierarchical section index."""
        section_index = {}
        
        for chunk in chunks:
            section_key = chunk.headings_path or "Document"
            if section_key not in section_index:
                section_index[section_key] = {
                    "chunks": [],
                    "chunk_types": set(),
                    "word_count": 0
                }
            
            section_index[section_key]["chunks"].append(chunk.chunk_id)
            section_index[section_key]["chunk_types"].add(chunk.chunk_type)
            section_index[section_key]["word_count"] += chunk.counts.get("alnum_words", 0)
        
        # Convert sets to lists for JSON serialization
        for section in section_index.values():
            section["chunk_types"] = list(section["chunk_types"])
        
        return section_index
    
    # ========================================================================
    # TABLE OF CONTENTS DETECTION AND PROCESSING
    # ========================================================================
    
    def _is_table_of_contents_label(self, label: str) -> bool:
        """
        Detect if a table label indicates it's a Table of Contents.
        Domain-agnostic detection using common TOC patterns.
        """
        label_lower = label.lower()
        
        # Common TOC indicators across different languages and domains
        toc_indicators = [
            "table of contents", "contents", "index", "toc",
            "document index", "section index", "content overview",
            "navigation", "outline", "structure",
            # International variants
            "índice", "indice", "table des matières", "inhaltsverzeichnis",
            "目次", "目录", "оглавление"
        ]
        
        return any(indicator in label_lower for indicator in toc_indicators)
    
    def _is_table_of_contents_content(self, table_text: str, headers: List[str]) -> bool:
        """
        Analyze table content to determine if it's a Table of Contents.
        Uses content patterns rather than labels.
        """
        text_lower = table_text.lower()
        
        # TOC content patterns
        toc_patterns = [
            # Section/Item references
            r'\b(?:section|item|chapter|part)\s+[\d\.]+',
            r'\b(?:page|p\.|pg\.)\s*\d+',
            # Common document sections
            r'\b(?:introduction|summary|conclusion|references|appendix)\b',
            r'\b(?:financial|business|risk|management|disclosure)\b',
            # Navigation elements
            r'\b(?:see|refer to|continued on)\b'
        ]
        
        pattern_matches = sum(1 for pattern in toc_patterns if re.search(pattern, text_lower))
        
        # Check headers for TOC-like structure
        header_indicators = 0
        if headers:
            header_text = " ".join(headers).lower()
            if any(word in header_text for word in ["page", "section", "item", "chapter"]):
                header_indicators += 1
        
        # TOC if multiple patterns match or has typical TOC headers
        return pattern_matches >= 2 or header_indicators > 0
    
    def _extract_document_structure_from_toc(self, table_chunks: List[EnhancedChunk]) -> None:
        """
        Extract document structure from Table of Contents chunks.
        Creates semantic mapping for enhanced navigation and cross-referencing.
        """
        for chunk in table_chunks:
            # Check if this chunk is a Table of Contents
            is_toc_by_title = (chunk.table_title and 
                             "table of contents" in chunk.table_title.lower())
            
            is_toc_by_content = self._is_table_of_contents_content(
                chunk.text, 
                chunk.col_headers + chunk.row_headers
            )
            
            if is_toc_by_title or is_toc_by_content:
                logger.info(f"Detected Table of Contents: {chunk.table_title}")
                self._parse_toc_structure(chunk)
                break  # Usually only one TOC per document
    
    def _parse_toc_structure(self, toc_chunk: EnhancedChunk) -> None:
        """
        Parse Table of Contents to extract document structure.
        Creates semantic mappings between sections and page numbers.
        """
        toc_text = toc_chunk.text
        lines = toc_text.split('\n')
        
        structure = {
            "sections": [],
            "page_mapping": {},
            "hierarchy": {}
        }
        
        current_section = None
        
        for line in lines:
            if not line.strip() or line.strip() in ['---', '|']:
                continue
                
            # Parse TOC entries (flexible patterns for different document types)
            section_match = self._parse_toc_line(line)
            if section_match:
                section_info = {
                    "section": section_match["section"],
                    "title": section_match["title"],
                    "page": section_match["page"],
                    "level": section_match.get("level", 1)
                }
                
                structure["sections"].append(section_info)
                structure["page_mapping"][section_match["title"]] = section_match["page"]
                
                # Build hierarchy
                if section_match.get("level") == 1:
                    current_section = section_match["title"]
                    structure["hierarchy"][current_section] = {"subsections": []}
                elif current_section and section_match.get("level") == 2:
                    structure["hierarchy"][current_section]["subsections"].append(section_match["title"])
        
        self.document_structure = structure
        self.toc_section_mapping = structure["page_mapping"]
        
        # Enhance the TOC chunk with extracted structure
        toc_chunk.metadata["is_table_of_contents"] = True
        toc_chunk.metadata["document_structure"] = structure
        toc_chunk.metadata["section_count"] = len(structure["sections"])
        
        logger.info(f"Extracted document structure: {len(structure['sections'])} sections, "
                   f"{len(structure['page_mapping'])} page mappings")
    
    def _parse_toc_line(self, line: str) -> Optional[Dict[str, Any]]:
        """
        Parse a single Table of Contents line to extract section info.
        Handles various TOC formats flexibly.
        """
        line = line.strip()
        if not line or line.startswith('#'):
            return None
        
        # Common TOC line patterns (flexible for different document types)
        patterns = [
            # "Item 1. Business | 5" or "Item 1.|Business|5"
            r'((?:Item|Section|Chapter|Part)\s+[\d\.A-Z]+\.?)\s*[|\-]?\s*(.+?)\s*[|\-]?\s*(\d+)\s*$',
            # "Business | 5" or "Business|5"
            r'^([^|\d]+?)\s*[|\-]?\s*(\d+)\s*$',
            # "1. Introduction 5" or "1.1 Overview 10"
            r'^([\d\.]+)\s+(.+?)\s+(\d+)\s*$',
            # "Introduction ................. 5"
            r'^(.+?)\s*[\.…]+\s*(\d+)\s*$'
        ]
        
        for pattern in patterns:
            match = re.match(pattern, line, re.IGNORECASE)
            if match:
                groups = match.groups()
                
                if len(groups) == 3:
                    section_num, title, page = groups
                    return {
                        "section": section_num.strip(),
                        "title": title.strip(),
                        "page": int(page) if page.isdigit() else None,
                        "level": self._determine_section_level(section_num.strip())
                    }
                elif len(groups) == 2:
                    title, page = groups
                    return {
                        "section": "",
                        "title": title.strip(),
                        "page": int(page) if page.isdigit() else None,
                        "level": 1
                    }
        
        return None
    
    def _determine_section_level(self, section_identifier: str) -> int:
        """Determine hierarchical level from section identifier."""
        if not section_identifier:
            return 1
            
        # Count dots or depth indicators
        if '.' in section_identifier:
            return section_identifier.count('.') + 1
        
        # Check for hierarchical patterns
        if re.match(r'^(Item|Section|Chapter|Part)\s+\d+$', section_identifier, re.IGNORECASE):
            return 1
        elif re.match(r'^(Item|Section|Chapter|Part)\s+\d+[A-Z]$', section_identifier, re.IGNORECASE):
            return 2
            
        return 1
    
    def _enrich_chunk_with_toc_context(self, chunk: EnhancedChunk) -> None:
        """
        Enrich chunk with context from Table of Contents structure.
        Adds semantic navigation and cross-referencing capabilities.
        """
        if not self.document_structure:
            return
            
        chunk_page = chunk.page
        chunk_heading = chunk.headings_path
        
        # Find relevant section from TOC
        relevant_section = None
        for section_info in self.document_structure["sections"]:
            if chunk_page and section_info["page"]:
                if chunk_page >= section_info["page"]:
                    relevant_section = section_info
            elif chunk_heading and section_info["title"]:
                if section_info["title"].lower() in chunk_heading.lower():
                    relevant_section = section_info
                    break
        
        if relevant_section:
            chunk.metadata["toc_section"] = relevant_section["title"]
            chunk.metadata["toc_section_page"] = relevant_section["page"]
            chunk.metadata["toc_section_level"] = relevant_section["level"]
            
            # Add navigation context
            current_idx = self.document_structure["sections"].index(relevant_section)
            if current_idx > 0:
                prev_section = self.document_structure["sections"][current_idx - 1]
                chunk.metadata["previous_section"] = prev_section["title"]
            if current_idx < len(self.document_structure["sections"]) - 1:
                next_section = self.document_structure["sections"][current_idx + 1]
                chunk.metadata["next_section"] = next_section["title"]