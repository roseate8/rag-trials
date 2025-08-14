"""
Layout-aware document chunking using Markdown + JSON fallback.

This implementation follows the Layout-Aware-Document-Extraction-Chunking-and-Indexing approach:
1. Hierarchical chunking: Title → Section → Subsection (never cross boundaries)
2. Element-specific handling: Tables, Lists, Paragraphs, Key-Value pairs
3. Header preservation: Each chunk includes its section/subsection context
4. Metadata preservation: Full lineage and element details stored
5. JSON fallback: For complex elements not well-represented in Markdown
"""
from __future__ import annotations

import json
import logging
import os
import re
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

# Import LLM classifier for intelligent table titles
try:
    from ..llm_table_classifier import LLMTableClassifier
except ImportError:
    LLMTableClassifier = None
    logger.warning("LLMTableClassifier not available - falling back to rule-based titles")


def count_alnum_words(text: str) -> int:
    """Count alphanumeric words in text."""
    tokens = re.findall(r"[A-Za-z0-9]+", text or "")
    return len(tokens)


def make_hash(payload: Any) -> str:
    """Generate hash for content deduplication."""
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


@dataclass
class Chunk:
    """Represents a single chunk with all metadata."""
    text: str
    method: str = "layout_aware_chunking"
    chunk_id: int = 0
    chunk_type: str = "paragraph"  # paragraph | list | table | kv
    lineage: Dict[str, Any] = field(default_factory=dict)
    headers: Dict[str, Any] = field(default_factory=dict)
    table_meta: Dict[str, Any] = field(default_factory=dict)
    counts: Dict[str, Any] = field(default_factory=dict)
    source: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class LayoutAwareChunker:
    """
    Layout-aware chunker that processes documents based on structure.
    
    Key principles:
    1. Respects document hierarchy (Title → Section → Subsection)
    2. Never creates chunks that cross subsection boundaries
    3. Preserves context headers for each chunk type
    4. Uses JSON for complex element detection when MD is insufficient
    """
    
    def __init__(
        self,
        max_words: int = 300,
        min_words: int = 15,
        external_table_dir: Optional[str] = None,
        table_payload_cap_bytes: int = 50_000,
        alias_enrichment: bool = False,
        openai_api_key: Optional[str] = None,
    ) -> None:
        self.max_words = max_words
        self.min_words = min_words
        self.alias_enrichment = alias_enrichment
        self.table_payload_cap_bytes = table_payload_cap_bytes
        
        # Initialize LLM classifier for intelligent table titles
        self.llm_classifier = None
        if LLMTableClassifier and openai_api_key:
            try:
                self.llm_classifier = LLMTableClassifier(openai_api_key)
                logger.info("LLM table classifier enabled for intelligent table titles")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM classifier: {e}")
        elif not openai_api_key:
            logger.debug("No OpenAI API key provided - using rule-based table titles")
        
        # Set up external table storage directory
        if external_table_dir is None:
            pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            self.external_table_dir = os.path.join(pkg_root, "data", "tables")
        else:
            self.external_table_dir = external_table_dir
        os.makedirs(self.external_table_dir, exist_ok=True)
    
    def chunk_document(
        self,
        file_path: str,
        source_format: str = "markdown",
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Main entry point for chunking documents.
        
        Args:
            file_path: Path to markdown file
            source_format: Must be "markdown" (only supported format)
            
        Returns:
            (chunks, section_index) - chunks list and hierarchical index
        """
        if source_format != "markdown":
            raise ValueError("Only 'markdown' source format is supported")
            
        # Load corresponding JSON for complex element detection
        # Try multiple strategies to find the JSON file
        json_path = None
        
        # Strategy 1: Replace /markdown/ with /json/ (original structure)
        if "/markdown/" in file_path:
            json_path = file_path.replace("/markdown/", "/json/").replace(".md", ".json")
        else:
            # Strategy 2: Same directory, just change extension
            json_path = file_path.replace(".md", ".json")
        
        json_data = self._load_json_fallback(json_path)
        
        return self._chunk_markdown_with_json_fallback(file_path, json_data)
    
    def _load_json_fallback(self, json_path: str) -> Optional[Dict[str, Any]]:
        """Load JSON data for complex element detection."""
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load JSON fallback from {json_path}: {e}")
            return None
    
    def _chunk_markdown_with_json_fallback(
        self, 
        md_path: str, 
        json_data: Optional[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Process markdown file with JSON enrichment for rich structure.
        
        Enhanced Workflow:
        1. Parse JSON to build element index with page numbers, bboxes, labels
        2. Parse markdown line-by-line for clean text extraction
        3. Enrich MD elements with JSON metadata:
           - Tables: Use row_span/col_span for merged cells
           - Table semantics: column_header/row_header/row_section flags
           - Page numbers: Add to every chunk from JSON bbox data
           - Spatial context: Include bbox for element positioning
        4. Build Title → Section → Subsection hierarchy respecting JSON groups
        5. Never cross subsection boundaries in chunks
        """
        
        with open(md_path, "r", encoding="utf-8") as f:
            md_content = f.read()
        
        lines = md_content.splitlines()
        chunks: List[Chunk] = []
        section_index: Dict[str, Any] = {}
        chunk_id = 0
        
        # Build JSON element index for enrichment
        json_index = self._build_json_element_index(json_data) if json_data else {}
        
        # Process JSON tables first to create comprehensive table chunks
        table_chunks = self._create_table_chunks_from_json(json_index, md_path) if json_index.get("tables") else []
        
        # Hierarchy tracking
        current = {"title": None, "section": None, "subsection": None}
        counters = {"title": 0, "section": 0, "subsection": 0}
        
        # Content buffers
        paragraph_buffer: List[str] = []
        last_paragraph_text: Optional[str] = None
        
        def ensure_subsection_entry() -> str:
            """Ensure we have a subsection context for content."""
            if current["subsection"] is None:
                counters["subsection"] += 1
                current["subsection"] = {"id": f"subsection_{counters['subsection']}", "text": ""}
                section_index[current["subsection"]["id"]] = {
                    "subsection_text": "",
                    "chunks": [],
                    "title": current["title"],
                    "section": current["section"],
                }
            return current["subsection"]["id"]
        
        def get_lineage() -> Dict[str, Any]:
            """Get current hierarchical context."""
            return {
                "title_id": current["title"]["id"] if current["title"] else None,
                "title_text": current["title"]["text"] if current["title"] else None,
                "section_id": current["section"]["id"] if current["section"] else None,
                "section_text": current["section"]["text"] if current["section"] else None,
                "subsection_id": current["subsection"]["id"] if current["subsection"] else None,
                "subsection_text": current["subsection"]["text"] if current["subsection"] else None,
            }
        
        def flush_paragraph_buffer():
            """Process accumulated paragraph lines."""
            nonlocal chunk_id, last_paragraph_text
            if not paragraph_buffer:
                return
            
            text = " ".join(line.strip() for line in paragraph_buffer if line.strip())
            if not text or count_alnum_words(text) == 0:
                paragraph_buffer.clear()
                return
            
            sub_id = ensure_subsection_entry()
            lineage = get_lineage()
            
            # Add subsection context for better retrieval
            if lineage.get("subsection_text"):
                text = f"{lineage['subsection_text']}: {text}"
            
            # Keep text as-is for semantic retrieval
            
            # Handle minimum word requirement with merging
            word_count = count_alnum_words(text)
            if word_count < self.min_words:
                if (chunks and chunks[-1].chunk_type == "paragraph" and 
                    chunks[-1].lineage.get("subsection_id") == lineage.get("subsection_id")):
                    merged_text = chunks[-1].text + " " + text
                    chunks[-1].text = merged_text
                    chunks[-1].counts["alnum_words"] = count_alnum_words(merged_text)
                    paragraph_buffer.clear()
                    last_paragraph_text = text
                    return
            
            chunk = Chunk(
                text=text,
                chunk_id=chunk_id,
                chunk_type="paragraph",
                lineage=lineage,
                counts={"alnum_words": word_count},
                source={"file_path": md_path, "source_format": "markdown"},
            )
            
            # Enrich with JSON metadata
            self._enrich_chunk_with_json_metadata(chunk, json_index)
            
            chunks.append(chunk)
            section_index[sub_id]["chunks"].append(chunk_id)
            chunk_id += 1
            last_paragraph_text = text
            paragraph_buffer.clear()
        
        # Main parsing loop
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # 1. HEADING DETECTION
            if self._is_heading(line):
                flush_paragraph_buffer()
                level, heading_text = self._parse_heading(line)
                
                if level == 1:  # Title
                    counters["title"] += 1
                    current["title"] = {"id": f"title_{counters['title']}", "text": heading_text}
                    current["section"] = None
                    current["subsection"] = None
                elif level == 2:  # Section
                    counters["section"] += 1
                    current["section"] = {"id": f"section_{counters['section']}", "text": heading_text}
                    current["subsection"] = None
                elif level == 3:  # Subsection
                    counters["subsection"] += 1
                    current["subsection"] = {"id": f"subsection_{counters['subsection']}", "text": heading_text}
                    section_index[current["subsection"]["id"]] = {
                        "subsection_text": heading_text,
                        "chunks": [],
                        "title": current["title"],
                        "section": current["section"],
                    }
                i += 1
                continue
            
            # 2. TABLE LINE DETECTION (SKIP - already processed via JSON)
            if self._is_table_line(line):
                i += 1
                continue  # Skip table lines since JSON processing handles them
            
            # 3. KEY-VALUE DETECTION
            if self._is_key_value_line(line):
                flush_paragraph_buffer()
                sub_id = ensure_subsection_entry()
                lineage = get_lineage()
                
                text = line
                if lineage.get("subsection_text"):
                    text = f"{lineage['subsection_text']}: {text}"
                # Keep KV pairs as-is for semantic retrieval
                
                chunk = Chunk(
                    text=text,
                    chunk_id=chunk_id,
                    chunk_type="kv",
                    lineage=lineage,
                    counts={"alnum_words": count_alnum_words(text)},
                    source={"file_path": md_path, "source_format": "markdown"},
                )
                
                # Enrich with JSON metadata
                self._enrich_chunk_with_json_metadata(chunk, json_index)
                
                chunks.append(chunk)
                section_index[sub_id]["chunks"].append(chunk_id)
                chunk_id += 1
                i += 1
                continue
            
            # 4. PARAGRAPH HANDLING
            if line.strip() == "":
                flush_paragraph_buffer()
                i += 1
                continue
            
            # Skip low-signal lines
            if count_alnum_words(line) == 0 or len(line.strip()) <= 1:
                i += 1
                continue
            
            # Add to paragraph buffer
            paragraph_buffer.append(line)
            i += 1
        
        # Final flush
        flush_paragraph_buffer()
        
        # Add comprehensive table chunks at the beginning (they have full context)
        all_chunks = table_chunks + chunks
        
        # Update chunk IDs to be sequential
        for i, chunk in enumerate(all_chunks):
            chunk.chunk_id = i
        
        output_chunks = [chunk.__dict__ for chunk in all_chunks]
        logger.info(f"Layout-aware chunking complete: {len(output_chunks)} chunks ({len(table_chunks)} table chunks + {len(chunks)} text chunks)")
        
        return output_chunks, section_index
    
    def _is_heading(self, line: str) -> bool:
        """Check if line is a markdown heading."""
        return bool(re.match(r"^#{1,3}\s+", line))
    
    def _parse_heading(self, line: str) -> Tuple[int, str]:
        """Parse heading level and text."""
        match = re.match(r"^(#{1,3})\s+(.+)", line)
        if match:
            level = len(match.group(1))
            text = match.group(2).strip()
            return level, text
        return 1, line.strip()
    
    def _is_key_value_line(self, line: str) -> bool:
        """
        Detect key-value pairs.
        
        Rationale: Look for patterns like "Key: Value", "Label - Value", etc.
        Common in financial documents for metadata.
        """
        # Pattern: text followed by colon/dash and more text
        patterns = [
            r"^[^:]+:\s*[^:]+$",  # "Key: Value"
            r"^[^-]+-\s*[^-]+$",  # "Key - Value"
            r"^[^=]+=\s*[^=]+$",  # "Key = Value"
        ]
        
        for pattern in patterns:
            if re.match(pattern, line.strip()):
                return True
        return False
    
    def _is_table_line(self, line: str) -> bool:
        """
        Check if line is part of a markdown table.
        
        Rationale: We already process tables via JSON, so skip markdown table lines
        to avoid creating duplicate fragmented chunks.
        """
        stripped = line.strip()
        if not stripped:
            return False
        
        # Markdown table patterns
        return (
            "|" in stripped and  # Contains pipe separator
            stripped != "|" and  # Not just a lone pipe
            not stripped.startswith("#")  # Not a heading
        )
    
    def _build_json_element_index(self, json_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build an index of JSON elements for enriching markdown chunks.
        
        Returns:
        {
            "tables": {...},  # Table structures with cell metadata
            "text_by_content": {...},  # Text elements indexed by content for matching
            "groups": {...},  # Grouped elements (key_value_area, lists, etc.)
            "page_map": {...}  # Content to page number mapping
        }
        """
        index = {
            "tables": {},
            "text_by_content": {},
            "groups": {},
            "page_map": {}
        }
        
        # Index tables with rich cell metadata
        if "tables" in json_data:
            for table in json_data["tables"]:
                if "data" in table and "table_cells" in table["data"]:
                    table_id = table["self_ref"]
                    cells = table["data"]["table_cells"]
                    
                    # Build cell matrix with metadata
                    cell_matrix = self._build_table_cell_matrix(cells)
                    
                    # Extract page number from table
                    page_no = None
                    if table.get("prov") and table["prov"]:
                        page_no = table["prov"][0].get("page_no")
                    
                    index["tables"][table_id] = {
                        "cells": cells,
                        "cell_matrix": cell_matrix,
                        "page_no": page_no,
                        "label": table.get("label", "table")
                    }
        
        # Index text elements by content for matching
        if "texts" in json_data:
            for text_elem in json_data["texts"]:
                text_content = text_elem.get("text", "").strip()
                if text_content:
                    # Extract page number and bbox
                    page_no = None
                    bbox = None
                    if text_elem.get("prov") and text_elem["prov"]:
                        prov = text_elem["prov"][0]
                        page_no = prov.get("page_no")
                        bbox = prov.get("bbox")
                    
                    index["text_by_content"][text_content] = {
                        "label": text_elem.get("label", "text"),
                        "level": text_elem.get("level"),
                        "page_no": page_no,
                        "bbox": bbox,
                        "self_ref": text_elem.get("self_ref")
                    }
                    
                    # Also index for page mapping
                    if page_no:
                        index["page_map"][text_content] = page_no
        
        # Index groups (key_value_area, lists, etc.)
        if "groups" in json_data:
            for group in json_data["groups"]:
                label = group.get("label", "group")
                if label in ["key_value_area", "list", "form_area"]:
                    group_id = group["self_ref"]
                    index["groups"][group_id] = {
                        "label": label,
                        "children": group.get("children", [])
                    }
        
        logger.info(f"JSON index built: {len(index['tables'])} tables, {len(index['text_by_content'])} texts, {len(index['groups'])} groups")
        return index
    
    def _build_table_cell_matrix(self, cells: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Build a structured cell matrix from JSON table cells.
        
        Handles merged cells by expanding row_span/col_span into individual positions.
        Returns metadata about headers, sections, and cell positions.
        """
        if not cells:
            return {"rows": [], "headers": [], "sections": []}
        
        # Find matrix dimensions
        max_row = max(cell["end_row_offset_idx"] for cell in cells)
        max_col = max(cell["end_col_offset_idx"] for cell in cells)
        
        # Initialize matrix
        matrix = [[None for _ in range(max_col)] for _ in range(max_row)]
        headers = {"column": [], "row": []}
        sections = []
        
        # Fill matrix with cell data
        for cell in cells:
            start_row = cell["start_row_offset_idx"]
            end_row = cell["end_row_offset_idx"]
            start_col = cell["start_col_offset_idx"]
            end_col = cell["end_col_offset_idx"]
            
            cell_data = {
                "text": cell.get("text", ""),
                "row_span": cell.get("row_span", 1),
                "col_span": cell.get("col_span", 1),
                "is_column_header": cell.get("column_header", False),
                "is_row_header": cell.get("row_header", False),
                "is_row_section": cell.get("row_section", False)
            }
            
            # Handle merged cells by duplicating content
            for r in range(start_row, end_row):
                for c in range(start_col, end_col):
                    if r < len(matrix) and c < len(matrix[r]):
                        matrix[r][c] = cell_data
            
            # Collect header and section information
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
    
    def _get_page_number_for_text(self, text: str, json_index: Dict[str, Any]) -> Optional[int]:
        """Get page number for a text chunk using JSON index."""
        # Try exact match first
        if text in json_index.get("page_map", {}):
            return json_index["page_map"][text]
        
        # Try partial matches for text that contains subsection prefixes
        text_clean = text
        if ":" in text:
            # Remove subsection prefix "Section Name: actual content"
            text_clean = text.split(":", 1)[1].strip()
        
        if text_clean in json_index.get("page_map", {}):
            return json_index["page_map"][text_clean]
        
        # Try fuzzy matching on text elements
        for json_text, metadata in json_index.get("text_by_content", {}).items():
            if json_text in text or text in json_text:
                return metadata.get("page_no")
        
        return None
    
    def _enrich_chunk_with_json_metadata(
        self, 
        chunk: Chunk, 
        json_index: Dict[str, Any]
    ) -> None:
        """Enrich chunk with JSON metadata including page numbers and spatial info."""
        # Add page number
        page_no = self._get_page_number_for_text(chunk.text, json_index)
        if page_no:
            chunk.metadata["page_no"] = page_no
        
        # Add spatial context if available
        text_clean = chunk.text
        if ":" in text_clean:
            text_clean = text_clean.split(":", 1)[1].strip()
        
        if text_clean in json_index.get("text_by_content", {}):
            text_meta = json_index["text_by_content"][text_clean]
            if text_meta.get("bbox"):
                chunk.metadata["bbox"] = text_meta["bbox"]
            if text_meta.get("label"):
                chunk.metadata["json_label"] = text_meta["label"]
    
    def _create_table_chunks_from_json(self, json_index: Dict[str, Any], md_path: str) -> List[Chunk]:
        """
        Create comprehensive table chunks using JSON structure.
        
        Key fix: Each table chunk includes:
        1. Table header/title for context
        2. Column headers for understanding data structure  
        3. Related data rows (grouped semantically)
        4. Never split context from data
        """
        table_chunks = []
        chunk_id = 0
        table_counter = 1  # For numbered fallback labels
        
        for table_id, table_data in json_index.get("tables", {}).items():
            cell_matrix = table_data["cell_matrix"]
            if not cell_matrix or not cell_matrix["rows"]:
                continue
                
            matrix = cell_matrix["rows"]
            column_headers = cell_matrix["headers"]["column"]
            row_headers = cell_matrix["headers"]["row"] 
            sections = cell_matrix["sections"]
            
            # Extract table title/context (look for preceding text or section headers)
            table_title = self._extract_table_title(table_data, json_index, table_counter)
            table_counter += 1
            
            # Group related rows together (never separate context from data)
            row_groups = self._group_table_rows_semantically(matrix, column_headers, sections)
            
            for group_idx, row_group in enumerate(row_groups):
                # Build comprehensive chunk text with full context
                chunk_text_lines = []
                
                # Always include table title for context
                if table_title:
                    chunk_text_lines.append(f"Table: {table_title}")
                
                # Always include column headers for data structure understanding
                if column_headers:
                    header_texts = [h["text"] for h in column_headers]
                    chunk_text_lines.append(" | ".join(header_texts))
                    chunk_text_lines.append(" | ".join(["---"] * len(header_texts)))
                
                # Include grouped data rows
                for row_idx in row_group["rows"]:
                    if row_idx < len(matrix):
                        row_cells = matrix[row_idx]
                        if row_cells:
                            # Filter out None cells and extract text
                            cell_texts = []
                            for cell in row_cells:
                                if cell and cell.get("text"):
                                    cell_texts.append(cell["text"])
                                else:
                                    cell_texts.append("")
                            chunk_text_lines.append(" | ".join(cell_texts))
                
                # Create chunk with rich metadata (no artificial aliases)
                chunk_text = "\n".join(chunk_text_lines)
                
                # Skip if chunk has no meaningful content
                if count_alnum_words(chunk_text) < self.min_words:
                    continue
                
                chunk = Chunk(
                    text=chunk_text,
                    chunk_id=chunk_id,
                    chunk_type="table",
                    lineage={
                        "table_id": table_id,
                        "table_title": table_title,
                        "row_group": group_idx
                    },
                    headers={
                        "table_title": table_title,
                        "column_headers": [h["text"] for h in column_headers],
                        "group_context": row_group.get("context", "")
                    },
                    table_meta={
                        "total_rows": len(matrix),
                        "row_range": (min(row_group["rows"]), max(row_group["rows"])),
                        "has_headers": bool(column_headers),
                        "table_label": table_data.get("label", "table")
                    },
                    counts={"alnum_words": count_alnum_words(chunk_text)},
                    source={"file_path": md_path, "source_format": "json_enhanced"},
                    metadata={
                        "page_no": table_data.get("page_no"),
                        "json_table_id": table_id
                    }
                )
                
                table_chunks.append(chunk)
                chunk_id += 1
                
                # Respect max_words limit
                if count_alnum_words(chunk_text) > self.max_words:
                    logger.warning(f"Table chunk {chunk_id-1} exceeds max_words ({count_alnum_words(chunk_text)} > {self.max_words})")
        
        logger.info(f"Created {len(table_chunks)} comprehensive table chunks from JSON")
        return table_chunks
    
    def _extract_table_title(self, table_data: Dict[str, Any], json_index: Dict[str, Any], table_number: int) -> str:
        """
        Extract meaningful table title based on content analysis.
        
        Strategy:
        1. Analyze cell content for financial patterns
        2. Check JSON label for semantic meaning
        3. Look for preceding text elements (captions)
        4. Use spatial analysis to find nearby headers
        5. Fall back to numbered label + content classification
        """
        
        # 1. Try JSON label first (clean it up)
        json_label = table_data.get("label", "")
        if json_label and json_label != "table":
            # Convert snake_case and clean up
            cleaned_label = json_label.replace("_", " ").replace("-", " ").title()
            if cleaned_label == "Document Index":
                return "Table of Contents"
            return cleaned_label
        
        # 2. Use LLM-based intelligent classification if available
        if self.llm_classifier:
            try:
                llm_title = self.llm_classifier.generate_table_title(table_data, table_number)
                # If LLM generated a specific title (not generic fallback), use it
                if not llm_title.startswith(f"Table {table_number}"):
                    return llm_title
            except Exception as e:
                logger.warning(f"LLM table classification failed: {e}")
        
        # 3. Skip hardcoded fallback - no pattern matching
        
        # 4. Look for spatial context (nearby text elements)
        caption_title = self._find_table_caption_from_spatial_context(table_data, json_index)
        if caption_title:
            return caption_title
        
        # 5. Final fallback: simple numbered label
        return f"Table {table_number}: Data Table"
    

    

    
    def _find_table_caption_from_spatial_context(
        self, 
        table_data: Dict[str, Any], 
        json_index: Dict[str, Any]
    ) -> Optional[str]:
        """
        Find table caption by analyzing nearby text elements using spatial positioning.
        
        Strategy: Look for text elements that appear just before the table
        based on page number and bounding box coordinates.
        """
        table_page = table_data.get("page_no")
        if not table_page:
            return None
        
        # Get table bounding box
        table_bbox = None
        if table_data.get("prov") and table_data["prov"]:
            table_bbox = table_data["prov"][0].get("bbox")
        
        if not table_bbox:
            return None
        
        # Look for text elements on the same page that appear above the table
        candidate_captions = []
        
        for text_content, text_meta in json_index.get("text_by_content", {}).items():
            if text_meta.get("page_no") == table_page and text_meta.get("bbox"):
                text_bbox = text_meta["bbox"]
                
                # Check if text appears above the table (smaller 't' coordinate in BOTTOMLEFT system)
                if (text_bbox.get("t", 0) > table_bbox.get("t", 0) and 
                    abs(text_bbox.get("l", 0) - table_bbox.get("l", 0)) < 100):  # Roughly same column
                    
                    # Check if it looks like a caption
                    if self._looks_like_table_caption(text_content):
                        distance = text_bbox.get("t", 0) - table_bbox.get("t", 0)
                        candidate_captions.append((distance, text_content))
        
        # Return the closest caption above the table
        if candidate_captions:
            candidate_captions.sort(key=lambda x: x[0])  # Sort by distance
            return candidate_captions[0][1].strip()
        
        return None
    
    def _looks_like_table_caption(self, text: str) -> bool:
        """Check if text looks like a table caption."""
        text_lower = text.lower().strip()
        
        # Caption indicators
        caption_patterns = [
            "table", "the following table", "statement", "summary",
            "consolidated", "financial", "shows", "presents"
        ]
        
        # Should be reasonably short (not a paragraph)
        if len(text.split()) > 20:
            return False
        
        # Should contain caption-like words
        return any(pattern in text_lower for pattern in caption_patterns)
    

    
    def _group_table_rows_semantically(
        self, 
        matrix: List[List[Dict]], 
        column_headers: List[Dict], 
        sections: List[Dict]
    ) -> List[Dict[str, Any]]:
        """
        Group table rows semantically to ensure context stays with data.
        
        Strategy:
        1. Keep section headers with their related data rows
        2. Group rows by semantic similarity (e.g., "Basic EPS" with "Diluted EPS")  
        3. Never exceed max_words per group
        4. Always include column headers in each group
        """
        if not matrix:
            return []
        
        groups = []
        current_group = {"rows": [], "context": "", "word_count": 0}
        
        # Calculate base word count (table title + column headers)
        base_words = len(column_headers) * 3  # Rough estimate
        
        for row_idx, row in enumerate(matrix):
            if not row:
                continue
                
            # Calculate words in this row
            row_text = " ".join(cell.get("text", "") for cell in row if cell)
            row_words = count_alnum_words(row_text)
            
            # Check if this row is a section header
            is_section_header = any(
                section["row"] == row_idx for section in sections
            )
            
            # Check if adding this row would exceed max_words
            projected_words = current_group["word_count"] + base_words + row_words
            
            # Start new group if:
            # 1. Would exceed max_words
            # 2. This is a section header and we already have content
            # 3. Current group is getting too large
            if (projected_words > self.max_words and current_group["rows"]) or \
               (is_section_header and current_group["rows"]):
                if current_group["rows"]:
                    groups.append(current_group)
                current_group = {"rows": [], "context": "", "word_count": 0}
            
            # Add row to current group
            current_group["rows"].append(row_idx)
            current_group["word_count"] += row_words
            
            # Update context if this is a section header
            if is_section_header:
                section_text = " ".join(cell.get("text", "") for cell in row if cell)
                current_group["context"] = section_text
        
        # Add final group
        if current_group["rows"]:
            groups.append(current_group)
        
        # Ensure minimum viable groups (merge very small groups)
        filtered_groups = []
        for group in groups:
            if group["word_count"] >= self.min_words or not filtered_groups:
                filtered_groups.append(group)
            else:
                # Merge with previous group if it won't exceed max_words
                if filtered_groups:
                    prev_group = filtered_groups[-1]
                    if prev_group["word_count"] + group["word_count"] + base_words <= self.max_words:
                        prev_group["rows"].extend(group["rows"])
                        prev_group["word_count"] += group["word_count"]
                    else:
                        filtered_groups.append(group)
        
        return filtered_groups if filtered_groups else [{"rows": list(range(len(matrix))), "context": "", "word_count": sum(count_alnum_words(" ".join(cell.get("text", "") for cell in row if cell)) for row in matrix)}]
    

