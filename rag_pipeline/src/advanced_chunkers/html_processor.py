"""
HTML document processing module for layout-aware chunking.
Maintains behavior parity with markdown+JSON processing.
"""
from typing import List, Dict, Any, Optional
import logging
import re
from dataclasses import asdict

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

from .layout_aware_chunker import EnhancedChunk, count_alnum_words

logger = logging.getLogger(__name__)


class HTMLProcessor:
    """
    HTML document processor with behavior parity to markdown+JSON chunking.
    Handles table extraction, hierarchy processing, and content structuring.
    """
    
    def __init__(self, chunker_instance):
        """Initialize with reference to main chunker for configuration access."""
        self.chunker = chunker_instance
        
    def process_html_document(self, html_path: str) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Process HTML document with same chunking behavior as markdown+JSON.
        
        Returns:
            (chunks, section_index) - enhanced chunks with rich metadata
        """
        if BeautifulSoup is None:
            raise ImportError("BeautifulSoup4 is required for HTML processing. Install with: pip install beautifulsoup4")
        
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract and process HTML tables first (equivalent to JSON table processing)
        table_chunks = self._extract_html_tables(soup, html_path)
        
        # Extract document structure from HTML TOC if present
        self._extract_document_structure_from_html_toc(table_chunks, soup)
        
        # Process HTML content with hierarchy tracking
        text_chunks = self._process_html_content(soup, html_path)
        
        # Combine and finalize chunks (same logic as markdown)
        all_chunks = table_chunks + text_chunks
        
        # Build section index
        section_index = self.chunker._build_section_index(all_chunks)
        
        # Convert to output format and add sequential chunk_id for compatibility
        output_chunks = []
        for i, chunk in enumerate(all_chunks):
            chunk_dict = asdict(chunk)
            chunk_dict["chunk_id"] = i
            chunk_dict["uuid"] = chunk.chunk_id
            output_chunks.append(chunk_dict)
        
        logger.info(f"HTML chunking complete: {len(output_chunks)} chunks "
                   f"({len(table_chunks)} table + {len(text_chunks)} text)")
        logger.info(f"Document: {self.chunker.doc_id} v{self.chunker.doc_version} ({self.chunker.source_type})")
        
        return output_chunks, section_index
    
    def _extract_html_tables(self, soup: Any, html_path: str) -> List[EnhancedChunk]:
        """Extract and process HTML tables with same behavior as JSON tables."""
        table_chunks = []
        table_counter = 1
        
        tables = soup.find_all('table')
        
        for table in tables:
            # Convert HTML table to matrix format (like JSON tables)
            table_data = self._html_table_to_matrix(table, table_counter)
            if not table_data or not table_data.get("cell_matrix", {}).get("rows"):
                continue
            
            cell_matrix = table_data["cell_matrix"]
            matrix = cell_matrix["rows"]
            column_headers = cell_matrix["headers"]["column"]
            sections = cell_matrix["sections"]
            
            # Generate table title (same logic as JSON tables)
            table_title = self._generate_html_table_title(table, table_data, table_counter)
            table_counter += 1
            
            # Group rows semantically (same logic as JSON tables)
            row_groups = self.chunker._group_table_rows(matrix, column_headers, sections)
            
            for group_idx, row_group in enumerate(row_groups):
                chunk_text = self.chunker._build_table_chunk_text(table_title, column_headers, matrix, row_group)
                
                if count_alnum_words(chunk_text) < self.chunker.min_words:
                    continue
                
                # Create enhanced chunk with table metadata
                chunk = self.chunker._create_enhanced_chunk(
                    text=chunk_text,
                    chunk_type="table",
                    md_path=html_path,
                    json_index={},  # No JSON enrichment for HTML
                    lineage={"table_id": f"html_table_{table_counter-1}", "table_title": table_title},
                    table_data={
                        "table_id": f"html_table_{table_counter-1}",
                        "table_title": table_title,
                        "cell_matrix": cell_matrix,
                        "html_attributes": dict(table.attrs) if table.attrs else {}
                    }
                )
                
                table_chunks.append(chunk)
        
        logger.info(f"Extracted {len(table_chunks)} HTML table chunks")
        return table_chunks
    
    def _is_numeric_value(self, text: str) -> bool:
        """Check if text represents a numeric value (including currency, percentages)."""
        if not text:
            return False
        
        # Remove common non-numeric characters
        cleaned = text.replace('$', '').replace(',', '').replace('%', '').replace('(', '').replace(')', '').strip()
        
        try:
            float(cleaned)
            return True
        except ValueError:
            return False
    
    def _html_table_to_matrix(self, table: Any, table_number: int) -> Dict[str, Any]:
        """Convert HTML table to matrix format equivalent to JSON table processing."""
        rows = table.find_all('tr')
        if not rows:
            return {}
        
        # Build matrix with proper dimensions
        max_cols = 0
        row_data = []
        
        for row in rows:
            cells = row.find_all(['td', 'th'])
            row_cells = []
            for cell in cells:
                cell_data = {
                    "text": cell.get_text(strip=True),
                    "is_header": cell.name == 'th',
                    "rowspan": int(cell.get('rowspan', 1)),
                    "colspan": int(cell.get('colspan', 1)),
                    "attributes": dict(cell.attrs) if cell.attrs else {}
                }
                row_cells.append(cell_data)
            row_data.append(row_cells)
            max_cols = max(max_cols, len(row_cells))
        
        # Normalize matrix and extract headers
        matrix = []
        column_headers = []
        row_headers = []
        sections = []
        
        for row_idx, row_cells in enumerate(row_data):
            matrix_row = []
            for col_idx, cell in enumerate(row_cells):
                matrix_row.append({
                    "text": cell["text"],
                    "is_column_header": cell["is_header"] and row_idx == 0,
                    "is_row_header": cell["is_header"] and col_idx == 0,
                    "is_row_section": False  # Could be enhanced with CSS class detection
                })
                
                # Enhanced header collection
                if cell["is_header"] and row_idx == 0:
                    column_headers.append({
                        "text": cell["text"],
                        "row": row_idx,
                        "col_range": (col_idx, col_idx + 1)
                    })
                elif cell["is_header"] and col_idx == 0:
                    row_headers.append({
                        "text": cell["text"],
                        "col": col_idx,
                        "row_range": (row_idx, row_idx + 1)
                    })
                # Also check for implied headers (first row/column with non-numeric content)
                elif row_idx == 0 and not self._is_numeric_value(cell["text"]) and cell["text"].strip():
                    column_headers.append({
                        "text": cell["text"],
                        "row": row_idx,
                        "col_range": (col_idx, col_idx + 1)
                    })
                elif col_idx == 0 and not self._is_numeric_value(cell["text"]) and cell["text"].strip():
                    row_headers.append({
                        "text": cell["text"],
                        "col": col_idx,
                        "row_range": (row_idx, row_idx + 1)
                    })
            
            matrix.append(matrix_row)
        
        return {
            "table_id": f"html_table_{table_number}",
            "cell_matrix": {
                "rows": matrix,
                "headers": {"column": column_headers, "row": row_headers},
                "sections": sections,
                "dimensions": (len(matrix), max_cols)
            },
            "html_element": table
        }
    
    def _generate_html_table_title(self, table: Any, table_data: Dict, table_number: int) -> str:
        """Generate meaningful table title for HTML tables."""
        # Check for HTML caption
        caption = table.find('caption')
        if caption:
            caption_text = caption.get_text(strip=True)
            if caption_text and len(caption_text) < 100:
                return caption_text
        
        # Check for nearby headings (look up the DOM tree)
        parent = table.parent
        for _ in range(3):  # Look up 3 levels
            if parent:
                prev_headings = parent.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
                if prev_headings:
                    # Get the closest heading before the table
                    for heading in reversed(prev_headings):
                        if heading.get_text(strip=True):
                            return f"Table: {heading.get_text(strip=True)}"
                parent = parent.parent
            else:
                break
        
        # Always use LLM classification for financial tables if available
        if self.chunker.llm_classifier:
            try:
                # Extract table content for LLM analysis
                sample_text = ""
                if table_data.get("cell_matrix", {}).get("rows"):
                    matrix = table_data["cell_matrix"]["rows"]
                    for row in matrix[:3]:  # First 3 rows
                        for cell in row[:5]:  # First 5 columns
                            if cell and cell.get("text"):
                                sample_text += cell["text"] + " "
                
                if sample_text.strip():
                    llm_title = self.chunker.llm_classifier.generate_table_title(
                        {"sample_content": sample_text}, table_number
                    )
                    # Use LLM title if it's different from generic fallback
                    if llm_title and not llm_title.startswith(f"Table {table_number}"):
                        return llm_title
            except Exception as e:
                logger.warning(f"LLM table classification failed: {e}")
        

        
        # Check CSS classes for semantic hints (generic patterns only)
        table_classes = table.get('class', [])
        if table_classes:
            class_text = " ".join(table_classes)
            # Only check for generic table of contents patterns
            if any(hint in class_text.lower() for hint in ['toc', 'contents', 'index']):
                return "Table of Contents"
        
        # Generic fallback
        return f"Table {table_number}"
    
    def _process_html_content(self, soup: Any, html_path: str) -> List[EnhancedChunk]:
        """Process HTML content with same behavior as markdown processing."""
        chunks = []
        
        # Remove tables since they're processed separately
        for table in soup.find_all('table'):
            table.decompose()
        
        # Extract text content while preserving structure
        content_elements = self._extract_html_structure(soup)
        
        # Process with hierarchy tracking (same logic as markdown)
        current_hierarchy = {"title": None, "section": None, "subsection": None}
        paragraph_buffer = []
        
        def flush_paragraph():
            if not paragraph_buffer:
                return
            
            text = " ".join(line.strip() for line in paragraph_buffer if line.strip())
            if not text or count_alnum_words(text) == 0:
                paragraph_buffer.clear()
                return
            
            lineage = self.chunker._get_current_lineage(current_hierarchy)
            
            # Add subsection context (same logic as markdown)
            if lineage.get("subsection_text"):
                text = f"{lineage['subsection_text']}: {text}"
            
            # Handle minimum word requirement with merging (same logic as markdown)
            word_count = count_alnum_words(text)
            if (word_count < self.chunker.min_words and chunks and 
                chunks[-1].chunk_type == "paragraph" and
                chunks[-1].lineage.get("subsection_id") == lineage.get("subsection_id")):
                # Merge with previous chunk
                chunks[-1].text = chunks[-1].text + " " + text
                chunks[-1].counts["alnum_words"] = count_alnum_words(chunks[-1].text)
                paragraph_buffer.clear()
                return
            
            chunk = self.chunker._create_enhanced_chunk(
                text=text,
                chunk_type="paragraph",
                md_path=html_path,
                json_index={},  # No JSON enrichment for HTML
                lineage=lineage
            )
            chunks.append(chunk)
            paragraph_buffer.clear()
        
        # Process elements
        for element in content_elements:
            if element["type"] == "heading":
                flush_paragraph()
                level = element["level"]
                heading_text = element["text"]
                self.chunker._update_hierarchy(current_hierarchy, level, heading_text)
            elif element["type"] == "key_value":
                flush_paragraph()
                lineage = self.chunker._get_current_lineage(current_hierarchy)
                
                text = element["text"]
                if lineage.get("subsection_text"):
                    text = f"{lineage['subsection_text']}: {text}"
                
                chunk = self.chunker._create_enhanced_chunk(
                    text=text,
                    chunk_type="kv",
                    md_path=html_path,
                    json_index={},
                    lineage=lineage
                )
                chunks.append(chunk)
            elif element["type"] == "list":
                flush_paragraph()
                lineage = self.chunker._get_current_lineage(current_hierarchy)
                
                text = element["text"]
                if lineage.get("subsection_text"):
                    text = f"{lineage['subsection_text']}: {text}"
                
                chunk = self.chunker._create_enhanced_chunk(
                    text=text,
                    chunk_type="list",
                    md_path=html_path,
                    json_index={},
                    lineage=lineage
                )
                chunks.append(chunk)
            elif element["type"] == "paragraph":
                if element["text"].strip():
                    paragraph_buffer.append(element["text"])
            elif element["type"] == "empty":
                flush_paragraph()
        
        # Final flush
        flush_paragraph()
        
        logger.info(f"Processed HTML content: {len(chunks)} text chunks")
        return chunks
    
    def _extract_clean_content_blocks(self, soup: Any) -> List[Dict[str, Any]]:
        """
        Extract clean content blocks from HTML using tags only for structure.
        
        Philosophy:
        - Tags define structure and boundaries
        - Only inner text content is extracted and stored
        - Structure guides chunking decisions and metadata
        """
        # Tags that should be completely ignored (including content)
        ignore_tags = {
            'script', 'style', 'meta', 'link', 'head', 'title',
            'nav', 'header', 'footer', 'aside', 'noscript'
        }
        
        # Remove ignored tags completely
        for tag_name in ignore_tags:
            for tag in soup.find_all(tag_name):
                tag.decompose()
        
        content_blocks = []
        self._extract_content_from_element(soup.body or soup, content_blocks, [])
        
        # Filter and clean content blocks
        return self._clean_content_blocks(content_blocks)
    
    def _extract_content_from_element(self, element: Any, content_blocks: List[Dict[str, Any]], parent_path: List[str]):
        """
        Recursively extract content from HTML elements.
        Uses element structure to guide extraction but only stores clean text.
        """
        if not element or not hasattr(element, 'name'):
            return
        
        current_path = parent_path + [element.name] if element.name else parent_path
        
        # Handle different element types based on structure
        if element.name and element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            text = self._get_clean_text_content(element)
            if text:
                level = int(element.name[1])  # h1 -> 1, h2 -> 2, etc.
                content_blocks.append({
                    "content": text,
                    "block_type": "heading",
                    "hierarchy_level": level,
                    "semantic_context": f"Level {level} heading",
                    "parent_structure": parent_path
                })
        
        elif element.name == 'table':
            # Skip tables as they're processed separately
            pass
        
        elif element.name in ['ul', 'ol']:
            list_content = self._extract_clean_list_content(element)
            if list_content:
                content_blocks.append({
                    "content": list_content,
                    "block_type": "list",
                    "semantic_context": f"{'Numbered' if element.name == 'ol' else 'Bulleted'} list",
                    "parent_structure": current_path[:-1]
                })
        
        elif element.name in ['p', 'div', 'section', 'article']:
            text = self._get_clean_text_content(element)
            if text:
                # Check if it's a key-value pattern
                if self._is_key_value_pattern(text):
                    content_blocks.append({
                        "content": text,
                        "block_type": "key_value",
                        "semantic_context": "Key-value information",
                        "parent_structure": current_path[:-1]
                    })
                else:
                    content_blocks.append({
                        "content": text,
                        "block_type": "paragraph",
                        "semantic_context": "Paragraph content",
                        "parent_structure": current_path[:-1]
                    })
        
        else:
            # For other elements, process children
            for child in element.children:
                if hasattr(child, 'name'):
                    self._extract_content_from_element(child, content_blocks, current_path)
    
    def _get_clean_text_content(self, element: Any) -> str:
        """
        Extract clean text from element, removing all HTML artifacts.
        This is the key method that ensures only content, not structure, is stored.
        """
        if not element:
            return ""
        
        # Get text content, excluding script/style
        text = element.get_text(separator=" ", strip=True) if element else ""
        
        # Clean the text thoroughly
        return self._clean_text_artifacts(text)
    
    def _clean_text_artifacts(self, text: str) -> str:
        """
        Thoroughly clean text content of all HTML artifacts and noise.
        """
        if not text:
            return ""
        
        # Remove HTML entities and artifacts
        text = re.sub(r'&[a-zA-Z0-9#]+;', ' ', text)
        
        # Remove GLYPH artifacts (common in PDF-to-HTML conversions)
        text = re.sub(r'GLYPH[^G]*GLYPH', '', text)
        text = re.sub(r'GLYPH.*?(?=\s|$)', '', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove navigation/UI artifacts
        noise_patterns = [
            r'Skip to (?:main )?content',
            r'Toggle navigation',
            r'Search\s*Search',
            r'Home\s*>\s*[^>]*>\s*',
            r'Copyright\s*©\s*\d{4}',
            r'All rights reserved',
            r'Privacy Policy',
            r'Terms of Service'
        ]
        
        for pattern in noise_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        return text.strip()
    
    def _extract_clean_list_content(self, element: Any) -> str:
        """Extract list content as clean, formatted text."""
        list_items = []
        
        for li in element.find_all('li', recursive=False):
            item_text = self._get_clean_text_content(li)
            if item_text:
                list_items.append(item_text)
        
        if not list_items:
            return ""
        
        # Format as clean list
        if element.name == 'ol':
            return "\n".join(f"{i+1}. {item}" for i, item in enumerate(list_items))
        else:
            return "\n".join(f"• {item}" for item in list_items)
    
    def _is_key_value_pattern(self, text: str) -> bool:
        """Detect if text represents a key-value pair."""
        if not text or len(text.split()) > 10:  # Too long to be a simple key-value
            return False
        
        # Common key-value patterns
        kv_patterns = [
            r'^[^:]+:\s*.+$',  # key: value
            r'^[^=]+=.+$',     # key=value
            r'^[^-]+\s*-\s*.+$',  # key - value
        ]
        
        return any(re.match(pattern, text.strip()) for pattern in kv_patterns)
    
    def _clean_content_blocks(self, content_blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter and clean content blocks."""
        cleaned = []
        
        for block in content_blocks:
            # Filter out empty or meaningless content
            if not block["content"] or len(block["content"].strip()) < 3:
                continue
            
            # Filter out pure navigation/UI content
            if self._is_ui_noise(block["content"]):
                continue
            
            # Additional cleaning
            block["content"] = self._clean_text_artifacts(block["content"])
            
            if block["content"]:  # Only keep if still has content after cleaning
                cleaned.append(block)
        
        return cleaned
    
    def _is_ui_noise(self, text: str) -> bool:
        """Check if text is UI noise that shouldn't be indexed."""
        text_lower = text.lower().strip()
        
        noise_indicators = [
            # Navigation
            'home', 'menu', 'search', 'login', 'logout', 'sign in', 'sign up',
            # UI elements
            'click here', 'read more', 'show more', 'expand', 'collapse',
            # Generic labels
            'loading', 'error', 'success', 'warning', 'info',
        ]
        
        return (
            len(text) < 10 or
            text_lower in noise_indicators or
            text.count(' ') == 0 and len(text) < 20  # Single short words
        )

    def _extract_html_structure(self, soup: Any) -> List[Dict[str, Any]]:
        """Extract structured content from HTML while preserving hierarchy."""
        elements = []
        
        # Find all content elements in document order
        content_tags = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'div', 'ul', 'ol', 'li', 'section', 'article'])
        
        for tag in content_tags:
            # Skip if this tag is inside another tag we'll process
            if any(parent.name in ['ul', 'ol'] for parent in tag.parents):
                if tag.name not in ['ul', 'ol']:
                    continue
            
            text = tag.get_text(strip=True)
            if not text:
                continue
            
            if tag.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                level = int(tag.name[1])  # Extract number from h1, h2, etc.
                # Map to configured hierarchy levels
                if level <= self.chunker.heading_config.get("max_heading_level", 3):
                    elements.append({
                        "type": "heading",
                        "level": level,
                        "text": text,
                        "element": tag
                    })
            
            elif tag.name in ['ul', 'ol']:
                # Process entire list as one chunk
                list_items = tag.find_all('li', recursive=False)
                if list_items:
                    list_text = "\n".join(f"• {li.get_text(strip=True)}" for li in list_items if li.get_text(strip=True))
                    if list_text:
                        elements.append({
                            "type": "list",
                            "text": list_text,
                            "element": tag
                        })
            
            elif tag.name in ['p', 'div', 'section', 'article']:
                # Check if it's a key-value pattern
                if self._is_html_key_value(tag):
                    elements.append({
                        "type": "key_value",
                        "text": text,
                        "element": tag
                    })
                else:
                    elements.append({
                        "type": "paragraph",
                        "text": text,
                        "element": tag
                    })
        
        return elements
    
    def _is_html_key_value(self, tag: Any) -> bool:
        """Detect key-value patterns in HTML content."""
        text = tag.get_text(strip=True)
        
        # Same regex patterns as markdown key-value detection
        patterns = [r"^[^:]+:\s*[^:]+$", r"^[^-]+-\s*[^-]+$", r"^[^=]+=\s*[^=]+$"]
        if any(re.match(pattern, text) for pattern in patterns):
            return True
        
        # HTML structure patterns
        if tag.find_all(['dt', 'dd']):  # Definition list
            return True
        
        # Check for label/value span structure
        spans = tag.find_all('span')
        if len(spans) == 2:
            classes_1 = spans[0].get('class', [])
            classes_2 = spans[1].get('class', [])
            if (any('label' in str(c).lower() for c in classes_1) and 
                any('value' in str(c).lower() for c in classes_2)):
                return True
        
        # CSS class hints
        classes = tag.get('class', [])
        if any('metric' in str(c).lower() or 'kv' in str(c).lower() or 'key-value' in str(c).lower() 
               for c in classes):
            return True
        
        return False
    
    def _extract_document_structure_from_html_toc(self, table_chunks: List[EnhancedChunk], soup: Any) -> None:
        """Extract document structure from HTML TOC elements."""
        # First check table-based TOC (same as existing logic)
        self.chunker._extract_document_structure_from_toc(table_chunks)
        
        # If no TOC found in tables, check HTML navigation structures
        if not self.chunker.document_structure:
            self._extract_html_nav_structure(soup)
    
    def _extract_html_nav_structure(self, soup: Any) -> None:
        """Extract document structure from HTML navigation elements."""
        # Look for common TOC patterns
        toc_selectors = [
            'nav.table-of-contents',
            'nav.toc',
            '.table-of-contents',
            '.document-index',
            '.toc',
            'nav ul',
            '.contents'
        ]
        
        for selector in toc_selectors:
            toc_element = soup.select_one(selector)
            if toc_element:
                self._parse_html_toc_element(toc_element)
                if self.chunker.document_structure:
                    logger.info(f"Extracted HTML navigation structure from: {selector}")
                    break
    
    def _parse_html_toc_element(self, toc_element: Any) -> None:
        """Parse HTML TOC element to extract document structure."""
        structure = {
            "sections": [],
            "page_mapping": {},
            "hierarchy": {}
        }
        
        # Find all links or structured elements
        links = toc_element.find_all('a')
        if not links:
            # Try to find structured content without links
            items = toc_element.find_all(['li', 'div'])
            links = items
        
        for item in links:
            text = item.get_text(strip=True)
            if not text:
                continue
            
            # Extract href for page/section mapping
            href = item.get('href', '')
            section_id = href.lstrip('#') if href.startswith('#') else ''
            
            section_info = {
                "section": section_id,
                "title": text,
                "page": None,  # HTML doesn't have page numbers typically
                "level": self._determine_html_toc_level(item)
            }
            
            structure["sections"].append(section_info)
            structure["page_mapping"][text] = section_id
        
        if structure["sections"]:
            self.chunker.document_structure = structure
            self.chunker.toc_section_mapping = structure["page_mapping"]
    
    def _determine_html_toc_level(self, item: Any) -> int:
        """Determine hierarchical level from HTML TOC item."""
        # Check nesting level in lists
        list_parents = [p for p in item.parents if p.name in ['ul', 'ol']]
        if list_parents:
            return len(list_parents)
        
        # Check CSS classes for level hints
        classes = item.get('class', [])
        for cls in classes:
            if 'level-' in str(cls).lower():
                try:
                    return int(str(cls).split('level-')[1][0])
                except (IndexError, ValueError):
                    pass
        
        return 1

