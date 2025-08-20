"""
Critical fixes for the chunking system to address:
1. Remove KV chunk type (merge into paragraphs)
2. Add content cleaning for GLYPH artifacts
3. Apply min_words check to ALL chunk types
4. Add post-processing to merge tiny chunks
5. Improve list aggregation
"""

import re
from typing import List, Dict, Any

def clean_content(text: str) -> str:
    """Clean corrupted content and artifacts."""
    if not text:
        return ""
    
    # Remove GLYPH artifacts from PDF extraction
    text = re.sub(r'GLYPH<[^>]*>[^<]*</GLYPH>', '', text)
    text = re.sub(r'GLYPH&lt;[^&]*&gt;[^&]*', '', text)
    text = re.sub(r'/[^/]*GLYPH[^/]*/', '', text)
    
    # Remove HTML artifacts
    text = re.sub(r'&lt;[^&]*&gt;', '', text)
    text = re.sub(r'<[^>]*>', '', text)
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove navigation patterns
    nav_patterns = [
        r'Home\s*>\s*Products\s*>\s*',
        r'Skip to main content',
        r'Toggle navigation',
        r'Search\s*Search\s*',
        r'Copyright\s*©\s*\d{4}',
    ]
    for pattern in nav_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    return text.strip()


def is_meaningful_content(text: str, min_words: int = 3) -> bool:
    """Check if content is meaningful (not just navigation/noise)."""
    if not text:
        return False
    
    # Clean text first
    cleaned = clean_content(text)
    
    # Count alphanumeric words
    tokens = re.findall(r"[A-Za-z0-9]+", cleaned or "")
    word_count = len(tokens)
    
    # Too short
    if word_count < min_words:
        return False
    
    # Check for noise patterns
    noise_patterns = [
        r'^[•\-\*\s]+$',  # Just bullets/dashes
        r'^\d+\.$',       # Just numbers
        r'^[:\s]*$',      # Just colons/spaces
        r'^(Home|Menu|Search|Login|Logout)$',  # Navigation words
    ]
    
    for pattern in noise_patterns:
        if re.match(pattern, cleaned, re.IGNORECASE):
            return False
    
    return True


def merge_tiny_chunks(chunks: List[Dict[str, Any]], min_words: int = 10) -> List[Dict[str, Any]]:
    """
    Post-process chunks to merge tiny adjacent chunks.
    This fixes the issue of too many small chunks.
    """
    if len(chunks) <= 1:
        return chunks
    
    merged = []
    current_chunk = chunks[0].copy()
    
    for next_chunk in chunks[1:]:
        # Check if chunks can be merged
        if should_merge_chunks(current_chunk, next_chunk, min_words):
            # Merge chunks
            merged_text = f"{current_chunk['text']}\n\n{next_chunk['text']}"
            current_chunk['text'] = clean_content(merged_text)
            current_chunk['chunk_type'] = "paragraph"  # Merged chunks become paragraphs
            
            # Update word count
            tokens = re.findall(r"[A-Za-z0-9]+", current_chunk['text'] or "")
            current_chunk['word_count'] = len(tokens)
        else:
            # Cannot merge, add current and move to next
            merged.append(current_chunk)
            current_chunk = next_chunk.copy()
    
    # Add the final chunk
    merged.append(current_chunk)
    
    return merged


def should_merge_chunks(chunk1: Dict[str, Any], chunk2: Dict[str, Any], min_words: int) -> bool:
    """Check if two chunks should be merged."""
    # Get word counts
    text1 = chunk1.get('text', '')
    text2 = chunk2.get('text', '')
    
    words1 = len(re.findall(r"[A-Za-z0-9]+", text1 or ""))
    words2 = len(re.findall(r"[A-Za-z0-9]+", text2 or ""))
    
    # Don't merge if either is already large enough
    if words1 >= min_words and words2 >= min_words:
        return False
    
    # Don't merge tables
    if chunk1.get('chunk_type') == "table" or chunk2.get('chunk_type') == "table":
        return False
    
    # Check combined size wouldn't be too large
    combined_words = words1 + words2
    if combined_words > 300:  # Max words limit
        return False
    
    # Check hierarchy compatibility (soft check)
    if chunk1.get('section_h1') != chunk2.get('section_h1'):
        return False
    
    return True


def convert_kv_to_paragraph(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert KV chunks to paragraphs to simplify chunk types.
    This addresses the issue of too many chunk types preventing merging.
    """
    converted = []
    
    for chunk in chunks:
        chunk_copy = chunk.copy()
        if chunk_copy.get('chunk_type') == 'kv':
            chunk_copy['chunk_type'] = 'paragraph'
        converted.append(chunk_copy)
    
    return converted


def aggregate_lists(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Aggregate adjacent list chunks to reduce fragmentation.
    """
    if len(chunks) <= 1:
        return chunks
    
    aggregated = []
    current_list = None
    
    for chunk in chunks:
        if chunk.get('chunk_type') == 'list':
            if current_list is None:
                current_list = chunk.copy()
            else:
                # Aggregate with previous list
                combined_text = f"{current_list['text']}\n{chunk['text']}"
                current_list['text'] = combined_text
                
                # Update word count
                tokens = re.findall(r"[A-Za-z0-9]+", combined_text or "")
                current_list['word_count'] = len(tokens)
        else:
            # Not a list, add current list if exists
            if current_list:
                aggregated.append(current_list)
                current_list = None
            aggregated.append(chunk)
    
    # Add final list if exists
    if current_list:
        aggregated.append(current_list)
    
    return aggregated


def filter_meaningless_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Filter out chunks that are not meaningful content.
    """
    filtered = []
    
    for chunk in chunks:
        text = chunk.get('text', '')
        if is_meaningful_content(text, min_words=3):
            filtered.append(chunk)
    
    return filtered


def apply_all_fixes(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Apply all chunking fixes in the correct order.
    """
    # Step 1: Clean content in all chunks
    for chunk in chunks:
        if 'text' in chunk:
            chunk['text'] = clean_content(chunk['text'])
    
    # Step 2: Convert KV to paragraphs
    chunks = convert_kv_to_paragraph(chunks)
    
    # Step 3: Filter meaningless chunks
    chunks = filter_meaningless_chunks(chunks)
    
    # Step 4: Aggregate lists
    chunks = aggregate_lists(chunks)
    
    # Step 5: Merge tiny chunks
    chunks = merge_tiny_chunks(chunks, min_words=10)
    
    # Step 6: Final filter
    chunks = filter_meaningless_chunks(chunks)
    
    return chunks
