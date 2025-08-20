"""
Provenance Trail Builder for Graph-RAG Wannabe

Builds structured responses with full provenance trail showing the 2-hop journey.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import sys
import os

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from hop_recipes.base_recipe import HopResult


@dataclass
class GraphRAGResponse:
    """Complete response with provenance trail"""
    
    # Core response
    query: str
    answer: str
    
    # 2-hop trail
    hop_1_chunks: List[Dict[str, Any]] = field(default_factory=list)
    hop_2_chunks: List[Dict[str, Any]] = field(default_factory=list)
    final_chunks: List[Dict[str, Any]] = field(default_factory=list)
    
    # Trail metadata
    expansion_signals: List[str] = field(default_factory=list)
    applied_filters: List[str] = field(default_factory=list)
    expansion_rationale: str = ""
    
    # Performance metrics
    intent_confidence: float = 0.0
    signal_confidence: float = 0.0
    total_time: float = 0.0
    hop_1_count: int = 0
    hop_2_count: int = 0
    
    # Strategy information
    intent_type: str = ""
    expansion_strategy: str = ""
    
    # Citations
    citations: List[Dict[str, Any]] = field(default_factory=list)


class ProvenanceTrailBuilder:
    """
    Builds structured responses with full 2-hop provenance trail.
    
    Shows users exactly how the system "hopped" through metadata
    to find related content.
    """
    
    def __init__(self, llm_client, model: str = "gpt-4o-mini"):
        self.llm_client = llm_client
        self.model = model
    
    def build_response(self, 
                      query: str,
                      hop_result: HopResult,
                      intent_type: str,
                      intent_confidence: float) -> GraphRAGResponse:
        """
        Build complete response with provenance trail.
        
        Args:
            query: Original user query
            hop_result: Result from 2-hop search
            intent_type: Classified intent type
            intent_confidence: Confidence in intent classification
            
        Returns:
            GraphRAGResponse with answer and full trail
        """
        
        # Generate answer using LLM
        answer = self._generate_answer(query, hop_result.final_chunks)
        
        # Build citations
        citations = self._build_citations(hop_result.final_chunks)
        
        # Create response
        response = GraphRAGResponse(
            query=query,
            answer=answer,
            hop_1_chunks=hop_result.hop_1_chunks,
            hop_2_chunks=hop_result.hop_2_chunks,
            final_chunks=hop_result.final_chunks,
            expansion_signals=self._extract_signals_summary(hop_result),
            applied_filters=hop_result.applied_filters,
            expansion_rationale=hop_result.expansion_rationale,
            intent_confidence=intent_confidence,
            signal_confidence=hop_result.signal_confidence,
            total_time=hop_result.total_time,
            hop_1_count=hop_result.hop_1_count,
            hop_2_count=hop_result.hop_2_count,
            intent_type=intent_type,
            expansion_strategy=hop_result.strategy_used,
            citations=citations
        )
        
        return response
    
    def _generate_answer(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        """Generate answer using LLM with retrieved chunks"""
        
        if not chunks:
            return "No relevant information found for this query."
        
        # Build context from chunks
        context_parts = []
        for i, chunk in enumerate(chunks[:10]):  # Limit to top 10 chunks
            payload = chunk.get('payload', {})
            text = payload.get('text', '')
            
            # Add source info for tables
            if payload.get('chunk_type') == 'table':
                table_title = payload.get('table_title', 'Table')
                page = payload.get('page', 'unknown')
                context_parts.append(f"[{table_title}, Page {page}]: {text}")
            else:
                context_parts.append(f"[Chunk {i+1}]: {text}")
        
        context = "\n\n".join(context_parts)
        
        # Build prompt
        prompt = f"""Based on the following retrieved document chunks, answer the user's question comprehensively.

User Question: {query}

Retrieved Context:
{context}

Please provide a clear, accurate answer based solely on the information provided. If specific numbers or data are mentioned, include them in your response. If the context doesn't fully answer the question, acknowledge what information is missing.

Answer:"""
        
        try:
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Answer generation error: {e}")
            return "I found relevant information but encountered an error generating the response."
    
    def _build_citations(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build structured citations for response"""
        
        citations = []
        
        for i, chunk in enumerate(chunks):
            payload = chunk.get('payload', {})
            
            citation = {
                'id': i + 1,
                'text_preview': payload.get('text', '')[:200] + "..." if len(payload.get('text', '')) > 200 else payload.get('text', ''),
                'source': {
                    'doc_id': payload.get('doc_id', 'unknown'),
                    'page': payload.get('page'),
                    'section': payload.get('headings_path', 'Unknown Section'),
                    'chunk_type': payload.get('chunk_type', 'text')
                },
                'relevance_score': chunk.get('score', 0.0),
                'rerank_score': chunk.get('rerank_score')
            }
            
            # Add table-specific metadata
            if payload.get('chunk_type') == 'table':
                citation['table_metadata'] = {
                    'title': payload.get('table_title', 'Table'),
                    'col_headers': payload.get('col_headers', []),
                    'cell_samples': payload.get('cell_samples', [])
                }
            
            citations.append(citation)
        
        return citations
    
    def _extract_signals_summary(self, hop_result: HopResult) -> List[str]:
        """Extract summary of signals used for expansion"""
        
        if not hop_result.extracted_signals:
            return []
        
        signals = hop_result.extracted_signals
        summary = []
        
        if signals.top_metric_terms:
            summary.extend([f"metric:{term}" for term in signals.top_metric_terms])
        
        if signals.top_doc_refs:
            summary.extend([f"reference:{ref}" for ref in signals.top_doc_refs])
        

        
        if signals.mentioned_dates:
            summary.extend([f"date:{date}" for date in signals.mentioned_dates[:2]])
        
        return summary
