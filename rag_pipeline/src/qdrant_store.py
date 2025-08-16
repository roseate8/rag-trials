"""
Qdrant client for vector storage operations
"""
import os
import logging
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition
import uuid

logger = logging.getLogger(__name__)

class QdrantVectorStore:
    def __init__(self, collection_name: str = "document_chunks", host: str = "localhost", port: int = 6333):
        """Initialize Qdrant client"""
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name
        self.embedding_dim = 384  # BAAI/bge-small-en-v1.5 dimension
        
    def clean_database(self):
        """Clean up existing vector database"""
        try:
            # Delete collection if it exists
            collections = self.client.get_collections().collections
            for collection in collections:
                if collection.name == self.collection_name:
                    self.client.delete_collection(collection_name=self.collection_name)
                    logger.info(f"Deleted existing collection: {self.collection_name}")
                    break
            else:
                logger.info(f"Collection {self.collection_name} does not exist")
        except Exception as e:
            logger.error(f"Error cleaning database: {e}")
    
    def create_collection(self):
        """Create a new collection"""
        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.embedding_dim, distance=Distance.COSINE)
            )
            logger.info(f"Created collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            raise
    
    def store_chunks(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]):
        """Store chunks with their embeddings, including enhanced metadata"""
        points = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            # Base payload with legacy fields
            payload = {
                "chunk_id": i,
                "text": chunk["text"],
                "method": chunk["method"],
                "start_char": chunk.get("start_char", 0),
                "end_char": chunk.get("end_char", 0),
                "chunk_size": chunk.get("chunk_size", 0),
                "overlap": chunk.get("overlap", 0),
                "document": chunk.get("document", ""),
                "metadata": chunk.get("metadata", {}),
            }
            
            # **NEW: Enhanced GraphRAG metadata fields**
            # Add all the enhanced metadata fields if they exist
            enhanced_fields = [
                # Canonical identification
                "doc_id", "doc_version", "source_type", "product_component", "confidentiality",
                "ingested_at", "effective_from", "effective_to",
                
                # Chunk taxonomy
                "chunk_type", "page", "bbox", "section_h1", "section_h2", "section_h3", "headings_path",
                
                # Deterministic metadata enrichment (mapping for compatibility)
                "metric_terms", "doc_refs", "entities",
                "mentioned_dates", "release_date", "author", "owner", "is_change_note",
                "owners", "policy_tags",
                
                # Table-aware indexing
                "table_id", "table_title", "row_headers", "col_headers", "units", "periods", "cell_samples",
                
                # Legacy compatibility
                "lineage", "headers", "table_meta", "counts", "source"
            ]
            
            # Add enhanced fields if they exist in the chunk
            for field in enhanced_fields:
                if field in chunk:
                    payload[field] = chunk[field]
            
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload=payload
            )
            points.append(point)
        
        # Upload points in batches
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i+batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch
            )
        
        logger.info(f"Stored {len(points)} chunks in collection")
    
    def get_chunk_by_method_and_index(self, method: str, chunk_index: int) -> Optional[Dict[str, Any]]:
        """Retrieve a specific chunk by method and index"""
        try:
            results = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(key="method", match={"value": method}),
                        FieldCondition(key="chunk_id", match={"value": chunk_index})
                    ]
                ),
                limit=1,
                with_vectors=True
            )
            
            if results[0]:  # results is a tuple (points, next_page_offset)
                point = results[0][0]
                return {
                    "id": point.id,
                    "payload": point.payload,
                    "vector": point.vector[:10] if point.vector else None  # Show only first 10 dimensions
                }
            return None
        except Exception as e:
            logger.error(f"Error retrieving chunk: {e}")
            return None
    
    def get_chunks_by_method(self, method: str) -> List[Dict[str, Any]]:
        """Get all chunks for a specific method"""
        try:
            results = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[FieldCondition(key="method", match={"value": method})]
                ),
                limit=10000  # Large limit to get all chunks
            )
            
            chunks = []
            for point in results[0]:
                chunks.append({
                    "id": point.id,
                    "payload": point.payload,
                    "vector": point.vector
                })
            
            return sorted(chunks, key=lambda x: x["payload"]["chunk_id"])
        except Exception as e:
            logger.error(f"Error retrieving chunks by method: {e}")
            return []
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection"""
        try:
            info = self.client.get_collection(collection_name=self.collection_name)
            return {
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": str(info.status)
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {}

    def delete_by_method(self, method: str) -> int:
        """Delete all points for a specific chunking method. Returns number of deleted (best-effort)."""
        try:
            flt = Filter(must=[FieldCondition(key="method", match={"value": method})])
            self.client.delete(collection_name=self.collection_name, points_selector=flt, wait=True)
            logger.info(f"Deleted points for method={method}")
            # Qdrant does not return count here; caller can verify via scroll
            return 0
        except Exception as e:
            logger.error(f"Error deleting by method {method}: {e}")
            return 0
