"""
Complete Pipeline Runner with Comprehensive Metrics
Runs chunking, embedding, and storage with detailed metrics tracking.
"""

import sys
import os
import time
import psutil
import logging
from typing import List, Dict, Any, Tuple
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from rag_pipeline.src.advanced_chunkers.layout_aware_chunker import LayoutAwareChunker
from rag_pipeline.src.embeddings import EmbeddingGenerator
from rag_pipeline.src.qdrant_store import QdrantVectorStore

logger = logging.getLogger(__name__)

class CompletePipelineRunner:
    """
    Complete pipeline runner that handles both single files and batch processing.
    Tracks comprehensive metrics across all phases:
    1. Chunking
    2. Embedding generation  
    3. Vector storage
    
    Supports:
    - Single file processing
    - Batch processing for 100s of files
    - Unified logging with detailed metrics
    - Token tracking by method and file
    """
    
    def __init__(self, openai_api_key: str = None, log_dir: str = None, domain_vocab: Dict[str, List[str]] = None):
        """Initialize the complete pipeline runner."""
        self.openai_api_key = openai_api_key
        self.embedder = EmbeddingGenerator()
        self.vector_store = QdrantVectorStore()
        self.domain_vocab = domain_vocab  # Optional domain-specific vocabulary
        
        # Set up unified log directory
        if log_dir is None:
            self.log_dir = os.path.join(os.path.dirname(__file__), "pipeline_logs")
        else:
            self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Pipeline-wide metrics (supports both single and batch)
        self.pipeline_metrics = {
            "session_start": None,
            "session_end": None,
            "total_files_processed": 0,
            "total_files_failed": 0,
            "total_chunks_created": 0,
            "total_embeddings_generated": 0,
            "total_chunks_stored": 0,
            "total_tokens_used": 0,
            "total_processing_time": 0,
            "total_embedding_time": 0,
            "total_storage_time": 0,
            "tokens_by_method": {},
            "files_metrics": [],
            "error_log": [],
            "memory_peak": 0
        }
        
        # Session ID for unified logging
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def run_complete_pipeline(
        self,
        file_path: str,
        source_format: str = "html",
        doc_name: str = None,
        source_type: str = "document"
    ) -> Dict[str, Any]:
        """
        Run the complete pipeline with comprehensive metrics tracking.
        
        Returns:
            Complete metrics including chunking, embedding, and storage phases
        """
        
        logger.info("🚀 STARTING COMPLETE PIPELINE WITH METRICS TRACKING")
        logger.info("=" * 70)
        
        # Start overall timing
        self.pipeline_metrics["overall"]["start_time"] = time.time()
        
        try:
            # Phase 1: Chunking
            logger.info("📦 PHASE 1: CHUNKING")
            chunks, chunking_metrics = self._run_chunking_phase(
                file_path, source_format, doc_name, source_type
            )
            
            # Phase 2: Embedding
            logger.info("🔤 PHASE 2: EMBEDDING GENERATION")
            embeddings, embedding_metrics = self._run_embedding_phase(chunks)
            
            # Phase 3: Storage
            logger.info("💾 PHASE 3: VECTOR STORAGE")
            storage_metrics = self._run_storage_phase(chunks, embeddings)
            
            # Complete overall metrics
            self.pipeline_metrics["overall"]["end_time"] = time.time()
            self.pipeline_metrics["overall"]["total_duration"] = (
                self.pipeline_metrics["overall"]["end_time"] - 
                self.pipeline_metrics["overall"]["start_time"]
            )
            
            # Combine all metrics
            complete_metrics = {
                "chunking": chunking_metrics,
                "embedding": embedding_metrics,
                "storage": storage_metrics,
                "overall": self.pipeline_metrics["overall"],
                "summary": self._generate_pipeline_summary(chunks, embeddings)
            }
            
            # Save comprehensive pipeline log
            self._save_pipeline_log(file_path, complete_metrics)
            
            logger.info("✅ COMPLETE PIPELINE FINISHED SUCCESSFULLY")
            return complete_metrics
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            raise
    
    def _run_chunking_phase(
        self, 
        file_path: str, 
        source_format: str, 
        doc_name: str, 
        source_type: str
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Run chunking phase with metrics tracking."""
        
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        
        # Configure controlled vocabulary (domain-agnostic, configurable)
        # These are just example categories - can be customized per use case
        controlled_vocab = {
            'products': [],  # No hardcoded products - domain agnostic
            'metrics': [],   # No hardcoded metrics - domain agnostic  
            'policy_tags': [] # No hardcoded tags - domain agnostic
        }
        
        # Optional: Add domain-specific vocabulary if provided
        # This allows the system to work for any domain without hardcoding
        if hasattr(self, 'domain_vocab') and self.domain_vocab:
            controlled_vocab.update(self.domain_vocab)
        
        # Initialize chunker
        chunker = LayoutAwareChunker(
            doc_name=doc_name or os.path.basename(file_path),
            source_type=source_type,
            openai_api_key=self.openai_api_key,
            controlled_vocab=controlled_vocab
        )
        
        # Run chunking (this will generate its own detailed log)
        chunks, section_index = chunker.chunk_document(file_path, source_format)
        
        # Extract chunking metrics from the chunker
        chunking_metrics = chunker.metrics.copy()
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        
        # Add phase-specific metrics
        chunking_metrics["phase_duration"] = end_time - start_time
        chunking_metrics["phase_memory_used"] = end_memory - start_memory
        
        logger.info(f"   ✅ Chunking complete: {len(chunks)} chunks in {end_time - start_time:.2f}s")
        
        return chunks, chunking_metrics
    
    def _run_embedding_phase(self, chunks: List[Dict[str, Any]]) -> Tuple[List[List[float]], Dict[str, Any]]:
        """Run embedding generation phase with metrics tracking."""
        
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        
        # Extract text from chunks
        texts = [chunk.get('text', '') for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.embedder.generate_embeddings(texts)
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        
        # Calculate embedding metrics
        embedding_metrics = {
            "total_texts": len(texts),
            "total_embeddings": len(embeddings),
            "embedding_dimension": len(embeddings[0]) if embeddings else 0,
            "phase_duration": end_time - start_time,
            "phase_memory_used": end_memory - start_memory,
            "embeddings_per_second": len(embeddings) / (end_time - start_time) if end_time > start_time else 0,
            "average_text_length": sum(len(text) for text in texts) / len(texts) if texts else 0
        }
        
        logger.info(f"   ✅ Embedding complete: {len(embeddings)} embeddings in {end_time - start_time:.2f}s")
        
        return embeddings, embedding_metrics
    
    def _run_storage_phase(
        self, 
        chunks: List[Dict[str, Any]], 
        embeddings: List[List[float]]
    ) -> Dict[str, Any]:
        """Run vector storage phase with metrics tracking."""
        
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        
        # Create collection
        self.vector_store.create_collection()
        
        # Store chunks with embeddings
        self.vector_store.store_chunks(chunks, embeddings)
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        
        # Calculate storage metrics
        storage_metrics = {
            "total_chunks_stored": len(chunks),
            "total_embeddings_stored": len(embeddings),
            "phase_duration": end_time - start_time,
            "phase_memory_used": end_memory - start_memory,
            "chunks_per_second": len(chunks) / (end_time - start_time) if end_time > start_time else 0,
            "collection_name": self.vector_store.collection_name
        }
        
        logger.info(f"   ✅ Storage complete: {len(chunks)} chunks stored in {end_time - start_time:.2f}s")
        
        return storage_metrics
    
    def _generate_pipeline_summary(
        self, 
        chunks: List[Dict[str, Any]], 
        embeddings: List[List[float]]
    ) -> Dict[str, Any]:
        """Generate overall pipeline summary."""
        
        total_duration = self.pipeline_metrics["overall"]["total_duration"]
        
        return {
            "total_chunks_processed": len(chunks),
            "total_embeddings_generated": len(embeddings),
            "total_pipeline_duration": total_duration,
            "overall_throughput": len(chunks) / total_duration if total_duration > 0 else 0,
            "pipeline_efficiency": "High" if total_duration < 60 else "Medium" if total_duration < 300 else "Low"
        }
    
    def _save_pipeline_log(self, file_path: str, metrics: Dict[str, Any]) -> None:
        """Save comprehensive pipeline log with all metrics."""
        
        # Create log directory
        log_dir = os.path.join(os.path.dirname(__file__), "chunking_logs")
        os.makedirs(log_dir, exist_ok=True)
        
        # Generate log filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_base = os.path.splitext(os.path.basename(file_path))[0]
        log_filename = f"complete_pipeline_{file_base}_{timestamp}.md"
        log_path = os.path.join(log_dir, log_filename)
        
        # Generate comprehensive markdown report
        report_content = self._generate_pipeline_report(file_path, metrics)
        
        # Save to file
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"📋 Complete pipeline log saved to: {log_path}")
    
    def run_batch_pipeline(
        self,
        file_paths: List[str],
        batch_name: str = "batch_processing",
        clear_vector_store: bool = True
    ) -> Dict[str, Any]:
        """
        Process multiple files through complete pipeline with unified logging.
        Optimized for 100s of files with memory management.
        
        Args:
            file_paths: List of file paths to process
            batch_name: Name for this batch (for logging)
            clear_vector_store: Whether to clear existing vector store
            
        Returns:
            Comprehensive batch metrics
        """
        
        logger.info(f"🚀 STARTING BATCH PIPELINE: {batch_name}")
        logger.info(f"📊 Files to process: {len(file_paths)}")
        logger.info(f"🔧 Pipeline: Chunking → Embedding → Storage")
        logger.info("=" * 70)
        
        # Start batch timing
        self.pipeline_metrics["session_start"] = time.time()
        
        # Clear vector store if requested
        if clear_vector_store:
            logger.info("🗑️  Clearing existing vector store...")
            self.vector_store.clean_database()
            self.vector_store.create_collection()
            logger.info("✅ Vector store cleared and ready")
        
        # Process files in batches to manage memory
        batch_size = 10  # Process 10 files at a time
        
        for i in range(0, len(file_paths), batch_size):
            batch_files = file_paths[i:i + batch_size]
            batch_num = i//batch_size + 1
            total_batches = (len(file_paths) + batch_size - 1) // batch_size
            
            logger.info(f"📦 Processing batch {batch_num}/{total_batches}")
            
            batch_chunks = []
            batch_embeddings = []
            
            # Process each file in the batch
            for j, file_path in enumerate(batch_files):
                file_index = i + j + 1
                logger.info(f"📄 Processing file {file_index}/{len(file_paths)}: {os.path.basename(file_path)}")
                
                try:
                    # Process single file
                    file_metrics = self._process_single_file_for_batch(file_path, file_index)
                    
                    if file_metrics["chunks"]:
                        batch_chunks.extend(file_metrics["chunks"])
                        batch_embeddings.extend(file_metrics["embeddings"])
                    
                    # Update metrics
                    self.pipeline_metrics["files_metrics"].append(file_metrics)
                    self.pipeline_metrics["total_files_processed"] += 1
                    self._update_batch_totals(file_metrics)
                    
                except Exception as e:
                    error_info = {
                        "file_path": file_path,
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    }
                    self.pipeline_metrics["error_log"].append(error_info)
                    self.pipeline_metrics["total_files_failed"] += 1
                    logger.error(f"❌ Failed to process {file_path}: {e}")
            
            # Store this batch
            if batch_chunks and batch_embeddings:
                logger.info(f"💾 Storing batch {batch_num} to vector database...")
                storage_start = time.time()
                self.vector_store.store_chunks(batch_chunks, batch_embeddings)
                storage_time = time.time() - storage_start
                self.pipeline_metrics["total_storage_time"] += storage_time
                self.pipeline_metrics["total_chunks_stored"] += len(batch_chunks)
                logger.info(f"✅ Stored {len(batch_chunks)} chunks in {storage_time:.2f}s")
        
        # Complete metrics
        self.pipeline_metrics["session_end"] = time.time()
        self.pipeline_metrics["total_processing_time"] = (
            self.pipeline_metrics["session_end"] - self.pipeline_metrics["session_start"]
        )
        
        # Save unified log
        self._finalize_unified_log(batch_name)
        
        logger.info("✅ BATCH PIPELINE COMPLETED")
        return self.pipeline_metrics
    
    def _process_single_file_for_batch(self, file_path: str, file_index: int) -> Dict[str, Any]:
        """Process a single file for batch processing."""
        
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        
        # Determine file format
        file_ext = os.path.splitext(file_path)[1].lower()
        format_map = {'.html': 'html', '.md': 'markdown', '.json': 'json'}
        source_format = format_map.get(file_ext)
        if not source_format:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        # Process chunking
        chunks, chunking_metrics = self._run_chunking_phase(file_path, source_format, f"Doc_{file_index}", "document")
        
        # Process embedding
        embeddings = []
        embedding_time = 0
        if chunks:
            embedding_start = time.time()
            texts = [chunk.get('text', '') for chunk in chunks]
            embeddings = self.embedder.generate_embeddings(texts)
            embedding_time = time.time() - embedding_start
            self.pipeline_metrics["total_embedding_time"] += embedding_time
            self.pipeline_metrics["total_embeddings_generated"] += len(embeddings)
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        
        # Track peak memory
        if end_memory > self.pipeline_metrics["memory_peak"]:
            self.pipeline_metrics["memory_peak"] = end_memory
        
        return {
            "file_path": file_path,
            "file_name": os.path.basename(file_path),
            "file_size_bytes": os.path.getsize(file_path) if os.path.exists(file_path) else 0,
            "source_format": source_format,
            "chunks_created": len(chunks),
            "chunks": chunks,
            "embeddings": embeddings,
            "processing_time": end_time - start_time,
            "embedding_time": embedding_time,
            "memory_used": end_memory - start_memory,
            "tokens_used": chunking_metrics.get("tokens", {}).get("total_used", 0),
            "tokens_by_method": chunking_metrics.get("tokens", {}).get("by_method", {}),
            "timestamp": datetime.now().isoformat()
        }
    
    def _update_batch_totals(self, file_metrics: Dict[str, Any]) -> None:
        """Update batch totals."""
        self.pipeline_metrics["total_chunks_created"] += file_metrics["chunks_created"]
        self.pipeline_metrics["total_tokens_used"] += file_metrics["tokens_used"]
        
        for method, tokens in file_metrics["tokens_by_method"].items():
            if method not in self.pipeline_metrics["tokens_by_method"]:
                self.pipeline_metrics["tokens_by_method"][method] = 0
            self.pipeline_metrics["tokens_by_method"][method] += tokens
    
    def _finalize_unified_log(self, batch_name: str) -> None:
        """Finalize and save unified log."""
        from .unified_logger import finalize_unified_log
        
        # Log embedding and storage metrics
        from .unified_logger import get_unified_logger
        unified_logger = get_unified_logger()
        unified_logger.log_embedding_batch(
            self.pipeline_metrics["total_embeddings_generated"],
            self.pipeline_metrics["total_embedding_time"]
        )
        unified_logger.log_storage_batch(
            self.pipeline_metrics["total_chunks_stored"],
            self.pipeline_metrics["total_storage_time"]
        )
        
        # Finalize unified log
        log_path = finalize_unified_log(batch_name)
        if log_path:
            logger.info(f"📋 Unified log saved to: {log_path}")
    
    # Removed: _generate_batch_pipeline_report() - now using unified logger
    
    def _generate_batch_pipeline_report_OLD(self, batch_name: str) -> str:
        """Generate unified batch pipeline report."""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        total_files = self.pipeline_metrics["total_files_processed"] + self.pipeline_metrics["total_files_failed"]
        success_rate = (self.pipeline_metrics["total_files_processed"] / total_files * 100) if total_files > 0 else 0
        
        report = f"""# 🚀 **BATCH PIPELINE UNIFIED REPORT**

## 📊 **BATCH SUMMARY**
- **Batch Name**: {batch_name}
- **Session ID**: {self.session_id}
- **Processed At**: {current_time}
- **Total Files**: {total_files:,}
- **Successful**: {self.pipeline_metrics["total_files_processed"]:,} ({success_rate:.1f}%)
- **Failed**: {self.pipeline_metrics["total_files_failed"]:,}

---

## ⏱️ **TIMING BREAKDOWN**
- **Total Pipeline Time**: {self.pipeline_metrics["total_processing_time"]:.2f} seconds
- **Embedding Time**: {self.pipeline_metrics["total_embedding_time"]:.2f}s
- **Storage Time**: {self.pipeline_metrics["total_storage_time"]:.2f}s

---

## 📦 **PIPELINE STATISTICS**
- **Chunks Created**: {self.pipeline_metrics["total_chunks_created"]:,}
- **Embeddings Generated**: {self.pipeline_metrics["total_embeddings_generated"]:,}
- **Chunks Stored**: {self.pipeline_metrics["total_chunks_stored"]:,}

---

## 🤖 **LLM TOKEN USAGE**
- **Total Tokens Used**: {self.pipeline_metrics["total_tokens_used"]:,}

### **Tokens by Method**:
"""
        
        if self.pipeline_metrics["tokens_by_method"]:
            for method, tokens in sorted(self.pipeline_metrics["tokens_by_method"].items()):
                report += f"- **{method}**: {tokens:,} tokens\n"
        else:
            report += "- No LLM calls made\n"
        
        if self.pipeline_metrics["total_tokens_used"] > 0:
            cost = self.pipeline_metrics["total_tokens_used"] * 0.00015 / 1000
            report += f"\n### **Estimated Cost**: ${cost:.4f} USD\n"
        
        report += f"""
---

## 📄 **FILE-LEVEL METRICS**

| File | Format | Size (KB) | Chunks | Tokens | Time (s) | Status |
|------|--------|-----------|--------|--------|----------|---------|
"""
        
        for file_metrics in self.pipeline_metrics["files_metrics"]:
            file_size_kb = file_metrics["file_size_bytes"] / 1024
            file_name = file_metrics["file_name"][:25] + "..." if len(file_metrics["file_name"]) > 25 else file_metrics["file_name"]
            
            report += f"| {file_name} | {file_metrics['source_format']} | {file_size_kb:.1f} | {file_metrics['chunks_created']} | {file_metrics['tokens_used']} | {file_metrics['processing_time']:.2f} | ✅ |\n"
        
        for error in self.pipeline_metrics["error_log"]:
            file_name = os.path.basename(error["file_path"])[:25] + "..." if len(os.path.basename(error["file_path"])) > 25 else os.path.basename(error["file_path"])
            report += f"| {file_name} | - | - | - | - | - | ❌ |\n"
        
        report += f"""
---

## 📈 **PERFORMANCE ANALYSIS**
- **Files per Second**: {self.pipeline_metrics['total_files_processed'] / self.pipeline_metrics['total_processing_time']:.2f}
- **Success Rate**: {success_rate:.1f}%
- **Peak Memory**: {self.pipeline_metrics['memory_peak']:.2f} MB

---

## ✅ **COMPLETION STATUS**
- **Status**: ✅ Batch Processing Complete
- **Vector Store**: ✅ {self.pipeline_metrics["total_chunks_stored"]:,} chunks stored
- **Ready for Queries**: ✅

---

*Generated by Complete Pipeline Runner v2.0*
*Timestamp: {current_time}*
"""
        
        return report
    
    def _generate_pipeline_report(self, file_path: str, metrics: Dict[str, Any]) -> str:
        """Generate comprehensive pipeline report in markdown format."""
        
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""# 🚀 **COMPLETE PIPELINE METRICS REPORT**

## 📄 **FILE INFORMATION**
- **File Path**: `{file_path}`
- **File Name**: `{os.path.basename(file_path)}`
- **Processed At**: {current_time}

---

## ⏱️ **OVERALL PIPELINE TIMING**
- **Total Pipeline Duration**: {metrics['overall']['total_duration']:.3f} seconds
- **Start Time**: {datetime.fromtimestamp(metrics['overall']['start_time']).strftime('%H:%M:%S')}
- **End Time**: {datetime.fromtimestamp(metrics['overall']['end_time']).strftime('%H:%M:%S')}
- **Overall Throughput**: {metrics['summary']['overall_throughput']:.1f} chunks/second

---

## 📦 **PHASE 1: CHUNKING METRICS**
- **Duration**: {metrics['chunking']['timing']['chunking_duration']:.3f} seconds
- **Chunks Created**: {metrics['chunking']['chunks']['total_created']:,}
- **Memory Used**: {metrics['chunking']['resources']['memory_used']:.2f} MB
- **Processing Speed**: {metrics['chunking']['performance']['chunks_per_second']:.1f} chunks/second

### **Chunk Type Breakdown**:
"""
        
        # Add chunking details
        for chunk_type, count in metrics['chunking']['chunks']['by_type'].items():
            report += f"- **{chunk_type}**: {count:,} chunks\n"
        
        report += f"""
### **LLM Token Usage**:
- **Total Tokens**: {metrics['chunking']['tokens']['total_used']:,}
- **LLM Calls**: {metrics['chunking']['tokens']['total_calls']:,}
- **Average per Call**: {metrics['chunking']['tokens']['average_per_call']:.1f}

---

## 🔤 **PHASE 2: EMBEDDING METRICS**
- **Duration**: {metrics['embedding']['phase_duration']:.3f} seconds
- **Embeddings Generated**: {metrics['embedding']['total_embeddings']:,}
- **Embedding Dimension**: {metrics['embedding']['embedding_dimension']}
- **Memory Used**: {metrics['embedding']['phase_memory_used']:.2f} MB
- **Processing Speed**: {metrics['embedding']['embeddings_per_second']:.1f} embeddings/second
- **Average Text Length**: {metrics['embedding']['average_text_length']:.0f} characters

---

## 💾 **PHASE 3: STORAGE METRICS**
- **Duration**: {metrics['storage']['phase_duration']:.3f} seconds
- **Chunks Stored**: {metrics['storage']['total_chunks_stored']:,}
- **Memory Used**: {metrics['storage']['phase_memory_used']:.2f} MB
- **Storage Speed**: {metrics['storage']['chunks_per_second']:.1f} chunks/second
- **Collection**: `{metrics['storage']['collection_name']}`

---

## 📊 **PERFORMANCE ANALYSIS**

### **Phase Performance Breakdown**:
"""
        
        # Calculate phase percentages
        total_time = metrics['overall']['total_duration']
        chunking_pct = (metrics['chunking']['timing']['chunking_duration'] / total_time * 100) if total_time > 0 else 0
        embedding_pct = (metrics['embedding']['phase_duration'] / total_time * 100) if total_time > 0 else 0
        storage_pct = (metrics['storage']['phase_duration'] / total_time * 100) if total_time > 0 else 0
        
        report += f"""- **Chunking**: {chunking_pct:.1f}% of total time
- **Embedding**: {embedding_pct:.1f}% of total time
- **Storage**: {storage_pct:.1f}% of total time

### **Efficiency Rating**: {metrics['summary']['pipeline_efficiency']}

### **Resource Utilization**:
- **Total Memory Used**: {metrics['chunking']['resources']['memory_used'] + metrics['embedding']['phase_memory_used'] + metrics['storage']['phase_memory_used']:.2f} MB
- **Peak Memory**: {metrics['chunking']['resources']['memory_peak']:.2f} MB

---

## 💰 **COST ANALYSIS**
"""
        
        # Add cost analysis
        if metrics['chunking']['tokens']['total_used'] > 0:
            cost_estimate = metrics['chunking']['tokens']['total_used'] * 0.00015 / 1000
            report += f"- **Estimated LLM Cost**: ${cost_estimate:.4f} USD\n"
        else:
            report += "- **LLM Cost**: $0.00 (no API key provided)\n"
        
        report += f"""
---

## ✅ **PIPELINE SUMMARY**
- **Status**: ✅ Completed Successfully
- **Total Processing Time**: {metrics['overall']['total_duration']:.3f} seconds
- **Final Output**: {metrics['summary']['total_chunks_processed']:,} chunks stored in vector database
- **Ready for Queries**: ✅ Vector store populated and ready for retrieval

---

*Generated by Complete Pipeline Runner v1.0*  
*Timestamp: {current_time}*
"""
        
        return report


def main():
    """Main function for testing the complete pipeline runner."""
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )
    
    # Test file path
    file_path = "input/folder 2/folder 3/investor reports/Elastic - Elastic Reports First Quarter Fiscal 2025 Financial Results.html"
    
    # Initialize and run pipeline
    runner = CompletePipelineRunner()
    
    try:
        metrics = runner.run_complete_pipeline(
            file_path=file_path,
            source_format="html",
            doc_name="Elastic Q1 2025 Financial Results",
            source_type="financial_report"
        )
        
        print("🎯 Pipeline completed successfully!")
        print(f"📊 Total chunks: {metrics['summary']['total_chunks_processed']}")
        print(f"⏱️ Total time: {metrics['overall']['total_duration']:.2f} seconds")
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
