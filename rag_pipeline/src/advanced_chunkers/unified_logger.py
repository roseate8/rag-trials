"""
Unified Logging System for All Chunking Operations
Single comprehensive log file across all processing activities.
"""

import os
import json
import time
from typing import Dict, Any, List
from datetime import datetime

class UnifiedLogger:
    """
    Single logger that handles all chunking, embedding, and storage operations.
    Creates one comprehensive log file with all metrics across all files processed.
    """
    
    def __init__(self, log_dir: str = None):
        """Initialize unified logger."""
        # Single log directory
        if log_dir is None:
            self.log_dir = os.path.join(os.path.dirname(__file__), "unified_logs")
        else:
            self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Single session for all operations
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_start = time.time()
        
        # Unified metrics across all operations
        self.unified_metrics = {
            "session_info": {
                "session_id": self.session_id,
                "started_at": datetime.now().isoformat(),
                "session_start_time": self.session_start
            },
            "totals": {
                "files_processed": 0,
                "files_failed": 0,
                "total_chunks": 0,
                "total_embeddings": 0,
                "total_chunks_stored": 0,
                "total_tokens": 0,
                "total_processing_time": 0,
                "total_embedding_time": 0,
                "total_storage_time": 0
            },
            "tokens_by_method": {},
            "files": [],
            "errors": [],
            "performance": {}
        }
    
    def log_file_processing(
        self, 
        file_path: str, 
        chunks_created: int,
        processing_time: float,
        tokens_used: int = 0,
        tokens_by_method: Dict[str, int] = None,
        embedding_time: float = 0,
        storage_time: float = 0,
        file_size_bytes: int = 0,
        source_format: str = "unknown",
        word_count: int = 0,
        char_count: int = 0,
        avg_chunk_size: float = 0,
        error: str = None
    ) -> None:
        """Log metrics for a single file processing."""
        
        file_metrics = {
            "file_path": file_path,
            "file_name": os.path.basename(file_path),
            "file_size_bytes": file_size_bytes,
            "file_size_kb": file_size_bytes / 1024 if file_size_bytes > 0 else 0,
            "source_format": source_format,
            "chunks_created": chunks_created,
            "processing_time": round(processing_time, 3),  # 3 decimal places
            "embedding_time": embedding_time,
            "storage_time": storage_time,
            "tokens_used": tokens_used,
            "tokens_by_method": tokens_by_method or {},
            "word_count": word_count,
            "char_count": char_count,
            "avg_chunk_size": round(avg_chunk_size, 1),
            "timestamp": datetime.now().isoformat(),
            "success": error is None,
            "error": error
        }
        
        # Add to files list
        self.unified_metrics["files"].append(file_metrics)
        
        # Update totals
        if error is None:
            self.unified_metrics["totals"]["files_processed"] += 1
            self.unified_metrics["totals"]["total_chunks"] += chunks_created
            self.unified_metrics["totals"]["total_tokens"] += tokens_used
            self.unified_metrics["totals"]["total_processing_time"] += processing_time
            self.unified_metrics["totals"]["total_embedding_time"] += embedding_time
            self.unified_metrics["totals"]["total_storage_time"] += storage_time
            
            # Update tokens by method
            for method, tokens in (tokens_by_method or {}).items():
                if method not in self.unified_metrics["tokens_by_method"]:
                    self.unified_metrics["tokens_by_method"][method] = 0
                self.unified_metrics["tokens_by_method"][method] += tokens
        else:
            self.unified_metrics["totals"]["files_failed"] += 1
            self.unified_metrics["errors"].append({
                "file_path": file_path,
                "error": error,
                "timestamp": datetime.now().isoformat()
            })
    
    def log_embedding_batch(self, embeddings_count: int, embedding_time: float) -> None:
        """Log batch embedding metrics."""
        self.unified_metrics["totals"]["total_embeddings"] += embeddings_count
        self.unified_metrics["totals"]["total_embedding_time"] += embedding_time
    
    def log_storage_batch(self, chunks_stored: int, storage_time: float) -> None:
        """Log batch storage metrics."""
        self.unified_metrics["totals"]["total_chunks_stored"] += chunks_stored
        self.unified_metrics["totals"]["total_storage_time"] += storage_time
    
    def finalize_and_save(self, operation_name: str = "processing_session") -> str:
        """
        Finalize the session and save unified log.
        Returns path to the saved log file.
        """
        
        # Calculate final metrics
        session_end = time.time()
        total_session_time = session_end - self.session_start
        
        self.unified_metrics["session_info"]["ended_at"] = datetime.now().isoformat()
        self.unified_metrics["session_info"]["total_session_time"] = total_session_time
        
        # Calculate performance metrics
        if total_session_time > 0:
            self.unified_metrics["performance"] = {
                "files_per_second": self.unified_metrics["totals"]["files_processed"] / total_session_time,
                "chunks_per_second": self.unified_metrics["totals"]["total_chunks"] / total_session_time,
                "success_rate": (self.unified_metrics["totals"]["files_processed"] / 
                               max(1, self.unified_metrics["totals"]["files_processed"] + self.unified_metrics["totals"]["files_failed"])) * 100
            }
        
        # Generate comprehensive report
        report_content = self._generate_unified_report(operation_name)
        
        # Save single unified log
        log_filename = f"unified_processing_log_{operation_name}_{self.session_id}.md"
        log_path = os.path.join(self.log_dir, log_filename)
        
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        return log_path
    
    def _generate_unified_report(self, operation_name: str) -> str:
        """Generate comprehensive unified report covering all operations."""
        
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        total_files = self.unified_metrics["totals"]["files_processed"] + self.unified_metrics["totals"]["files_failed"]
        
        report = f"""# 🚀 **UNIFIED PROCESSING LOG**

## 📊 **SESSION SUMMARY**
- **Operation Name**: {operation_name}
- **Session ID**: {self.unified_metrics["session_info"]["session_id"]}
- **Started**: {datetime.fromisoformat(self.unified_metrics["session_info"]["started_at"]).strftime("%Y-%m-%d %H:%M:%S")}
- **Completed**: {current_time}
- **Total Session Time**: {self.unified_metrics["session_info"]["total_session_time"]:.2f} seconds

---

## 📁 **FILE PROCESSING SUMMARY**
- **Total Files**: {total_files:,}
- **Successfully Processed**: {self.unified_metrics["totals"]["files_processed"]:,}
- **Failed**: {self.unified_metrics["totals"]["files_failed"]:,}
- **Success Rate**: {self.unified_metrics["performance"].get("success_rate", 0):.1f}%

---

## 📦 **CHUNKING STATISTICS**
- **Total Chunks Created**: {self.unified_metrics["totals"]["total_chunks"]:,}
- **Total Embeddings Generated**: {self.unified_metrics["totals"]["total_embeddings"]:,}
- **Total Chunks Stored**: {self.unified_metrics["totals"]["total_chunks_stored"]:,}

---

## ⏱️ **TIMING BREAKDOWN**
- **Total Processing Time**: {self.unified_metrics["totals"]["total_processing_time"]:.2f}s
- **Total Embedding Time**: {self.unified_metrics["totals"]["total_embedding_time"]:.2f}s
- **Total Storage Time**: {self.unified_metrics["totals"]["total_storage_time"]:.2f}s
- **Other Operations**: {self.unified_metrics["session_info"]["total_session_time"] - self.unified_metrics["totals"]["total_processing_time"]:.2f}s

---

## 🤖 **LLM TOKEN USAGE**
- **Total Tokens Used**: {self.unified_metrics["totals"]["total_tokens"]:,}

### **Tokens by Method**:
"""
        
        if self.unified_metrics["tokens_by_method"]:
            for method, tokens in sorted(self.unified_metrics["tokens_by_method"].items()):
                report += f"- **{method}**: {tokens:,} tokens\n"
        else:
            report += "- No LLM calls made\n"
        
        if self.unified_metrics["totals"]["total_tokens"] > 0:
            cost = self.unified_metrics["totals"]["total_tokens"] * 0.00015 / 1000
            report += f"\n### **Estimated Cost**: ${cost:.4f} USD\n"
        
        report += f"""
---

## 📈 **PERFORMANCE METRICS**
- **Files per Second**: {self.unified_metrics["performance"].get("files_per_second", 0):.2f}
- **Chunks per Second**: {self.unified_metrics["performance"].get("chunks_per_second", 0):.1f}
- **Average Chunks per File**: {self.unified_metrics["totals"]["total_chunks"] / max(1, self.unified_metrics["totals"]["files_processed"]):.1f}

---

## 📄 **DETAILED FILE-LEVEL METRICS**

| # | File | Format | Size (KB) | Words | Chars | Chunks | Avg Size | Time (s) | Tokens | Status |
|---|------|--------|-----------|-------|-------|--------|----------|----------|--------|--------|
"""
        
        # Add file-level details with enhanced metrics
        for i, file_metrics in enumerate(self.unified_metrics["files"], 1):
            file_name = file_metrics["file_name"][:25] + "..." if len(file_metrics["file_name"]) > 25 else file_metrics["file_name"]
            status = "✅" if file_metrics["success"] else "❌"
            
            report += f"| {i} | {file_name} | {file_metrics['source_format']} | {file_metrics['file_size_kb']:.1f} | {file_metrics.get('word_count', 0):,} | {file_metrics.get('char_count', 0):,} | {file_metrics['chunks_created']} | {file_metrics.get('avg_chunk_size', 0):.0f} | {file_metrics['processing_time']:.3f} | {file_metrics['tokens_used']} | {status} |\n"
        
        report += f"""
---

## ❌ **ERROR LOG**
"""
        
        if self.unified_metrics["errors"]:
            for error in self.unified_metrics["errors"]:
                report += f"- **{os.path.basename(error['file_path'])}**: {error['error']}\n"
        else:
            report += "- No errors encountered ✅\n"
        
        report += f"""
---

## ✅ **COMPLETION STATUS**
- **Status**: ✅ Session Complete
- **Total Files Processed**: {self.unified_metrics["totals"]["files_processed"]:,}
- **Total Chunks Created**: {self.unified_metrics["totals"]["total_chunks"]:,}
- **Vector Database**: {"✅ Populated" if self.unified_metrics["totals"]["total_chunks_stored"] > 0 else "❌ Not used"}
- **Ready for Queries**: {"✅" if self.unified_metrics["totals"]["total_chunks_stored"] > 0 else "❌"}

---

## 📊 **DATA BREAKDOWN**
"""
        
        # Add format breakdown
        format_counts = {}
        for file_metrics in self.unified_metrics["files"]:
            fmt = file_metrics["source_format"]
            format_counts[fmt] = format_counts.get(fmt, 0) + 1
        
        report += "### **Files by Format**:\n"
        for fmt, count in sorted(format_counts.items()):
            report += f"- **{fmt}**: {count} files\n"
        
        report += f"""
---

*Generated by Unified Logger v1.0*  
*Session: {self.unified_metrics["session_info"]["session_id"]}*  
*Timestamp: {current_time}*
"""
        
        return report


# Global unified logger instance
_global_logger = None

def get_unified_logger() -> UnifiedLogger:
    """Get or create global unified logger instance."""
    global _global_logger
    if _global_logger is None:
        _global_logger = UnifiedLogger()
    return _global_logger

def finalize_unified_log(operation_name: str = "processing_session") -> str:
    """Finalize and save the unified log."""
    global _global_logger
    if _global_logger is not None:
        log_path = _global_logger.finalize_and_save(operation_name)
        _global_logger = None  # Reset for next session
        return log_path
    return None
