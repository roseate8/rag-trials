#!/usr/bin/env python3
"""
Small runner to test LayoutAwareChunker on a single document and print summary stats.
Run with: PYTHONPATH=. python3 rag_pipeline/src/advanced_chunkers/layout_aware_runner.py
"""
import logging
from typing import Tuple

from layout_aware_chunker import LayoutAwareChunker


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    file_path = \
        "/Users/rudram.piplad/Documents/parsing-pocs/parsing_tool/output/markdown/10-Q4-2024-As-Filed.md"

    chunker = LayoutAwareChunker(max_words=300)
    chunks, section_index = chunker.chunk_document(file_path=file_path, source_format="markdown")

    # Summaries
    total = len(chunks)
    by_type = {}
    for c in chunks:
        by_type[c["chunk_type"]] = by_type.get(c["chunk_type"], 0) + 1

    print("\n=== Layout-Aware Chunking Summary ===")
    print(f"Document: {file_path}")
    print(f"Total chunks: {total}")
    print("By type:")
    for k, v in sorted(by_type.items()):
        print(f"  - {k}: {v}")
    print(f"Total subsections indexed: {len(section_index)}")


if __name__ == "__main__":
    main()


