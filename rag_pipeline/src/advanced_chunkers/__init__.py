"""Advanced chunkers package (layout-aware, etc.)."""

from .layout_aware_chunker import LayoutAwareChunker  # noqa: F401

try:
    from .html_processor import HTMLProcessor  # noqa: F401
    __all__ = ['LayoutAwareChunker', 'HTMLProcessor']
except ImportError:
    # HTMLProcessor requires BeautifulSoup4, gracefully degrade if not available
    __all__ = ['LayoutAwareChunker']


