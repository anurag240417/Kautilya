"""Graph construction and analysis module.

Represents relationships as graph structures. Must load and validate
the four native Elliptic++ edgelists rather than reconstructing edges
from raw records.
"""

from .builder import ChainTraceGraph

__all__ = ["ChainTraceGraph"]
